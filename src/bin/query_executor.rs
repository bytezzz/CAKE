/*! Execute JSONL-defined queries using the Query DSL executor.
 */

use arrow_quiver::{
    archon::Archon, deserialize_queries_jsonl, execute_query, perf_manager::perf_pause,
    selection::sel_intersect::Intersector, ArrowQuiver, Query,
};
use clap::Parser;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long)]
    job_path: Option<PathBuf>,
    #[arg(short, long, help = "Run a single query by index (0-based)")]
    query: Option<usize>,
    #[arg(long, help = "Run all queries (default if --query omitted)")]
    run_all: bool,
    #[arg(long, default_value = "1")]
    n_times: u32,
    #[arg(long, default_value = "1")]
    n_epochs: u32,
    #[arg(long, help = "Shuffle query order before starting each experiment")]
    shuffle: bool,
    #[arg(long, default_value = "config.yml")]
    config_path: Option<PathBuf>,
    #[arg(
        long,
        help = "Path to queries JSONL (from --export-jsonl)",
        default_value = "experiment_data/end_to_end.queries.jsonl"
    )]
    queries_path: PathBuf,
}

use arrow_quiver::configuration::{
    get_job_path, get_oracle_policies_path, get_policy, init_config, PolicyType,
};
use arrow_quiver::oracle_machine::init_oracle_machine_from_path;

fn run_query(
    queries: &[Query],
    query_id: usize,
    job_path: &Path,
    intersector: &Intersector,
) -> (ArrowQuiver, u128) {
    match queries.get(query_id) {
        Some(query) => execute_query(job_path, intersector, query),
        None => panic!("Invalid query index: {} (len={})", query_id, queries.len()),
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct QueryResult {
    order_id: u32,
    time: u128,
}

#[derive(Debug, Serialize, Deserialize)]
struct ExperimentResult {
    times: Vec<Vec<QueryResult>>,
}

fn run_all_queries_with_order(
    queries: &[Query],
    job_path: &Path,
    intersector: &Intersector,
    order: &[usize],
) -> Vec<QueryResult> {
    let mut times: Vec<QueryResult> = vec![];
    for &qid in order {
        let (_, time) = run_query(queries, qid, job_path, intersector);
        times.push(QueryResult {
            order_id: qid as u32,
            time,
        });
    }
    times
}

fn run_all_queries_n_times(
    queries: &[Query],
    job_path: &Path,
    intersector: &Intersector,
    n_times: u32,
    n_epochs: u32,
    shuffle: bool,
) {
    let mut base_order: Vec<usize> = (0..queries.len()).collect();
    let mut times: Vec<Vec<QueryResult>> = vec![];
    for i in 0..n_times {
        println!("--- Running experiment {}/{} ---", i + 1, n_times);
        if shuffle {
            base_order.shuffle(&mut rand::thread_rng());
        }
        for j in 0..n_epochs {
            println!("--- Running epoch {}/{} ---", j + 1, n_epochs);
            let query_result =
                run_all_queries_with_order(queries, job_path, intersector, &base_order);
            let _: u128 = query_result.iter().map(|qr| qr.time).sum();
            times.push(query_result);
            Archon::signal_end_of_epoch().unwrap();
        }
        Archon::signal_end_of_experiment().unwrap();
    }
    println!(
        "Total time: {}",
        times
            .iter()
            .map(|qr| qr.iter().map(|qr| qr.time).sum::<u128>())
            .sum::<u128>()
    );
    let experiment_result = ExperimentResult { times };
    let experiment_result_json = serde_json::to_string(&experiment_result).unwrap();
    println!("{}", experiment_result_json);
}

fn main() {
    perf_pause();
    let args = Args::parse();
    // Initialize configuration
    if let Some(config_path) = args.config_path {
        init_config(&config_path).expect("Failed to initialize config");
    }
    // Initialize Archon after config
    Archon::new_embedded().unwrap();
    env_logger::init();

    // Initialize OracleMachine if configured
    if matches!(get_policy().ok(), Some(PolicyType::OracleMachine)) {
        let oracle_path =
            get_oracle_policies_path().expect("Failed to read oracle policies path from config");
        let oracle_path = oracle_path
            .expect("oracle_policies_path must be set in config when policy=oracle_machine");
        init_oracle_machine_from_path(Path::new(oracle_path))
            .expect("Failed to initialize OracleMachine from path");
        println!("OracleMachine initialized from {}", oracle_path);
    }

    // Load queries from JSONL
    let queries = deserialize_queries_jsonl(&args.queries_path)
        .expect("Failed to deserialize queries from JSONL");
    println!(
        "Loaded {} queries from {}",
        queries.len(),
        args.queries_path.display()
    );

    // Resolve job path
    let job_path = match args.job_path {
        Some(path) => path,
        None => {
            let job_path = get_job_path().expect("Failed to get job_path from config");
            PathBuf::from(job_path)
        }
    };
    let intersector = Intersector::basic();

    if let Some(query_id) = args.query {
        for _ in 0..args.n_times {
            let _ = run_query(&queries, query_id, &job_path, &intersector);
            Archon::signal_end_of_epoch().unwrap();
        }
    } else {
        run_all_queries_n_times(
            &queries,
            &job_path,
            &intersector,
            args.n_times,
            args.n_epochs,
            args.shuffle,
        );
    }
}
