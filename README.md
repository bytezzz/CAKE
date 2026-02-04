# Official Implementation of the Paper "Piece of CAKE: Adaptive Execution Engines via Microsecond-Scale Learning"

**Overview**
CAKE (Counterfactual Adaptive Kernel Execution) learns to select the optimal kernel for each data "morsel" using a microsecond-scale contextual multi-armed bandit algorithm. This project is built upon [@RyanMarcus](https://github.com/RyanMarcus)'s project Arrow-quiver.

**Quick Start**
1. Download a dataset and extract it locally.
2. Edit `config.yml`, set `job_path` to the dataset root, and choose `policy` and hyperparameters.
3. Build and run:

```bash
cargo build --release
./target/release/query_executor --config-path config.yml --queries-path queries/imdb.jsonl
```

Or run via cargo:

```bash
cargo run --release --bin query_executor -- --config-path config.yml --queries-path queries/imdb.jsonl
```

**Datasets**
- [join-order-benchmark](https://huggingface.co/datasets/Veweew/join-order-benchmark)
- [dsb_parquet](https://huggingface.co/datasets/Veweew/dsb_parquet)
- [stack-parquet](https://huggingface.co/datasets/Veweew/stack-parquet)

`job_path` points to the dataset root. For join-order-benchmark, you need to run `join.py` first to pre-join tables as our system does not support joins at the moment.

**Build and Run**
- Build: `cargo build --release`
- Execute: `./target/release/query_executor --config-path config.yml --queries-path queries/imdb.jsonl`
- Run a single query (0-based): `--query 0`
- Repeat and epoch control flags: `--n-times 3 --n-epochs 2`

**Outputs**
- The console prints execution times for each query.
- At the end, it prints a total time and a JSON payload like `{"times":[[{"order_id":0,"time":123}]]}` where `time` is in nanoseconds.
- Reported times include dataset loading, chunking, and merging, which are not part of pure query processing. Remove those phases or exclude their time costs if you need more accurate query-processing time.
