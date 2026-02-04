use std::{fs::File, io::BufReader};

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use std::arch::x86_64::{
    __m256i, _mm256_cmpeq_epi32, _mm256_loadu_si256, _mm256_set1_epi32, _mm256_testz_si256,
};

use cpu_time::ProcessTime;
use indicatif::ProgressBar;
use itertools::Itertools;
use log::warn;
use ordered_float::OrderedFloat;
use serde::Serialize;
use serde_json::json;

use crate::regret_tree::{best_single_policy_regret, current_cpu_brand_string, RegretTree};

pub fn should_train_intersection() -> Result<RegretTree, String> {
    match File::open("intersection_model.json") {
        Ok(f) => {
            let br = BufReader::new(f);
            match serde_json::from_reader::<BufReader<File>, RegretTree>(br) {
                Ok(rt) => match current_cpu_brand_string() {
                    Some(runtime_cpu) if runtime_cpu != rt.cpu_trained_for() => Err(format!(
                        "Existing model was trained for CPU {}, but we are running on CPU {}",
                        rt.cpu_trained_for(),
                        runtime_cpu
                    )),
                    _ => Ok(rt),
                },
                Err(e) => Err(format!("unable to load existing model file: {}", e)),
            }
        }
        Err(e) => Err(format!("unable to read existing model file: {}", e)),
    }
}

#[derive(Debug, Serialize)]
pub struct IntersectionFeatures {
    len_ratio: f64,
    est: f64,
    sample_est: f64,
}

pub fn extract_features(v1: &[u32], v2: &[u32]) -> IntersectionFeatures {
    let lb = u32::max(v1[0], v2[0]) as f64;
    let ub = u32::min(*v1.last().unwrap(), *v2.last().unwrap()) as f64;

    let est = if ub < lb {
        0.0
    } else {
        f64::min(
            (ub - lb) / (*v1.last().unwrap() as f64 - v1[0] as f64),
            (ub - lb) / (*v2.last().unwrap() as f64 - v2[0] as f64),
        )
    };

    let sample_est = if v1.len() > 10 {
        if v2.len() > 10 {
            count_hits(&v1[0..10], &v2[0..10])
        } else {
            count_hits(&v1[0..10], v2)
        }
    } else if v2.len() > 10 {
        count_hits(v1, &v2[0..10])
    } else {
        count_hits(v1, v2)
    };

    IntersectionFeatures {
        len_ratio: v2.len() as f64 / v1.len() as f64,
        est,
        sample_est: sample_est as f64 / usize::max(v1.len(), v2.len()) as f64,
    }
}

pub fn train_model_if_needed() {
    #[derive(Debug, Serialize)]
    struct IntersectionLabels {
        linear: u128,
        binary: u128,
        mutual_partition: u128,
        avx2: u128,
    }

    match should_train_intersection() {
        Err(reason) => println!("Retraining intersection model because: {}", reason),
        Ok(_rt) => {
            return;
        }
    };

    let out = File::create("intersection.json").unwrap();
    let tree_out = File::create("intersection_model.json").unwrap();
    let mut x = Vec::new();
    let mut y = Vec::new();

    let bar = ProgressBar::new(1000);
    let v1 = (0..1_000_000).collect_vec();

    for _ in 0..1000 {
        bar.inc(1);
        for num_els in [
            10, 100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000,
        ] {
            let max_range = fastrand::u32(0..1_000_000);
            let v2 = (0..num_els)
                .map(|_| fastrand::u32(0..max_range))
                .sorted()
                .dedup()
                .collect_vec();

            let mut t_v1 = v1.clone();
            let start = ProcessTime::now();
            linear_intersect(&mut t_v1, &v2);
            let linear_intersect_time = start.elapsed().as_nanos();

            let mut t_v1 = v1.clone();
            let t_v2 = v2.clone();
            let start = ProcessTime::now();
            linear_intersect_avx2(&mut t_v1, &t_v2);
            let avx2_time = start.elapsed().as_nanos();

            let mut t_v2 = v2.clone();
            let start = ProcessTime::now();
            binary_probe(&mut t_v2, &v1);
            let binary_probe_time = start.elapsed().as_nanos();

            let start = ProcessTime::now();
            mutual_partition(&v2, &v1);
            let mutual_partition_time = start.elapsed().as_nanos();

            x.push(extract_features(&v1, &v2));
            y.push(IntersectionLabels {
                linear: linear_intersect_time,
                binary: binary_probe_time,
                mutual_partition: mutual_partition_time,
                avx2: avx2_time,
            });
        }
    }
    bar.finish();

    let mut all_data: Vec<Vec<f64>> = x
        .iter()
        .zip(y.iter())
        .map(|(x, y)| {
            vec![
                x.len_ratio,
                x.sample_est,
                x.est,
                y.linear as f64,
                y.avx2 as f64,
                y.mutual_partition as f64,
                y.binary as f64,
            ]
        })
        .collect_vec();

    fastrand::shuffle(&mut all_data);

    let train_size = (all_data.len() as f64 * 0.75).floor() as usize;
    let (train, test) = all_data.split_at_mut(train_size);

    println!(
        "Training samples: {}, test samples: {}",
        train.len(),
        test.len()
    );

    // sweep depths 1 through 8 to find the best on the test set
    let (regret, best_tree) = (1..8)
        .map(|depth| {
            let tree = RegretTree::fit(train, 3, depth);
            let regret = tree.regret(test);
            println!(
                "(depth = {}) RT regret (over test data): {:?}",
                depth, regret
            );
            (regret, tree)
        })
        .min_by_key(|(regret, _tree)| OrderedFloat::from(*regret))
        .unwrap();

    println!(
        "Best regret at depth = {}, regret: {}",
        best_tree.max_depth(),
        regret
    );
    println!(
        "Best single-policy regret:        {}",
        best_single_policy_regret(test, 3).1
    );

    let results = json!({"x": x, "y": y});
    serde_json::to_writer(out, &results).unwrap();
    serde_json::to_writer(tree_out, &best_tree).unwrap();
}

pub struct Intersector {
    policy: RegretTree,
}

impl Default for Intersector {
    fn default() -> Self {
        Self::new()
    }
}

impl Intersector {
    pub fn basic() -> Intersector {
        Intersector {
            policy: RegretTree::stub(0),
        }
    }

    pub fn new() -> Intersector {
        match should_train_intersection() {
            Ok(rt) => Intersector { policy: rt },
            Err(err) => {
                warn!(
                    "Could not load intersection model, consider training one. Reason: {}",
                    err
                );
                Intersector {
                    policy: RegretTree::stub(0),
                }
            }
        }
    }

    pub fn intersect(&self, mut v1: Vec<u32>, mut v2: Vec<u32>) -> Vec<u32> {
        if v2.len() > v1.len() {
            std::mem::swap(&mut v1, &mut v2);
        }

        let choice = self.policy.is_stub().unwrap_or_else(|| {
            let features = extract_features(&v1, &v2);
            self.policy
                .predict(&[features.len_ratio, features.sample_est, features.est])
        });

        match choice {
            0 => linear_intersect(&mut v1, &v2),
            1 => linear_intersect_avx2(&mut v1, &v2),
            2 => v1 = mutual_partition(&v2, &v1),
            3 => binary_probe(&mut v2, &v1),
            _ => {
                warn!(
                    "Policy returned invalid intersection policy {}, using linear intersection.\
                    Consider retraining the intersection model.",
                    choice
                );
                linear_intersect(&mut v1, &v2)
            }
        }

        v1
    }
}

fn count_hits(v1: &[u32], v2: &[u32]) -> usize {
    let mut out = 0;
    let mut v1_idx = 0;
    let mut v2_idx = 0;

    while v1_idx < v1.len() && v2_idx < v2.len() {
        match v1[v1_idx].cmp(&v2[v2_idx]) {
            std::cmp::Ordering::Less => v1_idx += 1,
            std::cmp::Ordering::Greater => v2_idx += 1,
            std::cmp::Ordering::Equal => {
                out += 1;
                v1_idx += 1;
                v2_idx += 1;
            }
        }
    }

    out
}

pub fn linear_intersect(v1: &mut Vec<u32>, v2: &[u32]) {
    let mut out_idx = 0;

    let mut v1_idx = 0;
    let mut v2_idx = 0;

    while v1_idx < v1.len() && v2_idx < v2.len() {
        // progress v1_idx until we get a candidate
        while v1_idx < v1.len() && v1[v1_idx] < v2[v2_idx] {
            v1_idx += 1;
        }

        if v1_idx >= v1.len() {
            break;
        }

        // progress v2_idx until we get a candidate
        while v2_idx < v2.len() && v1[v1_idx] > v2[v2_idx] {
            v2_idx += 1;
        }

        if v2_idx >= v2.len() {
            break;
        }

        // check for a match
        if v1[v1_idx] == v2[v2_idx] {
            v1[out_idx] = v1[v1_idx];
            out_idx += 1;
            v1_idx += 1;
            v2_idx += 1;
        }
    }

    v1.truncate(out_idx);
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub fn linear_intersect_avx2(v1: &mut Vec<u32>, v2: &[u32]) {
    assert!(
        v1.len() >= v2.len(),
        "first argument should be the larger vec"
    );

    if v1.len() < 8 {
        linear_intersect(v1, v2);
        return;
    }

    if v2.is_empty() {
        v1.clear();
        return;
    }

    v1.reserve_exact(v1.len() % 8);
    let sentinel = match u32::max(*v1.last().unwrap(), *v2.last().unwrap()).checked_add(1) {
        Some(v) => v,
        None => {
            linear_intersect(v1, v2);
            return;
        }
    };

    // pad with our sentinel value
    while v1.len() % 8 != 0 {
        v1.push(sentinel);
    }

    if v1.len() < v2.len() {
        // the padding made v1 longer than v2
        linear_intersect(v1, v2);
        return;
    }

    unsafe {
        div_linear_intersect_avx2(v1, v2);
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn div_linear_intersect_avx2(v1: &mut Vec<u32>, v2: &[u32]) {
    assert_eq!(v1.len() % 8, 0);
    let mut out_idx = 0;
    let mut v2_idx = 0;

    'outer: for chunk_start in (0..v1.len()).step_by(8) {
        let chunk = _mm256_loadu_si256(v1.as_ptr().add(chunk_start) as *const __m256i);

        loop {
            if v1[chunk_start + 7] < v2[v2_idx] {
                break;
            }

            let constant = _mm256_set1_epi32(i32::from_le_bytes(v2[v2_idx].to_le_bytes()));
            let eq = _mm256_cmpeq_epi32(chunk, constant);
            if _mm256_testz_si256(eq, eq) != 1 {
                // was not all zeros, which means the value exists
                v1[out_idx] = v2[v2_idx];
                out_idx += 1;
            }

            v2_idx += 1;

            if v2_idx >= v2.len() {
                break 'outer;
            }
        }
    }

    v1.truncate(out_idx);
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
pub fn linear_intersect_avx2(v1: &mut Vec<u32>, v2: &[u32]) {
    linear_intersect(v1, v2);
}

pub fn mutual_partition(small: &[u32], large: &[u32]) -> Vec<u32> {
    let mut out = vec![0; usize::min(small.len(), large.len())];
    let out_num = mutual_partition_recur(small, large, &mut out);
    out.truncate(out_num);
    out
}

fn mutual_partition_recur(small: &[u32], large: &[u32], out: &mut [u32]) -> usize {
    // Translated from D Lemire's code here https://github.com/lemire/SIMDCompressionAndIntersection/blob/98c0eb7271cb504b496a498fb0768d6a154911fe/src/intersection.cpp#L99

    if small.is_empty() || large.is_empty() {
        return 0;
    }
    if small.len() > large.len() {
        return mutual_partition_recur(large, small, out);
    }

    let mut mid_idx = small.len() / 2;
    let mid_val = small[mid_idx];

    let mut pos_in_large = large
        .binary_search_by(|el| match el.cmp(&mid_val) {
            std::cmp::Ordering::Equal => std::cmp::Ordering::Greater,
            ord => ord,
        })
        .unwrap_err();

    let mut out_num = mutual_partition_recur(&small[..mid_idx], &large[..pos_in_large], out);
    if pos_in_large >= large.len() {
        return out_num;
    }

    if large[pos_in_large] == mid_val {
        out[out_num] = mid_val;
        out_num += 1;
        pos_in_large += 1;
    }

    mid_idx += 1;
    out_num
        + mutual_partition_recur(
            &small[mid_idx..],
            &large[pos_in_large..],
            &mut out[out_num..],
        )
}

pub fn binary_probe(v1: &mut Vec<u32>, v2: &[u32]) {
    assert!(
        v1.len() <= v2.len(),
        "first argument should be the smaller vector"
    );
    let mut out_idx = 0;
    let mut v2_lb = 0;
    for v1_idx in 0..v1.len() {
        if v2_lb >= v2.len() {
            break;
        }
        match v2[v2_lb..].binary_search(&v1[v1_idx]) {
            Ok(idx) => {
                v1[out_idx] = v1[v1_idx];
                out_idx += 1;
                v2_lb = idx;
            }
            Err(idx) => {
                v2_lb = idx;
            }
        }
    }

    v1.truncate(out_idx);
}

#[cfg(test)]
mod tests {

    use super::*;

    use super::linear_intersect;

    #[test]
    fn test_count_intersect() {
        let v1 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let v2 = vec![0, 4, 8, 15];

        assert_eq!(count_hits(&v1, &v2), 2);
    }

    #[test]
    fn test_linear_intersect() {
        let mut v1 = vec![1, 2, 3, 4, 5, 6];
        let v2 = vec![4, 5, 6, 7, 8, 9];
        linear_intersect(&mut v1, &v2);
        assert_eq!(v1, vec![4, 5, 6]);

        let mut v1 = vec![1, 2, 3, 4, 5, 6];
        let v2 = vec![7, 8, 9];
        linear_intersect(&mut v1, &v2);
        assert_eq!(v1, Vec::<u32>::new());

        let mut v1 = vec![];
        let v2 = vec![7, 8, 9];
        linear_intersect(&mut v1, &v2);
        assert_eq!(v1, Vec::<u32>::new());
    }

    #[test]
    fn test_mutual_partition_intersect() {
        let v1 = vec![1, 2, 3, 4, 5, 6];
        let v2 = vec![4, 5, 6, 7, 8, 9];
        let r = mutual_partition(&v1, &v2);
        assert_eq!(r, vec![4, 5, 6]);

        let v1 = vec![1, 2, 3, 4, 5, 6];
        let v2 = vec![7, 8, 9];
        let r = mutual_partition(&v1, &v2);
        assert_eq!(r, Vec::<u32>::new());

        let v1 = vec![];
        let v2 = vec![7, 8, 9];
        let r = mutual_partition(&v1, &v2);
        assert_eq!(r, Vec::<u32>::new());
    }

    #[test]
    fn test_binary_probe() {
        let mut v1 = vec![1, 2, 3, 4, 5, 6];
        let v2 = vec![4, 5, 6, 7, 8, 9];
        binary_probe(&mut v1, &v2);
        assert_eq!(v1, vec![4, 5, 6]);

        let v1 = vec![1, 2, 3, 4, 5, 6];
        let mut v2 = vec![7, 8, 9];
        binary_probe(&mut v2, &v1);
        assert_eq!(v2, Vec::<u32>::new());

        let mut v1 = vec![];
        let v2 = vec![7, 8, 9];
        binary_probe(&mut v1, &v2);
        assert_eq!(v1, Vec::<u32>::new());
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn test_linear_intersect_avx2() {
        let mut v1 = (0..18).collect_vec();
        let v2 = vec![4, 8, 11, 12, 22];
        linear_intersect_avx2(&mut v1, &v2);
        assert_eq!(v1, vec![4, 8, 11, 12]);
    }
}
