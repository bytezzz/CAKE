use argminmax::ArgMinMax;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub fn current_cpu_brand_string() -> Option<String> {
    raw_cpuid::CpuId::new()
        .get_processor_brand_string()
        .map(|s| s.as_str().to_string())
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub fn current_cpu_brand_string() -> Option<String> {
    None
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RegretTree {
    num_features: usize,
    cpu: String,
    root: RegretTreeNode,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum RegretTreeNode {
    SplitNode {
        feature: usize,
        pt: f64,
        left: Box<RegretTreeNode>,
        right: Box<RegretTreeNode>,
    },
    LeafNode {
        policy: usize,
    },
}

// Data Format: [Vec<f64] : {1st_feature, 2nd_feature, ...., 1st_policy_regret,....nth_policy_regret}
impl RegretTree {
    pub fn stub(policy: usize) -> RegretTree {
        RegretTree {
            num_features: 0,
            cpu: String::from("stub RT"),
            root: RegretTreeNode::LeafNode { policy },
        }
    }

    pub fn fit(data: &mut [Vec<f64>], num_features: usize, max_depth: usize) -> RegretTree {
        let (best_single, _regret) = best_single_policy_regret(data, num_features);
        let cpu = current_cpu_brand_string().unwrap_or_else(|| "unknown cpu".to_string());
        RegretTree {
            num_features,
            cpu,
            root: induce_regret_tree(data, best_single, num_features, max_depth),
        }
    }

    pub fn max_depth(&self) -> usize {
        self.root.max_depth()
    }

    pub fn is_stub(&self) -> Option<usize> {
        match self.root {
            RegretTreeNode::SplitNode { .. } => None,
            RegretTreeNode::LeafNode { policy } => Some(policy),
        }
    }

    pub fn predict(&self, data: &[f64]) -> usize {
        self.root.predict(data)
    }

    pub fn regret(&self, data: &[Vec<f64>]) -> f64 {
        let mut total_regret = 0.0;
        for el in data {
            let policy_rewards = &el[self.num_features..];
            let best_reward = policy_rewards[policy_rewards.argmin()];

            let selected = self.predict(el);
            total_regret += el[self.num_features + selected] - best_reward;
        }
        total_regret
    }

    pub fn cpu_trained_for(&self) -> &str {
        &self.cpu
    }
}

impl RegretTreeNode {
    fn max_depth(&self) -> usize {
        match self {
            RegretTreeNode::SplitNode { left, right, .. } => {
                usize::max(left.max_depth(), right.max_depth()) + 1
            }
            RegretTreeNode::LeafNode { .. } => 0,
        }
    }

    fn predict(&self, data: &[f64]) -> usize {
        match self {
            RegretTreeNode::SplitNode {
                feature,
                pt,
                left,
                right,
            } => {
                if data[*feature] < *pt {
                    left.predict(data)
                } else {
                    right.predict(data)
                }
            }
            RegretTreeNode::LeafNode { policy } => *policy,
        }
    }
}

pub fn best_single_policy_regret(data: &[Vec<f64>], num_features: usize) -> (usize, f64) {
    // accum[i] is the total regret of given dataset when selecting the ith policy
    let mut accum = vec![0.0; data[0].len() - num_features];
    for row in data {
        let policy_rewards = &row[num_features..];
        let best = policy_rewards[policy_rewards.argmin()];
        for (reward, total_regret) in policy_rewards.iter().zip(accum.iter_mut()) {
            *total_regret += *reward - best;
        }
    }

    let best_policy = accum.argmin();
    (best_policy, accum[best_policy])
}

fn find_best_split(
    data: &mut [Vec<f64>],
    num_features: usize,
) -> Option<(usize, usize, usize, usize, f64)> {
    let mut best_curr_feat = 0;
    let mut best_curr_pt_idx = 0;
    let mut best_curr_regret = f64::INFINITY;
    let mut best_left = 0;
    let mut best_right = 0;

    for ft_idx in 0..num_features {
        data.sort_by_key(|k| OrderedFloat::from(k[ft_idx]));
        for idx in 1..data.len() {
            if data[idx][ft_idx] == data[idx - 1][ft_idx] {
                // cannot split here because the previous sample has the same feature
                // value
                continue;
            }
            let (l_policy, l_regret) = best_single_policy_regret(&data[..idx], num_features);
            let (r_policy, r_regret) = best_single_policy_regret(&data[idx..], num_features);
            let split_regret = l_regret + r_regret;

            if split_regret < best_curr_regret {
                best_curr_regret = split_regret;
                best_curr_feat = ft_idx;
                best_curr_pt_idx = idx;
                best_left = l_policy;
                best_right = r_policy;
            }
        }
    }

    if best_curr_regret < f64::INFINITY && best_left != best_right {
        Some((
            best_curr_feat,
            best_left,
            best_right,
            best_curr_pt_idx,
            best_curr_regret,
        ))
    } else {
        None
    }
}

fn induce_regret_tree(
    data: &mut [Vec<f64>],
    best_policy: usize,
    num_features: usize,
    max_depth: usize,
) -> RegretTreeNode {
    if max_depth == 0 {
        return RegretTreeNode::LeafNode {
            policy: best_policy,
        };
    }

    if let Some((feat, lp, rp, idx_pt, _regret)) = find_best_split(data, num_features) {
        data.sort_by_key(|k| OrderedFloat::from(k[feat]));
        let pt = data[idx_pt][feat];
        let left = induce_regret_tree(&mut data[..idx_pt], lp, num_features, max_depth - 1);
        let right = induce_regret_tree(&mut data[idx_pt..], rp, num_features, max_depth - 1);
        return RegretTreeNode::SplitNode {
            feature: feat,
            pt,
            left: Box::new(left),
            right: Box::new(right),
        };
    }
    RegretTreeNode::LeafNode {
        policy: best_policy,
    }
}

#[cfg(test)]
mod tests {
    use super::{best_single_policy_regret, find_best_split, RegretTree};

    #[test]
    fn test_best_single_policy() {
        let data = vec![
            vec![0.0, 1.0, 2.0],
            vec![0.0, 1.0, 3.0],
            vec![0.0, 90.0, 5.0],
        ];
        assert_eq!(best_single_policy_regret(&data, 1), (1, 3.0));

        let data = vec![
            vec![0.0, 1.0, 2.0],
            vec![0.0, 1.0, 3.0],
            vec![0.0, 2.0, 5.0],
        ];
        assert_eq!(best_single_policy_regret(&data, 1), (0, 0.0));
    }

    #[test]
    fn test_find_split_two() {
        let mut data = vec![vec![0.0, 1.0, 2.0], vec![1.0, 2.0, 1.0]];

        let (f, l, r, pt, reg) = find_best_split(&mut data, 1).unwrap();
        assert_eq!(f, 0);
        assert_eq!(l, 0);
        assert_eq!(r, 1);
        assert_eq!(pt, 1);
        assert_eq!(reg, 0.0);
    }
    #[test]
    fn test_find_split_duplicates() {
        let mut data = vec![
            vec![0.0, 1.0, 2.0],
            vec![0.0, 1.0, 2.0],
            vec![1.0, 1.0, 2.0],
            vec![1.0, 20.0, 1.0],
        ];

        let (f, l, r, pt, reg) = find_best_split(&mut data, 1).unwrap();
        assert_eq!(f, 0);
        assert!(
            l != r,
            "both the left and right policies were the same ({})",
            l
        );
        assert_eq!(pt, 2);
        assert_eq!(reg, 1.0);
    }

    #[test]
    fn test_find_split() {
        let mut data = vec![
            vec![0.0, 1.0, 2.0],
            vec![0.0, 1.0, 3.0],
            vec![2.0, 90.0, 5.0],
            vec![3.0, 95.0, 14.0],
        ];

        let (f, l, r, pt, reg) = find_best_split(&mut data, 1).unwrap();
        assert_eq!(f, 0);
        assert_eq!(l, 0);
        assert_eq!(r, 1);
        assert_eq!(pt, 2);
        assert_eq!(reg, 0.0);

        let mut data = vec![
            vec![0.0, 1.0, 2.0],
            vec![0.0, 1.0, 3.0],
            vec![2.0, 90.0, 5.0],
            vec![3.0, 95.0, 96.0],
        ];

        let (f, l, r, pt, reg) = find_best_split(&mut data, 1).unwrap();
        assert_eq!(f, 0);
        assert_eq!(l, 0);
        assert_eq!(r, 1);
        assert_eq!(pt, 2);
        assert_eq!(reg, 1.0);
    }

    #[test]
    fn test_induce_tree() {
        let mut data = vec![
            vec![0.0, 1.0, 10.0, 20.0],
            vec![0.0, 2.0, 11.0, 21.0],
            vec![1.0, 1.0, 12.0, 22.0],
            vec![1.0, 2.0, 80.0, 24.0],
        ];

        let tree = RegretTree::fit(&mut data, 2, 9);
        assert_eq!(tree.max_depth(), 2);
        assert_eq!(tree.predict(&[0.0, 1.0]), 0);
        assert_eq!(tree.predict(&[1.0, 3.0]), 1);
    }
}
