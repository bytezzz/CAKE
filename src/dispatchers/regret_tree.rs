use argminmax::ArgMinMax;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RegretTree {
    num_features: usize,
    root: RegretTreeNode,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum RegretTreeNode {
    SplitNode {
        feature: usize,
        threshold: f32,
        left: Box<RegretTreeNode>,
        right: Box<RegretTreeNode>,
    },
    LeafNode {
        policy: usize,
    },
}

impl RegretTree {
    /// Creates a "stub" tree that always returns the same policy.
    pub fn stub(policy: usize) -> Self {
        Self {
            num_features: 0,
            root: RegretTreeNode::LeafNode { policy },
        }
    }

    /// Trains a new `RegretTree` from the given data.
    ///
    /// # Arguments
    ///
    /// * `data`: A 2D array where each row is a sample. The format for each row is
    ///           `[feature_1, ..., feature_n, reward_1, ..., reward_m]`.
    /// * `num_features`: The number of feature columns in the data.
    /// * `max_depth`: The maximum depth of the tree.
    pub fn fit(mut data: Array2<f32>, num_features: usize, max_depth: usize) -> Self {
        let (best_policy, _) = best_single_policy_regret(data.view(), num_features);
        let root = induce_regret_tree(&mut data, best_policy, num_features, max_depth);
        Self { num_features, root }
    }

    /// Predicts the best policy for a given feature vector.
    pub fn predict(&self, features: ArrayView1<f32>) -> usize {
        self.root.predict(features)
    }

    /// Checks if the tree is a stub (i.e., just a single leaf node).
    pub fn is_stub(&self) -> bool {
        matches!(self.root, RegretTreeNode::LeafNode { .. })
    }
}

impl RegretTreeNode {
    fn predict(&self, features: ArrayView1<f32>) -> usize {
        match self {
            RegretTreeNode::SplitNode {
                feature,
                threshold,
                left,
                right,
            } => {
                if features[*feature] < *threshold {
                    left.predict(features)
                } else {
                    right.predict(features)
                }
            }
            RegretTreeNode::LeafNode { policy } => *policy,
        }
    }
}

/// For a given dataset, finds the single best policy and its total regret.
/// The total regret is the sum of `(chosen_policy_reward - best_possible_reward)` for each row.
fn best_single_policy_regret(data: ArrayView2<f32>, num_features: usize) -> (usize, f32) {
    if data.is_empty() {
        return (0, 0.0);
    }
    let rewards = data.slice(s![.., num_features..]);
    let best_rewards_per_row = rewards
        .map_axis(Axis(1), |row| {
            if row.len() == 0 {
                0.0
            } else {
                let best_idx = row.argmin();
                row[best_idx]
            }
        })
        .insert_axis(Axis(1));

    let regrets = &rewards - &best_rewards_per_row;
    let total_regrets_per_policy: Array1<f32> = regrets.sum_axis(Axis(0));

    if total_regrets_per_policy.is_empty() {
        return (0, 0.0);
    }
    let best_policy = total_regrets_per_policy.argmin();
    (best_policy, total_regrets_per_policy[best_policy])
}

/// Finds the best feature and threshold to split the data on.
fn find_best_split(
    data: &Array2<f32>,
    num_features: usize,
) -> Option<(usize, f32, usize, usize, Array1<usize>)> {
    let mut best_regret = f32::INFINITY;
    let mut best_split = None;

    for ft_idx in 0..num_features {
        let feature_col = data.column(ft_idx);
        let mut sorted_indices: Vec<usize> = (0..feature_col.len()).collect();
        // Sort indices based on the feature column
        sorted_indices
            .sort_unstable_by(|&a, &b| feature_col[a].partial_cmp(&feature_col[b]).unwrap());

        for i in 1..data.nrows() {
            let idx_curr = sorted_indices[i];
            let idx_prev = sorted_indices[i - 1];
            if data[[idx_curr, ft_idx]] == data[[idx_prev, ft_idx]] {
                continue; // Cannot split on identical values
            }

            // A more efficient way would be to update regrets incrementally
            // instead of recomputing from scratch for each split point.
            // For now, clarity is prioritized.
            let left_indices: Vec<usize> = sorted_indices.iter().take(i).cloned().collect();
            let right_indices: Vec<usize> = sorted_indices.iter().skip(i).cloned().collect();

            let left_data = data.select(Axis(0), &left_indices);
            let right_data = data.select(Axis(0), &right_indices);

            let (l_policy, l_regret) = best_single_policy_regret(left_data.view(), num_features);
            let (r_policy, r_regret) = best_single_policy_regret(right_data.view(), num_features);

            if l_policy == r_policy {
                continue; // Split is not useful if policies are the same
            }

            let total_regret = l_regret + r_regret;
            if total_regret < best_regret {
                best_regret = total_regret;
                let threshold = (data[[idx_curr, ft_idx]] + data[[idx_prev, ft_idx]]) / 2.0;
                best_split = Some((
                    ft_idx,
                    threshold,
                    l_policy,
                    r_policy,
                    sorted_indices.iter().cloned().collect(),
                ));
            }
        }
    }
    best_split
}

/// Recursively builds the regret tree.
fn induce_regret_tree(
    data: &mut Array2<f32>,
    best_policy: usize,
    num_features: usize,
    max_depth: usize,
) -> RegretTreeNode {
    if max_depth == 0 || data.nrows() < 2 {
        return RegretTreeNode::LeafNode {
            policy: best_policy,
        };
    }

    if let Some((feat, threshold, l_policy, r_policy, sorted_indices)) =
        find_best_split(data, num_features)
    {
        // Find the split index
        let split_idx = sorted_indices
            .iter()
            .position(|&idx| data[[idx, feat]] >= threshold)
            .unwrap_or(data.nrows());

        // Create partitioned data views without copying the underlying data
        let left_indices_view = sorted_indices.slice(s![..split_idx]);
        let right_indices_view = sorted_indices.slice(s![split_idx..]);
        let mut left_data = data.select(Axis(0), left_indices_view.as_slice().unwrap());
        let mut right_data = data.select(Axis(0), right_indices_view.as_slice().unwrap());

        let left_node = induce_regret_tree(&mut left_data, l_policy, num_features, max_depth - 1);
        let right_node = induce_regret_tree(&mut right_data, r_policy, num_features, max_depth - 1);

        RegretTreeNode::SplitNode {
            feature: feat,
            threshold,
            left: Box::new(left_node),
            right: Box::new(right_node),
        }
    } else {
        RegretTreeNode::LeafNode {
            policy: best_policy,
        }
    }
}
