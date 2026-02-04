use crate::dispatchers::ModelDecision;

use super::{DecisionModel, ExplorationRequest, Heuristic, SmartWareConfiguration};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct UcbState {
    num_policies: usize,
    total_pulls: u64,
    counts: Vec<u64>,
    sum_costs: Vec<f64>,
    /// Exploration scale. For UCB1 LCB on cost: mean - c * sqrt(2 ln t / n)
    c: f64,
}

impl UcbState {
    fn new(num_policies: usize, c: f64) -> Self {
        Self {
            num_policies,
            total_pulls: 0,
            counts: vec![0; num_policies],
            sum_costs: vec![0.0; num_policies],
            c,
        }
    }

    #[inline]
    fn has_unseen(&self) -> bool {
        self.counts.iter().any(|&n| n == 0)
    }

    fn means(&self) -> Vec<f64> {
        (0..self.num_policies)
            .map(|j| {
                if self.counts[j] > 0 {
                    self.sum_costs[j] / self.counts[j] as f64
                } else {
                    f64::INFINITY
                }
            })
            .collect()
    }

    fn lcbs(&self) -> Vec<f64> {
        // Lower confidence bound for cost minimization
        let t = self.total_pulls.max(1) as f64;
        (0..self.num_policies)
            .map(|j| {
                if self.counts[j] == 0 {
                    // Encourage at least one pull per arm
                    f64::NEG_INFINITY
                } else {
                    let mean = self.sum_costs[j] / self.counts[j] as f64;
                    let bonus = (2.0 * (t.ln()) / self.counts[j] as f64).sqrt();
                    mean - self.c * bonus
                }
            })
            .collect()
    }

    fn observe_single(&mut self, policy: usize, cost: f64) {
        self.total_pulls += 1;
        self.counts[policy] += 1;
        self.sum_costs[policy] += cost;
    }

    // Standard UCB observes only the selected arm per round.
}

pub struct NonContextualUcbSmartWare<H: Heuristic> {
    #[allow(dead_code)]
    config: SmartWareConfiguration,
    heuristic: H,
    state: UcbState,
    pending: Option<ModelDecision>,
}

impl<H: Heuristic> NonContextualUcbSmartWare<H> {
    pub fn new(heuristic: H, config: SmartWareConfiguration) -> Self {
        // Use `ucb.alpha` as the exploration scale for UCB.
        let c = config.ucb_params().alpha;
        let state = UcbState::new(config.num_policies, c);
        Self {
            config,
            heuristic,
            state,
            pending: None,
        }
    }

    fn select_policy(&self) -> (usize, usize, f64, f64) {
        let means = self.state.means();
        let lcbs = self.state.lcbs();

        // predictive (best mean among seen arms)
        let mut any_seen = false;
        let mut best_mean_idx = 0usize;
        let mut best_mean_val = f64::MAX;
        for j in 0..means.len() {
            if self.state.counts[j] > 0 {
                any_seen = true;
                if means[j] < best_mean_val {
                    best_mean_val = means[j];
                    best_mean_idx = j;
                }
            }
        }
        if !any_seen {
            best_mean_val = 0.0; // avoid Infinity in logs/serialization
        }

        // optimistic (best LCB)
        let mut best_lcb_idx = 0usize;
        let mut best_lcb_val = lcbs[0];
        for j in 1..lcbs.len() {
            if lcbs[j] < best_lcb_val {
                best_lcb_val = lcbs[j];
                best_lcb_idx = j;
            }
        }
        if best_lcb_val.is_infinite() {
            // unseen arms produce -inf LCB; clamp for logs
            best_lcb_val = 0.0;
        }

        let selected = if self.state.has_unseen() {
            // Prefer unseen arm to ensure one initial sample per arm
            self.state
                .counts
                .iter()
                .enumerate()
                .find(|(_, &n)| n == 0)
                .map(|(idx, _)| idx)
                .unwrap_or(best_lcb_idx)
        } else {
            best_lcb_idx
        };

        (selected, best_mean_idx, best_mean_val, best_lcb_val)
    }
}

impl<H: Heuristic> DecisionModel for NonContextualUcbSmartWare<H> {
    fn start_decision(&mut self, _features: &[f32]) -> &ModelDecision {
        debug_assert!(self.pending.is_none(), "pending decision leaked");

        let (selected_policy, predictive_policy, predictive_mean, optimisitc_mean) =
            self.select_policy();

        self.pending = Some(ModelDecision::Predict {
            selected_policy,
            predictive_policy,
            optimisitc_policy: selected_policy,
            predictive_mean,
            optimisitc_mean,
        });

        self.pending.as_ref().unwrap()
    }

    fn after_selected(&mut self, latency: f32) -> ExplorationRequest {
        let pending = self.pending.take().unwrap();
        match pending {
            ModelDecision::Predict {
                selected_policy, ..
            } => {
                // Standard UCB: update only the selected arm per round.
                self.state.observe_single(selected_policy, latency as f64);
                ExplorationRequest::Skip
            }
            ModelDecision::Initialize => ExplorationRequest::RequireAll,
        }
    }

    fn finalize(&mut self, _features: &[f32], _latencies: &[Option<f64>]) {
        // Standard UCB does not use side observations; nothing to do.
    }

    fn heuristic(&self, features: &[f32]) -> usize {
        (self.heuristic)(features)
    }
}
