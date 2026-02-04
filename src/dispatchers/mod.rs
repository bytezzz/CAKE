use serde::{Deserialize, Serialize};

pub mod cake;
pub mod regret_tree;
pub mod simple_models;
pub mod cake_bs;
pub mod ucb;
pub use cake::CakeCLT;
pub use simple_models::{
    FixedPolicyDecisionModel, HeuristicDecisionModel, OracleMachineDecisionModel,
};
pub use cake_bs::CakeBS;
pub use ucb::NonContextualUcbSmartWare;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExplorationRequest {
    Skip,
    RequireAll,
}

pub type PolicyId = usize;

pub enum ModelDecision {
    Initialize,
    Predict {
        selected_policy: PolicyId,
        predictive_policy: PolicyId,
        optimisitc_policy: PolicyId,
        predictive_mean: f64,
        optimisitc_mean: f64,
    },
}

impl ExplorationRequest {
    pub fn require_all(self) -> Self {
        ExplorationRequest::RequireAll
    }
}

/// Configuration for the SmartWare model. This mirrors the Python implementation's configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartWareConfiguration {
    pub name_in_log: String,
    pub num_features: usize,
    pub num_policies: usize,
    pub cake: Option<CakeHyperParams>,
    pub ucb: Option<UcbHyperParams>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UcbHyperParams {
    pub alpha: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CakeHyperParams {
    #[serde(default = "CakeHyperParams::default_bandwidth")]
    pub bandwidth: f64,
    #[serde(default = "CakeHyperParams::default_n_min")]
    pub n_min: usize,
    #[serde(default)]
    pub buffer_size: Option<usize>,
    #[serde(default)]
    pub alpha: Option<f64>,
    #[serde(default)]
    pub max_depth: Option<usize>,
    #[serde(default)]
    pub regret_tree_after_decisions: Option<usize>,
}

impl Default for CakeHyperParams {
    fn default() -> Self {
        Self {
            bandwidth: 1.0,
            n_min: 3,
            buffer_size: None,
            alpha: None,
            max_depth: None,
            regret_tree_after_decisions: None,
        }
    }
}

impl SmartWareConfiguration {
    pub fn cake_params(&self) -> &CakeHyperParams {
        self.cake
            .as_ref()
            .expect("Cake hyperparameters missing from configuration")
    }

    pub fn ucb_params(&self) -> &UcbHyperParams {
        self.ucb
            .as_ref()
            .expect("UCB hyperparameters missing from configuration")
    }
}

impl CakeHyperParams {
    fn default_bandwidth() -> f64 {
        1.0
    }

    fn default_n_min() -> usize {
        3
    }
}

/// A trait for the heuristic function, which provides a default policy choice.
/// The `Fn` trait bound allows closures and function pointers to be used as heuristics.
pub trait Heuristic: Fn(&[f32]) -> usize + Send + Sync {}
impl<T: Fn(&[f32]) -> usize + Send + Sync> Heuristic for T {}

/// A trait that provides a common interface for different decision models
pub trait DecisionModel: Send + Sync {
    fn start_decision(&mut self, features: &[f32]) -> &ModelDecision;
    fn after_selected(&mut self, latency: f32) -> ExplorationRequest;
    fn finalize(&mut self, features: &[f32], latencies: &[Option<f64>]);
    fn heuristic(&self, features: &[f32]) -> PolicyId;
}
