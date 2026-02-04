use arrow_schema::ArrowError;
use num_enum::TryFromPrimitive;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use super::{DecisionModel, ExplorationRequest, Heuristic, ModelDecision, PolicyId};

pub struct OracleMachine {
    oracle_policy_sequence: Vec<OraclePolicy>,
    curr_idx: usize,
}

pub struct OraclePolicy {
    pub op_type: OpType,
    pub policy_id: PolicyId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, TryFromPrimitive)]
#[repr(u8)]
pub enum OpType {
    Filter = 0,
    Sort = 1,
    QueryPlan = 2,
}

static ORACLE_MACHINE: OnceLock<Mutex<OracleMachine>> = OnceLock::new();

pub fn get_oracle_machine() -> &'static Mutex<OracleMachine> {
    ORACLE_MACHINE.get().unwrap()
}

/// Initialize the global OracleMachine from a policy file.
///
/// The file must contain one whitespace-separated pair per line:
/// "op_type policy_id" where `op_type` is a u8 matching `OpType` and
/// `policy_id` is the strategy index for that op.
/// - Filter (0): 0 => slice_iter, 1 => index_iter
/// - Sort   (1): 0 => quicksort, 1 => heapsort, 2 => insertion sort
/// - Query  (2): 0 => parallel, 1 => sequential
///
/// Note: This may be called only once (global OnceLock). Subsequent attempts
/// to re-initialize will return an error.
pub fn init_oracle_machine_from_path(path: &Path) -> Result<(), ArrowError> {
    ORACLE_MACHINE
        .set(Mutex::new(OracleMachine::from(path)))
        .map_err(|_| ArrowError::ExternalError("OracleMachine already initialized".into()))
}

pub fn reset_oracle_machine() -> Result<(), ArrowError> {
    if let Some(oracle_machine) = ORACLE_MACHINE.get() {
        oracle_machine.lock().unwrap().reset_cursor();
    }
    Ok(())
}

impl OracleMachine {
    pub fn from(path: &Path) -> Self {
        let oracle_policy_sequence = std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| {
                let parts: Vec<&str> = line.split_whitespace().collect();
                let op_type = parts[0].parse::<u8>().unwrap().try_into().unwrap();
                let policy_id = parts[1].parse::<PolicyId>().unwrap();
                OraclePolicy { op_type, policy_id }
            })
            .collect();
        Self {
            oracle_policy_sequence,
            curr_idx: 0,
        }
    }

    pub fn reset_cursor(&mut self) {
        self.curr_idx = 0;
    }

    pub fn next_policy(&mut self, op_type: OpType) -> PolicyId {
        if self.curr_idx >= self.oracle_policy_sequence.len() {
            panic!(
                "OracleMachine exhausted: requested {:?} at idx {}, but sequence length is {}",
                op_type,
                self.curr_idx,
                self.oracle_policy_sequence.len()
            );
        }
        let expected_op = self.oracle_policy_sequence[self.curr_idx].op_type;
        if expected_op != op_type {
            panic!(
                "OracleMachine op_type mismatch at idx {}: expected {:?} from oracle file, got {:?} from caller",
                self.curr_idx, expected_op, op_type
            );
        }
        let policy = self.oracle_policy_sequence[self.curr_idx].policy_id;
        self.curr_idx += 1;
        policy
    }
}

pub struct HeuristicDecisionModel {
    heuristic: Box<dyn Heuristic>,
    decision: ModelDecision,
}

impl HeuristicDecisionModel {
    pub fn new(heuristic: Box<dyn Heuristic>) -> Self {
        Self {
            heuristic,
            decision: ModelDecision::Initialize,
        }
    }
}

impl DecisionModel for HeuristicDecisionModel {
    fn start_decision(&mut self, features: &[f32]) -> &ModelDecision {
        let policy = (self.heuristic)(features);
        self.decision = ModelDecision::Predict {
            selected_policy: policy,
            predictive_policy: policy,
            optimisitc_policy: policy,
            predictive_mean: 0.0,
            optimisitc_mean: 0.0,
        };
        &self.decision
    }

    fn after_selected(&mut self, _latency: f32) -> ExplorationRequest {
        ExplorationRequest::Skip
    }

    fn finalize(&mut self, _features: &[f32], _latencies: &[Option<f64>]) {}

    fn heuristic(&self, features: &[f32]) -> PolicyId {
        (self.heuristic)(features)
    }
}

pub struct FixedPolicyDecisionModel {
    policy: PolicyId,
    decision: ModelDecision,
}

impl FixedPolicyDecisionModel {
    pub fn new(policy: PolicyId) -> Self {
        Self {
            policy,
            decision: ModelDecision::Initialize,
        }
    }
}

impl DecisionModel for FixedPolicyDecisionModel {
    fn start_decision(&mut self, _features: &[f32]) -> &ModelDecision {
        let policy = self.policy;
        self.decision = ModelDecision::Predict {
            selected_policy: policy,
            predictive_policy: policy,
            optimisitc_policy: policy,
            predictive_mean: 0.0,
            optimisitc_mean: 0.0,
        };
        &self.decision
    }

    fn after_selected(&mut self, _latency: f32) -> ExplorationRequest {
        ExplorationRequest::Skip
    }

    fn finalize(&mut self, _features: &[f32], _latencies: &[Option<f64>]) {}

    fn heuristic(&self, _features: &[f32]) -> PolicyId {
        self.policy
    }
}

pub struct OracleMachineDecisionModel {
    op_type: OpType,
    last_policy: PolicyId,
    decision: ModelDecision,
}

impl OracleMachineDecisionModel {
    pub fn new(op_type: OpType) -> Self {
        Self {
            op_type,
            last_policy: 0,
            decision: ModelDecision::Initialize,
        }
    }
}

impl DecisionModel for OracleMachineDecisionModel {
    fn start_decision(&mut self, _features: &[f32]) -> &ModelDecision {
        let policy = get_oracle_machine()
            .lock()
            .unwrap()
            .next_policy(self.op_type);
        self.last_policy = policy;
        self.decision = ModelDecision::Predict {
            selected_policy: policy,
            predictive_policy: policy,
            optimisitc_policy: policy,
            predictive_mean: 0.0,
            optimisitc_mean: 0.0,
        };
        &self.decision
    }

    fn after_selected(&mut self, _latency: f32) -> ExplorationRequest {
        ExplorationRequest::Skip
    }

    fn finalize(&mut self, _features: &[f32], _latencies: &[Option<f64>]) {}

    fn heuristic(&self, _features: &[f32]) -> PolicyId {
        self.last_policy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oracle_machine_reset_cursor_rewinds() {
        let mut om = OracleMachine {
            oracle_policy_sequence: vec![
                OraclePolicy {
                    op_type: OpType::Filter,
                    policy_id: 1,
                },
                OraclePolicy {
                    op_type: OpType::Sort,
                    policy_id: 2,
                },
            ],
            curr_idx: 0,
        };

        assert_eq!(om.next_policy(OpType::Filter), 1);
        assert_eq!(om.next_policy(OpType::Sort), 2);

        om.reset_cursor();
        assert_eq!(om.next_policy(OpType::Filter), 1);
    }
}
