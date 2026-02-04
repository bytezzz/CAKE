use arrow_schema::ArrowError;
use serde::Serialize;

use crate::{
    configuration::{
        get_model_config, get_policy, get_single_policy_index, runtime_dir_or_default, ModelConfig,
        PolicyType,
    },
    dispatchers::{
        CakeBS, CakeCLT, DecisionModel, ExplorationRequest, FixedPolicyDecisionModel, Heuristic,
        HeuristicDecisionModel, ModelDecision, NonContextualUcbSmartWare,
        OracleMachineDecisionModel, SmartWareConfiguration,
    },
    oracle_machine::OpType,
};
use std::collections::HashMap;
use std::fs;
use std::sync::{Mutex, OnceLock};

fn build_smartware_config(name: &str, model_config: &ModelConfig) -> SmartWareConfiguration {
    SmartWareConfiguration {
        name_in_log: format!("{}_model", name),
        num_features: model_config.num_features,
        num_policies: model_config.num_policies,
        cake: model_config.cake.clone(),
        ucb: model_config.ucb.clone(),
    }
}

pub struct PolicyRun<T> {
    pub policy: usize,
    pub payload: T,
    pub latency: f64,
}

fn collect_latencies_into<'a, P>(
    petition: &mut P,
    latencies: &mut [Option<f64>],
) -> Result<(), ArrowError>
where
    P: AdaptivePetition<'a>,
{
    for policy_idx in 0..latencies.len() {
        // if latencies[policy_idx].is_some() {
        //     continue;
        // }
        let PolicyRun { latency, .. } = petition.eval(policy_idx)?;
        latencies[policy_idx] = Some(latency);
    }
    Ok(())
}

pub struct Archon {
    models: HashMap<String, Box<dyn DecisionModel>>,
}

static ARCHON: OnceLock<Mutex<Archon>> = OnceLock::new();

impl Archon {
    fn new() -> Result<Self, ArrowError> {
        let policy = get_policy()?;
        let mut models: HashMap<String, Box<dyn DecisionModel>> = HashMap::new();

        fn build_model(
            policy: PolicyType,
            op_name: &str,
            oracle_op: OpType,
            heuristic: Box<dyn Heuristic>,
            model_config: &ModelConfig,
        ) -> Result<Box<dyn DecisionModel>, ArrowError> {
            match policy {
                PolicyType::Cake => {
                    let config = build_smartware_config(op_name, model_config);
                    Ok(Box::new(CakeCLT::new(heuristic, config)))
                }
                PolicyType::CakeBS => Ok(Box::new(CakeBS::new(
                    heuristic,
                    build_smartware_config(op_name, model_config),
                ))),
                PolicyType::NonContextualUcb => Ok(Box::new(NonContextualUcbSmartWare::new(
                    heuristic,
                    build_smartware_config(op_name, model_config),
                ))),
                PolicyType::Heuristic => Ok(Box::new(HeuristicDecisionModel::new(heuristic))),
                PolicyType::SinglePolicy => {
                    let policy_idx = get_single_policy_index(op_name).unwrap_or(0);
                    if policy_idx >= model_config.num_policies {
                        return Err(ArrowError::InvalidArgumentError(format!(
                            "single_policy_config for {op_name} is {policy_idx}, but {op_name}.num_policies is {}",
                            model_config.num_policies
                        )));
                    }
                    Ok(Box::new(FixedPolicyDecisionModel::new(policy_idx)))
                }
                PolicyType::OracleMachine => {
                    Ok(Box::new(OracleMachineDecisionModel::new(oracle_op)))
                }
            }
        }

        let filter_config = get_model_config("filter")?;
        let filter_model = build_model(
            policy,
            "filter",
            OpType::Filter,
            Box::new(|x: &[f32]| if x[0] > 0.8 { 0 } else { 1 }),
            filter_config,
        )?;
        models.insert("filter".to_string(), filter_model);

        let query_plan_config = get_model_config("query_plan")?;
        let query_plan_model = build_model(
            policy,
            "query_plan",
            OpType::QueryPlan,
            Box::new(|_: &[f32]| 1),
            query_plan_config,
        )?;
        models.insert("query_plan".to_string(), query_plan_model);

        let sort_config = get_model_config("sort")?;
        let sort_model = build_model(
            policy,
            "sort",
            OpType::Sort,
            Box::new(|_: &[f32]| 0),
            sort_config,
        )?;
        models.insert("sort".to_string(), sort_model);

        Ok(Self { models })
    }

    fn adaptive_execute<'a, P: AdaptivePetition<'a>>(
        &mut self,
        mut petition: P,
    ) -> Result<P::Output, ArrowError> {
        let op_type = petition.op_type();
        let features = petition.feature_vector();
        let policy_count = petition.policy_count();

        if policy_count == 0 {
            return Err(ArrowError::InvalidArgumentError(format!(
                "No policies registered for {}",
                op_type
            )));
        }

        let model = self
            .models
            .get_mut(op_type)
            .ok_or_else(|| ArrowError::NotYetImplemented(format!("No model for {}", op_type)))?;

        // Limit the borrow of `model` from `start_decision` to this block by
        // extracting the needed data immediately, then dropping the reference.
        let maybe_selected_policy = {
            let md = model.start_decision(&features);

            let maybe_selected_policy = match md {
                ModelDecision::Predict {
                    selected_policy, ..
                } => Some(selected_policy.clone()),
                ModelDecision::Initialize => None,
            };

            maybe_selected_policy
        };

        let policy_to_evaluate =
            maybe_selected_policy.unwrap_or_else(|| model.heuristic(&features));

        let PolicyRun {
            policy: selected_policy,
            payload,
            latency: selected_latency,
        } = petition.eval(policy_to_evaluate)?;

        let mut latencies = vec![None; policy_count];
        latencies[selected_policy] = Some(selected_latency);

        let exploration_request = model.after_selected(selected_latency as f32);

        let require_all = matches!(exploration_request, ExplorationRequest::RequireAll);

        if require_all {
            collect_latencies_into(&mut petition, &mut latencies)?;
        }

        if require_all {
            model.finalize(&features, &latencies);
        }

        Ok(payload)
    }

    fn reset_state(&mut self) -> Result<(), ArrowError> {
        // Create runtime directory if it doesn't exist
        let runtime_dir = runtime_dir_or_default();
        fs::create_dir_all(runtime_dir).map_err(|e| {
            ArrowError::ExternalError(format!("Failed to create runtime directory: {}", e).into())
        })?;

        *self = Self::new()?;
        println!("Archon has been reset");
        Ok(())
    }
}

pub trait AdaptivePetition<'a>: Sized + Serialize + 'a {
    type Output;

    fn op_type(&self) -> &'static str;
    fn feature_vector(&self) -> Vec<f32>;
    fn policy_count(&self) -> usize;
    fn eval(&mut self, policy: usize) -> Result<PolicyRun<Self::Output>, ArrowError>;

    fn adaptive_execute(self) -> Result<Self::Output, ArrowError> {
        let archon = ARCHON.get().unwrap();
        archon.lock().unwrap().adaptive_execute(self)
    }
}

impl Archon {
    pub fn new_embedded() -> Result<(), ArrowError> {
        println!("Archon created");
        ARCHON
            .set(Mutex::new(Self::new()?))
            .map_err(|_| ArrowError::ExternalError("Failed to set Archon".to_string().into()))?;
        Ok(())
    }

    pub fn signal_end_of_epoch() -> Result<(), ArrowError> {
        crate::oracle_machine::reset_oracle_machine()
    }

    pub fn signal_end_of_experiment() -> Result<(), ArrowError> {
        crate::oracle_machine::reset_oracle_machine()?;
        let archon = ARCHON.get().unwrap();
        archon.lock().unwrap().reset_state()
    }
}
