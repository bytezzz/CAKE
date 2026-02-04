use arrow_schema::ArrowError;
use serde::Deserialize;
use std::{
    path::{Path, PathBuf},
    sync::OnceLock,
};

static CONFIG: OnceLock<SmartWareConfig> = OnceLock::new();

const DEFAULT_RUNTIME_DIR: &str = "smartware_runtime";

fn default_runtime_dir() -> String {
    DEFAULT_RUNTIME_DIR.to_string()
}

#[derive(Deserialize, Clone, Copy, Debug, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PolicyType {
    Heuristic,
    SinglePolicy,
    NonContextualUcb,
    OracleMachine,
    Cake,
    #[serde(rename = "cake_bs")]
    CakeBS,
}

#[derive(Deserialize)]
pub struct ModelConfig {
    pub num_features: usize,
    pub num_policies: usize,
    pub metric_type: Option<String>, // "time", "cycles", "instructions", "cache_misses"
    pub cake: Option<crate::dispatchers::CakeHyperParams>, // Cake learner hyperparameters
    pub ucb: Option<crate::dispatchers::UcbHyperParams>, // UCB hyperparameters
}

#[derive(Deserialize)]
pub struct ChunkSizeConfig {
    pub query_plan: usize,
    pub filter: usize,
    pub sort: usize,
}

#[derive(Deserialize)]
pub struct SmartWareConfig {
    pub job_path: String,
    #[serde(default = "default_runtime_dir")]
    pub runtime_dir: String,
    pub parquet_cache_memory_budget_gb: Option<f64>,
    pub policy: PolicyType,
    pub single_policy_config: Option<SinglePolicyConfig>,
    pub oracle_policies_path: Option<String>,
    pub chunk_sizes: ChunkSizeConfig,
    pub string_contains_feature_sample_ratio: Option<f64>,
    pub sort_feature_sample_ratio: Option<f64>,
    pub filter: ModelConfig,
    pub query_plan: ModelConfig,
    pub sort: ModelConfig,
}

#[derive(Deserialize)]
pub struct SinglePolicyConfig {
    pub sort_policy: usize,
    pub filter_policy: usize,
    pub query_plan_policy: usize,
}

pub fn init_config(path: &Path) -> Result<(), ArrowError> {
    let config = load_config_from_path(path)?;
    CONFIG
        .set(config)
        .map_err(|_| ArrowError::ExternalError("Failed to set config".into()))?;
    println!("Config initialized from {}", path.display());
    Ok(())
}

fn load_config_from_path(path: &Path) -> Result<SmartWareConfig, ArrowError> {
    let config_str =
        std::fs::read_to_string(path).map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
    let mut config: SmartWareConfig =
        serde_yaml::from_str(&config_str).map_err(|e| ArrowError::ExternalError(Box::new(e)))?;

    if config.runtime_dir.trim().is_empty() {
        config.runtime_dir = default_runtime_dir();
    }

    validate_config(&config)?;

    Ok(config)
}

fn validate_config(config: &SmartWareConfig) -> Result<(), ArrowError> {
    let models = [
        ("filter", &config.filter),
        ("query_plan", &config.query_plan),
        ("sort", &config.sort),
    ];

    match config.policy {
        PolicyType::NonContextualUcb => {
            let mut missing = Vec::new();
            for (name, model) in models {
                if model.ucb.as_ref().map(|params| params.alpha).is_none() {
                    missing.push(format!("{name}.ucb.alpha"));
                }
            }
            if !missing.is_empty() {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "Missing configuration values for {:?}: {}",
                    config.policy,
                    missing.join(", ")
                )));
            }
        }
        PolicyType::Cake | PolicyType::CakeBS => {
            let mut missing = Vec::new();
            for (name, model) in models {
                let Some(cake) = model.cake.as_ref() else {
                    missing.push(format!("{name}.cake"));
                    continue;
                };
                if cake.alpha.is_none() {
                    missing.push(format!("{name}.cake.alpha"));
                }
                if cake.max_depth.is_none() {
                    missing.push(format!("{name}.cake.max_depth"));
                }
            }
            if !missing.is_empty() {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "Missing configuration values for {:?}: {}",
                    config.policy,
                    missing.join(", ")
                )));
            }
        }
        _ => {}
    }

    Ok(())
}

fn get_config() -> Result<&'static SmartWareConfig, ArrowError> {
    let config_result = CONFIG.get();
    config_result.ok_or_else(|| ArrowError::ExternalError("Config not initialized".into()))
}

// Macro to generate simple getter functions
macro_rules! config_getter {
    // For reference types
    ($fn_name:ident, &$lifetime:lifetime $return_type:ty, $field:ident) => {
        pub fn $fn_name() -> Result<&$lifetime $return_type, ArrowError> {
            let config = get_config()?;
            Ok(&config.$field)
        }
    };
    // For value types (bool, usize, etc.)
    ($fn_name:ident, $return_type:ty, $field:ident) => {
        pub fn $fn_name() -> Result<$return_type, ArrowError> {
            let config = get_config()?;
            Ok(config.$field)
        }
    };
}

pub fn get_config_sample_ratio(_op_type: &str, ratio_name: &str) -> Result<f64, ArrowError> {
    let config = get_config()?;
    let ratio = match ratio_name {
        "sort_feature_sample_ratio" => config.sort_feature_sample_ratio,
        "string_contains_feature_sample_ratio" => config.string_contains_feature_sample_ratio,
        _ => {
            return Err(ArrowError::InvalidArgumentError(format!(
                "Unknown ratio name: {}",
                ratio_name
            )))
        }
    };

    ratio.ok_or_else(|| {
        ArrowError::InvalidArgumentError(format!(
            "Missing configuration value for {}",
            ratio_name
        ))
    })
}

pub fn get_model_config(op_type: &str) -> Result<&'static ModelConfig, ArrowError> {
    let config = get_config()?;
    match op_type {
        "sort" => Ok(&config.sort),
        "filter" => Ok(&config.filter),
        "query_plan" => Ok(&config.query_plan),
        _ => Err(ArrowError::InvalidArgumentError(format!(
            "Unknown operation type: {}",
            op_type
        ))),
    }
}

// Generate getter functions using the macro
config_getter!(get_job_path, &'static str, job_path);
config_getter!(get_runtime_dir, &'static str, runtime_dir);
config_getter!(get_chunk_sizes, &'static ChunkSizeConfig, chunk_sizes);
config_getter!(get_policy, PolicyType, policy);

pub fn runtime_dir_or_default() -> &'static str {
    CONFIG
        .get()
        .map(|config| config.runtime_dir.as_str())
        .unwrap_or(DEFAULT_RUNTIME_DIR)
}

pub fn runtime_path_or_default<P: AsRef<Path>>(relative: P) -> PathBuf {
    Path::new(runtime_dir_or_default()).join(relative)
}

/// Returns the parquet cache budget in bytes if configured and greater than zero.
pub fn get_parquet_cache_budget_bytes() -> Result<Option<usize>, ArrowError> {
    let config = get_config()?;
    let gb = config.parquet_cache_memory_budget_gb.unwrap_or(0.0);
    if gb <= 0.0 {
        return Ok(None);
    }

    // Use base-2 gibibytes to align with typical memory reporting.
    let bytes = (gb * (1u128 << 30) as f64).round() as u128;
    let clamped = bytes.min(usize::MAX as u128);
    Ok(Some(clamped as usize))
}

pub fn get_single_policy_config() -> Result<&'static SinglePolicyConfig, ArrowError> {
    let config = get_config()?;
    config.single_policy_config.as_ref().ok_or_else(|| {
        ArrowError::InvalidArgumentError(
            "Single policy configuration not provided in config file".to_string(),
        )
    })
}

pub fn get_single_policy_index(op_type: &str) -> Result<usize, ArrowError> {
    let single_policy_config = get_single_policy_config()?;
    match op_type {
        "sort" => Ok(single_policy_config.sort_policy),
        "filter" => Ok(single_policy_config.filter_policy),
        "query_plan" => Ok(single_policy_config.query_plan_policy),
        _ => Err(ArrowError::InvalidArgumentError(format!(
            "Unknown operation type: {}",
            op_type
        ))),
    }
}

pub fn get_metric_type(op_type: &str) -> Result<String, ArrowError> {
    let model_config = get_model_config(op_type)?;
    Ok(model_config
        .metric_type
        .clone()
        .unwrap_or_else(|| "time".to_string()))
}

// Specific chunk size getters
pub fn get_chunk_size(op_type: &str) -> Result<usize, ArrowError> {
    let chunk_sizes = get_chunk_sizes()?;
    match op_type {
        "query_plan" => Ok(chunk_sizes.query_plan),
        "filter" => Ok(chunk_sizes.filter),
        "sort" => Ok(chunk_sizes.sort),
        _ => Err(ArrowError::InvalidArgumentError(format!(
            "Unknown operation type for chunk size: {}",
            op_type
        ))),
    }
}

/// Optional path to oracle policies file if running with `policy: oracle_machine`.
pub fn get_oracle_policies_path() -> Result<Option<&'static str>, ArrowError> {
    let config = get_config()?;
    Ok(config.oracle_policies_path.as_deref())
}
