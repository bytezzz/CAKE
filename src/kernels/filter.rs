use crate::utils::get_array_slice_logical_size;
use arrow_array::{new_empty_array, Array, ArrayRef, BooleanArray};
use arrow_schema::ArrowError;
use arrow_select::{
    concat,
    filter::{index_iter_filter, slice_iter_filter},
};
// Timing aggregation is handled at the DSL/top level.
use serde::Serialize;

use super::{
    chunking::get_chunk_size_filter,
    metrics::{create_metric_collector, MetricCollectorWrapper, TimeMetricCollector},
};
use crate::{
    archon::{AdaptivePetition, PolicyRun},
    execute_with_adaptive_chunking,
};

#[derive(Clone, Copy)]
pub enum FilterExecutionStrategy {
    SliceIter,
    IndexIter,
}

// Static strategy list - no need to store in each petition
const FILTER_STRATEGIES: [FilterExecutionStrategy; 2] = [
    FilterExecutionStrategy::SliceIter,
    FilterExecutionStrategy::IndexIter,
];

#[derive(Clone, Serialize)]
pub struct FilterOpFeatures {
    pub num_rows: usize,
    pub selectivity: f64,
    pub total_size: usize,
}

#[derive(Serialize)]
pub struct FilterPetition<'a> {
    pub op_type: &'static str,

    #[serde(flatten)]
    pub features: FilterOpFeatures,

    // Store references instead of owned data
    #[serde(skip)]
    pub column: &'a dyn Array,
    #[serde(skip)]
    pub ba: &'a BooleanArray,
}

/// Get the metric collector based on configuration
fn get_filter_metric_collector() -> MetricCollectorWrapper {
    create_metric_collector("filter").unwrap_or(MetricCollectorWrapper::Time(TimeMetricCollector))
}

impl FilterPetition<'_> {
    fn execute_strategy(&self, strategy: FilterExecutionStrategy) -> Result<ArrayRef, ArrowError> {
        match strategy {
            FilterExecutionStrategy::SliceIter => slice_iter_filter(self.column, self.ba),
            FilterExecutionStrategy::IndexIter => index_iter_filter(self.column, self.ba),
        }
    }
}

impl<'a> AdaptivePetition<'a> for FilterPetition<'a> {
    type Output = ArrayRef;

    fn op_type(&self) -> &'static str {
        self.op_type
    }

    fn feature_vector(&self) -> Vec<f32> {
        vec![self.features.selectivity as f32]
    }

    fn policy_count(&self) -> usize {
        FILTER_STRATEGIES.len()
    }

    fn eval(&mut self, policy: usize) -> Result<PolicyRun<Self::Output>, ArrowError> {
        if policy >= FILTER_STRATEGIES.len() {
            return Err(ArrowError::InvalidArgumentError(format!(
                "Invalid strategy index: {}, available strategies: {}",
                policy,
                FILTER_STRATEGIES.len()
            )));
        }

        let metric_collector = get_filter_metric_collector();
        let measurement = metric_collector.start_measurement();
        let result = self.execute_strategy(FILTER_STRATEGIES[policy])?;
        let latency = measurement.finish();
        Ok(PolicyRun {
            policy,
            payload: result,
            latency,
        })
    }
}

#[inline]
pub fn extract_filter_op_features(column: &dyn Array, ba: &BooleanArray) -> FilterOpFeatures {
    let num_rows = ba.len();
    let num_set_bits = ba.true_count();
    let selectivity = if num_rows > 0 {
        num_set_bits as f64 / num_rows as f64
    } else {
        0.0
    };
    let total_size = get_array_slice_logical_size(column);
    FilterOpFeatures {
        num_rows,
        selectivity,
        total_size,
    }
}

pub fn execute(column: &dyn Array, ba: &BooleanArray) -> Result<ArrayRef, ArrowError> {
    execute_smart(column, ba)
}

fn execute_smart(column: &dyn Array, ba: &BooleanArray) -> Result<ArrayRef, ArrowError> {
    // Only return the filtering result; top-level sets total time.
    execute_with_adaptive_chunking!(
        data: (column, ba),
        chunk_size: get_chunk_size_filter(),
        process_chunk: |_offset, chunk| {
            let (column_chunk, ba_chunk) = chunk;

            let ba_bool = ba_chunk.as_any().downcast_ref::<BooleanArray>()
                .expect("Failed to downcast to BooleanArray");

            // Fast path: if selectivity is zero, avoid model invocation and return empty array
            if ba_bool.true_count() == 0 {
                Ok::<ArrayRef, ArrowError>(new_empty_array(column_chunk.data_type()))
            } else {
                let features = extract_filter_op_features(column_chunk.as_ref(), ba_bool);

                let petition = FilterPetition {
                    op_type: "filter",
                    features,
                    column: column_chunk.as_ref(),
                    ba: ba_bool,
                };

                petition.adaptive_execute()
            }
        },
        merge_results: |results| {
            let result_chunks: Vec<ArrayRef> = results
                .into_iter()
                .collect::<Result<Vec<_>, _>>()?;
            let result_chunks_ref: Vec<&dyn Array> =
                result_chunks.iter().map(|c| c.as_ref()).collect();
            concat::concat(&result_chunks_ref)
        }
    )
}
