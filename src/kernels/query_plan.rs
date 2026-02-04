use arrow_array::{cast::AsArray, Array, StringArray};
use arrow_buffer::bit_iterator::BitIndexIterator;
use arrow_schema::{ArrowError, DataType};
use serde::Serialize;
use std::sync::Arc;

use super::{
    chunking::get_chunk_size_query_plan,
    metrics::{create_metric_collector, MetricCollectorWrapper, TimeMetricCollector},
};
use crate::{
    archon::{AdaptivePetition, PolicyRun},
    configuration::get_config_sample_ratio,
    execute_with_adaptive_chunking,
    predicate::Predicate,
    selection::Selection,
    ArrowQuiver, ColumnIdentifier, Intersector,
};

#[derive(Clone, Copy)]
pub enum QueryPlanExecutionStrategy {
    Parallel,
    Sequential,
}

// Static strategy list - no need to store in each petition
const QUERY_PLAN_STRATEGIES: [QueryPlanExecutionStrategy; 2] = [
    QueryPlanExecutionStrategy::Parallel,
    QueryPlanExecutionStrategy::Sequential,
];

#[derive(Serialize)]
pub struct QueryPlanPetition<'a> {
    pub op_type: &'static str,
    pub inner_selectivity: f64,
    pub needle_length: f64,

    // Store only references to avoid cloning
    #[serde(skip)]
    pub base_selection: Selection,
    #[serde(skip)]
    pub str_col_ci: &'a ColumnIdentifier,
    #[serde(skip)]
    pub needle: &'a str,
    #[serde(skip)]
    pub aq_chunk: &'a ArrowQuiver,
    #[serde(skip)]
    pub intersector: &'a Intersector,
}

/// Get the metric collector based on configuration
fn get_query_plan_metric_collector() -> MetricCollectorWrapper {
    create_metric_collector("query_plan")
        .unwrap_or(MetricCollectorWrapper::Time(TimeMetricCollector))
}

impl QueryPlanPetition<'_> {
    fn execute_strategy(
        &self,
        strategy: QueryPlanExecutionStrategy,
    ) -> Result<Selection, ArrowError> {
        match strategy {
            QueryPlanExecutionStrategy::Parallel => {
                // Use SIMD optimized StringContains
                let string_pred = Predicate::StringContains(
                    self.str_col_ci.clone(),
                    Arc::new(StringArray::new_scalar(self.needle)),
                );
                let string_sel = string_pred.apply(self.aq_chunk, self.intersector);
                // Intersect with base_selection
                Ok(self
                    .base_selection
                    .clone()
                    .intersect(self.intersector, string_sel))
            }
            QueryPlanExecutionStrategy::Sequential => {
                // Use refactored ParitialStringContains with Selection
                let pred = Predicate::ParitialStringContains(
                    self.str_col_ci.clone(),
                    Arc::new(StringArray::new_scalar(self.needle)),
                    Box::new(self.base_selection.clone()),
                );
                Ok(pred.apply(self.aq_chunk, self.intersector))
            }
        }
    }
}
impl<'a> AdaptivePetition<'a> for QueryPlanPetition<'a> {
    type Output = Selection;

    fn op_type(&self) -> &'static str {
        self.op_type
    }

    fn feature_vector(&self) -> Vec<f32> {
        // 1. Inner selectivity remains unchanged (already in [0,1])
        let inner_selectivity = self.inner_selectivity as f32;

        // 2. Needle feature using inverse transform
        let needle_ratio = if self.needle_length > 0.0 {
            self.needle.len() as f32 / self.needle_length as f32
        } else {
            1.0
        };
        let needle_feature = 1.0 / (1.0 + needle_ratio);

        vec![inner_selectivity, needle_feature]
    }

    fn policy_count(&self) -> usize {
        QUERY_PLAN_STRATEGIES.len()
    }

    fn eval(&mut self, policy: usize) -> Result<PolicyRun<Self::Output>, ArrowError> {
        if policy >= QUERY_PLAN_STRATEGIES.len() {
            return Err(ArrowError::InvalidArgumentError(format!(
                "Invalid strategy index: {}, available strategies: {}",
                policy,
                QUERY_PLAN_STRATEGIES.len()
            )));
        }

        let metric_collector = get_query_plan_metric_collector();
        let measurement = metric_collector.start_measurement();
        let result = self.execute_strategy(QUERY_PLAN_STRATEGIES[policy])?;
        let latency = measurement.finish();
        Ok(PolicyRun {
            policy,
            payload: result,
            latency,
        })
    }
}

fn smart_string_contains_on_top_of_impl(
    aq: &ArrowQuiver,
    base_predicate: Predicate,
    str_col_ci: ColumnIdentifier,
    needle: &str,
    intersector: &Intersector,
) -> Result<Selection, ArrowError> {
    execute_with_adaptive_chunking!(
        data: aq,
        chunk_size: get_chunk_size_query_plan(),
        process_chunk: |offset, aq_chunk| {
            // Step 1: Evaluate base predicate
            let base_selection = base_predicate.apply(&aq_chunk, intersector);

            // Step 2: Calculate exact selectivity and average string length
            let (exact_selectivity, avg_length) = {
                // Calculate exact selectivity from actual result
                let num_rows = aq_chunk.num_rows();
                let selected_count = match &base_selection {
                    Selection::SelVec(v) => v.len(),
                    Selection::Bitmap(b) => b.count_set_bits(),
                    Selection::AllValid => num_rows,
                    Selection::NoneValid => 0,
                };
                let selectivity = selected_count as f64 / num_rows as f64;

                // Keep original string length sampling logic
                let str_col_chunk = &aq_chunk[&str_col_ci];
                let sample_fraction = get_config_sample_ratio("query_plan",
                    "string_contains_feature_sample_ratio"
                ).unwrap_or(if offset == 0 { 0.01 } else { 0.001 });

                let slice_len = num_rows;
                let n_points = (slice_len as f64 * sample_fraction).round() as usize;

                let avg_length = if n_points == 0 {
                    0.0
                } else {
                    match str_col_chunk.data_type() {
                        DataType::Utf8 => {
                            let str_arr = str_col_chunk.as_string::<i32>();
                            let offset_arr = str_arr.value_offsets();
                            let str_offset = str_arr.offset();

                            let start_off = offset_arr[str_offset] as usize;
                            let end_off = offset_arr[str_offset + n_points] as usize;
                            let total_length = end_off - start_off;

                            total_length as f64 / n_points as f64
                        },
                        DataType::LargeUtf8 => {
                            let str_arr = str_col_chunk.as_string::<i64>();
                            let offset_arr = str_arr.value_offsets();
                            let str_offset = str_arr.offset();

                            let start_off = offset_arr[str_offset] as usize;
                            let end_off = offset_arr[str_offset + n_points] as usize;
                            let total_length = end_off - start_off;

                            total_length as f64 / n_points as f64
                        },
                        _ => panic!(
                            "Unsupported data type for string contains: {:?}",
                            str_col_chunk.data_type()
                        ),
                    }
                };

                (selectivity, avg_length)
            };

            // Step 3: Create petition with evaluated selection
            let partition = QueryPlanPetition {
                op_type: "query_plan",
                inner_selectivity: exact_selectivity,
                needle_length: avg_length,
                base_selection,
                str_col_ci: &str_col_ci,
                needle,
                aq_chunk: &aq_chunk,
                intersector,
            };

            let local_selection = partition.adaptive_execute()?;

            // Transform local indices to global indices if needed
            if offset == 0 {
                Ok::<Selection, ArrowError>(local_selection)
            } else {
                let num_rows_in_chunk = aq_chunk.num_rows();
                Ok(match local_selection {
                    Selection::SelVec(mut v) => {
                        v.iter_mut().for_each(|i| *i += offset as u32);
                        Selection::SelVec(v)
                    }
                    Selection::Bitmap(b) => {
                        let indices = BitIndexIterator::new(
                            b.values(),
                            b.offset(),
                            b.len(),
                        )
                        .map(|i| i as u32 + offset as u32);
                        Selection::SelVec(indices.collect())
                    }
                    Selection::AllValid => {
                        let indices =
                            (0..num_rows_in_chunk as u32).map(|i| i + offset as u32);
                        Selection::SelVec(indices.collect())
                    }
                    Selection::NoneValid => Selection::NoneValid,
                })
            }
        },
        merge_results: |chunk_selections| {
            let mut all_indices = Vec::new();
            for selection_result in chunk_selections {
                let selection = selection_result?;
                match selection {
                    Selection::SelVec(mut indices) => {
                        all_indices.append(&mut indices);
                    }
                    Selection::AllValid => {
                        // This shouldn't happen after transformation to global indices
                        panic!("AllValid selection should not exist after global transformation");
                    }
                    Selection::Bitmap(b) => {
                        // Convert bitmap to indices and add to all_indices
                        let indices = BitIndexIterator::new(b.values(), b.offset(), b.len())
                            .map(|i| i as u32)
                            .collect::<Vec<_>>();
                        all_indices.extend(indices);
                    }
                    Selection::NoneValid => {
                        // No indices to add
                    }
                }
            }

            let final_selection = if all_indices.is_empty() {
                Selection::NoneValid
            } else {
                Selection::SelVec(all_indices)
            };
            Ok(final_selection.into_sel_vec())
        }
    )
}

pub fn execute(
    aq: &ArrowQuiver,
    base_predicate: Box<Predicate>,
    str_col_ci: ColumnIdentifier,
    needle: &str,
    intersector: &Intersector,
) -> Result<Selection, ArrowError> {
    smart_string_contains_on_top_of_impl(aq, *base_predicate, str_col_ci, needle, intersector)
}
