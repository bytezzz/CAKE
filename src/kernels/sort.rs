use arrow_array::{Array, ArrowPrimitiveType, PrimitiveArray};
use arrow_schema::ArrowError;
use serde::Serialize;
use std::collections::{BTreeMap, BinaryHeap};

use super::{
    chunking::get_chunk_size_sort,
    metrics::{create_metric_collector, MetricCollectorWrapper, TimeMetricCollector},
};
use crate::{
    archon::{AdaptivePetition, PolicyRun},
    configuration::get_config_sample_ratio,
    execute_with_adaptive_chunking,
    selection::Selection,
    sort::{sort_array_to_indices, SortAlgorithm, SortDirection},
    ArrowQuiver, ColumnIdentifier,
};

#[derive(Clone, Copy)]
pub enum SortExecutionStrategy {
    Quicksort,
    Heapsort,
}

// Static strategy list - no need to store in each petition
const SORT_STRATEGIES: [SortExecutionStrategy; 2] = [
    SortExecutionStrategy::Quicksort,
    SortExecutionStrategy::Heapsort,
];

#[derive(Serialize)]
pub struct SortPetition<'a> {
    pub op_type: &'static str,

    pub inversion_pairs: f64,
    pub unique_count: f64,

    // Store references instead of owned data
    #[serde(skip)]
    pub column: &'a dyn Array,
    #[serde(skip)]
    pub direction: SortDirection,
}

/// Get the metric collector based on configuration
fn get_sort_metric_collector() -> MetricCollectorWrapper {
    create_metric_collector("sort").unwrap_or(MetricCollectorWrapper::Time(TimeMetricCollector))
}

impl SortPetition<'_> {
    fn execute_strategy(&self, strategy: SortExecutionStrategy) -> Result<Vec<u32>, ArrowError> {
        let algorithm = match strategy {
            SortExecutionStrategy::Quicksort => SortAlgorithm::Quicksort,
            SortExecutionStrategy::Heapsort => SortAlgorithm::Heapsort,
        };
        sort_array_to_indices(self.column, algorithm, self.direction)
    }
}

impl<'a> AdaptivePetition<'a> for SortPetition<'a> {
    type Output = Vec<u32>;

    fn op_type(&self) -> &'static str {
        self.op_type
    }

    fn feature_vector(&self) -> Vec<f32> {
        vec![self.inversion_pairs as f32, self.unique_count as f32]
    }

    fn policy_count(&self) -> usize {
        SORT_STRATEGIES.len()
    }

    fn eval(&mut self, policy: usize) -> Result<PolicyRun<Self::Output>, ArrowError> {
        if policy >= SORT_STRATEGIES.len() {
            return Err(ArrowError::InvalidArgumentError(format!(
                "Invalid strategy index: {}, available strategies: {}",
                policy,
                SORT_STRATEGIES.len()
            )));
        }

        let metric_collector = get_sort_metric_collector();
        let measurement = metric_collector.start_measurement();
        let result = self.execute_strategy(SORT_STRATEGIES[policy])?;
        let latency = measurement.finish();
        Ok(PolicyRun {
            policy,
            payload: result,
            latency,
        })
    }
}

struct MergeItem<N: Ord> {
    value: N,
    chunk_idx: usize,
    pos_in_chunk: usize,
}

impl<N: Ord> PartialEq for MergeItem<N> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}
impl<N: Ord> Eq for MergeItem<N> {}

impl<N: Ord> PartialOrd for MergeItem<N> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<N: Ord> Ord for MergeItem<N> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.value.cmp(&self.value)
    }
}

fn k_way_merge_indices<T>(
    original_array: &PrimitiveArray<T>,
    chunk_sorted_local_indices: &[Vec<u32>],
    chunk_offsets: &[usize],
) -> Vec<u32>
where
    T: ArrowPrimitiveType,
    T::Native: Ord,
{
    if original_array.is_empty() {
        return Vec::new();
    }

    // Pre-allocate with exact capacity to avoid reallocation
    let mut result = Vec::with_capacity(original_array.len());

    // Pre-allocate heap with number of chunks to avoid initial resizing
    let mut heap = BinaryHeap::with_capacity(chunk_sorted_local_indices.len());

    for i in 0..chunk_sorted_local_indices.len() {
        if !chunk_sorted_local_indices[i].is_empty() {
            let local_idx = chunk_sorted_local_indices[i][0] as usize;
            let global_idx = chunk_offsets[i] + local_idx;
            let value = unsafe { original_array.value_unchecked(global_idx) };
            heap.push(MergeItem {
                value,
                chunk_idx: i,
                pos_in_chunk: 0,
            });
        }
    }

    while let Some(item) = heap.pop() {
        let local_idx = chunk_sorted_local_indices[item.chunk_idx][item.pos_in_chunk] as usize;
        result.push((chunk_offsets[item.chunk_idx] + local_idx) as u32);

        let next_pos = item.pos_in_chunk + 1;
        if next_pos < chunk_sorted_local_indices[item.chunk_idx].len() {
            let next_local_idx = chunk_sorted_local_indices[item.chunk_idx][next_pos] as usize;
            let next_global_idx = chunk_offsets[item.chunk_idx] + next_local_idx;
            let next_value = unsafe { original_array.value_unchecked(next_global_idx) };
            heap.push(MergeItem {
                value: next_value,
                chunk_idx: item.chunk_idx,
                pos_in_chunk: next_pos,
            });
        }
    }

    result
}

fn subsample<T>(arr: &PrimitiveArray<T>, sub_sample_ratio: f64) -> Vec<T::Native>
where
    T: ArrowPrimitiveType,
{
    let len = arr.len();
    assert!(
        len >= 2,
        "array needs at least two elements (index 0 and ≥1)"
    );
    assert!(
        (0.0..=1.0).contains(&sub_sample_ratio),
        "ratio must be between 0 and 1"
    );

    let n_points = ((len as f64) * sub_sample_ratio).round() as usize;
    if n_points == 0 {
        return Vec::new();
    }
    if n_points == 1 {
        return vec![unsafe { arr.value_unchecked(1) }];
    }

    let step = (len - 2) as f64 / (n_points - 1) as f64;

    let mut out = Vec::with_capacity(n_points);
    for k in 0..n_points {
        let idx = (1.0 + k as f64 * step).round() as usize;
        unsafe {
            out.push(arr.value_unchecked(idx));
        }
    }
    out
}

pub fn calculate_inversions_and_unique_count<T>(
    arr: &PrimitiveArray<T>,
    sub_sample_ratio: f64,
) -> (usize, usize)
where
    T: ArrowPrimitiveType,
    T::Native: Ord,
{
    let mut freq = BTreeMap::new();
    let mut inv_count = 0;

    let sampled = subsample(arr, sub_sample_ratio);

    for x in sampled.iter().rev() {
        let less_count: usize = freq.range(..*x).map(|(_, &cnt)| cnt).sum();
        inv_count += less_count;
        *freq.entry(*x).or_insert(0) += 1;
    }

    (inv_count, freq.len())
}

fn smart_sort_to_indices_impl<T>(
    column_data: &PrimitiveArray<T>,
    direction: SortDirection,
) -> Result<Vec<u32>, ArrowError>
where
    T: ArrowPrimitiveType,
    T::Native: Ord,
{
    execute_with_adaptive_chunking!(
        data: column_data,
        chunk_size: get_chunk_size_sort(),
        process_chunk: |offset, chunk| {
            let indices = process_sort_chunk(chunk.as_ref(), direction)?;
            Ok::<(usize, Vec<u32>), ArrowError>((offset, indices))
        },
        single_chunk: |_offset, chunk| {
            process_sort_chunk(chunk.as_ref(), direction)
        },
        merge_results: |results| {
            let results: Vec<(usize, Vec<u32>)> =
                results.into_iter().collect::<Result<Vec<_>, _>>()?;

            let chunk_offsets: Vec<usize> = results.iter().map(|(offset, _)| *offset).collect();
            let locally_sorted_indices: Vec<Vec<u32>> =
                results.into_iter().map(|(_, indices)| indices).collect();

            Ok(k_way_merge_indices(
                column_data,
                &locally_sorted_indices,
                &chunk_offsets,
            ))
        }
    )
}

fn process_sort_chunk<T>(
    chunk: &PrimitiveArray<T>,
    direction: SortDirection,
) -> Result<Vec<u32>, ArrowError>
where
    T: ArrowPrimitiveType,
    T::Native: Ord,
{
    if chunk.len() < 2 {
        return Ok(vec![0; chunk.len()]);
    }

    let sample_ratio = get_config_sample_ratio("sort", "sort_feature_sample_ratio").unwrap_or(0.01);

    let (inversions, unique_count) = calculate_inversions_and_unique_count(chunk, sample_ratio);

    let num_sampled = (chunk.len() as f64 * sample_ratio) as usize;

    let normalized_inversion_pairs = if num_sampled < 2 {
        0.0
    } else {
        (inversions * 2) as f64 / (num_sampled * (num_sampled - 1)) as f64
    };

    let normalized_unique_count = if num_sampled == 0 {
        0.0
    } else {
        unique_count as f64 / num_sampled as f64
    };

    let petition = SortPetition {
        op_type: "sort",
        inversion_pairs: normalized_inversion_pairs,
        unique_count: normalized_unique_count,
        column: chunk,
        direction,
    };

    petition.adaptive_execute()
}

macro_rules! execute_fn_on_column_with_type_dispatch {
    ($column_data:expr, $execute_fn:expr) => {
        match $column_data.data_type() {
            arrow_schema::DataType::Int32 => {
                let arr = $column_data
                    .as_any()
                    .downcast_ref::<PrimitiveArray<arrow_array::types::Int32Type>>()
                    .unwrap();
                $execute_fn(arr)
            }
            arrow_schema::DataType::Int64 => {
                let arr = $column_data
                    .as_any()
                    .downcast_ref::<PrimitiveArray<arrow_array::types::Int64Type>>()
                    .unwrap();
                $execute_fn(arr)
            }
            arrow_schema::DataType::UInt64 => {
                let arr = $column_data
                    .as_any()
                    .downcast_ref::<PrimitiveArray<arrow_array::types::UInt64Type>>()
                    .unwrap();
                $execute_fn(arr)
            }
            arrow_schema::DataType::Decimal128(_, _) => {
                let arr = $column_data
                    .as_any()
                    .downcast_ref::<PrimitiveArray<arrow_array::types::Decimal128Type>>()
                    .unwrap();
                $execute_fn(arr)
            }
            _ => {
                return Err(ArrowError::NotYetImplemented(format!(
                    "Sort not implemented for type: {:?}",
                    $column_data.data_type()
                )))
            }
        }
    };
}

pub fn execute(
    data: &ArrowQuiver,
    column_name: &str,
    direction: SortDirection,
) -> Result<Selection, ArrowError> {
    execute_smart(data, column_name, direction)
}

fn execute_smart(
    data: &ArrowQuiver,
    column_name: &str,
    direction: SortDirection,
) -> Result<Selection, ArrowError> {
    let col_id = ColumnIdentifier::Name(column_name.into());
    let column_data = &data[&col_id];

    execute_fn_on_column_with_type_dispatch!(column_data, |arr| {
        smart_sort_to_indices_impl(arr, direction)
    })
    .map(Selection::SelVec)
}
