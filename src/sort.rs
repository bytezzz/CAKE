use arrow_array::{Array, Int32Array};
use arrow_schema::{ArrowError, DataType};

// Added imports for new array types
use arrow_array::{
    Decimal128Array, Float32Array, Float64Array, Int16Array, Int64Array, Int8Array, UInt16Array,
    UInt32Array, UInt64Array, UInt8Array,
};
use ordered_float::OrderedFloat; // Added import for OrderedFloat

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortAlgorithm {
    Quicksort,
    Heapsort,
    InsertionSort,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortDirection {
    Ascending,
    Descending,
}

fn compare_items<T: Ord>(a: &T, b: &T, direction: SortDirection) -> std::cmp::Ordering {
    match direction {
        SortDirection::Ascending => a.cmp(b),
        SortDirection::Descending => b.cmp(a),
    }
}

// Compare two (value, index) pairs with a stable tie-breaker on index.
// - Primary key: value (according to `direction`)
// - Tie-break: original index ascending, regardless of `direction`.
// This guarantees stability with respect to the original order.
fn compare_pairs<T: Ord>(
    a: &(T, u32),
    b: &(T, u32),
    direction: SortDirection,
) -> std::cmp::Ordering {
    let ord = compare_items(&a.0, &b.0, direction);
    if ord.is_eq() {
        // Keep original relative order among equal values
        a.1.cmp(&b.1)
    } else {
        ord
    }
}

// Quicksort
fn quicksort_tuples<T: Ord + Clone>(arr: &mut [(T, u32)], direction: SortDirection) {
    if arr.len() <= 1 {
        return;
    }
    // median_of_three_tuples modifies arr and ensures arr[pivot_idx] is the pivot.
    let pivot_idx = median_of_three_tuples(arr, direction);
    // partition_tuples then partitions around arr[pivot_idx]
    let (left, right) = partition_tuples(arr, pivot_idx, direction);
    quicksort_tuples(left, direction);
    quicksort_tuples(right, direction);
}

fn median_of_three_tuples<T: Ord>(arr: &mut [(T, u32)], direction: SortDirection) -> usize {
    let mid = arr.len() / 2;
    if arr.len() < 3 {
        // For very small arrays, mid is fine or could be 0.
        return mid;
    }
    let last = arr.len() - 1;

    // Sort arr[0], arr[mid], arr[last] by their values to find the median
    // and place it at arr[mid]
    if compare_pairs(&arr[0], &arr[mid], direction).is_gt() {
        arr.swap(0, mid);
    }
    if compare_pairs(&arr[mid], &arr[last], direction).is_gt() {
        arr.swap(mid, last);
    }
    if compare_pairs(&arr[0], &arr[mid], direction).is_gt() {
        arr.swap(0, mid);
    }

    mid // The element at mid is now the median of the three
}

#[allow(clippy::type_complexity)]
fn partition_tuples<T: Ord>(
    arr: &mut [(T, u32)],
    pivot_idx: usize,
    direction: SortDirection,
) -> (&mut [(T, u32)], &mut [(T, u32)]) {
    let len = arr.len();
    // Move pivot (which is at arr[pivot_idx]) to the end for partitioning
    arr.swap(pivot_idx, len - 1);
    let mut i = 0; // Index for elements smaller/equal to pivot

    for j in 0..(len - 1) {
        // If arr[j] should come before pivot (arr[len-1])
        if compare_pairs(&arr[j], &arr[len - 1], direction).is_le() {
            arr.swap(i, j);
            i += 1;
        }
    }
    // Move pivot to its final sorted place
    arr.swap(i, len - 1);

    let (left_slice, right_slice_with_pivot) = arr.split_at_mut(i);
    // The pivot is at right_slice_with_pivot[0], so exclude it from the right sub-array for recursion.
    (left_slice, &mut right_slice_with_pivot[1..])
}

// Heapsort
fn heapsort_tuples<T: Ord + Clone>(arr: &mut [(T, u32)], direction: SortDirection) {
    if arr.len() <= 1 {
        return;
    }
    let len = arr.len();
    // Build heap
    for i in (0..len / 2).rev() {
        heapify_tuples(arr, len, i, direction);
    }
    // Extract elements from heap
    for i in (1..len).rev() {
        arr.swap(0, i); // Move current root to end
        heapify_tuples(arr, i, 0, direction); // Call heapify on the reduced heap
    }
}

fn heapify_tuples<T: Ord>(arr: &mut [(T, u32)], n: usize, i: usize, direction: SortDirection) {
    let mut largest_or_smallest = i; // Initialize root
    let left_child = 2 * i + 1;
    let right_child = 2 * i + 2;

    // If left child is larger/smaller (based on direction) than root
    if left_child < n
        && compare_pairs(&arr[left_child], &arr[largest_or_smallest], direction).is_gt()
    {
        largest_or_smallest = left_child;
    }
    // If right child is larger/smaller than current largest_or_smallest
    if right_child < n
        && compare_pairs(&arr[right_child], &arr[largest_or_smallest], direction).is_gt()
    {
        largest_or_smallest = right_child;
    }

    // If largest_or_smallest is not root
    if largest_or_smallest != i {
        arr.swap(i, largest_or_smallest);
        // Recursively heapify the affected sub-tree
        heapify_tuples(arr, n, largest_or_smallest, direction);
    }
}

// Insertion sort
fn insertion_sort_tuples<T: Ord + Clone>(arr: &mut [(T, u32)], direction: SortDirection) {
    for i in 1..arr.len() {
        let mut j = i;
        // While j > 0 and arr[j] is less/greater (based on direction) than arr[j-1]
        while j > 0 && compare_pairs(&arr[j], &arr[j - 1], direction).is_lt() {
            arr.swap(j, j - 1);
            j -= 1;
        }
    }
}

// Macro to handle sorting for different numeric array types (integers)
macro_rules! sort_numeric_array {
    ($column_data:expr, $algorithm:expr, $direction:expr, $array_type:ty, $rust_type:ty) => {{
        let arr = $column_data
            .as_any()
            .downcast_ref::<$array_type>()
            .ok_or_else(|| {
                ArrowError::InvalidArgumentError(format!(
                    "Internal error: Failed to downcast to {}",
                    stringify!($array_type)
                ))
            })?;

        if arr.is_empty() {
            return Ok(Vec::new());
        }

        let mut val_idx_pairs: Vec<($rust_type, u32)> = arr
            .iter()
            .enumerate()
            .map(|(idx, val_opt)| (val_opt.unwrap_or_default(), idx as u32))
            .collect();

        match $algorithm {
            SortAlgorithm::Quicksort => quicksort_tuples(&mut val_idx_pairs, $direction),
            SortAlgorithm::Heapsort => heapsort_tuples(&mut val_idx_pairs, $direction),
            SortAlgorithm::InsertionSort => insertion_sort_tuples(&mut val_idx_pairs, $direction),
        }

        let indices = val_idx_pairs.into_iter().map(|(_, idx)| idx).collect();
        Ok(indices)
    }};
}

// Macro to handle sorting for float array types
macro_rules! sort_float_array {
    ($column_data:expr, $algorithm:expr, $direction:expr, $array_type:ty, $rust_type:ty) => {{
        let arr = $column_data
            .as_any()
            .downcast_ref::<$array_type>()
            .ok_or_else(|| {
                ArrowError::InvalidArgumentError(format!(
                    "Internal error: Failed to downcast to {}",
                    stringify!($array_type)
                ))
            })?;

        if arr.is_empty() {
            return Ok(Vec::new());
        }

        // Use OrderedFloat for sorting floats
        let mut val_idx_pairs: Vec<(OrderedFloat<$rust_type>, u32)> = arr
            .iter()
            .enumerate()
            .map(|(idx, val_opt)| (OrderedFloat(val_opt.unwrap_or_default()), idx as u32))
            .collect();

        match $algorithm {
            SortAlgorithm::Quicksort => quicksort_tuples(&mut val_idx_pairs, $direction),
            SortAlgorithm::Heapsort => heapsort_tuples(&mut val_idx_pairs, $direction),
            SortAlgorithm::InsertionSort => insertion_sort_tuples(&mut val_idx_pairs, $direction),
        }

        let indices = val_idx_pairs.into_iter().map(|(_, idx)| idx).collect();
        Ok(indices)
    }};
}

pub fn sort_array_to_indices(
    column_data: &dyn Array,
    algorithm: SortAlgorithm,
    direction: SortDirection,
) -> Result<Vec<u32>, ArrowError> {
    match column_data.data_type() {
        DataType::Int32 => {
            sort_numeric_array!(column_data, algorithm, direction, Int32Array, i32)
        }
        DataType::Int8 => {
            sort_numeric_array!(column_data, algorithm, direction, Int8Array, i8)
        }
        DataType::Int16 => {
            sort_numeric_array!(column_data, algorithm, direction, Int16Array, i16)
        }
        DataType::Int64 => {
            sort_numeric_array!(column_data, algorithm, direction, Int64Array, i64)
        }
        DataType::UInt8 => {
            sort_numeric_array!(column_data, algorithm, direction, UInt8Array, u8)
        }
        DataType::UInt16 => {
            sort_numeric_array!(column_data, algorithm, direction, UInt16Array, u16)
        }
        DataType::UInt32 => {
            sort_numeric_array!(column_data, algorithm, direction, UInt32Array, u32)
        }
        DataType::UInt64 => {
            sort_numeric_array!(column_data, algorithm, direction, UInt64Array, u64)
        }
        DataType::Float32 => {
            sort_float_array!(column_data, algorithm, direction, Float32Array, f32)
        }
        DataType::Float64 => {
            sort_float_array!(column_data, algorithm, direction, Float64Array, f64)
        }
        DataType::Decimal128(_, _) => {
            // Compare using the unscaled i128 native representation.
            // This preserves decimal ordering for a uniform precision/scale column.
            sort_numeric_array!(column_data, algorithm, direction, Decimal128Array, i128)
        }
        // Extend here for other DataTypes like Float64, Utf8, etc.
        // Example:
        // DataType::Float64 => { /* similar logic for Float64Array, T = f64 */ }
        // DataType::Utf8 => { /* similar logic for StringArray, T = String or &str */ }
        dt => Err(ArrowError::InvalidArgumentError(format!(
            "Sorting for DataType {:?} is not yet implemented. Only Int32 is currently supported.",
            dt
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int32Array;
    use std::sync::Arc; // Ensure Arc is in scope for tests too

    fn get_test_array() -> Arc<dyn Array> {
        // Original data: [Some(5), Some(1), Some(9), Some(3), Some(7), None, Some(2)]
        // Indices:         0,       1,       2,       3,       4,       5,       6
        // Values with None as 0: [5, 1, 9, 3, 7, 0, 2]
        Arc::new(Int32Array::from(vec![
            Some(5),
            Some(1),
            Some(9),
            Some(3),
            Some(7),
            None,
            Some(2),
        ]))
    }

    fn test_sorting_algorithm_impl(algo: SortAlgorithm) {
        let array = get_test_array();

        // Ascending Order
        // Expected sorted values (None as 0): [0, 1, 2, 3, 5, 7, 9]
        // Corresponding original indices:    [5, 1, 6, 3, 0, 4, 2]
        let indices_asc = sort_array_to_indices(&*array, algo, SortDirection::Ascending).unwrap();
        assert_eq!(
            indices_asc,
            vec![5, 1, 6, 3, 0, 4, 2],
            "Ascending sort failed for {:?}",
            algo
        );

        // Descending Order
        // Expected sorted values (None as 0): [9, 7, 5, 3, 2, 1, 0]
        // Corresponding original indices:    [2, 4, 0, 3, 6, 1, 5]
        let indices_desc = sort_array_to_indices(&*array, algo, SortDirection::Descending).unwrap();
        assert_eq!(
            indices_desc,
            vec![2, 4, 0, 3, 6, 1, 5],
            "Descending sort failed for {:?}",
            algo
        );

        // Test with empty array
        let empty_array: Arc<dyn Array> = Arc::new(Int32Array::from(Vec::<Option<i32>>::new()));
        let empty_indices =
            sort_array_to_indices(&*empty_array, algo, SortDirection::Ascending).unwrap();
        assert!(
            empty_indices.is_empty(),
            "Empty array sort failed for {:?}",
            algo
        );

        // Test with single element array
        let single_array: Arc<dyn Array> = Arc::new(Int32Array::from(vec![Some(42)]));
        let single_indices =
            sort_array_to_indices(&*single_array, algo, SortDirection::Ascending).unwrap();
        assert_eq!(
            single_indices,
            vec![0],
            "Single element array sort failed for {:?}",
            algo
        );

        // Test with already sorted array
        let sorted_array_asc: Arc<dyn Array> =
            Arc::new(Int32Array::from(vec![Some(1), Some(2), Some(3)]));
        let sorted_indices_asc =
            sort_array_to_indices(&*sorted_array_asc, algo, SortDirection::Ascending).unwrap();
        assert_eq!(
            sorted_indices_asc,
            vec![0, 1, 2],
            "Already sorted ascending failed for {:?}",
            algo
        );
        let sorted_indices_desc =
            sort_array_to_indices(&*sorted_array_asc, algo, SortDirection::Descending).unwrap();
        assert_eq!(
            sorted_indices_desc,
            vec![2, 1, 0],
            "Already sorted ascending, to descending sort, failed for {:?}",
            algo
        );
    }

    #[test]
    fn test_quicksort() {
        test_sorting_algorithm_impl(SortAlgorithm::Quicksort);
    }

    #[test]
    fn test_heapsort() {
        test_sorting_algorithm_impl(SortAlgorithm::Heapsort);
    }

    #[test]
    fn test_insertion_sort() {
        test_sorting_algorithm_impl(SortAlgorithm::InsertionSort);
    }
}
