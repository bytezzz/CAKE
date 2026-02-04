use arrow_array::{
    types::{ArrowPrimitiveType, ByteArrayType, Int32Type, Int64Type},
    Array, GenericByteArray, GenericListArray, GenericStringArray, LargeListArray, ListArray,
    OffsetSizeTrait, PrimitiveArray, StructArray,
};
use arrow_schema::DataType;
use cpu_time::ProcessTime;
use std::time::{Duration, Instant};

use arrow_buffer::{ArrowNativeType, BooleanBuffer};
use itertools::Itertools;
use random_word::Lang;

pub fn generate_str_vector(len: usize, repeat: usize) -> Vec<String> {
    (0..len)
        .map(|_| random_word::gen(Lang::En).repeat(repeat))
        .collect_vec()
}

pub fn generate_str_vector_random_contains(
    len: usize,
    repeat: usize,
    percent_contains: f64,
    pattern: &str,
) -> Vec<String> {
    let strarr = generate_str_vector(len, repeat);

    strarr
        .into_iter()
        .map(|mut s| {
            if fastrand::f64() < percent_contains {
                let pos = fastrand::usize(0..s.len());
                s.insert_str(pos, pattern);
            }
            s // Return `s` without unnecessary cloning
        })
        .collect()
}

#[macro_export]
macro_rules! get_env_or_default {
    ($env_var:expr, $default:expr, $t:ty) => {{
        std::env::var($env_var)
            .ok()
            .and_then(|v| v.trim().parse::<$t>().ok())
            .unwrap_or($default)
    }};
}

pub fn measure_time_avg<F>(f: F, n: usize) -> (Duration, Duration)
where
    F: FnMut(),
{
    let result = measure_time_raw(f, n);

    let total_wall_duration = result.wall_time.iter().sum::<Duration>();
    let total_cpu_duration = result.cpu_time.iter().sum::<Duration>();

    (
        total_wall_duration / n as u32,
        total_cpu_duration / n as u32,
    )
}

pub struct RawMeasureResult {
    pub wall_time: Vec<Duration>,
    pub cpu_time: Vec<Duration>,
}

pub fn measure_time_raw<F>(mut f: F, n: usize) -> RawMeasureResult
where
    F: FnMut(),
{
    let mut result = RawMeasureResult {
        wall_time: Vec::new(),
        cpu_time: Vec::new(),
    };

    let mut total_wall_duration = Duration::new(0, 0);
    let mut total_cpu_duration = Duration::new(0, 0);

    for _ in 0..n {
        let wall_start = Instant::now();
        let cpu_start = ProcessTime::now();
        f();
        result.wall_time.push(wall_start.elapsed());
        result.cpu_time.push(cpu_start.elapsed());
        total_wall_duration += wall_start.elapsed();
        total_cpu_duration += cpu_start.elapsed();
    }

    result
}

pub fn compute_runs(arr: &BooleanBuffer) -> u32 {
    let mut prev = false;
    let mut count = 0;
    for i in 0..arr.len() {
        let value = unsafe { arr.value_unchecked(i) };
        if prev && !value {
            count += 1;
        }
        prev = value;
    }
    count
}

// 辅助函数 1: GenericByteArray 的大小计算 (无变化)
fn get_generic_byte_array_slice_logical_size<T: ByteArrayType>(arr: &GenericByteArray<T>) -> usize
where
    T::Offset: OffsetSizeTrait,
{
    let slice_offset = arr.offset();
    let slice_length = arr.len();
    let offsets = arr.value_offsets();
    let values_size = offsets[slice_offset + slice_length] - offsets[slice_offset];
    let offsets_size = (slice_length + 1) * std::mem::size_of::<T::Offset>();
    let null_size = arr.nulls().map_or(0, |_| (slice_length + 7) / 8);
    values_size.to_usize().unwrap() + offsets_size + null_size
}

// 辅助函数 2 (新): 为 GenericListArray 实现的通用大小计算函数
fn get_generic_list_slice_logical_size<O: OffsetSizeTrait>(arr: &GenericListArray<O>) -> usize {
    let slice_offset = arr.offset();
    let slice_length = arr.len();

    // 计算 list 自身的 null 和 offset buffer 大小 (使用泛型 O)
    let self_offsets_size = (slice_length + 1) * std::mem::size_of::<O>();
    let self_null_size = arr.nulls().map_or(0, |_| (slice_length + 7) / 8);

    // 找出 values 数组中对应的 slice (使用 to_usize() 进行安全转换)
    let offsets = arr.value_offsets();
    let start_value_offset = offsets[slice_offset].to_usize().unwrap();
    let end_value_offset = offsets[slice_offset + slice_length].to_usize().unwrap();
    let values_len = end_value_offset - start_value_offset;
    let values_slice = arr.values().slice(start_value_offset, values_len);

    // 递归调用主函数
    self_offsets_size + self_null_size + get_array_slice_logical_size(&values_slice)
}

// 辅助宏: PrimitiveArray (无变化)
macro_rules! get_primitive_array_slice_logical_size {
    ($arr:expr, $T:ty) => {{
        let primitive_arr = $arr.as_any().downcast_ref::<PrimitiveArray<$T>>().unwrap();
        let values_size =
            primitive_arr.len() * std::mem::size_of::<<$T as ArrowPrimitiveType>::Native>();
        let null_size = primitive_arr
            .nulls()
            .map_or(0, |_| (primitive_arr.len() + 7) / 8);
        values_size + null_size
    }};
}

// 主函数: 现在 List 和 LargeList 的分支都调用了新的辅助函数
pub fn get_array_slice_logical_size(arr: &dyn Array) -> usize {
    match arr.data_type() {
        DataType::Utf8 => get_generic_byte_array_slice_logical_size(
            arr.as_any()
                .downcast_ref::<GenericStringArray<i32>>()
                .unwrap(),
        ),
        DataType::LargeUtf8 => get_generic_byte_array_slice_logical_size(
            arr.as_any()
                .downcast_ref::<GenericStringArray<i64>>()
                .unwrap(),
        ),
        DataType::Int32 => get_primitive_array_slice_logical_size!(arr, Int32Type),
        DataType::Int64 => get_primitive_array_slice_logical_size!(arr, Int64Type),
        DataType::Float32 => {
            get_primitive_array_slice_logical_size!(arr, arrow_array::types::Float32Type)
        }
        DataType::Float64 => {
            get_primitive_array_slice_logical_size!(arr, arrow_array::types::Float64Type)
        }
        DataType::Date32 => {
            get_primitive_array_slice_logical_size!(arr, arrow_array::types::Date32Type)
        }
        DataType::Date64 => {
            get_primitive_array_slice_logical_size!(arr, arrow_array::types::Date64Type)
        }
        DataType::Decimal128(_, _) => {
            get_primitive_array_slice_logical_size!(arr, arrow_array::types::Decimal128Type)
        }
        DataType::Decimal256(_, _) => {
            get_primitive_array_slice_logical_size!(arr, arrow_array::types::Decimal256Type)
        }
        // ... 其他原始类型
        DataType::Struct(_) => {
            let struct_arr = arr.as_any().downcast_ref::<StructArray>().unwrap();
            let self_null_size = struct_arr.nulls().map_or(0, |_| (struct_arr.len() + 7) / 8);
            let children_size: usize = struct_arr
                .columns()
                .iter()
                .map(|col| {
                    let child_slice = col.slice(struct_arr.offset(), struct_arr.len());
                    get_array_slice_logical_size(&child_slice)
                })
                .sum();
            self_null_size + children_size
        }

        // --- 重构后的代码 ---
        DataType::List(_) => {
            get_generic_list_slice_logical_size(arr.as_any().downcast_ref::<ListArray>().unwrap())
        }
        DataType::LargeList(_) => get_generic_list_slice_logical_size(
            arr.as_any().downcast_ref::<LargeListArray>().unwrap(),
        ),
        // --- 结束 ---
        _ => arr.get_buffer_memory_size(),
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::generate_str_vector_random_contains;

    #[test]
    fn test_generate_str_vector_random_contains() {
        let x = generate_str_vector_random_contains(10, 2, 0.5, "hello");
        println!("{:?}", x);
    }
}
