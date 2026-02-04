use arrow_array::{
    cast::AsArray,
    types::{
        Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type,
        UInt32Type, UInt64Type, UInt8Type,
    },
    Array, UInt64Array,
};
use arrow_schema::DataType;
use paste::paste;

const MURMUR_C1: u64 = 0xff51afd7ed558ccd;
const MURMUR_C2: u64 = 0xc4ceb9fe1a85ec53;
const UPPER_MASK: u64 = 0x00000000FFFFFFFF;

macro_rules! generate_prim_murmur {
    ($p: ty, $sp: ty) => {
        paste! {
            pub fn [<hash_ $p>](val: $p) -> u64 {
                let v = u64::from_le_bytes((val as $sp).to_le_bytes());
                let v = v ^ (v >> 33);
                let v = (v & UPPER_MASK).wrapping_mul(MURMUR_C1 & UPPER_MASK);
                let v = v ^ (v >> 33);
                let v = (v & UPPER_MASK).wrapping_mul(MURMUR_C2 & UPPER_MASK);
                let v = v ^ (v >> 33);
                v as u64
            }
        }
    };
}

generate_prim_murmur!(i8, i64);
generate_prim_murmur!(i16, i64);
generate_prim_murmur!(i32, i64);
generate_prim_murmur!(i64, i64);
generate_prim_murmur!(u8, u64);
generate_prim_murmur!(u16, u64);
generate_prim_murmur!(u32, u64);
generate_prim_murmur!(u64, u64);
generate_prim_murmur!(f32, f64);
generate_prim_murmur!(f64, f64);

// manual SIMD, which is nearly identical to what the compiler generates for the above
/*#[cfg(target_feature = "avx2")]
fn hash_i32_chunk(vals: &[i32; 4], out: &mut [u64; 4]) {
    unsafe {
        let v = _mm_loadu_si128(vals.as_ptr() as *const __m128i);
        let v = _mm256_cvtepi32_epi64(v);
        let v = _mm256_xor_si256(v, _mm256_srli_epi64::<33>(v));
        let v = _mm256_mul_epu32(v, _mm256_set1_epi64x(MURMUR_C1 as i64));
        let v = _mm256_xor_si256(v, _mm256_srli_epi64::<33>(v));
        let v = _mm256_mul_epu32(v, _mm256_set1_epi64x(MURMUR_C2 as i64));
        let v = _mm256_xor_si256(v, _mm256_srli_epi64::<33>(v));
        _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, v);
    }
}*/

/// Computes a hash of an array. Uses a modified SIMD-enabled murmur2.
/// ```
/// # use arrow_quiver::hash;
/// # use arrow_quiver::ArrowQuiver;
/// let aq = ArrowQuiver::i32_col(vec![200, 100, 50, 50, 200]);
/// let hash_vals = hash(&aq[&0.into()]);
/// assert_eq!(hash_vals.len(), 5);
/// ```
pub fn hash(data: &dyn Array) -> UInt64Array {
    match data.data_type() {
        DataType::Null => todo!(),
        DataType::Boolean => todo!(),
        DataType::Int8 => data.as_primitive::<Int8Type>().unary(hash_i8),
        DataType::Int16 => data.as_primitive::<Int16Type>().unary(hash_i16),
        DataType::Int32 => data.as_primitive::<Int32Type>().unary(hash_i32),
        DataType::Int64 => data.as_primitive::<Int64Type>().unary(hash_i64),
        DataType::UInt8 => data.as_primitive::<UInt8Type>().unary(hash_u8),
        DataType::UInt16 => data.as_primitive::<UInt16Type>().unary(hash_u16),
        DataType::UInt32 => data.as_primitive::<UInt32Type>().unary(hash_u32),
        DataType::UInt64 => data.as_primitive::<UInt64Type>().unary(hash_u64),
        DataType::Float16 => todo!(),
        DataType::Float32 => data.as_primitive::<Float32Type>().unary(hash_f32),
        DataType::Float64 => data.as_primitive::<Float64Type>().unary(hash_f64),
        DataType::Timestamp(_, _) => todo!(),
        DataType::Date32 => todo!(),
        DataType::Date64 => todo!(),
        DataType::Time32(_) => todo!(),
        DataType::Time64(_) => todo!(),
        DataType::Duration(_) => todo!(),
        DataType::Interval(_) => todo!(),
        DataType::Binary => todo!(),
        DataType::FixedSizeBinary(_) => todo!(),
        DataType::LargeBinary => todo!(),
        DataType::BinaryView => todo!(),
        DataType::Utf8 => todo!(),
        DataType::LargeUtf8 => todo!(),
        DataType::Utf8View => todo!(),
        DataType::List(_) => todo!(),
        DataType::ListView(_) => todo!(),
        DataType::FixedSizeList(_, _) => todo!(),
        DataType::LargeList(_) => todo!(),
        DataType::LargeListView(_) => todo!(),
        DataType::Struct(_) => todo!(),
        DataType::Union(_, _) => todo!(),
        DataType::Dictionary(_, _) => todo!(),
        DataType::Decimal128(_, _) => todo!(),
        DataType::Decimal256(_, _) => todo!(),
        DataType::Map(_, _) => todo!(),
        DataType::RunEndEncoded(_, _) => todo!(),
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::Int32Array;
    use itertools::Itertools;

    use super::*;

    #[test]
    fn test_vec_h_i32() {
        let data = (-1000..1000).collect_vec();
        let arr = Int32Array::from(data.clone());
        let hashes = hash(&arr);
        assert_eq!(hashes.len(), data.len());

        for (res, el) in hashes.values().iter().zip(data.iter()) {
            let should_be = hash_i32(*el);
            assert_eq!(*res, should_be);
        }
    }
}
