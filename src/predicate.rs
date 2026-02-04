use arrow_array::{
    cast::{as_largestring_array, as_string_array, AsArray},
    downcast_primitive_array,
    types::ArrowPrimitiveType,
    Array, Datum, GenericStringArray, Int64Array, OffsetSizeTrait, PrimitiveArray, Scalar,
    UInt64Array,
};
use arrow_buffer::BooleanBuffer;
use arrow_cast::cast;
use arrow_ord::cmp;
use arrow_schema::DataType;
use itertools::Itertools;
use log::warn;
use memchr::memmem;
use std::{ops::Not, sync::Arc};

use crate::{hash::hash, xor_filter::AQXorFilter16};
use crate::{selection::sel_intersect::Intersector, ArrowQuiver, ColumnIdentifier, Selection};

#[derive(Clone)]
pub enum Predicate {
    IsFiltered,
    NotNull(ColumnIdentifier),
    IsNull(ColumnIdentifier),
    LessThanConst(ColumnIdentifier, Arc<dyn Datum>),
    LessThanOrEqualConst(ColumnIdentifier, Arc<dyn Datum>),
    GreaterThanConst(ColumnIdentifier, Arc<dyn Datum>),
    GreaterThanOrEqualConst(ColumnIdentifier, Arc<dyn Datum>),
    NotEqConst(ColumnIdentifier, Arc<dyn Datum>),
    Between(ColumnIdentifier, Arc<dyn Datum>, Arc<dyn Datum>),
    PartialLessThanConst(ColumnIdentifier, Arc<dyn Datum>, Box<Predicate>),
    EqConst(ColumnIdentifier, Arc<dyn Datum>),
    Eq(ColumnIdentifier, ColumnIdentifier),
    And(Box<Predicate>, Box<Predicate>),
    Or(Box<Predicate>, Box<Predicate>),
    Not(Box<Predicate>),
    In(ColumnIdentifier, Vec<Arc<dyn Datum>>),
    StringContains(ColumnIdentifier, Arc<dyn Datum>),
    Like(ColumnIdentifier, Arc<dyn Datum>),
    ParitialStringContains(ColumnIdentifier, Arc<dyn Datum>, Box<Selection>),

    XorFilterIn(ColumnIdentifier, Arc<AQXorFilter16>),
    PartialXorFilterIn(ColumnIdentifier, Arc<AQXorFilter16>, Box<Predicate>),

    HintSelectionVector(Box<Predicate>),
    HintBitmap(Box<Predicate>),
}

unsafe impl Send for Predicate {}
unsafe impl Sync for Predicate {} // safe because our Datums are always singletons

/// Cast a scalar `Datum` to match the column's data type.
/// This avoids runtime errors like `Invalid comparison operation: Int32 < Int64`.
fn cast_singleton_to_column_type<'a>(col: &dyn Array, k: &dyn Datum) -> Arc<dyn Array> {
    let (arr, sing) = k.get();
    assert!(sing, "cannot cast non-singleton datum for comparison");
    if arr.data_type() == col.data_type() {
        // Ensure we return an owned Array of the same type as `arr`.
        // `cast` with the same type is cheap and returns an `ArrayRef`.
        cast(arr, col.data_type()).expect("cast to same type should not fail")
    } else {
        cast(arr, col.data_type())
            .unwrap_or_else(|_| panic!("could not convert {:?} to {}", arr, col.data_type()))
    }
}

/// Cast a scalar `Datum` to a scalar of the column's data type, preserving
/// singleton semantics expected by Arrow's comparison kernels.
fn cast_singleton_to_column_scalar(col: &dyn Array, k: &dyn Datum) -> Arc<dyn Datum> {
    match col.data_type() {
        DataType::Int8 => Arc::new(PrimitiveArray::<arrow_array::types::Int8Type>::new_scalar(
            cast_singleton_to_native::<arrow_array::types::Int8Type>(k),
        )),
        DataType::Int16 => Arc::new(PrimitiveArray::<arrow_array::types::Int16Type>::new_scalar(
            cast_singleton_to_native::<arrow_array::types::Int16Type>(k),
        )),
        DataType::Int32 => Arc::new(PrimitiveArray::<arrow_array::types::Int32Type>::new_scalar(
            cast_singleton_to_native::<arrow_array::types::Int32Type>(k),
        )),
        DataType::Int64 => Arc::new(PrimitiveArray::<arrow_array::types::Int64Type>::new_scalar(
            cast_singleton_to_native::<arrow_array::types::Int64Type>(k),
        )),
        DataType::UInt8 => Arc::new(PrimitiveArray::<arrow_array::types::UInt8Type>::new_scalar(
            cast_singleton_to_native::<arrow_array::types::UInt8Type>(k),
        )),
        DataType::UInt16 => Arc::new(
            PrimitiveArray::<arrow_array::types::UInt16Type>::new_scalar(
                cast_singleton_to_native::<arrow_array::types::UInt16Type>(k),
            ),
        ),
        DataType::UInt32 => Arc::new(
            PrimitiveArray::<arrow_array::types::UInt32Type>::new_scalar(
                cast_singleton_to_native::<arrow_array::types::UInt32Type>(k),
            ),
        ),
        DataType::UInt64 => Arc::new(
            PrimitiveArray::<arrow_array::types::UInt64Type>::new_scalar(
                cast_singleton_to_native::<arrow_array::types::UInt64Type>(k),
            ),
        ),
        DataType::Float32 => Arc::new(
            PrimitiveArray::<arrow_array::types::Float32Type>::new_scalar(
                cast_singleton_to_native::<arrow_array::types::Float32Type>(k),
            ),
        ),
        DataType::Float64 => Arc::new(
            PrimitiveArray::<arrow_array::types::Float64Type>::new_scalar(
                cast_singleton_to_native::<arrow_array::types::Float64Type>(k),
            ),
        ),
        // Decimal128 requires preserving the precision and scale of the column
        // type. Construct a scalar by casting the input singleton to the
        // column's exact Decimal128(precision, scale) then wrapping it.
        DataType::Decimal128(precision, scale) => {
            // Cast input to the exact Decimal128(precision, scale)
            let casted = cast_singleton_to_column_type(col, k);
            let v = casted
                .as_primitive::<arrow_array::types::Decimal128Type>()
                .value(0);
            // Create a Decimal128 scalar with the same precision/scale
            let arr = PrimitiveArray::<arrow_array::types::Decimal128Type>::new_scalar(v)
                .into_inner()
                .with_precision_and_scale(*precision, *scale)
                .expect("invalid decimal precision/scale");
            Arc::new(Scalar::new(arr))
        }
        other => panic!(
            "cast_singleton_to_column_scalar does not support type {}",
            other
        ),
    }
}

fn cast_singleton_to_matching_native<T: ArrowPrimitiveType>(
    _to_match: &PrimitiveArray<T>,
    k: &dyn Datum,
) -> T::Native {
    cast_singleton_to_native::<T>(k)
}

pub fn cast_singleton_to_native<T: ArrowPrimitiveType>(k: &dyn Datum) -> T::Native {
    let (k, sing) = k.get();
    assert!(sing, "cannot cast {:?} to singleton", k);

    let k = cast(k, &T::DATA_TYPE)
        .unwrap_or_else(|_| panic!("could not convert {:?} to {}", k, T::DATA_TYPE));

    k.as_primitive::<T>().value(0)
}

fn cast_singleton_to_string(k: &dyn Datum) -> &str {
    let (k, sing) = k.get();
    assert!(sing, "cannot cast {:?} to singleton", k);

    match k.data_type() {
        DataType::Utf8 => as_string_array(k).value(0),
        DataType::LargeUtf8 => as_largestring_array(k).value(0),
        _ => panic!("cannot cast {:?} to string", k),
    }
}

fn simd_string_contains<T: OffsetSizeTrait>(
    arr: &GenericStringArray<T>,
    pattern: &str,
) -> BooleanBuffer {
    use super::selection::idx_list_to_bb;
    // Fast path – if pattern is empty, nothing matches.
    if pattern.is_empty() {
        return idx_list_to_bb(&Vec::<u32>::new(), arr.len());
    }

    // Offsets are still relative to the original concatenated buffer, even after
    // slicing.  We normalise them so the first string starts at 0 **and** trim the
    // search window so it ends exactly at the last byte of the slice.

    let value_offsets = arr.value_offsets();
    let first_byte = value_offsets[0].as_usize();
    let last_byte = value_offsets[value_offsets.len() - 1].as_usize();

    // Safe because these indices lie inside the original value buffer.
    let cat_arr = &arr.value_data()[first_byte..last_byte];

    // Pre-compute relative offsets so we can translate a hit position back to a row idx.
    // Doing this once keeps the hot loop small and branch-free.
    let rel_offsets: Vec<usize> = value_offsets
        .iter()
        .map(|o| o.as_usize() - first_byte)
        .collect();

    let mut finder = memmem::find_iter(cat_arr, pattern.as_bytes());

    let mut offset_idx = 0;
    let mut results = vec![];
    let mut base = 0;

    while let Some(uncorrected_idx) = finder.next() {
        let idx = base + uncorrected_idx;
        while offset_idx < arr.len() - 1 && idx >= rel_offsets[offset_idx + 1] {
            offset_idx += 1;
        }
        if offset_idx != arr.len() - 1 && idx + pattern.len() > rel_offsets[offset_idx + 1] {
            finder = memmem::find_iter(&cat_arr[rel_offsets[offset_idx + 1]..], pattern.as_bytes());
            base = rel_offsets[offset_idx + 1];
        } else if results.is_empty() || results[results.len() - 1] != offset_idx as u32 {
            results.push(offset_idx as u32);
        }
    }
    idx_list_to_bb(&results, arr.len())
}

impl Predicate {
    pub fn apply(&self, aq: &ArrowQuiver, intersector: &Intersector) -> Selection {
        match self {
            Predicate::IsFiltered => aq.sel().clone(),
            Predicate::NotNull(ci) => {
                let col = &aq[ci];
                match col.nulls() {
                    Some(nm) => Selection::Bitmap(nm.inner().clone()),
                    None => Selection::AllValid,
                }
            }
            Predicate::IsNull(ci) => {
                let col = &aq[ci];
                match col.nulls() {
                    Some(nm) => Selection::Bitmap(nm.inner().clone().not()),
                    None => Selection::NoneValid,
                }
            }
            Predicate::Eq(ci1, ci2) => {
                let col1 = &aq[ci1];
                let col2 = &aq[ci2];
                Selection::Bitmap(cmp::eq(col1, col2).unwrap().into_parts().0)
            }
            Predicate::EqConst(ci, k) => {
                let col = &aq[ci];
                let k_cast = cast_singleton_to_column_scalar(col.as_ref(), k.as_ref());
                Selection::Bitmap(cmp::eq(col, k_cast.as_ref()).unwrap().into_parts().0)
            }
            Predicate::In(ci, k) => {
                let col = &aq[ci];
                let k = k
                    .iter()
                    .map(|x| {
                        let xc = cast_singleton_to_column_scalar(col.as_ref(), x.as_ref());
                        cmp::eq(col, xc.as_ref()).unwrap().into_parts().0
                    })
                    .collect_vec();
                let k = k.into_iter().reduce(|a, b| &a | &b).unwrap();
                Selection::Bitmap(k)
            }
            Predicate::LessThanConst(ci, k) => {
                let col = &aq[ci];
                let k_cast = cast_singleton_to_column_scalar(col.as_ref(), k.as_ref());
                let (r, _n) = cmp::lt(col, k_cast.as_ref()).unwrap().into_parts();
                Selection::Bitmap(r)
            }
            Predicate::Between(ci, low, high) => {
                let col = &aq[ci];
                let low_cast = cast_singleton_to_column_scalar(col.as_ref(), low.as_ref());
                let high_cast = cast_singleton_to_column_scalar(col.as_ref(), high.as_ref());
                let (r1, _n) = cmp::gt(col, low_cast.as_ref()).unwrap().into_parts();
                let (r2, _n) = cmp::lt(col, high_cast.as_ref()).unwrap().into_parts();
                let r = &r1 & &r2;
                Selection::Bitmap(r)
            }
            Predicate::LessThanOrEqualConst(ci, k) => {
                let col = &aq[ci];
                let k_cast = cast_singleton_to_column_scalar(col.as_ref(), k.as_ref());
                let (r, _n) = cmp::lt_eq(col, k_cast.as_ref()).unwrap().into_parts();
                Selection::Bitmap(r)
            }
            Predicate::GreaterThanConst(ci, k) => {
                let col = &aq[ci];
                let k_cast = cast_singleton_to_column_scalar(col.as_ref(), k.as_ref());
                let (r, _n) = cmp::gt(col, k_cast.as_ref()).unwrap().into_parts();
                Selection::Bitmap(r)
            }
            Predicate::GreaterThanOrEqualConst(ci, k) => {
                let col = &aq[ci];
                let k_cast = cast_singleton_to_column_scalar(col.as_ref(), k.as_ref());
                let (r, _n) = cmp::gt_eq(col, k_cast.as_ref()).unwrap().into_parts();
                Selection::Bitmap(r)
            }
            Predicate::NotEqConst(ci, k) => {
                let col = &aq[ci];
                let k_cast = cast_singleton_to_column_scalar(col.as_ref(), k.as_ref());
                let (r, _n) = cmp::neq(col, k_cast.as_ref()).unwrap().into_parts();
                Selection::Bitmap(r)
            }
            Predicate::PartialLessThanConst(ci, k, sel) => {
                let col = &aq[ci];
                let sel = sel.apply(aq, intersector);

                // special case: check for a SIMD selection kernel
                #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                match &sel {
                    Selection::SelVec(vec) => match col.data_type() {
                        DataType::Int32 => {
                            let arr: &PrimitiveArray<Int32Type> = col.as_primitive();
                            let k = cast_singleton_to_native::<Int32Type>(k.as_ref());
                            return Selection::SelVec(simd_lt_const_i32(arr.values(), vec, k));
                        }
                        _ => {
                            warn!("No SIMD kernel for selection over type {}", col.data_type());
                        }
                    },
                    Selection::Bitmap(bb) => match col.data_type() {
                        DataType::Int32 => {
                            let arr: &PrimitiveArray<Int32Type> = col.as_primitive();
                            let k = cast_singleton_to_native::<Int32Type>(k.as_ref());
                            return Selection::SelVec(simd_lt_const_i32_bm(arr.values(), bb, k));
                        }
                        _ => {
                            warn!("No SIMD kernel for selection over type {}", col.data_type());
                        }
                    },
                    _ => {}
                }

                // no special kernel -- use an Arrow kernel or a non-explicit SIMD
                downcast_primitive_array!(
                    col => {
                        let k = cast_singleton_to_matching_native(col, k.as_ref());
                        sel.filter(col, |x| x < k)
                    }
                    _t => {
                        warn!("No partial less implementation for type {}, using full evaluation", col.data_type());
                        let k_cast = cast_singleton_to_column_scalar(col.as_ref(), k.as_ref());
                        let (r, _n) = cmp::lt(col, k_cast.as_ref()).unwrap().into_parts();
                        Selection::Bitmap(r)
                    }
                )
            }
            Predicate::And(p1, p2) => {
                let p1 = p1.apply(aq, intersector);
                let p2 = p2.apply(aq, intersector);
                p1.intersect(intersector, p2)
            }
            Predicate::Or(p1, p2) => {
                let p1 = p1.apply(aq, intersector);
                let p2 = p2.apply(aq, intersector);
                p1.union(p2)
            }
            Predicate::Not(p) => p.apply(aq, intersector).invert(aq.num_rows()),
            Predicate::HintSelectionVector(p) => p.apply(aq, intersector).into_sel_vec(),
            Predicate::HintBitmap(p) => p.apply(aq, intersector).into_bitmap(aq.num_rows()),
            Predicate::StringContains(ci, substring) => {
                let col = &aq[ci];
                let bitmap = match col.data_type() {
                    DataType::Utf8 => simd_string_contains(
                        as_string_array(col),
                        cast_singleton_to_string(substring.as_ref()),
                    ),
                    DataType::LargeUtf8 => simd_string_contains(
                        as_largestring_array(col),
                        cast_singleton_to_string(substring.as_ref()),
                    ),
                    _ => {
                        panic!("No SIMD kernel for selection over type {}", col.data_type());
                    }
                };
                Selection::Bitmap(bitmap)
            }
            Predicate::Like(ci, substring) => {
                let col = &aq[ci];
                let bb = arrow_string::like::like(col, substring.as_ref()).unwrap();
                Selection::Bitmap(bb.into_parts().0)
            }
            Predicate::ParitialStringContains(ci, substring, selection) => {
                let array_ref = &aq[ci];
                let pattern_str = cast_singleton_to_string(substring.as_ref());
                let finder = memchr::memmem::Finder::new(pattern_str.as_bytes());

                match array_ref.data_type() {
                    DataType::Utf8 => {
                        let string_array = as_string_array(array_ref);
                        selection.filter_string_array(string_array, |val: &str| {
                            finder.find(val.as_bytes()).is_some()
                        })
                    }
                    DataType::LargeUtf8 => {
                        let largestring_array = as_largestring_array(array_ref);
                        selection.filter_string_array(largestring_array, |val: &str| {
                            finder.find(val.as_bytes()).is_some()
                        })
                    }
                    _ => panic!(
                        "ParitialStringContains is not supported for type {}",
                        array_ref.data_type()
                    ),
                }
            }
            Predicate::XorFilterIn(ci, filter) => filter.test(ci, aq),
            Predicate::PartialXorFilterIn(ci, filter, sel) => {
                let col = &aq[ci];
                let sel = sel.apply(aq, intersector);
                let hash_arr = match col.data_type() {
                    DataType::Int64 => {
                        let arr: &Int64Array = col.as_primitive();
                        UInt64Array::from(sel.apply_primitive(arr, |x| x as u64))
                    }
                    DataType::UInt64 => {
                        let arr: &UInt64Array = col.as_primitive();
                        UInt64Array::from(sel.apply_primitive(arr, |x| x))
                    }
                    _ => {
                        let after_filter = sel.clone_and_filter(col.clone());
                        hash(&after_filter)
                    }
                };
                let sel_vec = sel.into_sel_vec();
                let idxarr = sel_vec.as_sel_vec().unwrap();
                let sv = hash_arr
                    .values()
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, v)| filter.test_with_hash(v).then_some(idxarr[idx]))
                    .collect_vec();
                Selection::SelVec(sv)
            }
        }
    }

    /// Recursive depth of the predicate (in terms of logical operators, AND, OR, NOT).
    pub fn max_depth(&self) -> usize {
        match self {
            Predicate::And(p1, p2) => usize::max(p1.max_depth(), p2.max_depth()) + 1,
            Predicate::Or(p1, p2) => usize::max(p1.max_depth(), p2.max_depth()) + 1,
            Predicate::Not(p) => p.max_depth() + 1,
            Predicate::PartialLessThanConst(_, _, p) => p.max_depth() + 1,
            Predicate::ParitialStringContains(_, _, _) => 0,
            Predicate::PartialXorFilterIn(_, _, p) => p.max_depth() + 1,
            Predicate::HintSelectionVector(p) => p.max_depth(),
            Predicate::HintBitmap(p) => p.max_depth(),
            _ => 0,
        }
    }
}

/// Transforms a vector of predicates into a single AND predicate, with a balanced tree of
/// ANDs (i.e., as few ANDs as possible). This can perform better than a left-deep chain
/// of ANDs since fewer total boolean operations are performed.
///
/// ```
/// use arrow_quiver::predicate::{Predicate, multi_and};
/// let to_combine = vec![Box::new(Predicate::Eq(0.into(), 1.into())),
///                       Box::new(Predicate::Eq(1.into(), 2.into())),
///                       Box::new(Predicate::Eq(2.into(), 3.into())),
///                       Box::new(Predicate::Eq(3.into(), 4.into()))];
///
/// let combined = multi_and(to_combine);
///
/// assert_eq!(combined.max_depth(), 2);
/// ```
pub fn multi_and(mut expr: Vec<Box<Predicate>>) -> Box<Predicate> {
    match expr.len() {
        0 => panic!("cannot build multi-and with zero predicates"),
        1 => expr.into_iter().next().unwrap(),
        _ => {
            let rhs = multi_and(expr.split_off(expr.len() / 2));
            let lhs = multi_and(expr);
            Box::new(Predicate::And(lhs, rhs))
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
const fn perm_for_byte(b: u8) -> [i32; 8] {
    let mut result = [0_i32; 8];
    let mut next_output = 0;

    let mut i = 0;
    while i < 8 {
        if b & (1 << i) > 0 {
            result[i] = next_output;
            next_output += 1;
        }
        i += 1;
    }

    i = 0;
    while i < 8 {
        if b & (1 << i) == 0 {
            result[i] = next_output;
            next_output += 1;
        }
        i += 1;
    }

    result
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
const fn all_perms() -> [[i32; 8]; 256] {
    let mut result = [[0_i32; 8]; 256];

    let mut b: u8 = 0;
    loop {
        result[b as usize] = perm_for_byte(b);
        b = b.wrapping_add(1);

        if b == 0 {
            break;
        }
    }

    result
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
const PERMUTE: [[i32; 8]; 256] = all_perms();

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub fn simd_lt_const_i32(data: &[i32], s: &[u32], k: i32) -> Vec<u32> {
    assert!(data.len() < i32::MAX as usize);

    use std::{
        arch::x86_64::{
            __m256i, _mm256_castsi256_ps, _mm256_cmpgt_epi32, _mm256_i32gather_epi32,
            _mm256_loadu_si256, _mm256_movemask_ps, _mm256_permutevar8x32_epi32, _mm256_set1_epi32,
        },
        mem,
    };

    let mut results = Vec::with_capacity(s.len());
    let k_vec = unsafe { _mm256_set1_epi32(k) };

    s.chunks_exact(8).for_each(|c| unsafe {
        let offsets = _mm256_loadu_si256(c.as_ptr() as *const __m256i);
        let data = _mm256_i32gather_epi32::<4>(data.as_ptr(), offsets);
        let gt = _mm256_cmpgt_epi32(k_vec, data);
        let mask = _mm256_movemask_ps(_mm256_castsi256_ps(gt)) as usize;

        let num_match = mask.count_ones();
        if num_match > 0 {
            let perm = PERMUTE[mask];
            let packed: [u32; 8] = mem::transmute(_mm256_permutevar8x32_epi32(
                offsets,
                mem::transmute::<[i32; 8], std::arch::x86_64::__m256i>(perm),
            ));
            results.extend_from_slice(&packed[..num_match as usize]);
        }
    });

    // deal with the tail
    for tail_idx in &s[(s.len() / 8) * 8..s.len()] {
        if data[*tail_idx as usize] < k {
            results.push(*tail_idx);
        }
    }

    results
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
fn simd_lt_const_i32_bm(data: &[i32], bm: &BooleanBuffer, k: i32) -> Vec<u32> {
    use std::{
        arch::x86_64::{
            _mm256_castsi256_ps, _mm256_cmpgt_epi32, _mm256_i32gather_epi32, _mm256_movemask_ps,
            _mm256_permutevar8x32_epi32, _mm256_set1_epi32,
        },
        mem,
    };

    let mut results = Vec::with_capacity(bm.len());
    let mut indexes: [i32; 8] = [0, 0, 0, 0, 0, 0, 0, 0];
    let mut idx_pos = 0;

    for (byte_idx, byte) in bm.values().iter().enumerate() {
        if *byte == 0 {
            continue;
        }

        let perm = PERMUTE[*byte as usize];
        for i in 0..byte.count_ones() as usize {
            indexes[idx_pos] = (byte_idx * 8) as i32 + perm[i];
            idx_pos += 1;

            if idx_pos == 8 {
                unsafe {
                    let offsets = mem::transmute::<[i32; 8], std::arch::x86_64::__m256i>(indexes);
                    let data = _mm256_i32gather_epi32::<4>(data.as_ptr(), offsets);
                    let gt = _mm256_cmpgt_epi32(_mm256_set1_epi32(k), data);
                    let mask = _mm256_movemask_ps(_mm256_castsi256_ps(gt)) as usize;
                    let num_match = mask.count_ones();
                    if num_match > 0 {
                        let perm = PERMUTE[mask];
                        let packed: [u32; 8] = mem::transmute(_mm256_permutevar8x32_epi32(
                            offsets,
                            mem::transmute::<[i32; 8], std::arch::x86_64::__m256i>(perm),
                        ));
                        results.extend_from_slice(&packed[..num_match as usize]);
                    }
                }
                idx_pos = 0;
            }
        }
    }

    // deal with the tail
    for idx in &indexes[0..idx_pos] {
        if data[*idx as usize] < k {
            results.push(*idx as u32);
        }
    }

    results
}

#[cfg(test)]
mod tests {

    use std::sync::Arc;

    use arrow_array::{Int32Array, Int64Array};
    use itertools::Itertools;

    use crate::{
        predicate::Predicate,
        selection::{sel_intersect::Intersector, Selection},
        ArrowQuiver, ColumnIdentifier,
    };

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn test_simd_lt_i32() {
        use itertools::Itertools;

        use super::simd_lt_const_i32;
        let data = (500..520).collect_vec();
        let indexes = (0..20).collect_vec();
        let r = simd_lt_const_i32(&data, &indexes, 510);
        assert_eq!(r, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn test_simd_lt_i32_bm() {
        use super::simd_lt_const_i32_bm;
        use itertools::Itertools;

        let data = (500..520).collect_vec();
        let indexes = Selection::SelVec((0..20).collect_vec()).into_bitmap(20);
        let r = simd_lt_const_i32_bm(&data, indexes.as_bitmap().unwrap(), 510);
        assert_eq!(r, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_string_contains() {
        use super::*;
        use crate::utils::generate_str_vector_random_contains;
        use arrow_array::StringArray;
        use random_word::Lang;
        use std::sync::Arc;

        for _ in 0..1000000 {
            let pattern = random_word::gen(Lang::En);
            let array = StringArray::from(generate_str_vector_random_contains(10, 1, 0.6, pattern));
            let strr = pattern;
            let pattern = Arc::new(StringArray::new_scalar(pattern));

            let aq = ArrowQuiver::new("test".to_string(), vec!["a".into()], vec![Arc::new(array)]);

            let pred = Predicate::StringContains(ColumnIdentifier::Index(0), pattern.clone());
            let intersector = Intersector::new();
            let sel = pred.apply(&aq, &intersector);
            let ground_truth = arrow_string::like::contains(&aq[0], pattern.as_ref());
            let (bb, _) = ground_truth.unwrap().into_parts();
            assert_eq!(
                sel.as_bitmap().unwrap().clone(),
                bb,
                "pattern: {}, array: {:?}",
                strr,
                aq[0]
            );
        }
    }
    #[test]
    fn test_in() {
        let aq = ArrowQuiver::i32_col((0..10).collect_vec());
        let pred = Predicate::In(
            ColumnIdentifier::Index(0),
            vec![
                Arc::new(Int32Array::new_scalar(8)),
                Arc::new(Int32Array::new_scalar(9)),
            ],
        );
        let intersector = Intersector::new();
        let sel = pred.apply(&aq, &intersector);
        let bb = sel.as_bitmap().unwrap();
        assert_eq!(bb.len(), 10);
        assert!(bb.value(8));
        assert!(bb.value(9));
        assert!(!bb.value(0));
        assert!(!bb.value(1));
        assert!(!bb.value(2));
        assert!(!bb.value(3));
        assert!(!bb.value(4));
        assert!(!bb.value(5));
        assert!(!bb.value(6));
        assert!(!bb.value(7));
    }

    #[test]
    fn test_mixed_int_scalar_cast_lt() {
        // Int32 column with Int64 scalar constant should work via implicit cast
        let aq = ArrowQuiver::i32_col((0..10).collect_vec());
        let pred = Predicate::LessThanConst(
            ColumnIdentifier::Index(0),
            Arc::new(Int64Array::new_scalar(5)),
        );
        let intersector = Intersector::new();
        let sel = pred.apply(&aq, &intersector).into_sel_vec();
        let v = sel.as_sel_vec().unwrap().to_vec();
        assert_eq!(v, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_hash_predicate() {
        use crate::xor_filter::AQXorFilter16;
        use std::sync::Arc;
        let aq = ArrowQuiver::i32_col((0..10).collect_vec());

        let xorfilter = Arc::new(AQXorFilter16::build_from(&Int32Array::from(vec![
            4, 5, 7, 9,
        ])));

        let prev_pred = Box::new(Predicate::LessThanConst(
            ColumnIdentifier::Index(0),
            Arc::new(Int32Array::new_scalar(8)),
        ));

        let pred = Predicate::PartialXorFilterIn(ColumnIdentifier::Index(0), xorfilter, prev_pred);

        let intersector = Intersector::new();

        let result = pred.apply(&aq, &intersector);

        assert_eq!(result.as_sel_vec().unwrap(), [4u32, 5u32, 7u32]);
    }

    #[test]
    fn test_simd_string_contains_slice_regression() {
        use super::simd_string_contains;
        use arrow_array::cast::as_string_array;
        use arrow_array::StringArray;

        // Original buffer with a match at index 0 and no match afterwards
        let full = StringArray::from(vec![
            "hello world",   // idx 0 – contains pattern
            "goodbye world", // idx 1 – does NOT contain pattern
            "hello there",   // idx 2 – contains pattern
            "goodbye world", // idx 3 – does NOT contain pattern
        ]);

        // Take a slice that begins *after* a row containing the pattern and spans
        // only rows without the pattern. The bitmap returned should therefore be
        // all false.
        let slice_arr = full.slice(1, 1); // only row 1 ("goodbye world")
        let slice = as_string_array(&slice_arr);
        let bb = simd_string_contains(slice, "hello");
        assert_eq!(bb.count_set_bits(), 0, "slice erroneously reported a match");
    }
}
