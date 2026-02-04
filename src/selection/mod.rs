pub mod sel_intersect;

use arrow_array::{types::ArrowPrimitiveType, PrimitiveArray};
use arrow_array::{ArrayRef, BooleanArray, UInt32Array};
use arrow_buffer::{bit_iterator::BitIndexIterator, BooleanBuffer, BooleanBufferBuilder, Buffer};
use arrow_select::take::TakeOptions;
use itertools::Itertools;
use sel_intersect::Intersector;
use std::ops::Not;

use arrow_array::array::ArrayAccessor;
use arrow_array::iterator::ArrayIter;

#[derive(Clone)]
pub enum Selection {
    NoneValid,
    AllValid,
    SelVec(Vec<u32>),
    Bitmap(BooleanBuffer),
}

impl std::fmt::Debug for Selection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Selection::NoneValid => write!(f, "NoneValid"),
            Selection::AllValid => write!(f, "AllValid"),
            Selection::SelVec(v) => write!(f, "SelVec({:?})", v),
            Selection::Bitmap(_) => write!(f, "Bitmap"),
        }
    }
}

pub fn idx_list_to_bb(sel_vec: &Vec<u32>, len: usize) -> BooleanBuffer {
    let mut buf = vec![0_u8; std::cmp::max(len.next_multiple_of(8), 1) as usize];
    for el in sel_vec {
        let byte_index = (el / 8) as usize;
        let byte_offset = el % 8;
        let mask = 1 << byte_offset;
        buf[byte_index] |= mask;
    }
    let buf = Buffer::from_vec(buf);
    BooleanBuffer::new(buf, 0, len)
}

impl Selection {
    pub fn is_all_valid(&self) -> bool {
        matches!(self, Selection::AllValid)
    }

    pub fn is_none_valid(&self) -> bool {
        matches!(self, Selection::NoneValid)
    }

    pub fn check(&self, idx: usize) -> bool {
        match self {
            Selection::NoneValid => false,
            Selection::AllValid => true,
            Selection::SelVec(sv) => sv.binary_search(&(idx as u32)).is_ok(),
            Selection::Bitmap(b) => (idx < b.len()).then(|| b.value(idx)).unwrap_or(false),
        }
    }

    pub fn filter_string_array<'a, V: ArrayAccessor<Item = &'a str>, F: FnMut(&str) -> bool>(
        &self,
        arr: V,
        mut f: F,
    ) -> Selection {
        match self {
            Selection::NoneValid => Selection::NoneValid,
            Selection::AllValid => Selection::SelVec(if arr.null_count() == 0 {
                // no nulls, so we can just use the indices
                (0..arr.len())
                    .filter_map(|x| f(unsafe { arr.value_unchecked(x) }).then_some(x as u32))
                    .collect_vec()
            } else {
                // there are nulls, use the iterator that will skip them
                ArrayIter::new(arr)
                    .enumerate()
                    .filter_map(|(idx, string)| {
                        string.map(|s| f(s).then_some(idx as u32)).unwrap_or(None)
                    })
                    .collect_vec()
            }),
            Selection::SelVec(v) => Selection::SelVec(
                v.iter()
                    .filter_map(|idx| {
                        debug_assert!(
                            (*idx as usize) < arr.len(),
                            "sel vec had index {}, but arr len was {}",
                            *idx,
                            arr.len()
                        );
                        unsafe { f(arr.value_unchecked(*idx as usize)).then_some(*idx) }
                    })
                    .collect_vec(),
            ),
            Selection::Bitmap(bm) => {
                let mut r = Vec::new();
                BitIndexIterator::new(bm.values(), bm.offset(), bm.len()).for_each(|idx| {
                    debug_assert!(
                        idx < arr.len(),
                        "bitmap idx {} was set, but arr len is {}",
                        idx,
                        arr.len()
                    );
                    if f(unsafe { arr.value_unchecked(idx) }) {
                        r.push(idx as u32);
                    }
                });
                Selection::SelVec(r)
            }
        }
    }

    pub fn filter<K: ArrowPrimitiveType, F: FnMut(K::Native) -> bool>(
        &self,
        arr: &PrimitiveArray<K>,
        mut f: F,
    ) -> Selection {
        match self {
            Selection::NoneValid => Selection::NoneValid,
            Selection::AllValid => Selection::SelVec(
                arr.values()
                    .iter()
                    .copied()
                    .enumerate()
                    .filter_map(|(idx, i)| f(i).then_some(idx as u32))
                    .collect_vec(),
            ),
            Selection::SelVec(v) => Selection::SelVec(
                v.iter()
                    .filter_map(|idx| {
                        debug_assert!(
                            (*idx as usize) < arr.len(),
                            "sel vec had index {}, but arr len was {}",
                            *idx,
                            arr.len()
                        );
                        unsafe { f(arr.value_unchecked(*idx as usize)).then_some(*idx) }
                    })
                    .collect_vec(),
            ),
            Selection::Bitmap(bm) => {
                let mut r = Vec::new();
                BitIndexIterator::new(bm.values(), bm.offset(), bm.len()).for_each(|idx| {
                    debug_assert!(
                        idx < arr.len(),
                        "bitmap idx {} was set, but arr len is {}",
                        idx,
                        arr.len()
                    );
                    if f(unsafe { arr.value_unchecked(idx) }) {
                        r.push(idx as u32);
                    }
                });
                Selection::SelVec(r)
            }
        }
    }

    pub fn apply_primitive<K: ArrowPrimitiveType, V, F: FnMut(K::Native) -> V>(
        &self,
        arr: &PrimitiveArray<K>,
        f: F,
    ) -> Vec<V> {
        match self {
            Selection::NoneValid => vec![],
            Selection::AllValid => arr.values().iter().copied().map(f).collect_vec(),
            Selection::SelVec(v) => v
                .iter()
                .map(|idx| {
                    debug_assert!(
                        (*idx as usize) < arr.len(),
                        "sel vec had index {}, but arr len was {}",
                        idx,
                        arr.len()
                    );
                    unsafe { arr.value_unchecked(*idx as usize) }
                })
                .map(f)
                .collect_vec(),
            Selection::Bitmap(b) => BitIndexIterator::new(b.values(), b.offset(), b.len())
                .map(|idx| {
                    debug_assert!(
                        idx < arr.len(),
                        "bitmap had {} set, but arr len was {}",
                        idx,
                        arr.len()
                    );
                    unsafe { arr.value_unchecked(idx) }
                })
                .map(f)
                .collect_vec(),
        }
    }

    pub fn intersect(self, intersector: &Intersector, other: Selection) -> Selection {
        match self {
            Selection::NoneValid => self,
            Selection::AllValid => other,
            Selection::SelVec(mut v1) => match other {
                Selection::NoneValid => other,
                Selection::AllValid => Selection::SelVec(v1),
                Selection::SelVec(v2) => Selection::SelVec(intersector.intersect(v1, v2)),
                Selection::Bitmap(bm) => {
                    v1.retain(|idx| bm.value(*idx as usize));
                    Selection::SelVec(v1)
                }
            },
            Selection::Bitmap(bm) => match other {
                Selection::NoneValid => other,
                Selection::AllValid => Selection::Bitmap(bm),
                Selection::SelVec(mut sv) => {
                    sv.retain(|idx| bm.value(*idx as usize));
                    Selection::SelVec(sv)
                }
                Selection::Bitmap(bm2) => {
                    let l = usize::min(bm.len(), bm2.len());
                    Selection::Bitmap(&bm.slice(0, l) & &bm2.slice(0, l))
                }
            },
        }
    }

    pub fn union(self, other: Selection) -> Selection {
        match self {
            Selection::NoneValid => other,
            Selection::AllValid => Selection::AllValid,
            Selection::SelVec(v1) => match other {
                Selection::NoneValid => Selection::SelVec(v1),
                Selection::AllValid => Selection::AllValid,
                Selection::SelVec(v2) => {
                    Selection::SelVec(v1.into_iter().merge(v2).dedup().collect_vec())
                }
                Selection::Bitmap(b) => {
                    let mut bbb = BooleanBufferBuilder::new(b.len());
                    bbb.append_buffer(&b);

                    for idx in v1 {
                        bbb.set_bit(idx as usize, true);
                    }
                    Selection::Bitmap(bbb.finish())
                }
            },
            Selection::Bitmap(b1) => match other {
                Selection::NoneValid => Selection::Bitmap(b1),
                Selection::AllValid => Selection::AllValid,
                Selection::SelVec(v) => {
                    let mut bbb = BooleanBufferBuilder::new(b1.len());
                    bbb.append_buffer(&b1);

                    for idx in v {
                        bbb.set_bit(idx as usize, true);
                    }
                    Selection::Bitmap(bbb.finish())
                }
                Selection::Bitmap(b2) => Selection::Bitmap(&b1 | &b2),
            },
        }
    }

    pub fn invert(self, max_rows: usize) -> Selection {
        match self {
            Selection::NoneValid => Selection::AllValid,
            Selection::AllValid => Selection::NoneValid,
            Selection::SelVec(v) => {
                let mut inverted = Vec::with_capacity(max_rows - v.len());
                let mut v_iter = v.iter().peekable();
                for i in 0..max_rows {
                    if v_iter.peek() == Some(&&(i as u32)) {
                        v_iter.next();
                    } else {
                        inverted.push(i as u32);
                    }
                }
                Selection::SelVec(inverted)
            }
            Selection::Bitmap(b) => Selection::Bitmap(b.not()),
        }
    }

    pub fn into_sel_vec(self) -> Selection {
        match self {
            Selection::SelVec(_) => self,
            Selection::AllValid => Selection::AllValid,
            Selection::NoneValid => Selection::NoneValid,
            Selection::Bitmap(bm) if bm.is_empty() => Selection::NoneValid,
            Selection::Bitmap(bm) => Selection::SelVec(
                BitIndexIterator::new(bm.values(), bm.offset(), bm.len())
                    .map(|idx| idx as u32)
                    .collect_vec(),
            ),
        }
    }

    pub fn into_bitmap(self, total_len: usize) -> Selection {
        match self {
            Selection::Bitmap(_) => self,
            Selection::AllValid => self,
            Selection::NoneValid => self,
            Selection::SelVec(v) => Selection::Bitmap(idx_list_to_bb(&v, total_len)),
        }
    }

    pub fn as_sel_vec(&self) -> Option<&[u32]> {
        match self {
            Selection::NoneValid => None,
            Selection::AllValid => None,
            Selection::SelVec(s) => Some(s),
            Selection::Bitmap(_) => None,
        }
    }

    pub fn as_bitmap(&self) -> Option<&BooleanBuffer> {
        match self {
            Selection::NoneValid => None,
            Selection::AllValid => None,
            Selection::SelVec(_) => None,
            Selection::Bitmap(b) => Some(b),
        }
    }

    pub fn extract_as_bitmap(&self, len: usize) -> BooleanBuffer {
        match self {
            Selection::NoneValid => BooleanBuffer::new_unset(len),
            Selection::AllValid => BooleanBuffer::new_set(len),
            Selection::Bitmap(bb) => bb.clone(),
            Selection::SelVec(sel) => idx_list_to_bb(sel, len),
        }
    }

    pub fn clone_and_filter(&self, col: ArrayRef) -> ArrayRef {
        match self {
            Selection::NoneValid => todo!(),
            Selection::AllValid => col.clone(),
            Selection::SelVec(vec) => {
                let take_conf = Some(TakeOptions {
                    check_bounds: false,
                });
                let arr = UInt32Array::from(vec.clone());
                arrow_select::take::take(col.as_ref(), &arr, take_conf.clone()).unwrap()
            }
            Selection::Bitmap(boolean_buffer) => {
                let ba = BooleanArray::from(boolean_buffer.clone());
                arrow_select::filter::filter(col.as_ref(), &ba).unwrap()
            }
        }
    }

    /// Create a slice of this selection vector suitable for a sliced quiver. If
    /// the selection is all valid or all invalid, the same is returned. If the
    /// selection is a bitmap, the bitmap is sliced. If the selection is a
    /// vector, a new selection vector of shifted values is created.
    /// ```
    /// use arrow_quiver::selection::Selection;
    /// let sv = Selection::SelVec(vec![5, 6, 10, 11]);
    /// let sliced = sv.slice(2..8).as_sel_vec().unwrap().to_vec();
    /// assert_eq!(sliced, vec![3, 4]);
    /// ```
    pub fn slice(&self, r: std::ops::Range<usize>) -> Selection {
        match self {
            Selection::NoneValid => Selection::NoneValid,
            Selection::AllValid => Selection::AllValid,
            Selection::SelVec(vec) => {
                let start_idx = vec.binary_search(&(r.start as u32)).unwrap_or_else(|x| x);
                let stop_idx = vec[start_idx..]
                    .binary_search(&(r.end as u32))
                    .unwrap_or_else(|x| x);

                Selection::SelVec(
                    vec[start_idx..start_idx + stop_idx]
                        .iter()
                        .map(|i| i - r.start as u32)
                        .collect_vec(),
                )
            }
            Selection::Bitmap(boolean_buffer) => {
                Selection::Bitmap(boolean_buffer.slice(r.start, r.end - r.start))
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use std::sync::Arc;

    use arrow_array::{cast::as_string_array, StringArray};

    use crate::{ArrowQuiver, ColumnIdentifier};

    use super::Selection;

    #[test]
    fn test_vec_to_bitmap() {
        let sv = Selection::SelVec(vec![1, 3, 9, 22, 47]).into_bitmap(50);
        match sv {
            Selection::NoneValid => panic!(),
            Selection::AllValid => panic!(),
            Selection::SelVec(_) => panic!(),
            Selection::Bitmap(bb) => {
                assert!(!bb.value(0));
                assert!(bb.value(1));
                assert!(!bb.value(2));
                assert!(bb.value(3));
                assert!(!bb.value(7));
                assert!(bb.value(22));
                assert!(!bb.value(23));
                assert!(!bb.value(46));
                assert!(bb.value(47));
            }
        };
    }

    #[test]
    fn test_bitmap_to_vec() {
        let sv = Selection::SelVec(vec![1, 3, 9, 22, 47])
            .into_bitmap(50)
            .into_sel_vec();
        match sv {
            Selection::NoneValid => panic!(),
            Selection::AllValid => panic!(),
            Selection::SelVec(v) => assert_eq!(v, vec![1, 3, 9, 22, 47]),
            Selection::Bitmap(_) => panic!(),
        };
    }

    #[test]
    fn test_string_filter() {
        let sample = StringArray::from(vec!["apple", "pear", "watermelon", "blueberry"]);

        let aq = ArrowQuiver::new("test".to_string(), vec!["a".into()], vec![Arc::new(sample)]);

        let sv = Selection::SelVec(vec![0, 2, 3]);

        let f = |s: &str| s.contains("a");

        let ci: ColumnIdentifier = "a".into();

        let col = &aq[&ci];

        match sv.filter_string_array(as_string_array(col), f) {
            Selection::NoneValid => panic!(),
            Selection::AllValid => panic!(),
            Selection::SelVec(v) => assert_eq!(v, vec![0, 2]),
            Selection::Bitmap(_) => panic!(),
        }
    }

    #[test]
    fn test_slice_sel() {
        let v = vec![4, 10, 12, 15, 22, 25];
        let sel = Selection::SelVec(v.clone());

        let sliced = sel.slice(11..21);
        let r = sliced.as_sel_vec().unwrap().to_vec();
        assert_eq!(r, vec![12 - 11, 15 - 11]);

        let sliced = sel.slice(0..50);
        let r = sliced.as_sel_vec().unwrap().to_vec();
        assert_eq!(r, v);

        let sliced = sel.slice(1..25);
        let r = sliced.as_sel_vec().unwrap().to_vec();
        assert_eq!(r, vec![3, 9, 11, 14, 21]);
    }
}
