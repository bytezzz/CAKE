use std::hash::Hash;
use std::sync::Arc;

use arrow_array::{
    cast::AsArray,
    types::{
        ArrowPrimitiveType, Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type, UInt32Type,
        UInt64Type, UInt8Type,
    },
    Array, Int16Array, Int32Array, Int64Array, Int8Array, PrimitiveArray, UInt16Array, UInt32Array,
    UInt64Array, UInt8Array,
};
use arrow_schema::DataType;
use fnv::FnvHashMap;
use itertools::Itertools;

use crate::{ArrowQuiver, ColumnIdentifier, Selection};

type AQHashMap<K, V> = FnvHashMap<K, V>;

/// A "grouping" hash table which assigns distinct elements of the input a unique and
/// dense "ticket" number. Therefore, the first distinct value will be assigned 0, the
/// second distinct value will be assigned 1, etc.
///
/// This table can trivially be used to identify distinct elements, but can also be used
/// for more advanced operations like group-by aggregations.
pub enum TicketHashTable {
    Int8(AQHashMap<i8, usize>),
    Int16(AQHashMap<i16, usize>),
    Int32(AQHashMap<i32, usize>),
    Int64(AQHashMap<i64, usize>),
    UInt8(AQHashMap<u8, usize>),
    UInt16(AQHashMap<u16, usize>),
    UInt32(AQHashMap<u32, usize>),
    UInt64(AQHashMap<u64, usize>),
    MultiCol(AQHashMap<Vec<Box<dyn Array>>, usize>),
}

impl TicketHashTable {
    /// Create a new hash table with the key type `dt`
    /// ```
    /// # use arrow_schema::DataType;
    /// # use arrow_quiver::ticket_ht::TicketHashTable;
    /// // a new hash table with i32 keys
    /// let tht = TicketHashTable::new(&[&DataType::Int32]);
    /// ```
    pub fn new(dt: &[&DataType]) -> TicketHashTable {
        if dt.len() > 1 {
            TicketHashTable::MultiCol(AQHashMap::default())
        } else {
            match dt[0] {
                DataType::Null => todo!(),
                DataType::Boolean => todo!(),
                DataType::Int8 => TicketHashTable::Int8(AQHashMap::default()),
                DataType::Int16 => TicketHashTable::Int16(AQHashMap::default()),
                DataType::Int32 => TicketHashTable::Int32(AQHashMap::default()),
                DataType::Int64 => TicketHashTable::Int64(AQHashMap::default()),
                DataType::UInt8 => TicketHashTable::UInt8(AQHashMap::default()),
                DataType::UInt16 => TicketHashTable::UInt16(AQHashMap::default()),
                DataType::UInt32 => TicketHashTable::UInt32(AQHashMap::default()),
                DataType::UInt64 => TicketHashTable::UInt64(AQHashMap::default()),
                DataType::Float16 => todo!(),
                DataType::Float32 => todo!(),
                DataType::Float64 => todo!(),
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
                DataType::Dictionary(_kt, vt) => TicketHashTable::new(&[vt]),
                DataType::Decimal128(_, _) => todo!(),
                DataType::Decimal256(_, _) => todo!(),
                DataType::Map(_, _) => todo!(),
                DataType::RunEndEncoded(_re, vals) => TicketHashTable::new(&[vals.data_type()]),
            }
        }
    }

    /// Perform ticketing for an input quiver. The result in a vector of group IDs for
    /// each element in `data`.
    /// ```
    /// # use arrow_schema::DataType;
    /// # use arrow_quiver::ticket_ht::TicketHashTable;
    /// # use arrow_quiver::ArrowQuiver;
    /// let mut tht = TicketHashTable::new(&[&DataType::Int32]);
    /// let aq = ArrowQuiver::i32_col(vec![200, 100, 50, 50, 200]);
    /// let result = tht.ticket(&[0.into()], &aq);
    /// assert_eq!(result, vec![0, 1, 2, 2, 0]);
    /// ```
    pub fn ticket(&mut self, key: &[ColumnIdentifier], data: &ArrowQuiver) -> Vec<usize> {
        match self {
            TicketHashTable::Int8(hm) => {
                ticket_primitives(hm, &data.sel, data[&key[0]].as_primitive::<Int8Type>())
            }
            TicketHashTable::Int16(hm) => {
                ticket_primitives(hm, &data.sel, data[&key[0]].as_primitive::<Int16Type>())
            }
            TicketHashTable::Int32(hm) => {
                ticket_primitives(hm, &data.sel, data[&key[0]].as_primitive::<Int32Type>())
            }
            TicketHashTable::Int64(hm) => {
                ticket_primitives(hm, &data.sel, data[&key[0]].as_primitive::<Int64Type>())
            }
            TicketHashTable::UInt8(hm) => {
                ticket_primitives(hm, &data.sel, data[&key[0]].as_primitive::<UInt8Type>())
            }
            TicketHashTable::UInt16(hm) => {
                ticket_primitives(hm, &data.sel, data[&key[0]].as_primitive::<UInt16Type>())
            }
            TicketHashTable::UInt32(hm) => {
                ticket_primitives(hm, &data.sel, data[&key[0]].as_primitive::<UInt32Type>())
            }
            TicketHashTable::UInt64(hm) => {
                ticket_primitives(hm, &data.sel, data[&key[0]].as_primitive::<UInt64Type>())
            }
            TicketHashTable::MultiCol(hm) => todo!(),
        }
    }

    /// Extract the keys in order of ticket number from the hash table. This is a `O(n)`
    /// operation (does not require sorting the hash table keys).
    /// ```
    /// # use arrow_schema::DataType;
    /// # use arrow_quiver::ticket_ht::TicketHashTable;
    /// # use arrow_quiver::ArrowQuiver;
    /// # use arrow_array::cast::AsArray;
    /// # use arrow_array::types::Int32Type;
    /// let mut tht = TicketHashTable::new(&[&DataType::Int32]);
    /// let aq = ArrowQuiver::i32_col(vec![200, 100, 50, 50, 200]);
    /// tht.ticket(&[0.into()], &aq);
    ///
    /// let keys = tht.keys();
    /// assert_eq!(keys.as_primitive::<Int32Type>().values(), &[200, 100, 50]);
    /// ```
    ///
    pub fn keys(&self) -> Arc<dyn Array> {
        match self {
            TicketHashTable::Int8(hm) => Arc::new(Int8Array::from(extract_keys(hm))),
            TicketHashTable::Int16(hm) => Arc::new(Int16Array::from(extract_keys(hm))),
            TicketHashTable::Int32(hm) => Arc::new(Int32Array::from(extract_keys(hm))),
            TicketHashTable::Int64(hm) => Arc::new(Int64Array::from(extract_keys(hm))),
            TicketHashTable::UInt8(hm) => Arc::new(UInt8Array::from(extract_keys(hm))),
            TicketHashTable::UInt16(hm) => Arc::new(UInt16Array::from(extract_keys(hm))),
            TicketHashTable::UInt32(hm) => Arc::new(UInt32Array::from(extract_keys(hm))),
            TicketHashTable::UInt64(hm) => Arc::new(UInt64Array::from(extract_keys(hm))),
            TicketHashTable::MultiCol(_) => todo!(),
        }
    }

    /// Returns the number of entries in the hash table (which is always one more than the
    /// largest ticket ID).
    /// ```
    /// # use arrow_schema::DataType;
    /// # use arrow_quiver::ticket_ht::TicketHashTable;
    /// # use arrow_quiver::ArrowQuiver;
    /// let mut tht = TicketHashTable::new(&[&DataType::Int32]);
    /// let aq = ArrowQuiver::i32_col(vec![200, 100, 50, 50, 200]);
    /// tht.ticket(&[0.into()], &aq);
    ///
    /// assert_eq!(tht.len(), 3);
    /// ```
    ///
    pub fn len(&self) -> usize {
        match self {
            TicketHashTable::Int8(hm) => hm.len(),
            TicketHashTable::Int16(hm) => hm.len(),
            TicketHashTable::Int32(hm) => hm.len(),
            TicketHashTable::Int64(hm) => hm.len(),
            TicketHashTable::UInt8(hm) => hm.len(),
            TicketHashTable::UInt16(hm) => hm.len(),
            TicketHashTable::UInt32(hm) => hm.len(),
            TicketHashTable::UInt64(hm) => hm.len(),
            TicketHashTable::MultiCol(hm) => hm.len(),
        }
    }
}

fn extract_keys<K: Copy + Default>(hm: &AQHashMap<K, usize>) -> Vec<K> {
    let mut data = (0..hm.len()).map(|_| K::default()).collect_vec();
    for (k, v) in hm.iter() {
        data[*v] = *k;
    }
    data
}

fn ticket_primitives<K: ArrowPrimitiveType>(
    hm: &mut AQHashMap<K::Native, usize>,
    sel: &Selection,
    data: &PrimitiveArray<K>,
) -> Vec<usize>
where
    <K as ArrowPrimitiveType>::Native: Hash,
    <K as ArrowPrimitiveType>::Native: Eq,
{
    sel.apply_primitive(data, |v| {
        let curr_max = hm.len();
        *hm.entry(v).or_insert(curr_max)
    })
}

#[cfg(test)]
mod tests {
    use crate::ArrowQuiver;

    use super::TicketHashTable;

    #[test]
    fn simple_ticket_ht() {
        let aq = ArrowQuiver::i32_col(vec![1, 2, 2, 3, 3, 1, 2, -1000]);
        let mut tht = TicketHashTable::new(&[aq.data_type(&0.into())]);
        let r = tht.ticket(&[0.into()], &aq);
        assert_eq!(r, vec![0, 1, 1, 2, 2, 0, 1, 3]);
    }
}
