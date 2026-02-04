use std::sync::{Arc, Mutex};

use arrow_array::{cast::AsArray, Array, Int64Array, UInt64Array};
use arrow_buffer::ScalarBuffer;
use arrow_schema::DataType;
use itertools::Itertools;
use xorf::{BinaryFuse16, Filter};

use crate::{
    hash, selection::Selection, AQExecutorContext, ArrowQuiver, ColumnIdentifier, Operator, Sink,
};

pub struct AQXorFilterSink {
    pool: Mutex<Vec<Arc<dyn Array>>>,
    col: ColumnIdentifier,
}

impl AQXorFilterSink {
    pub fn new(col: ColumnIdentifier) -> Self {
        AQXorFilterSink {
            pool: Mutex::new(vec![]),
            col,
        }
    }
}

impl Operator for AQXorFilterSink {
    fn name(&self) -> String {
        format!("Build Xor Filter")
    }
}

impl Sink for AQXorFilterSink {
    type Output = AQXorFilter16;

    fn sink(&self, ctx: &AQExecutorContext, quiver: ArrowQuiver) {
        assert!(
            quiver.sel().is_all_valid(),
            "materialize quiver before building xor filter"
        );
        let data = &quiver[&self.col];
        assert!(
            !data.is_nullable(),
            "cannot construct xor filter over nullable column, filter out nulls first"
        );
        self.pool.lock().unwrap().push(data.clone());
    }

    fn finish(self: Box<Self>, ctx: &AQExecutorContext) -> Self::Output {
        let data = self.pool.into_inner().unwrap();
        let data: Vec<&dyn Array> = data.iter().map(|x| x.as_ref()).collect();
        let pooled = arrow_select::concat::concat(&data).unwrap();

        AQXorFilter16::build_from(pooled.as_ref())
    }
}

pub struct AQXorFilter16 {
    filter: BinaryFuse16,
    dtype: DataType,
}

impl AQXorFilter16 {
    pub fn build_from(col: &dyn Array) -> Self {
        assert!(
            !col.is_nullable(),
            "cannot construct xor filter over nullable column, filter out nulls first"
        );
        match col.data_type() {
            DataType::Int64 => {
                // interpret the i64s as u64s, then build the buffer
                let arr: &Int64Array = col.as_primitive();
                let sb = ScalarBuffer::<u64>::new(
                    arr.values().clone().into_inner(),
                    arr.offset(),
                    arr.len(),
                );

                let filter = BinaryFuse16::try_from(sb.as_ref()).unwrap();
                AQXorFilter16 {
                    filter,
                    dtype: col.data_type().clone(),
                }
            }
            DataType::UInt64 => {
                let arr: &UInt64Array = col.as_primitive();
                let filter = BinaryFuse16::try_from(arr.values().as_ref()).unwrap();

                AQXorFilter16 {
                    filter,
                    dtype: col.data_type().clone(),
                }
            }
            _ => {
                let hashes = hash(col);
                let filter = BinaryFuse16::try_from(hashes.values().as_ref()).unwrap();
                AQXorFilter16 {
                    filter,
                    dtype: col.data_type().clone(),
                }
            }
        }
    }

    pub fn test_with_hash(&self, hashval: &u64) -> bool {
        self.filter.contains(hashval)
    }

    pub fn test(&self, ci: &ColumnIdentifier, data: &ArrowQuiver) -> Selection {
        let col = &data[ci];
        let fps = match col.data_type() {
            DataType::Int64 => {
                // interpret the i64s as u64s, then build the buffer
                let arr: &Int64Array = col.as_primitive();
                let sb = ScalarBuffer::<u64>::new(
                    arr.values().clone().into_inner(),
                    arr.offset(),
                    arr.len(),
                );
                UInt64Array::new(sb, col.nulls().cloned())
            }
            DataType::UInt64 => col.as_primitive().clone(),
            _ => hash(col),
        };

        let sv = fps
            .values()
            .iter()
            .enumerate()
            .filter_map(|(idx, v)| self.filter.contains(v).then_some(idx as u32))
            .collect_vec();
        Selection::SelVec(sv)
    }
}

#[cfg(test)]
mod tests {

    use crate::{xor_filter::AQXorFilterSink, AQExecutorContext, ArrowQuiver, ColumnIdentifier};

    use super::*;

    #[test]
    fn test_xor_i32() {
        let aq1 = ArrowQuiver::i32_col(vec![1, 5, 10, 20]);
        let aq2 = ArrowQuiver::i32_col(vec![900, 6, 950, 2]);

        let ctx = AQExecutorContext::default();
        let sink = Box::new(AQXorFilterSink::new(ColumnIdentifier::Index(0)));
        sink.sink(&ctx, aq1);
        sink.sink(&ctx, aq2);
        let f = sink.finish(&ctx);

        let q = ArrowQuiver::i32_col(vec![1, 5, 1000, 10000, -400, 950]);
        let r = f.test(&ColumnIdentifier::Index(0), &q);
        assert!(r.check(0));
        assert!(r.check(1));
        assert!(!r.check(2));
        assert!(!r.check(3));
        assert!(!r.check(4));
        assert!(r.check(5));
    }
}
