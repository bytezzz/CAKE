use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::ArrowPrimitiveType;
use arrow_array::{Array, PrimitiveArray, StringArray, UInt64Array};
use arrow_schema::DataType;
use itertools::Itertools;

use crate::ticket_ht::TicketHashTable;
use crate::{ArrowQuiver, ColumnIdentifier, Operator, Sink};

/// A trait representing a generic aggregation. Aggregations work in a vectorized fashion,
/// storing multiple groups. Each group is identified by an index (i.e., a ticket ID from
/// a ticket hash table).
pub trait Aggregation: Sync {
    /// Prepare the aggregator to receive data with at least `num_groups` distinct groups.
    /// The `update` function may assume that this function has always been called with
    /// the maximum group size.
    fn reserve_up_to(&mut self, num_groups: usize);

    /// Perform vectorized aggregation. Taking the input data `i`, aggregate each entry
    /// into the appropriate group given by `indexes.`
    fn update(&mut self, i: &[&dyn Array], indexes: &[usize]);

    /// Produce the result of the aggregation. This can "destroy" the state of the
    /// aggregator, so implementors may assume this is only called once after all updates
    /// are given.
    fn results(&mut self) -> Arc<dyn Array>;
}

pub struct MinAgg<T: ArrowPrimitiveType> {
    data: Vec<Option<T::Native>>,
}

impl<T: ArrowPrimitiveType> Aggregation for MinAgg<T> {
    fn reserve_up_to(&mut self, num_groups: usize) {
        while self.data.len() < num_groups {
            self.data.push(None)
        }
    }

    fn update(&mut self, i: &[&dyn Array], indexes: &[usize]) {
        assert_eq!(i.len(), 1, "min agg is unary");
        let arr: &PrimitiveArray<T> = i[0].as_primitive();
        arr.values()
            .iter()
            .zip(indexes.iter())
            .for_each(|(v, idx)| {
                if let Some(curr) = self.data[*idx].as_ref() {
                    if curr < v {
                        return;
                    }
                }

                self.data[*idx] = Some(v.clone());
            });
    }

    fn results(&mut self) -> Arc<dyn Array> {
        let mut tmp = Vec::new();
        std::mem::swap(&mut tmp, &mut self.data);
        Arc::new(PrimitiveArray::<T>::from_iter(tmp.into_iter()))
    }
}

impl<K: ArrowPrimitiveType> MinAgg<K> {
    pub fn new() -> Box<dyn Aggregation> {
        Box::new(Self { data: Vec::new() })
    }
}

pub struct MinStringAgg {
    data: Vec<Option<String>>,
}

impl Aggregation for MinStringAgg {
    fn reserve_up_to(&mut self, num_groups: usize) {
        while self.data.len() < num_groups {
            self.data.push(None)
        }
    }

    fn update(&mut self, i: &[&dyn Array], indexes: &[usize]) {
        assert_eq!(i.len(), 1, "min string agg is unary");
        let arr = i[0]
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("Expected StringArray");
        arr.iter().zip(indexes.iter()).for_each(|(v, idx)| {
            if let Some(value) = v {
                if let Some(curr) = self.data[*idx].as_ref() {
                    if curr.as_str() < value {
                        return;
                    }
                }
                self.data[*idx] = Some(value.to_string());
            }
        });
    }

    fn results(&mut self) -> Arc<dyn Array> {
        let mut tmp = Vec::new();
        std::mem::swap(&mut tmp, &mut self.data);
        Arc::new(StringArray::from_iter(tmp.into_iter()))
    }
}

impl MinStringAgg {
    pub fn new() -> Box<dyn Aggregation> {
        Box::new(Self { data: Vec::new() })
    }
}

pub struct SumAgg<T: ArrowPrimitiveType> {
    data: Vec<Option<T::Native>>,
}

impl<T: ArrowPrimitiveType> Aggregation for SumAgg<T>
where
    T::Native: std::ops::AddAssign + Clone,
{
    fn reserve_up_to(&mut self, num_groups: usize) {
        while self.data.len() < num_groups {
            self.data.push(None)
        }
    }

    fn update(&mut self, i: &[&dyn Array], indexes: &[usize]) {
        assert_eq!(i.len(), 1, "sum agg is unary");
        let arr: &PrimitiveArray<T> = i[0].as_primitive();
        arr.values()
            .iter()
            .zip(indexes.iter())
            .for_each(|(v, idx)| {
                if let Some(curr) = self.data[*idx].as_mut() {
                    *curr += v.clone();
                } else {
                    self.data[*idx] = Some(v.clone());
                }
            });
    }

    fn results(&mut self) -> Arc<dyn Array> {
        let mut tmp = Vec::new();
        std::mem::swap(&mut tmp, &mut self.data);
        Arc::new(PrimitiveArray::<T>::from_iter(tmp.into_iter()))
    }
}

impl<K: ArrowPrimitiveType> SumAgg<K>
where
    K::Native: std::ops::AddAssign + Clone,
{
    pub fn new() -> Box<dyn Aggregation> {
        Box::new(Self { data: Vec::new() })
    }
}

pub struct MeanAgg {
    sums: Vec<Option<f64>>,
    counts: Vec<u64>,
}

impl Aggregation for MeanAgg {
    fn reserve_up_to(&mut self, num_groups: usize) {
        while self.sums.len() < num_groups {
            self.sums.push(None);
            self.counts.push(0);
        }
    }

    fn update(&mut self, i: &[&dyn Array], indexes: &[usize]) {
        assert_eq!(i.len(), 1, "mean agg is unary");

        // Convert to f64 for averaging
        let values: Vec<f64> = match i[0].data_type() {
            DataType::Int32 => {
                let arr = i[0].as_primitive::<arrow_array::types::Int32Type>();
                arr.values().iter().map(|&v| v as f64).collect()
            }
            DataType::Int64 => {
                let arr = i[0].as_primitive::<arrow_array::types::Int64Type>();
                arr.values().iter().map(|&v| v as f64).collect()
            }
            DataType::Float32 => {
                let arr = i[0].as_primitive::<arrow_array::types::Float32Type>();
                arr.values().iter().map(|&v| v as f64).collect()
            }
            DataType::Float64 => {
                let arr = i[0].as_primitive::<arrow_array::types::Float64Type>();
                arr.values().iter().copied().collect()
            }
            _ => panic!("Unsupported data type for mean aggregation"),
        };

        values.iter().zip(indexes.iter()).for_each(|(v, idx)| {
            if let Some(curr) = self.sums[*idx].as_mut() {
                *curr += v;
            } else {
                self.sums[*idx] = Some(*v);
            }
            self.counts[*idx] += 1;
        });
    }

    fn results(&mut self) -> Arc<dyn Array> {
        let mut tmp_sums = Vec::new();
        let mut tmp_counts = Vec::new();
        std::mem::swap(&mut tmp_sums, &mut self.sums);
        std::mem::swap(&mut tmp_counts, &mut self.counts);

        let means: Vec<Option<f64>> = tmp_sums
            .into_iter()
            .zip(tmp_counts.into_iter())
            .map(|(sum, count)| sum.map(|s| if count > 0 { s / count as f64 } else { s }))
            .collect();

        Arc::new(arrow_array::Float64Array::from_iter(means.into_iter()))
    }
}

impl MeanAgg {
    pub fn new() -> Box<dyn Aggregation> {
        Box::new(Self {
            sums: Vec::new(),
            counts: Vec::new(),
        })
    }
}

pub struct MaxAgg<T: ArrowPrimitiveType> {
    data: Vec<Option<T::Native>>,
}

impl<T: ArrowPrimitiveType> Aggregation for MaxAgg<T> {
    fn reserve_up_to(&mut self, num_groups: usize) {
        while self.data.len() < num_groups {
            self.data.push(None)
        }
    }

    fn update(&mut self, i: &[&dyn Array], indexes: &[usize]) {
        assert_eq!(i.len(), 1, "max agg is unary");
        let arr: &PrimitiveArray<T> = i[0].as_primitive();
        arr.values()
            .iter()
            .zip(indexes.iter())
            .for_each(|(v, idx)| {
                if let Some(curr) = self.data[*idx].as_ref() {
                    if curr > v {
                        return;
                    }
                }

                self.data[*idx] = Some(v.clone());
            });
    }

    fn results(&mut self) -> Arc<dyn Array> {
        let mut tmp = Vec::new();
        std::mem::swap(&mut tmp, &mut self.data);
        Arc::new(PrimitiveArray::<T>::from_iter(tmp.into_iter()))
    }
}

impl<K: ArrowPrimitiveType> MaxAgg<K> {
    pub fn new() -> Box<dyn Aggregation> {
        Box::new(Self { data: Vec::new() })
    }
}

pub struct CountAgg {
    data: Vec<u64>,
}

impl Aggregation for CountAgg {
    fn reserve_up_to(&mut self, num_groups: usize) {
        while self.data.len() < num_groups {
            self.data.push(0);
        }
    }

    fn update(&mut self, _i: &[&dyn Array], indexes: &[usize]) {
        for idx in indexes {
            self.data[*idx] += 1;
        }
    }

    fn results(&mut self) -> Arc<dyn Array> {
        let mut tmp = Vec::new();
        std::mem::swap(&mut tmp, &mut self.data);
        Arc::new(UInt64Array::from(tmp))
    }
}

impl CountAgg {
    pub fn new() -> Box<dyn Aggregation> {
        Box::new(CountAgg { data: Vec::new() })
    }
}

pub struct HashAggregatorSink {
    aggs: Vec<(Vec<ColumnIdentifier>, Box<dyn Aggregation>)>,
    keys: Vec<ColumnIdentifier>,
    tht: TicketHashTable,
}

impl Operator for HashAggregatorSink {
    fn name(&self) -> String {
        format!("Hash Aggregator")
    }
}

impl Sink for HashAggregatorSink {
    type Output = ArrowQuiver;

    fn sink(&self, ctx: &crate::AQExecutorContext, quiver: ArrowQuiver) {
        /*let indexes = self.tht.ticket(&self.keys, &quiver);
        if indexes.is_empty() {
            return;
        }
        let max_index = self.tht.len();

        for (inputs, agg) in self.aggs.iter() {
            let inputs = inputs.iter().map(|ci| quiver[ci].as_ref()).collect_vec();
            agg.reserve_up_to(max_index);
            agg.update(&inputs, &indexes);
        }*/
    }

    fn finish(self: Box<Self>, ctx: &crate::AQExecutorContext) -> Self::Output {
        todo!()
    }
}

impl HashAggregatorSink {
    pub fn new(
        keys: Vec<ColumnIdentifier>,
        key_type: &[&DataType],
        aggs: Vec<(Vec<ColumnIdentifier>, Box<dyn Aggregation>)>,
    ) -> HashAggregatorSink {
        HashAggregatorSink {
            keys,
            aggs,
            tht: TicketHashTable::new(key_type),
        }
    }

    pub fn aggregate(&mut self, data: &ArrowQuiver) {
        let indexes = self.tht.ticket(&self.keys, data);
        if indexes.is_empty() {
            return;
        }
        let max_index = self.tht.len();

        for (inputs, agg) in self.aggs.iter_mut() {
            let inputs = inputs.iter().map(|ci| data[ci].as_ref()).collect_vec();
            agg.reserve_up_to(max_index);
            agg.update(&inputs, &indexes);
        }
    }

    pub fn results(self) -> ArrowQuiver {
        let mut all_cols = Vec::new();

        let gb_col = self.tht.keys();
        all_cols.push(gb_col);

        for (_, mut agg) in self.aggs {
            all_cols.push(agg.results());
        }

        ArrowQuiver::new_unnamed(all_cols)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        cast::AsArray,
        types::{Int32Type, UInt64Type},
        Int32Array,
    };
    use arrow_schema::DataType;

    use crate::ArrowQuiver;

    use super::{CountAgg, HashAggregatorSink, MaxAgg, MeanAgg, MinAgg, MinStringAgg, SumAgg};

    #[test]
    fn test_count_agg() {
        let aq = ArrowQuiver::i32_col(vec![1, 1, 1, 2, 2, 2, 9, 9]);
        let mut agg = HashAggregatorSink::new(
            vec![0.into()],
            &[&DataType::Int32],
            vec![(vec![], CountAgg::new())],
        );

        agg.aggregate(&aq);
        agg.aggregate(&aq);

        let r = agg.results();
        let counts = r[&1.into()].as_primitive::<UInt64Type>();
        assert_eq!(counts.values(), &[6, 6, 4]);
    }

    #[test]
    fn test_count_ooo_agg() {
        let aq = ArrowQuiver::i32_col(vec![3, 2, 1, 1, 2, 3]);
        let mut agg = HashAggregatorSink::new(
            vec![0.into()],
            &[&DataType::Int32],
            vec![(vec![], CountAgg::new())],
        );

        agg.aggregate(&aq);
        agg.aggregate(&aq);

        let r = agg.results();
        let counts = r[&1.into()].as_primitive::<UInt64Type>();
        assert_eq!(counts.values(), &[4, 4, 4]);
    }

    #[test]
    fn test_min_agg() {
        let k = Arc::new(Int32Array::from(vec![1, 2, 3, 1, 2, 3, 100]));
        let v = Arc::new(Int32Array::from(vec![100, 200, 300, -100, -200, 900, 100]));
        let aq = ArrowQuiver::new_unnamed(vec![k, v]);

        let mut agg = HashAggregatorSink::new(
            vec![0.into()],
            &[&DataType::Int32],
            vec![(vec![1.into()], MinAgg::<Int32Type>::new())],
        );

        agg.aggregate(&aq);

        let r = agg.results();
        let mins = r[&1.into()].as_primitive::<Int32Type>();
        assert_eq!(mins.values(), &[-100, -200, 300, 100]);
    }

    #[test]
    fn test_sum_agg() {
        let k = Arc::new(Int32Array::from(vec![1, 2, 3, 1, 2, 3]));
        let v = Arc::new(Int32Array::from(vec![10, 20, 30, 40, 50, 60]));
        let aq = ArrowQuiver::new_unnamed(vec![k, v]);

        let mut agg = HashAggregatorSink::new(
            vec![0.into()],
            &[&DataType::Int32],
            vec![(vec![1.into()], SumAgg::<Int32Type>::new())],
        );

        agg.aggregate(&aq);

        let r = agg.results();
        let sums = r[&1.into()].as_primitive::<Int32Type>();
        assert_eq!(sums.values(), &[50, 70, 90]);
    }

    #[test]
    fn test_mean_agg() {
        let k = Arc::new(Int32Array::from(vec![1, 2, 3, 1, 2, 3]));
        let v = Arc::new(Int32Array::from(vec![10, 20, 30, 40, 50, 60]));
        let aq = ArrowQuiver::new_unnamed(vec![k, v]);

        let mut agg = HashAggregatorSink::new(
            vec![0.into()],
            &[&DataType::Int32],
            vec![(vec![1.into()], MeanAgg::new())],
        );

        agg.aggregate(&aq);

        let r = agg.results();
        let means = r[&1.into()].as_primitive::<arrow_array::types::Float64Type>();
        assert_eq!(means.values(), &[25.0, 35.0, 45.0]);
    }

    #[test]
    fn test_max_agg() {
        let k = Arc::new(Int32Array::from(vec![1, 2, 3, 1, 2, 3, 100]));
        let v = Arc::new(Int32Array::from(vec![100, 200, 300, -100, -200, 900, 100]));
        let aq = ArrowQuiver::new_unnamed(vec![k, v]);

        let mut agg = HashAggregatorSink::new(
            vec![0.into()],
            &[&DataType::Int32],
            vec![(vec![1.into()], MaxAgg::<Int32Type>::new())],
        );

        agg.aggregate(&aq);

        let r = agg.results();
        let maxs = r[&1.into()].as_primitive::<Int32Type>();
        assert_eq!(maxs.values(), &[100, 200, 900, 100]);
    }
}
