#[allow(unused_extern_crates)]
extern crate blas_src;

#[allow(warnings)]
pub mod aggregate;
pub mod archon;
pub mod configuration;
pub mod cusfilter;
pub mod dispatchers;
pub mod expression;
mod hash;
#[allow(warnings)]
pub mod hashjoin;
pub mod kernels;
#[allow(warnings)]
pub mod lang;
#[allow(warnings)]
pub mod parquet;
pub mod parquet_cache;
pub mod perf_manager;
pub mod pipeline;
pub mod predicate;
pub mod query_dsl;
pub mod regret_tree;
pub mod selection;
pub mod sort;
pub mod spool;
#[allow(warnings)]
pub mod ticket_ht;
pub mod utils;
#[allow(warnings)]
pub mod xor_filter;

pub mod oracle_machine {
    pub use crate::dispatchers::simple_models::{
        get_oracle_machine, init_oracle_machine_from_path, reset_oracle_machine, OpType,
        OracleMachine, OraclePolicy,
    };
}

// Re-export commonly used Query DSL items for simpler imports from binaries/tools
pub use crate::query_dsl::{deserialize_queries_jsonl, execute_query, Query};

use std::collections::HashMap;
use std::error::Error;
use std::ops::{Index, Range};
use std::path::Path;
use std::sync::Arc;

use arrow_array::{
    Array, BooleanArray, Int32Array, RecordBatch, RunArray, StringArray, UInt32Array,
};
use arrow_schema::DataType;
use arrow_select::take::TakeOptions;
pub use hash::hash;
pub use hash::{hash_i32, hash_i64};

use itertools::Itertools;
#[cfg(not(test))]
use log::warn;
use parquet::ParquetSource;
use selection::sel_intersect::Intersector;
use selection::Selection;

#[derive(Clone, Debug, Hash)]
pub enum ColumnIdentifier {
    Name(String),
    Index(usize),
}

impl From<usize> for ColumnIdentifier {
    fn from(value: usize) -> Self {
        ColumnIdentifier::Index(value)
    }
}

impl From<&str> for ColumnIdentifier {
    fn from(value: &str) -> Self {
        ColumnIdentifier::Name(value.to_string())
    }
}

impl From<String> for ColumnIdentifier {
    fn from(value: String) -> Self {
        ColumnIdentifier::Name(value)
    }
}

#[derive(Clone)]
pub struct ArrowQuiver {
    name: String,
    col_names: HashMap<String, usize>,
    columns: Vec<Arc<dyn Array>>,
    sel: Selection,
}

// Custom Debug implementation to show column names and lengths
impl std::fmt::Debug for ArrowQuiver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("ArrowQuiver");

        // Add column info
        for (name, &_) in &self.col_names {
            debug_struct
                .field("name", &name)
                .field("len", &self.num_rows())
                .field(
                    "data_type",
                    &self.data_type(&ColumnIdentifier::Name(name.clone())),
                );
        }
        debug_struct.finish()
    }
}

impl ArrowQuiver {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn empty() -> ArrowQuiver {
        ArrowQuiver::new("empty".to_string(), vec![], vec![])
    }

    pub fn new_unnamed(columns: Vec<Arc<dyn Array>>) -> ArrowQuiver {
        let names = (0..columns.len()).map(|i| format!("${}", i)).collect_vec();
        ArrowQuiver::new("unnamed".to_string(), names, columns)
    }
    pub fn new(
        table_name: String,
        col_names: Vec<String>,
        columns: Vec<Arc<dyn Array>>,
    ) -> ArrowQuiver {
        let num_cols = columns.len();
        let col_names: HashMap<String, usize> = col_names
            .into_iter()
            .enumerate()
            .map(|(idx, v)| (v, idx))
            .collect();

        assert_eq!(
            num_cols,
            col_names.len(),
            "all column names must be unique when building a quiver"
        );

        ArrowQuiver {
            name: table_name,
            col_names,
            columns,
            sel: Selection::AllValid,
        }
    }

    pub fn new_with_selection(
        col_names: Vec<String>,
        columns: Vec<Arc<dyn Array>>,
        sel: Selection,
    ) -> ArrowQuiver {
        let col_names = col_names
            .into_iter()
            .enumerate()
            .map(|(idx, v)| (v, idx))
            .collect();

        ArrowQuiver {
            name: "unnamed".to_string(),
            col_names,
            columns,
            sel,
        }
    }

    pub fn from_parquet_file(fp: &Path) -> Result<ArrowQuiver, Box<dyn Error>> {
        let mut all_rows = Vec::new();
        let mut source = ParquetSource::new(fp)?;
        let ctx = AQExecutorContext::default();

        while let Some(r) = source.produce(&ctx) {
            let r = r?;
            all_rows.push(r);
        }
        ArrowQuiver::concat(&all_rows)
    }

    pub fn rename_cols(self, col_names: Vec<String>) -> ArrowQuiver {
        ArrowQuiver::new(self.name, col_names, self.columns)
    }

    pub fn i32_col(data: Vec<i32>) -> ArrowQuiver {
        let data = Int32Array::from(data);
        ArrowQuiver::new(
            "i32_col".to_string(),
            vec![String::from("a")],
            vec![Arc::new(data)],
        )
    }

    pub fn i32_cols(data: Vec<Vec<i32>>) -> ArrowQuiver {
        let data: Vec<Arc<dyn Array>> = data
            .into_iter()
            .map(Int32Array::from)
            .map(|arr| Arc::new(arr) as Arc<dyn Array>)
            .collect_vec();
        let col_names = (0..data.len()).map(|i| format!("c{}", i)).collect_vec();
        ArrowQuiver::new("i32_cols".to_string(), col_names, data)
    }

    #[cfg(test)]
    pub fn as_i32_col(&self, ci: &ColumnIdentifier) -> Vec<i32> {
        self[ci]
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values()
            .to_vec()
    }

    pub fn arity(&self) -> usize {
        self.columns.len()
    }

    pub fn column_names(&self) -> impl Iterator<Item = &str> {
        self.col_names
            .iter()
            .sorted_by_key(|i| *i.1)
            .map(|i| i.0.as_str())
    }

    pub fn columns(&self) -> &[Arc<dyn Array>] {
        &self.columns
    }

    pub fn named_columns(&self) -> impl Iterator<Item = (&str, &dyn Array)> {
        self.col_names
            .iter()
            .map(|(k, v)| (k.as_str(), self.columns[*v].as_ref()))
    }

    pub fn select_columns(self, cols: &[&str]) -> ArrowQuiver {
        let target_cols = cols
            .iter()
            .map(|&name| self.col_names.get(name).unwrap())
            .map(|idx| self.columns[*idx].clone())
            .collect_vec();

        ArrowQuiver {
            name: self.name,
            col_names: cols
                .iter()
                .map(|name| String::from(*name))
                .zip(0..cols.len())
                .collect::<HashMap<_, _>>(),
            columns: target_cols,
            sel: Selection::AllValid,
        }
    }

    pub fn data_type(&self, ci: &ColumnIdentifier) -> &DataType {
        self[ci].data_type()
    }

    pub fn with_selection(&self, sel: Selection) -> ArrowQuiver {
        ArrowQuiver {
            name: self.name.clone(),
            col_names: self.col_names.clone(),
            columns: self.columns.clone(),
            sel,
        }
    }
    /// Copies all rows that are currently selected into a new ArrowQuiver. The new
    /// ArrowQuiver will always have a selection vector which is all valid (i.e.,
    /// `Selection::AllValid`).
    pub fn materialize_smart(self) -> ArrowQuiver {
        let ba = BooleanArray::from(self.sel.extract_as_bitmap(self.num_rows()));

        let columns = self
            .columns
            .into_iter()
            .map(|col| crate::kernels::filter::execute(col.as_ref(), &ba).unwrap())
            .collect_vec();

        ArrowQuiver {
            name: self.name.clone(),
            col_names: self.col_names.clone(),
            columns,
            sel: Selection::AllValid,
        }
    }

    /// Copies all rows that are currently selected into a new ArrowQuiver. The new
    /// ArrowQuiver will always have a selection vector which is all valid (i.e.,
    /// `Selection::AllValid`).
    pub fn materialize(self) -> ArrowQuiver {
        match self.sel {
            Selection::NoneValid => todo!(),
            Selection::AllValid => self.clone(),
            Selection::SelVec(vec) => {
                let take_conf = Some(TakeOptions {
                    check_bounds: false,
                });
                let arr = UInt32Array::from(vec);
                let columns = self
                    .columns
                    .iter()
                    .map(|col| arrow_select::take::take(col, &arr, take_conf.clone()).unwrap())
                    .collect_vec();

                ArrowQuiver {
                    name: self.name.clone(),
                    col_names: self.col_names.clone(),
                    columns,
                    sel: Selection::AllValid,
                }
            }
            Selection::Bitmap(boolean_buffer) => {
                let ba = BooleanArray::from(boolean_buffer);
                let columns = self
                    .columns
                    .iter()
                    .map(|col| arrow_select::filter::filter(col, &ba).unwrap())
                    .collect();

                ArrowQuiver {
                    name: self.name.clone(),
                    col_names: self.col_names.clone(),
                    columns,
                    sel: Selection::AllValid,
                }
            }
        }
    }

    pub fn with_run_ends(&self, ends: Vec<i32>) -> ArrowQuiver {
        let ends = Int32Array::from(ends);

        ArrowQuiver {
            name: self.name.clone(),
            col_names: self.col_names.clone(),
            columns: self
                .columns
                .iter()
                .map(|c| RunArray::try_new(&ends, c).unwrap())
                .map(|c| Arc::new(c) as Arc<dyn Array>)
                .collect_vec(),
            sel: self.sel.clone(),
        }
    }

    pub fn sel(&self) -> &Selection {
        &self.sel
    }

    pub fn num_rows(&self) -> usize {
        self.columns[0].len()
    }

    pub fn concat(data: &[ArrowQuiver]) -> Result<ArrowQuiver, Box<dyn Error>> {
        if data.is_empty() {
            return Err("Cannot concatenate empty data".into());
        }

        let mut concat_cols = Vec::new();
        let mut concat_col_names = Vec::new();
        for col in data[0].column_names() {
            let ci = ColumnIdentifier::Name(col.to_string());
            let arrays: Vec<&dyn Array> = data.iter().map(|x| x[&ci].as_ref()).collect();

            // Check if this is a string column that might overflow
            if matches!(arrays[0].data_type(), arrow_schema::DataType::Utf8) {
                // Calculate total string bytes to check for overflow risk
                let total_string_bytes: usize =
                    arrays.iter().map(|arr| arr.get_array_memory_size()).sum();

                // If risk of overflow, convert to LargeStringArray
                if total_string_bytes > (i32::MAX as usize / 2) {
                    let large_arrays: Result<Vec<_>, _> = arrays
                        .iter()
                        .map(|arr| {
                            let string_arr = arr.as_any().downcast_ref::<StringArray>().unwrap();
                            arrow_cast::cast(string_arr, &arrow_schema::DataType::LargeUtf8)
                        })
                        .collect();

                    match large_arrays {
                        Ok(large_arrays) => {
                            let large_array_refs: Vec<&dyn Array> =
                                large_arrays.iter().map(|a| a.as_ref()).collect();
                            concat_cols.push(arrow_select::concat::concat(&large_array_refs)?);
                        }
                        Err(_) => {
                            // Fallback to regular concat if conversion fails
                            concat_cols.push(arrow_select::concat::concat(&arrays)?);
                        }
                    }
                } else {
                    concat_cols.push(arrow_select::concat::concat(&arrays)?);
                }
            } else {
                concat_cols.push(arrow_select::concat::concat(&arrays)?);
            }

            concat_col_names.push(col.to_string());
        }

        Ok(ArrowQuiver::new(
            data[0].name.clone(),
            concat_col_names,
            concat_cols,
        ))
    }

    pub fn horizontal_stack(data: &[&ArrowQuiver]) -> ArrowQuiver {
        let mut concat_cols = Vec::new();
        let mut concat_col_names = Vec::new();

        // First pass: collect all column names and identify conflicts
        let mut all_names: Vec<Vec<String>> = Vec::new();
        let mut name_counts = std::collections::HashMap::new();

        for quiver in data {
            let names: Vec<String> = quiver.column_names().map(|s| s.to_string()).collect();
            for name in &names {
                *name_counts.entry(name.clone()).or_insert(0) += 1;
            }
            all_names.push(names);
        }

        // Second pass: build the result with resolved names
        for (quiver_idx, quiver) in data.iter().enumerate() {
            for col in quiver.column_names() {
                let final_name = if name_counts[col] > 1 {
                    // There's a conflict, add prefix
                    if quiver_idx == 0 {
                        format!("L_{}", col) // Left side prefix
                    } else {
                        format!("R_{}", col) // Right side prefix
                    }
                } else {
                    // No conflict, keep original name
                    col.to_string()
                };

                concat_col_names.push(final_name);
                let ci = ColumnIdentifier::Name(col.to_string());
                concat_cols.push(quiver[&ci].clone());
            }
        }
        let name = data.iter().map(|q| q.name.clone()).join("_");

        ArrowQuiver::new(name, concat_col_names, concat_cols)
    }

    /// Creates a slice of the quiver, containing rows that fall in the range
    /// given by `r`. Note that `r` is relative to the base data, not the
    /// selected data (e.g., selection is ignored when slicing).
    pub fn slice(&self, r: Range<usize>) -> ArrowQuiver {
        let columns = self
            .columns()
            .iter()
            .map(|c| c.slice(r.start, r.end - r.start))
            .collect_vec();
        let sel = self.sel.slice(r);
        ArrowQuiver {
            name: self.name.clone(),
            col_names: self.col_names.clone(),
            columns,
            sel,
        }
    }
}

impl Index<&ColumnIdentifier> for ArrowQuiver {
    type Output = Arc<dyn Array>;

    fn index(&self, index: &ColumnIdentifier) -> &Self::Output {
        let i = match index {
            ColumnIdentifier::Name(cname) => self.col_names[cname],
            ColumnIdentifier::Index(idx) => *idx,
        };
        &self.columns[i]
    }
}

impl Index<usize> for ArrowQuiver {
    type Output = Arc<dyn Array>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.columns[index]
    }
}

impl From<RecordBatch> for ArrowQuiver {
    fn from(value: RecordBatch) -> Self {
        let cnames = value
            .schema()
            .flattened_fields()
            .iter()
            .map(|f| f.name().clone())
            .collect_vec();
        ArrowQuiver::new("record_batch".to_string(), cnames, value.columns().to_vec())
    }
}

pub struct AQExecutorContext {
    intersector: Intersector,
}

impl Default for AQExecutorContext {
    fn default() -> Self {
        #[cfg(not(test))]
        warn!("Using default (test) intersector without training data in non-test mode");

        Self {
            intersector: Intersector::basic(),
        }
    }
}

pub trait Operator {
    fn name(&self) -> String;
    fn explain(&self) -> String {
        self.name()
    }
}

pub trait Sink: Operator + Sync {
    type Output;
    fn sink(&self, ctx: &AQExecutorContext, quiver: ArrowQuiver);
    fn finish(self: Box<Self>, ctx: &AQExecutorContext) -> Self::Output;
}

pub trait Source: Operator {
    fn produce(&mut self, ctx: &AQExecutorContext) -> Option<Result<ArrowQuiver, Box<dyn Error>>>;
}

pub trait Transform: Operator + Sync {
    fn transform(&self, ctx: &AQExecutorContext, quiver: ArrowQuiver) -> Vec<ArrowQuiver>;
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_materialize() {
        let aq = ArrowQuiver::i32_col(vec![1, 2, 3]);
        let aq = aq.with_selection(Selection::SelVec(vec![0, 2]));
        let aq = aq.materialize();
        assert!(aq.sel().is_all_valid());

        let r = aq[0].as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(r.values(), &[1, 3]);

        let aq = ArrowQuiver::i32_col(vec![1, 2, 3]);
        let aq = aq.with_selection(Selection::SelVec(vec![0, 2]).into_bitmap(3));
        let aq = aq.materialize();
        assert!(aq.sel().is_all_valid());

        let r = aq[0].as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(r.values(), &[1, 3]);
    }

    #[test]
    fn test_concat() {
        let aq1 = ArrowQuiver::i32_cols(vec![vec![1, 2, 3], vec![10, 20, 30]]);
        let aq2 = ArrowQuiver::i32_cols(vec![vec![4, 5, 6], vec![40, 50, 60]]);
        let r = ArrowQuiver::concat(&[aq1, aq2]).unwrap();

        assert_eq!(r.as_i32_col(&0.into()), vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(r.as_i32_col(&1.into()), vec![10, 20, 30, 40, 50, 60]);
    }

    #[test]
    fn test_hstack() {
        let aq = ArrowQuiver::i32_col(vec![1, 2, 3]);
        let aq2 = ArrowQuiver::i32_col(vec![4, 5, 6]).rename_cols(vec!["b".to_string()]);
        let aq = ArrowQuiver::horizontal_stack(&[&aq, &aq2]);
        assert_eq!(aq.num_rows(), 3);
        assert_eq!(aq.arity(), 2);

        let c1 = aq.as_i32_col(&0.into());
        assert_eq!(c1, vec![1, 2, 3]);
        let c2 = aq.as_i32_col(&1.into());
        assert_eq!(c2, vec![4, 5, 6]);

        let aq3 = ArrowQuiver::i32_cols(vec![vec![1, 2, 3], vec![9, 8, 7]]);
        let r = ArrowQuiver::horizontal_stack(&[&aq3, &aq2]);
        assert_eq!(r.as_i32_col(&1.into()), vec![9, 8, 7]);
    }

    #[test]
    fn test_slice() {
        let aq = ArrowQuiver::i32_col(vec![1, 2, 3, 4, 5]);
        let aq = aq.slice(1..4);
        assert_eq!(aq.num_rows(), 3);
        let r = aq.as_i32_col(&0.into());
        assert_eq!(r, vec![2, 3, 4]);

        let aq = ArrowQuiver::i32_col(vec![1, 2, 3, 4, 5]);
        let aq = aq.with_selection(Selection::SelVec(vec![0, 2, 4]));
        let aq = aq.slice(1..3);
        let r = aq.as_i32_col(&0.into());
        assert_eq!(r, &[2, 3]);
        assert_eq!(aq.sel().as_sel_vec().unwrap().to_vec(), vec![1]);
    }
}
