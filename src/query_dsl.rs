use crate::{
    kernels,
    parquet_cache::load_parquet_cached,
    perf_manager::{perf_pause, perf_resume},
    predicate::Predicate,
    selection::{sel_intersect::Intersector, Selection},
    sort::SortDirection,
    ArrowQuiver, ColumnIdentifier,
};
use arrow_array::Int64Array;
use arrow_buffer::BooleanBufferBuilder;
use arrow_schema::ArrowError;
use cpu_time;
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{BufRead, BufReader, Write},
};
use std::{path::Path, sync::Arc};

#[derive(Clone)]
pub enum QueryOp {
    LoadParquet(&'static str),
    LessThan(&'static str, i64),
    StringContains(&'static str, &'static str),
    SmartStringOnTop(&'static str, &'static str, &'static str, i64), // string_col, search_term, filter_col, filter_val
    SelectColumns(&'static [&'static str]),
    SetSelection(Selection),
    SortBy(&'static str, SortDirection),
    GreaterThan(&'static str, i64),
    Equals(&'static str, i64),
    And,
    Or,
    FilterTop(usize),
    GreaterThanOrEqual(&'static str, i64),
    LessThanOrEqual(&'static str, i64),
    NotEquals(&'static str, i64),
    Between(&'static str, i64, i64),
    NotNull(&'static str),
    IsNull(&'static str),
    In(&'static str, &'static [i64]),
}

pub struct Query {
    pub name: &'static str,
    pub ops: &'static [QueryOp],
}

pub fn execute_query(
    job_path: &Path,
    intersector: &Intersector,
    query: &Query,
) -> (ArrowQuiver, u128) {
    let mut start_time = None;

    let mut aq: Option<ArrowQuiver> = None;

    for op in query.ops {
        match op {
            QueryOp::LoadParquet(file) => {
                aq = Some(load_parquet_cached(&job_path.join(file)));
                start_time = Some(cpu_time::ProcessTime::now());

                perf_resume();
            }
            QueryOp::SmartStringOnTop(string_col, search_term, filter_col, filter_val) => {
                let data = aq.as_ref().unwrap();
                let base_predicate = Predicate::LessThanConst(
                    ColumnIdentifier::Name((*filter_col).into()),
                    Arc::new(Int64Array::new_scalar(*filter_val)),
                );

                let selection = kernels::query_plan::execute(
                    data,
                    Box::new(base_predicate),
                    ColumnIdentifier::Name((*string_col).into()),
                    search_term,
                    intersector,
                )
                .unwrap();

                aq = Some(aq.unwrap().with_selection(selection).materialize_smart());
            }
            QueryOp::SelectColumns(cols) => aq = Some(aq.unwrap().select_columns(cols)),
            QueryOp::SortBy(col, direction) => {
                let selection =
                    kernels::sort::execute(aq.as_ref().unwrap(), col, *direction).unwrap();

                aq = Some(aq.unwrap().with_selection(selection).materialize())
            }
            QueryOp::LessThan(col, val) => {
                let data = aq.as_ref().unwrap();
                let pred = Predicate::LessThanConst(
                    ColumnIdentifier::Name((*col).into()),
                    Arc::new(Int64Array::new_scalar(*val)),
                );
                let selection = pred.apply(data, intersector);

                aq = Some(aq.unwrap().with_selection(selection).materialize_smart())
            }
            QueryOp::StringContains(col, term) => {
                let data = aq.as_ref().unwrap();
                let pred = Predicate::StringContains(
                    ColumnIdentifier::Name((*col).into()),
                    Arc::new(arrow_array::StringArray::new_scalar(*term)),
                );
                let selection = pred.apply(data, intersector);

                aq = Some(aq.unwrap().with_selection(selection).materialize_smart())
            }
            QueryOp::SetSelection(sel) => aq = Some(aq.unwrap().with_selection(sel.clone())),
            QueryOp::GreaterThan(col, val) => {
                let data = aq.as_ref().unwrap();
                let pred = Predicate::GreaterThanConst(
                    ColumnIdentifier::Name((*col).into()),
                    Arc::new(Int64Array::new_scalar(*val)),
                );
                let selection = pred.apply(data, intersector);

                aq = Some(aq.unwrap().with_selection(selection).materialize_smart())
            }
            QueryOp::Equals(col, val) => {
                let data = aq.as_ref().unwrap();
                let pred = Predicate::EqConst(
                    ColumnIdentifier::Name((*col).into()),
                    Arc::new(Int64Array::new_scalar(*val)),
                );
                let selection = pred.apply(data, intersector);

                aq = Some(aq.unwrap().with_selection(selection).materialize_smart())
            }
            QueryOp::FilterTop(n) => {
                let num_rows = aq.as_ref().unwrap().num_rows();
                let mut bbb = BooleanBufferBuilder::new(num_rows);
                let m = num_rows.min(*n);
                bbb.append_n(m, true);
                bbb.append_n(num_rows - m, false);
                let selection = Selection::Bitmap(bbb.finish());

                aq = Some(aq.unwrap().with_selection(selection).materialize_smart())
            }
            QueryOp::GreaterThanOrEqual(col, val) => {
                let data = aq.as_ref().unwrap();
                let pred = Predicate::GreaterThanOrEqualConst(
                    ColumnIdentifier::Name((*col).into()),
                    Arc::new(Int64Array::new_scalar(*val)),
                );
                let selection = pred.apply(data, intersector);

                aq = Some(aq.unwrap().with_selection(selection).materialize_smart())
            }
            QueryOp::LessThanOrEqual(col, val) => {
                let data = aq.as_ref().unwrap();
                let pred = Predicate::LessThanOrEqualConst(
                    ColumnIdentifier::Name((*col).into()),
                    Arc::new(Int64Array::new_scalar(*val)),
                );
                let selection = pred.apply(data, intersector);

                aq = Some(aq.unwrap().with_selection(selection).materialize_smart())
            }
            QueryOp::NotEquals(col, val) => {
                let data = aq.as_ref().unwrap();
                let pred = Predicate::NotEqConst(
                    ColumnIdentifier::Name((*col).into()),
                    Arc::new(Int64Array::new_scalar(*val)),
                );
                let selection = pred.apply(data, intersector);

                aq = Some(aq.unwrap().with_selection(selection).materialize_smart())
            }
            QueryOp::Between(col, low, high) => {
                let data = aq.as_ref().unwrap();
                let pred = Predicate::Between(
                    ColumnIdentifier::Name((*col).into()),
                    Arc::new(Int64Array::new_scalar(*low)),
                    Arc::new(Int64Array::new_scalar(*high)),
                );
                let selection = pred.apply(data, intersector);

                aq = Some(aq.unwrap().with_selection(selection).materialize_smart())
            }
            QueryOp::NotNull(col) => {
                let data = aq.as_ref().unwrap();
                let pred = Predicate::NotNull(ColumnIdentifier::Name((*col).into()));
                let selection = pred.apply(data, intersector);

                aq = Some(aq.unwrap().with_selection(selection).materialize_smart())
            }
            QueryOp::IsNull(col) => {
                let data = aq.as_ref().unwrap();
                let pred = Predicate::IsNull(ColumnIdentifier::Name((*col).into()));
                let selection = pred.apply(data, intersector);

                aq = Some(aq.unwrap().with_selection(selection).materialize_smart())
            }
            QueryOp::In(col, vals) => {
                let data = aq.as_ref().unwrap();
                let values = vals
                    .iter()
                    .map(|v| Arc::new(Int64Array::new_scalar(*v)) as Arc<dyn arrow_array::Datum>)
                    .collect();
                let pred = Predicate::In(ColumnIdentifier::Name((*col).into()), values);
                let selection = pred.apply(data, intersector);

                aq = Some(aq.unwrap().with_selection(selection).materialize_smart())
            }
            QueryOp::And => {
                // And is handled by default behavior - predicates accumulate with intersection
            }
            QueryOp::Or => {
                // Or would need special handling - for now we'll skip
                todo!("OR operation needs special handling")
            }
        }
    }
    perf_pause();

    let duration = start_time.unwrap().elapsed();
    println!(
        "Execution time for query {:?} is {:?}",
        query.name, duration
    );

    (aq.unwrap(), duration.as_nanos())
}

// =========================
// JSONL (De)Serialization
// =========================

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
enum SortDirSpec {
    Ascending,
    Descending,
}

impl From<SortDirection> for SortDirSpec {
    fn from(d: SortDirection) -> Self {
        match d {
            SortDirection::Ascending => SortDirSpec::Ascending,
            SortDirection::Descending => SortDirSpec::Descending,
        }
    }
}

impl From<SortDirSpec> for SortDirection {
    fn from(d: SortDirSpec) -> Self {
        match d {
            SortDirSpec::Ascending => SortDirection::Ascending,
            SortDirSpec::Descending => SortDirection::Descending,
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
enum QueryOpSpec {
    LoadParquet {
        file: String,
    },
    LessThan {
        col: String,
        val: i64,
    },
    StringContains {
        col: String,
        term: String,
    },
    SmartStringOnTop {
        string_col: String,
        search_term: String,
        filter_col: String,
        filter_val: i64,
    },
    SelectColumns {
        cols: Vec<String>,
    },
    SortBy {
        col: String,
        direction: SortDirSpec,
    },
    GreaterThan {
        col: String,
        val: i64,
    },
    Equals {
        col: String,
        val: i64,
    },
    And,
    Or,
    FilterTop {
        n: usize,
    },
    GreaterThanOrEqual {
        col: String,
        val: i64,
    },
    LessThanOrEqual {
        col: String,
        val: i64,
    },
    NotEquals {
        col: String,
        val: i64,
    },
    Between {
        col: String,
        low: i64,
        high: i64,
    },
    NotNull {
        col: String,
    },
    IsNull {
        col: String,
    },
    In {
        col: String,
        vals: Vec<i64>,
    },
}

#[derive(Serialize, Deserialize)]
struct QuerySpec {
    name: String,
    ops: Vec<QueryOpSpec>,
}

fn leak_string(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

fn leak_str_vec(v: Vec<String>) -> &'static [&'static str] {
    let leaked: Vec<&'static str> = v.into_iter().map(leak_string).collect();
    Box::leak(leaked.into_boxed_slice())
}

fn leak_i64_vec(v: Vec<i64>) -> &'static [i64] {
    Box::leak(v.into_boxed_slice())
}

impl From<&Query> for QuerySpec {
    fn from(q: &Query) -> Self {
        let ops: Vec<QueryOpSpec> = q
            .ops
            .iter()
            .map(|op| match op {
                QueryOp::LoadParquet(file) => QueryOpSpec::LoadParquet {
                    file: (*file).to_string(),
                },
                QueryOp::LessThan(col, val) => QueryOpSpec::LessThan {
                    col: (*col).to_string(),
                    val: *val,
                },
                QueryOp::StringContains(col, term) => QueryOpSpec::StringContains {
                    col: (*col).to_string(),
                    term: (*term).to_string(),
                },
                QueryOp::SmartStringOnTop(string_col, search_term, filter_col, filter_val) => {
                    QueryOpSpec::SmartStringOnTop {
                        string_col: (*string_col).to_string(),
                        search_term: (*search_term).to_string(),
                        filter_col: (*filter_col).to_string(),
                        filter_val: *filter_val,
                    }
                }
                QueryOp::SelectColumns(cols) => QueryOpSpec::SelectColumns {
                    cols: cols.iter().map(|s| (*s).to_string()).collect(),
                },
                QueryOp::SortBy(col, dir) => QueryOpSpec::SortBy {
                    col: (*col).to_string(),
                    direction: (*dir).into(),
                },
                QueryOp::GreaterThan(col, val) => QueryOpSpec::GreaterThan {
                    col: (*col).to_string(),
                    val: *val,
                },
                QueryOp::Equals(col, val) => QueryOpSpec::Equals {
                    col: (*col).to_string(),
                    val: *val,
                },
                QueryOp::And => QueryOpSpec::And,
                QueryOp::Or => QueryOpSpec::Or,
                QueryOp::FilterTop(n) => QueryOpSpec::FilterTop { n: *n },
                QueryOp::GreaterThanOrEqual(col, val) => QueryOpSpec::GreaterThanOrEqual {
                    col: (*col).to_string(),
                    val: *val,
                },
                QueryOp::LessThanOrEqual(col, val) => QueryOpSpec::LessThanOrEqual {
                    col: (*col).to_string(),
                    val: *val,
                },
                QueryOp::NotEquals(col, val) => QueryOpSpec::NotEquals {
                    col: (*col).to_string(),
                    val: *val,
                },
                QueryOp::Between(col, low, high) => QueryOpSpec::Between {
                    col: (*col).to_string(),
                    low: *low,
                    high: *high,
                },
                QueryOp::NotNull(col) => QueryOpSpec::NotNull {
                    col: (*col).to_string(),
                },
                QueryOp::IsNull(col) => QueryOpSpec::IsNull {
                    col: (*col).to_string(),
                },
                QueryOp::In(col, vals) => QueryOpSpec::In {
                    col: (*col).to_string(),
                    vals: vals.iter().copied().collect(),
                },
                QueryOp::SetSelection(_) => {
                    // Not serializable using this protocol
                    // We intentionally skip it by converting to a placeholder unsupported op
                    // Actual serialization will fail if encountered below.
                    QueryOpSpec::Or // sentinel that will cause error if round-tripped
                }
            })
            .collect();
        QuerySpec {
            name: q.name.to_string(),
            ops,
        }
    }
}

impl TryFrom<QuerySpec> for Query {
    type Error = ArrowError;
    fn try_from(spec: QuerySpec) -> Result<Self, ArrowError> {
        // Convert each op spec into a QueryOp, erroring on unsupported variants.
        let ops: Vec<QueryOp> = spec
            .ops
            .into_iter()
            .map(TryInto::try_into)
            .collect::<Result<_, _>>()?;

        // Leak ops Vec to obtain a 'static slice of ops for Query
        let ops_static: &'static [QueryOp] = Box::leak(ops.into_boxed_slice());
        Ok(Query {
            name: leak_string(spec.name),
            ops: ops_static,
        })
    }
}

impl TryFrom<QueryOpSpec> for QueryOp {
    type Error = ArrowError;
    fn try_from(op: QueryOpSpec) -> Result<Self, ArrowError> {
        Ok(match op {
            QueryOpSpec::LoadParquet { file } => QueryOp::LoadParquet(leak_string(file)),
            QueryOpSpec::LessThan { col, val } => QueryOp::LessThan(leak_string(col), val),
            QueryOpSpec::StringContains { col, term } => {
                QueryOp::StringContains(leak_string(col), leak_string(term))
            }
            QueryOpSpec::SmartStringOnTop {
                string_col,
                search_term,
                filter_col,
                filter_val,
            } => QueryOp::SmartStringOnTop(
                leak_string(string_col),
                leak_string(search_term),
                leak_string(filter_col),
                filter_val,
            ),
            QueryOpSpec::SelectColumns { cols } => QueryOp::SelectColumns(leak_str_vec(cols)),
            QueryOpSpec::SortBy { col, direction } => {
                QueryOp::SortBy(leak_string(col), direction.into())
            }
            QueryOpSpec::GreaterThan { col, val } => QueryOp::GreaterThan(leak_string(col), val),
            QueryOpSpec::Equals { col, val } => QueryOp::Equals(leak_string(col), val),
            QueryOpSpec::And => QueryOp::And,
            QueryOpSpec::Or => QueryOp::Or,
            QueryOpSpec::FilterTop { n } => QueryOp::FilterTop(n),
            QueryOpSpec::GreaterThanOrEqual { col, val } => {
                QueryOp::GreaterThanOrEqual(leak_string(col), val)
            }
            QueryOpSpec::LessThanOrEqual { col, val } => {
                QueryOp::LessThanOrEqual(leak_string(col), val)
            }
            QueryOpSpec::NotEquals { col, val } => QueryOp::NotEquals(leak_string(col), val),
            QueryOpSpec::Between { col, low, high } => {
                QueryOp::Between(leak_string(col), low, high)
            }
            QueryOpSpec::NotNull { col } => QueryOp::NotNull(leak_string(col)),
            QueryOpSpec::IsNull { col } => QueryOp::IsNull(leak_string(col)),
            QueryOpSpec::In { col, vals } => QueryOp::In(leak_string(col), leak_i64_vec(vals)),
        })
    }
}

/// Serialize a single `Query` as one JSONL line to the provided writer.
/// Protocol: JSON object with fields {"name": string, "ops": [ ... ]} where each
/// op is an object with a discriminant field `type` and op-specific fields.
pub fn serialize_query_jsonl<W: Write>(writer: &mut W, query: &Query) -> Result<(), ArrowError> {
    // Validate unsupported ops before converting
    for op in query.ops.iter() {
        match op {
            QueryOp::SetSelection(_) => {
                return Err(ArrowError::ExternalError(
                    "SetSelection is not supported".into(),
                ))
            }
            _ => {}
        }
    }
    let spec: QuerySpec = query.into();
    serde_json::to_writer(&mut *writer, &spec)
        .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
    writer.write_all(b"\n")?;
    Ok(())
}

/// Deserialize all queries from a JSONL file. Each line contains one query object.
/// Returns a vector of `Query`. Strings are leaked to satisfy the existing
/// `'static` lifetimes in the `Query`/`QueryOp` types.
pub fn deserialize_queries_jsonl<P: AsRef<Path>>(path: P) -> Result<Vec<Query>, ArrowError> {
    let file = File::open(path).map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for line_res in reader.lines() {
        let line = line_res?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let spec: QuerySpec =
            serde_json::from_str(trimmed).map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
        let q: Query = Query::try_from(spec)?;
        out.push(q);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn roundtrip_query_jsonl() {
        let q = Query {
            name: "TestQuery",
            ops: &[
                QueryOp::LoadParquet("data.parquet"),
                QueryOp::GreaterThan("id", 10),
                QueryOp::LessThan("id", 100),
                QueryOp::SelectColumns(&["id", "name"]),
                QueryOp::SortBy("id", SortDirection::Ascending),
                QueryOp::FilterTop(5),
            ],
        };

        let mut buf = Cursor::new(Vec::<u8>::new());
        serialize_query_jsonl(&mut buf, &q).unwrap();
        let s = String::from_utf8(buf.into_inner()).unwrap();

        // Should parse back
        let spec: QuerySpec = serde_json::from_str(s.trim()).unwrap();
        let q2: Query = Query::try_from(spec).unwrap();

        assert_eq!(q.name, q2.name);
        assert_eq!(q.ops.len(), q2.ops.len());
    }
}
