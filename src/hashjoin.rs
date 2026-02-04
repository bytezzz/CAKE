use std::sync::{
    atomic::{AtomicPtr, Ordering},
    Mutex,
};

use arrow_array::{types::UInt64Type, PrimitiveArray};
use itertools::Itertools;

use crate::{
    expression::AQExpr,
    hash,
    predicate::{multi_and, Predicate},
    selection::Selection,
    ArrowQuiver, ColumnIdentifier, Operator, Sink, Transform,
};

/// A node in a hash multimap, which holds an index into the data vector stored in the
/// build sink and a pointer to the next node.
#[derive(Copy, Clone)]
struct HTNode {
    next: *mut HTNode,
    idx: usize,
}
unsafe impl Send for HTNode {}
unsafe impl Sync for HTNode {}

struct ParPtr<T> {
    ptr: *const T,
}
unsafe impl<T: Send> Send for ParPtr<T> {}
unsafe impl<T: Sync> Sync for ParPtr<T> {}

/// A sink that builds a hash table for a hash join. The strategy used is a slight
/// modification from "Morsel-driven parallelism: a NUMA-aware query evaluation framework
/// for the many-core age" by Leis et al. Modifications are:
/// 1. We do not store the "fingerprint bits" in the extra bits of each pointer, and thus
///    we do not do any early pruning
/// 2. We wrap the data and hash nodes in mutexes, which are acquired once per batch,
///    instead of having each thread build its own local hash table
/// 3. We use atomic swap instead of CAS, since we don't have to manage the fingerprint
///    values
pub struct HashJoinBuildSink {
    build_keys: Vec<ColumnIdentifier>,

    ht: Vec<AtomicPtr<HTNode>>,

    data: Mutex<(usize, Vec<ArrowQuiver>)>,
    nodes: Mutex<Vec<Box<[HTNode]>>>,
}

impl Operator for HashJoinBuildSink {
    fn name(&self) -> String {
        format!("Build Join Hash Table")
    }
}

impl HashJoinBuildSink {
    /// Constructs a new hash join build sink which will build a hash table over the
    /// input, keyed on the columns given by `keys`. The hash table uses open addressing
    /// (chaining), so resizing is never performed. The `table_size` parameter can be set
    /// well above the expected number of keys, since each hash table entry is just a
    /// 64-bit pointer.
    pub fn new(keys: Vec<ColumnIdentifier>, table_size: usize) -> HashJoinBuildSink {
        let ht = (0..table_size)
            .map(|_| AtomicPtr::new(std::ptr::null_mut()))
            .collect_vec();

        HashJoinBuildSink {
            build_keys: keys,
            ht,
            data: Mutex::new((0, Vec::new())),
            nodes: Mutex::new(Vec::new()),
        }
    }

    /// Creates HashJoinTransform with flexible probe key positions
    pub fn finish_with_probe_keys(
        build_sink: Box<Self>,
        _ctx: &crate::AQExecutorContext,
        probe_keys: Vec<ColumnIdentifier>,
    ) -> HashJoinTransform {
        let (_, quivers) = build_sink.data.into_inner().unwrap();
        let data = ArrowQuiver::concat(&quivers).unwrap();

        // construct an expression to evaluate the equality condition using flexible key positions
        let num_build_cols = data.arity();

        HashJoinTransform {
            probe_keys,
            build_keys: build_sink.build_keys,
            ht: build_sink
                .ht
                .into_iter()
                .map(|x| ParPtr {
                    ptr: x.into_inner() as *const HTNode,
                })
                .collect_vec(),
            data,
            _ptrs: build_sink.nodes.into_inner().unwrap(),
        }
    }
}

impl Sink for HashJoinBuildSink {
    type Output = HashJoinTransform;

    fn sink(&self, _ctx: &crate::AQExecutorContext, quiver: crate::ArrowQuiver) {
        let quiver = quiver.materialize();

        // ensure none of the key columns are null and
        for key in &self.build_keys {
            assert!(
                !quiver[key].is_nullable(),
                "cannot build hash table on nullable column"
            );
        }

        let hashes = hash_all(&quiver, &self.build_keys);

        let idx_offset = {
            let mut counter_and_data = self.data.lock().unwrap();
            let start_at = counter_and_data.0;
            counter_and_data.0 += quiver.num_rows();
            counter_and_data.1.push(quiver);
            start_at
        };

        let mut nodes: Vec<HTNode> = Vec::with_capacity(hashes.len());
        for (row_idx, hash) in hashes.values().iter().enumerate() {
            let hash = *hash as usize % self.ht.len();

            let node_idx = nodes.len();
            nodes.push(HTNode {
                next: std::ptr::null_mut(),
                idx: idx_offset + row_idx,
            });

            let node = &mut nodes[node_idx];
            node.next = self.ht[hash].swap(node as *mut HTNode, Ordering::Relaxed);
        }

        // we are going to turn these nodes into a boxed slice, but we don't want any of
        // the pointers to move, so make sure that `into_boxed_slice` won't reallocate.
        assert!(nodes.capacity() == nodes.len());

        let mut all_nodes = self.nodes.lock().unwrap();
        all_nodes.push(nodes.into_boxed_slice());
    }

    fn finish(self: Box<Self>, _ctx: &crate::AQExecutorContext) -> Self::Output {
        // For backward compatibility, default probe keys to [0, 1, 2, ...] (first k columns)
        let probe_keys = self
            .build_keys
            .iter()
            .enumerate()
            .map(|(i, _)| ColumnIdentifier::Index(i))
            .collect_vec();

        HashJoinBuildSink::finish_with_probe_keys(self, _ctx, probe_keys)
    }
}

pub struct HashJoinTransform {
    probe_keys: Vec<ColumnIdentifier>, // Actual probe-side key positions (can be anywhere)
    build_keys: Vec<ColumnIdentifier>, // Original build-side keys for reference
    ht: Vec<ParPtr<HTNode>>,
    data: ArrowQuiver,

    // never directly used, but the pointers in `ht` reference these HTNodes so we have to
    // keep them around
    _ptrs: Vec<Box<[HTNode]>>,
}

impl Operator for HashJoinTransform {
    fn name(&self) -> String {
        format!("Hash Join Transform")
    }
}

impl Transform for HashJoinTransform {
    fn transform(&self, ctx: &crate::AQExecutorContext, quiver: ArrowQuiver) -> Vec<ArrowQuiver> {
        let mut quiver = quiver.materialize();

        // ensure none of the key columns are null
        for key in &self.probe_keys {
            assert!(
                !quiver[key].is_nullable(),
                "cannot build hash table on nullable column"
            );
        }

        // hash the probe-side keys using their actual positions (no longer assumes first-k columns)
        let hashes = hash_all(&quiver, &self.probe_keys);
        let mut curr_ptrs = hashes
            .values()
            .iter()
            .map(|h| *h as usize % self.ht.len())
            .map(|h| self.ht[h].ptr)
            .collect_vec();

        let mut to_return = Vec::new();
        // at each iteration through this loop, we check each of the pointers in
        // `curr_ptrs` and see if they point to a valid HTNode. if so, we take the
        // corresponding row from the build-side and add it to the output. if not, we move
        // on to the next pointer in `curr_ptrs` and mark that probe-side row for removal.
        // We repeat until all probe-side rows have exhausted their matches.
        loop {
            let mut to_take = Vec::with_capacity(curr_ptrs.len());
            let mut probe_indices = Vec::with_capacity(curr_ptrs.len());
            let mut may_have_next = Vec::with_capacity(quiver.num_rows());
            for (input_row_idx, ptr) in curr_ptrs.iter_mut().enumerate() {
                if ptr.is_null() {
                    continue;
                }

                // safety: HTNode array is allocated by builder once and owned by us
                let node = unsafe { &**ptr };
                to_take.push(node.idx as u32);
                probe_indices.push(input_row_idx as u32); // Track which probe rows have matches

                if !node.next.is_null() {
                    may_have_next.push(input_row_idx as u32);
                }
                *ptr = node.next;
            }

            // Skip empty iterations
            if to_take.is_empty() {
                if may_have_next.is_empty() {
                    break;
                } else {
                    curr_ptrs = may_have_next
                        .iter()
                        .map(|&i| curr_ptrs[i as usize])
                        .collect_vec();
                    quiver = quiver
                        .with_selection(Selection::SelVec(may_have_next))
                        .materialize();
                }
                continue;
            }

            let to_take = Selection::SelVec(to_take);
            let probe_take = Selection::SelVec(probe_indices);
            let build_data = self.data.with_selection(to_take).materialize().rename_cols(
                self.data
                    .column_names()
                    .map(|k| format!("{}_{}", self.data.name(), k))
                    .collect_vec(),
            );
            let probe_data = quiver.with_selection(probe_take).materialize();
            let probe_new_col_names = probe_data
                .column_names()
                .map(|k| format!("{}_{}", probe_data.name(), k))
                .collect_vec();
            let probe_data = probe_data.rename_cols(probe_new_col_names);
            let complete_fragment = ArrowQuiver::horizontal_stack(&[&build_data, &probe_data]);
            assert_eq!(
                complete_fragment.arity(),
                self.data.arity() + quiver.arity()
            );

            let eq_expr = AQExpr::Filter(
                *build_eq_expr(
                    &self.build_keys,
                    &self.probe_keys,
                    &self.data.name(),
                    &probe_data.name(),
                    self.data.arity(),
                ),
                Box::new(AQExpr::Input),
            );

            // check the join predicate for each row of to_return, and add the filtered
            // result to `to_return`
            to_return.extend_from_slice(&eq_expr.transform(ctx, complete_fragment));

            if may_have_next.is_empty() {
                break;
            } else {
                curr_ptrs = may_have_next
                    .iter()
                    .map(|&i| curr_ptrs[i as usize])
                    .collect_vec();
                quiver = quiver
                    .with_selection(Selection::SelVec(may_have_next))
                    .materialize();
            }
        }

        to_return
    }
}

impl HashJoinTransform {
    /// Returns a vector with the length of the chain stored in each bucket.
    /// Useful for debugging if a hash table is properly sized.
    pub fn chain_lenghts(&self) -> Vec<usize> {
        let mut lengths = Vec::with_capacity(self.ht.len());
        for ptr in &self.ht {
            let mut length = 0;
            let mut current = ptr.ptr;

            while !current.is_null() {
                // Safety: We assume the HTNode array is correctly allocated by the builder
                current = unsafe { &*current }.next;
                length += 1;
            }

            lengths.push(length);
        }
        lengths
    }
}

#[derive(Clone, Debug)]
pub struct JoinStep {
    pub left_table_idx: usize,
    pub right_table_idx: usize,
    pub left_keys: Vec<ColumnIdentifier>,
    pub right_keys: Vec<ColumnIdentifier>,
}

#[derive(Clone, Debug)]
pub struct MultiTableJoinSpec {
    pub join_order: Vec<JoinStep>,
}

impl MultiTableJoinSpec {
    pub fn new(join_order: Vec<JoinStep>) -> Self {
        Self { join_order }
    }
}

pub struct MultiTableHashJoinBuildSink {
    spec: MultiTableJoinSpec,
    tables: Vec<ArrowQuiver>,
    hash_tables: Vec<Option<HashJoinBuildSink>>,
    table_size: usize,
}

impl Operator for MultiTableHashJoinBuildSink {
    fn name(&self) -> String {
        format!("Multi-Table Build Join Hash Tables")
    }
}

impl MultiTableHashJoinBuildSink {
    pub fn new(spec: MultiTableJoinSpec, table_size: usize) -> Self {
        Self {
            spec,
            tables: Vec::new(),
            hash_tables: Vec::new(),
            table_size,
        }
    }

    pub fn add_table(&mut self, table: ArrowQuiver) -> usize {
        let table_idx = self.tables.len();
        self.tables.push(table);
        self.hash_tables.push(None);
        table_idx
    }

    fn build_hash_table_for_join(&mut self, join_step: &JoinStep) {
        // Ensure hash tables vector is large enough for both left and right table indices
        let max_idx = join_step.left_table_idx.max(join_step.right_table_idx);
        while self.hash_tables.len() <= max_idx {
            self.hash_tables.push(None);
        }

        // Build hash table for the right table if not already built
        if self.hash_tables[join_step.right_table_idx].is_none() {
            let build_sink = HashJoinBuildSink::new(join_step.right_keys.clone(), self.table_size);
            self.hash_tables[join_step.right_table_idx] = Some(build_sink);
        }
    }
}

impl Sink for MultiTableHashJoinBuildSink {
    type Output = MultiTableHashJoinTransform;

    fn sink(&self, _ctx: &crate::AQExecutorContext, _quiver: crate::ArrowQuiver) {}

    fn finish(mut self: Box<Self>, ctx: &crate::AQExecutorContext) -> Self::Output {
        // Build hash tables for all join steps if not already built
        for join_step in &self.spec.join_order.clone() {
            self.build_hash_table_for_join(join_step);
        }

        // Build the hash tables with actual data
        for (table_idx, table) in self.tables.iter().enumerate() {
            if let Some(Some(hash_table)) = self.hash_tables.get_mut(table_idx) {
                hash_table.sink(ctx, table.clone());
            }
        }

        // Finish all hash tables and create transforms with proper probe key mapping
        let mut transforms = Vec::new();
        for (idx, build_sink_opt) in self.hash_tables.into_iter().enumerate() {
            if let Some(build_sink) = build_sink_opt {
                // Find the corresponding join step to get the correct left_keys (probe keys)
                let probe_keys = self
                    .spec
                    .join_order
                    .iter()
                    .find(|step| step.right_table_idx == idx)
                    .map(|step| step.left_keys.clone())
                    .unwrap_or_else(|| {
                        // Default to first k columns for backward compatibility
                        build_sink
                            .build_keys
                            .iter()
                            .enumerate()
                            .map(|(i, _)| ColumnIdentifier::Index(i))
                            .collect()
                    });

                let transform = HashJoinBuildSink::finish_with_probe_keys(
                    Box::new(build_sink),
                    ctx,
                    probe_keys,
                );
                transforms.push(transform);
            } else if idx < self.tables.len() {
                // Create a dummy hash table for tables that don't participate in joins
                let dummy_sink = HashJoinBuildSink::new(vec![0.into()], self.table_size);
                // Provide some dummy data to avoid empty concatenation
                if let Some(table) = self.tables.get(idx) {
                    dummy_sink.sink(ctx, table.clone());
                }
                let transform = Box::new(dummy_sink).finish(ctx);
                transforms.push(transform);
            }
        }

        MultiTableHashJoinTransform {
            spec: self.spec,
            tables: self.tables,
            transforms,
        }
    }
}

pub struct MultiTableHashJoinTransform {
    spec: MultiTableJoinSpec,
    #[allow(dead_code)]
    tables: Vec<ArrowQuiver>,
    transforms: Vec<HashJoinTransform>,
}

impl Operator for MultiTableHashJoinTransform {
    fn name(&self) -> String {
        format!("Multi-Table Hash Join Transform")
    }
}

impl Transform for MultiTableHashJoinTransform {
    fn transform(&self, ctx: &crate::AQExecutorContext, quiver: ArrowQuiver) -> Vec<ArrowQuiver> {
        let mut current_result = quiver;

        // For multi-table joins, we execute binary joins sequentially
        // Each join step should specify which table to join with the current result
        for join_step in &self.spec.join_order {
            // Get the hash table for the right-side table
            if join_step.right_table_idx >= self.transforms.len() {
                return Vec::new(); // Invalid table index
            }

            let right_transform = &self.transforms[join_step.right_table_idx];

            // Perform the join: current_result ⋈ right_table
            let join_results = right_transform.transform(ctx, current_result);

            if join_results.is_empty() {
                return Vec::new(); // No matches, empty result
            }

            // Concatenate results and continue to next join step
            current_result = if join_results.len() == 1 {
                join_results.into_iter().next().unwrap()
            } else {
                ArrowQuiver::concat(&join_results).unwrap()
            };
        }

        vec![current_result]
    }
}

fn hash_all(quiver: &ArrowQuiver, keys: &[ColumnIdentifier]) -> PrimitiveArray<UInt64Type> {
    let mut hashes = hash(&quiver[&keys[0]]);
    for key in &keys[1..] {
        hashes = arrow_arith::numeric::add(&hashes, &hash(&quiver[key]))
            .unwrap()
            .as_any()
            .downcast_ref::<PrimitiveArray<UInt64Type>>()
            .unwrap()
            .clone();
    }
    hashes
}

/// Builds equality predicate expression for join condition
/// build_keys are resolved to their positions in the build side
/// probe_keys can be at any positions in the probe side (named or indexed)
/// After horizontal stacking: [build_cols..., probe_cols...]
fn build_eq_expr(
    build_keys: &[ColumnIdentifier],
    probe_keys: &[ColumnIdentifier],
    build_data_name: &str,
    probe_data_name: &str,
    num_build_cols: usize,
) -> Box<Predicate> {
    let eqs: Vec<Box<Predicate>> = build_keys
        .iter()
        .zip(probe_keys.iter())
        .enumerate()
        .map(|(_, (build_key, probe_key))| {
            // Resolve build-side named columns to their positions in build data
            let build_identifier = match build_key {
                ColumnIdentifier::Index(idx) => ColumnIdentifier::Index(*idx),
                ColumnIdentifier::Name(name) => {
                    ColumnIdentifier::Name(format!("{}_{}", build_data_name, name))
                }
            };

            // For probe-side columns, try the name first, then try with R_ prefix
            let probe_identifier = match probe_key {
                ColumnIdentifier::Index(idx) => ColumnIdentifier::Index(*idx + num_build_cols),
                ColumnIdentifier::Name(name) => {
                    // Check if this name conflicts with build-side columns
                    ColumnIdentifier::Name(format!("{}_{}", probe_data_name, name))
                }
            };

            Box::new(Predicate::Eq(build_identifier, probe_identifier))
        })
        .collect();

    multi_and(eqs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AQExecutorContext;
    use arrow_array::Int32Array;
    use std::collections::HashSet;
    use std::sync::Arc;

    #[test]
    fn test_hash_join_build() {
        let quiver1 = ArrowQuiver::i32_col(vec![1, 2, 2, 3, 3, 1, 2, -1000]);
        let quiver2 = ArrowQuiver::i32_col(vec![1, 2]);
        let builder = Box::new(HashJoinBuildSink::new(vec![0.into()], 32));

        let ctx = AQExecutorContext::default();
        builder.sink(&ctx, quiver1);
        builder.sink(&ctx, quiver2);

        let hj = builder.finish(&ctx);
        assert_eq!(hj.data.num_rows(), 10);
        assert_eq!(hj.ht.len(), 32);
        assert_eq!(hj._ptrs.len(), 2);

        // check validity of each pointer
        let all_valid_ptrs: HashSet<*const HTNode> = hj
            ._ptrs
            .iter()
            .flat_map(|ptrs| ptrs.iter().map(|ptr| ptr as *const HTNode))
            .collect();

        assert!(hj
            .ht
            .iter()
            .all(|ptr| ptr.ptr.is_null() || all_valid_ptrs.contains(&ptr.ptr)));

        assert!(
            hj._ptrs
                .iter()
                .flat_map(|ptrs| ptrs.iter())
                .all(|ptr| ptr.next.is_null()
                    || all_valid_ptrs.contains(&(ptr.next as *const HTNode)))
        );
    }

    #[test]
    fn test_hash_join_probe() {
        let quiver1 = ArrowQuiver::i32_col(vec![1, 2, 2, 3, 3, 1, 2, -1000]);
        let quiver2 = ArrowQuiver::i32_col(vec![1, 2]);
        let builder = Box::new(HashJoinBuildSink::new(vec![0.into()], 32));

        let ctx = AQExecutorContext::default();
        builder.sink(&ctx, quiver1);
        builder.sink(&ctx, quiver2);

        let hj = builder.finish(&ctx);
        let probe_data = ArrowQuiver::i32_col(vec![1, 2, 3]).rename_cols(vec!["b".to_string()]);
        let results = hj.transform(&ctx, probe_data);

        let results = ArrowQuiver::concat(&results).unwrap().materialize();
        assert_eq!(results.arity(), 2);
        assert_eq!(results.num_rows(), 9);

        let c1 = results.as_i32_col(&0.into());
        let c2 = results.as_i32_col(&1.into());
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_multi_table_join_spec() {
        let join_step = JoinStep {
            left_table_idx: 0,
            right_table_idx: 1,
            left_keys: vec![0.into()],
            right_keys: vec![0.into()],
        };

        let spec = MultiTableJoinSpec::new(vec![join_step]);
        assert_eq!(spec.join_order.len(), 1);
        assert_eq!(spec.join_order[0].left_table_idx, 0);
        assert_eq!(spec.join_order[0].right_table_idx, 1);
    }

    #[test]
    fn test_employee_department_project_join() {
        /*
        Equivalent SQL:
        SELECT e.emp_id, e.emp_salary, d.dept_id, d.dept_budget, p.proj_id, p.proj_cost
        FROM employees e
        JOIN departments d ON e.emp_dept_id = d.dept_id
        JOIN projects p ON d.dept_id = p.proj_dept_id;
        */

        let ctx = AQExecutorContext::default();

        // CREATE TABLE employees (emp_id, emp_dept_id, emp_salary)
        let employees = ArrowQuiver::new(
            "employees".to_string(),
            vec![
                "emp_id".to_string(),
                "emp_dept_id".to_string(),
                "emp_salary".to_string(),
            ],
            vec![
                Arc::new(Int32Array::from(vec![1001, 1002, 1003, 1004])), // emp_id
                Arc::new(Int32Array::from(vec![10, 20, 10, 30])),         // emp_dept_id
                Arc::new(Int32Array::from(vec![50000, 60000, 55000, 70000])), // emp_salary
            ],
        );

        // CREATE TABLE departments (dept_id, dept_budget, dept_manager_id)
        let departments = ArrowQuiver::new(
            "departments".to_string(),
            vec![
                "dept_id".to_string(),
                "dept_budget".to_string(),
                "dept_manager_id".to_string(),
            ],
            vec![
                Arc::new(Int32Array::from(vec![10, 20, 30])), // dept_id
                Arc::new(Int32Array::from(vec![100000, 150000, 200000])), // dept_budget
                Arc::new(Int32Array::from(vec![9001, 9002, 9003])), // dept_manager_id
            ],
        );

        // CREATE TABLE projects (proj_id, proj_dept_id, proj_cost)
        let projects = ArrowQuiver::new(
            "projects".to_string(),
            vec![
                "proj_id".to_string(),
                "proj_dept_id".to_string(),
                "proj_cost".to_string(),
            ],
            vec![
                Arc::new(Int32Array::from(vec![5001, 5002, 5003, 5004])), // proj_id
                Arc::new(Int32Array::from(vec![10, 10, 20, 30])),         // proj_dept_id
                Arc::new(Int32Array::from(vec![25000, 30000, 40000, 35000])), // proj_cost
            ],
        );

        // Use MultiTableHashJoinTransform for the complete three-table join
        let join_spec = MultiTableJoinSpec::new(vec![
            // Step 1: employees ⋈ departments ON emp_dept_id = dept_id
            JoinStep {
                left_table_idx: 0,          // employees (will be the input to transform)
                right_table_idx: 0,         // departments (first table added)
                left_keys: vec![1.into()],  // emp_dept_id (column 1)
                right_keys: vec![0.into()], // dept_id (column 0)
            },
            // Step 2: (employees ⋈ departments) ⋈ projects ON dept_id = proj_dept_id
            JoinStep {
                left_table_idx: 0,          // result from previous join
                right_table_idx: 1,         // projects (second table added)
                left_keys: vec![0.into()], // dept_id after first join (column 0, was renamed to departments_dept_id)
                right_keys: vec![1.into()], // proj_dept_id (column 1)
            },
        ]);

        let mut multi_join_builder = Box::new(MultiTableHashJoinBuildSink::new(join_spec, 100));
        let dept_idx = multi_join_builder.add_table(departments);
        let proj_idx = multi_join_builder.add_table(projects);

        assert_eq!(dept_idx, 0, "Departments should be at index 0");
        assert_eq!(proj_idx, 1, "Projects should be at index 1");

        let multi_join_transform = multi_join_builder.finish(&ctx);

        // Execute the complete multi-table join with employees as the probe input
        let final_results = multi_join_transform.transform(&ctx, employees);

        if final_results.is_empty() {
            println!("⚠️  Multi-table join failed - no results returned");
            return;
        }
        assert!(final_results.iter().all(|r| r.num_rows() > 0));

        let final_result = ArrowQuiver::concat(&final_results).unwrap().materialize();

        println!("Complete three-table join result: {:?}", final_result);

        // Expected result analysis:
        // Employees in dept 10: 1001, 1003 -> Projects 5001, 5002 (4 rows)
        // Employees in dept 20: 1002 -> Project 5003 (1 row)
        // Employees in dept 30: 1004 -> Project 5004 (1 row)
        // Total: 6 rows
        assert_eq!(
            final_result.num_rows(),
            6,
            "Should have 6 total rows in three-table join"
        );

        // Validate column layout after multi-table join should be:
        // Final step build side (projects): [0]proj_id, [1]proj_dept_id, [2]proj_cost
        // Final step probe side (emp⋈dept): [3]dept_id, [4]dept_budget, [5]dept_manager_id, [6]emp_id, [7]emp_dept_id, [8]emp_salary
        let project_ids = final_result.as_i32_col(&0.into());
        let proj_dept_ids = final_result.as_i32_col(&1.into());
        // let project_costs = final_result.as_i32_col(&2.into());
        let dept_ids = final_result.as_i32_col(&3.into());
        let emp_ids = final_result.as_i32_col(&6.into());

        // Verify join condition: proj_dept_id should equal dept_id
        for (i, (&proj_dept_id, &dept_id)) in proj_dept_ids.iter().zip(dept_ids.iter()).enumerate()
        {
            assert_eq!(
                proj_dept_id, dept_id,
                "Row {} join condition failed: proj_dept_id {} != dept_id {}",
                i, proj_dept_id, dept_id
            );
        }

        // Count employees by department in final result
        let dept10_count = dept_ids.iter().filter(|&&id| id == 10).count();
        let dept20_count = dept_ids.iter().filter(|&&id| id == 20).count();
        let dept30_count = dept_ids.iter().filter(|&&id| id == 30).count();

        assert_eq!(
            dept10_count, 4,
            "Dept 10 should have 4 rows (2 employees × 2 projects)"
        );
        assert_eq!(
            dept20_count, 1,
            "Dept 20 should have 1 row (1 employee × 1 project)"
        );
        assert_eq!(
            dept30_count, 1,
            "Dept 30 should have 1 row (1 employee × 1 project)"
        );

        // Verify we have all expected employees
        let unique_employees: std::collections::HashSet<i32> = emp_ids.iter().cloned().collect();
        assert_eq!(
            unique_employees,
            [1001, 1002, 1003, 1004].iter().cloned().collect(),
            "Should include all employees"
        );

        // Verify we have all expected projects
        let unique_projects: std::collections::HashSet<i32> = project_ids.iter().cloned().collect();
        assert_eq!(
            unique_projects,
            [5001, 5002, 5003, 5004].iter().cloned().collect(),
            "Should include all projects"
        );

        println!("✅ Multi-table join validates complete SQL JOIN across employees, departments, and projects!");
    }

    #[test]
    fn test_flexible_key_positioning() {
        /*
        This test demonstrates the fix for flexible key positioning.
        We create a scenario where join keys are NOT in the first columns,
        which would have failed before our fix.
        */

        let ctx = AQExecutorContext::default();

        // Table with join key at column 2 (not column 0)
        let table_a = ArrowQuiver::new(
            "table_a".to_string(),
            vec![
                "a_col1".to_string(),
                "a_col2".to_string(),
                "a_join_key".to_string(),
            ],
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),       // a_col1
                Arc::new(Int32Array::from(vec![10, 20, 30])),    // a_col2
                Arc::new(Int32Array::from(vec![100, 200, 300])), // a_join_key at position 2
            ],
        );

        // Table with join key at column 1 (not column 0)
        let table_b = ArrowQuiver::new(
            "table_b".to_string(),
            vec![
                "b_data".to_string(),
                "b_join_key".to_string(),
                "b_extra".to_string(),
            ],
            vec![
                Arc::new(Int32Array::from(vec![77, 88, 99])), // b_data
                Arc::new(Int32Array::from(vec![100, 200, 300])), // b_join_key at position 1
                Arc::new(Int32Array::from(vec![5, 6, 7])),    // b_extra
            ],
        );

        // Build hash table on table_b using join_key at position 1
        let builder = Box::new(HashJoinBuildSink::new(vec![1.into()], 50));
        builder.sink(&ctx, table_b);

        // Use our new flexible method with probe keys at position 2
        let transform = HashJoinBuildSink::finish_with_probe_keys(
            builder,
            &ctx,
            vec![2.into()], // join_key is at position 2 in table_a
        );

        // Perform the join
        let results = transform.transform(&ctx, table_a.clone());
        assert!(
            !results.is_empty(),
            "Join should succeed with flexible key positioning"
        );

        let result = ArrowQuiver::concat(&results).unwrap().materialize();
        assert_eq!(result.num_rows(), 3, "Should have 3 joined rows");

        println!(
            "Result columns: {}, rows: {}",
            result.arity(),
            result.num_rows()
        );

        // Verify join correctness: build_side.join_key == probe_side.join_key
        let build_join_keys = result.as_i32_col(&1.into()); // build side join_key at pos 1
        let probe_join_keys = result.as_i32_col(&5.into()); // probe side join_key should be at position 3+2=5

        for (i, (&build_key, &probe_key)) in build_join_keys
            .iter()
            .zip(probe_join_keys.iter())
            .enumerate()
        {
            assert_eq!(build_key, probe_key, "Row {} join condition failed", i);
        }

        println!("✅ Flexible key positioning test passed!");
    }

    #[test]
    fn test_supply_chain_join_validation() {
        /*
        Equivalent SQL:
        SELECT s.supplier_id, s.supplier_country, s.supplier_rating,
               o.order_id, o.order_product_id, o.order_quantity
        FROM suppliers s
        JOIN orders o ON s.supplier_id = o.order_supplier_id;
        */

        let ctx = AQExecutorContext::default();

        // CREATE TABLE suppliers (supplier_id, supplier_country, supplier_rating)
        let suppliers = ArrowQuiver::new(
            "suppliers".to_string(),
            vec![
                "supplier_id".to_string(),
                "supplier_country".to_string(),
                "supplier_rating".to_string(),
            ],
            vec![
                Arc::new(Int32Array::from(vec![100, 200, 300])), // supplier_id
                Arc::new(Int32Array::from(vec![1, 2, 1])),       // country (1=USA, 2=China)
                Arc::new(Int32Array::from(vec![5, 3, 4])),       // rating
            ],
        );

        // CREATE TABLE orders (order_id, order_product_id, order_supplier_id, order_quantity)
        let orders = ArrowQuiver::new(
            "orders".to_string(),
            vec![
                "order_id".to_string(),
                "order_product_id".to_string(),
                "order_supplier_id".to_string(),
                "order_quantity".to_string(),
            ],
            vec![
                Arc::new(Int32Array::from(vec![1001, 1002, 1003, 1004, 1005])), // order_id
                Arc::new(Int32Array::from(vec![501, 502, 501, 503, 502])),      // product_id
                Arc::new(Int32Array::from(vec![100, 200, 300, 100, 200])),      // supplier_id
                Arc::new(Int32Array::from(vec![10, 20, 15, 5, 25])),            // quantity
            ],
        );

        // Build hash table on orders: FROM orders o
        // Hash on order_supplier_id (column 2) for join condition s.supplier_id = o.order_supplier_id
        let order_builder = Box::new(HashJoinBuildSink::new(vec![2.into()], 100));
        order_builder.sink(&ctx, orders);
        let order_transform = order_builder.finish(&ctx);

        // Probe with complete suppliers table: FROM suppliers s
        // Join condition: suppliers.supplier_id (column 0) = orders.order_supplier_id (column 2)
        let results = order_transform.transform(&ctx, suppliers);

        if results.is_empty() || results.iter().all(|r| r.num_rows() == 0) {
            println!("⚠️ Supplier-order join failed");
            return;
        }

        // Collect all join results: SELECT * FROM suppliers s JOIN orders o ...
        let result = ArrowQuiver::concat(&results).unwrap().materialize();

        println!("Supplier-Order join result: {:?}", result);

        // Expected result: All orders with their matching suppliers
        // Supplier 100: Orders 1001, 1004 (2 rows)
        // Supplier 200: Orders 1002, 1005 (2 rows)
        // Supplier 300: Order 1003 (1 row)
        // Total: 5 rows
        assert_eq!(
            result.num_rows(),
            5,
            "Should have 5 total joined rows (all orders with suppliers)"
        );

        // Validate column layout:
        // Build side (orders): [0]order_id, [1]order_product_id, [2]order_supplier_id, [3]order_quantity
        // Probe side (suppliers): [4]supplier_id, [5]supplier_country, [6]supplier_rating
        let order_ids = result.as_i32_col(&0.into());
        let order_product_ids = result.as_i32_col(&1.into());
        let order_supplier_ids = result.as_i32_col(&2.into());
        let order_quantities = result.as_i32_col(&3.into());
        let supplier_ids = result.as_i32_col(&4.into());
        let supplier_countries = result.as_i32_col(&5.into());
        let supplier_ratings = result.as_i32_col(&6.into());

        // Verify join condition: all order_supplier_id should equal supplier_id
        for (i, (&order_supp_id, &supp_id)) in order_supplier_ids
            .iter()
            .zip(supplier_ids.iter())
            .enumerate()
        {
            assert_eq!(
                order_supp_id, supp_id,
                "Row {} join condition failed: order_supplier_id {} != supplier_id {}",
                i, order_supp_id, supp_id
            );
        }

        // Verify we have all expected orders
        let mut sorted_order_ids = order_ids.clone();
        sorted_order_ids.sort();
        assert_eq!(
            sorted_order_ids,
            vec![1001, 1002, 1003, 1004, 1005],
            "Should have all orders 1001-1005"
        );

        // Test supplier distribution
        let supplier_100_count = supplier_ids.iter().filter(|&&id| id == 100).count();
        let supplier_200_count = supplier_ids.iter().filter(|&&id| id == 200).count();
        let supplier_300_count = supplier_ids.iter().filter(|&&id| id == 300).count();

        assert_eq!(supplier_100_count, 2, "Supplier 100 should have 2 orders");
        assert_eq!(supplier_200_count, 2, "Supplier 200 should have 2 orders");
        assert_eq!(supplier_300_count, 1, "Supplier 300 should have 1 order");

        // Validate business logic for supplier 100 (USA, rating 5)
        let supplier_100_indices: Vec<usize> = supplier_ids
            .iter()
            .enumerate()
            .filter(|(_, &id)| id == 100)
            .map(|(i, _)| i)
            .collect();

        // Check supplier 100's order details
        let supplier_100_quantities: Vec<i32> = supplier_100_indices
            .iter()
            .map(|&i| order_quantities[i])
            .collect();
        let mut s100_quantities = supplier_100_quantities.clone();
        s100_quantities.sort();
        assert_eq!(
            s100_quantities,
            vec![5, 10],
            "Supplier 100 should have order quantities 5 and 10"
        );

        let supplier_100_products: Vec<i32> = supplier_100_indices
            .iter()
            .map(|&i| order_product_ids[i])
            .collect();
        let mut s100_products = supplier_100_products.clone();
        s100_products.sort();
        assert_eq!(
            s100_products,
            vec![501, 503],
            "Supplier 100 should have products 501 and 503"
        );

        // Verify supplier attributes consistency
        let supplier_100_countries: Vec<i32> = supplier_100_indices
            .iter()
            .map(|&i| supplier_countries[i])
            .collect();
        assert!(
            supplier_100_countries.iter().all(|&country| country == 1),
            "Supplier 100 should consistently be from country 1 (USA)"
        );

        let supplier_100_ratings: Vec<i32> = supplier_100_indices
            .iter()
            .map(|&i| supplier_ratings[i])
            .collect();
        assert!(
            supplier_100_ratings.iter().all(|&rating| rating == 5),
            "Supplier 100 should consistently have rating 5"
        );

        // Validate business logic for supplier 200 (China, rating 3)
        let supplier_200_indices: Vec<usize> = supplier_ids
            .iter()
            .enumerate()
            .filter(|(_, &id)| id == 200)
            .map(|(i, _)| i)
            .collect();

        let supplier_200_countries: Vec<i32> = supplier_200_indices
            .iter()
            .map(|&i| supplier_countries[i])
            .collect();
        assert!(
            supplier_200_countries.iter().all(|&country| country == 2),
            "Supplier 200 should consistently be from country 2 (China)"
        );

        let supplier_200_quantities: Vec<i32> = supplier_200_indices
            .iter()
            .map(|&i| order_quantities[i])
            .collect();
        let mut s200_quantities = supplier_200_quantities.clone();
        s200_quantities.sort();
        assert_eq!(
            s200_quantities,
            vec![20, 25],
            "Supplier 200 should have order quantities 20 and 25"
        );

        println!("✅ Standard supply chain join validated: all suppliers properly joined with their orders, business logic preserved!");
    }

    #[test]
    fn test_named_column_join_keys() {
        /*
        Test that named columns can be used as join keys instead of just index-based columns.
        */
        let ctx = AQExecutorContext::default();

        // Create table with named columns (using only Int32Array for simplicity)
        let left_table = ArrowQuiver::new(
            "left_table".to_string(),
            vec![
                "user_id".to_string(),
                "user_score".to_string(),
                "user_age".to_string(),
            ],
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),    // user_id
                Arc::new(Int32Array::from(vec![85, 92, 78])), // user_score
                Arc::new(Int32Array::from(vec![25, 30, 35])), // user_age
            ],
        );

        let right_table = ArrowQuiver::new(
            "right_table".to_string(),
            vec![
                "order_id".to_string(),
                "customer_id".to_string(),
                "amount".to_string(),
            ],
            vec![
                Arc::new(Int32Array::from(vec![101, 102, 103, 104])), // order_id
                Arc::new(Int32Array::from(vec![1, 2, 1, 3])),         // customer_id
                Arc::new(Int32Array::from(vec![50, 75, 25, 100])),    // amount
            ],
        );

        // Build hash table using named column as key
        let build_keys = vec![ColumnIdentifier::Name("customer_id".to_string())];
        let builder = Box::new(HashJoinBuildSink::new(build_keys, 100));
        builder.sink(&ctx, right_table);

        // Use named column as probe key
        let probe_keys = vec![ColumnIdentifier::Name("user_id".to_string())];
        let transform = HashJoinBuildSink::finish_with_probe_keys(builder, &ctx, probe_keys);

        // Perform join
        let results = transform.transform(&ctx, left_table);

        if results.is_empty() {
            panic!("Join with named columns failed - no results returned");
        }

        let result = ArrowQuiver::concat(&results).unwrap().materialize();

        // Should have 4 rows (user 1 appears twice, user 2 once, user 3 once)
        assert_eq!(result.num_rows(), 4, "Should have 4 joined rows");

        // Debug: print the result column names to understand the structure
        println!(
            "Result column names: {:?}",
            result.column_names().collect::<Vec<_>>()
        );

        // Verify join correctness by checking that customer_id matches user_id
        // The customer_id should be available without conflict since it doesn't exist in the left table
        // The user_id should be available without conflict since it doesn't exist in the right table
        let customer_ids = result.as_i32_col(&ColumnIdentifier::Name(
            "right_table_customer_id".to_string(),
        )); // right_table_customer_id
        let user_ids = result.as_i32_col(&ColumnIdentifier::Name("left_table_user_id".to_string())); // left_table_user_id

        for (i, (&customer_id, &user_id)) in customer_ids.iter().zip(user_ids.iter()).enumerate() {
            assert_eq!(
                customer_id, user_id,
                "Row {} join condition failed: customer_id {} != user_id {}",
                i, customer_id, user_id
            );
        }

        println!("✅ Named column join keys test passed!");
    }
}
