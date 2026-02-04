use std::{
    error::Error,
    iter,
    sync::atomic::{AtomicBool, Ordering},
    thread,
};

use crossbeam::deque::{Injector, Stealer, Worker};
use itertools::Itertools;

use crate::{AQExecutorContext, Sink, Source, Transform};

pub struct Pipeline<O> {
    pub source: Box<dyn Source>,
    pub transforms: Vec<Box<dyn Transform>>,
    pub sink: Box<dyn Sink<Output = O>>,
}

impl<O> Pipeline<O> {
    pub fn execute_serial(mut self, ctx: &AQExecutorContext) -> Result<O, Box<dyn Error>> {
        while let Some(quiver) = self.source.produce(ctx) {
            let mut quivers = vec![quiver?];
            for t in &self.transforms {
                quivers = quivers
                    .into_iter()
                    .flat_map(|quiver| t.transform(ctx, quiver))
                    .collect_vec();
            }

            for quiver in quivers {
                self.sink.sink(ctx, quiver);
            }
        }

        let output = self.sink.finish(ctx);
        Ok(output)
    }

    pub fn execute_parallel(
        mut self,
        ctx: &AQExecutorContext,
        num_workers: usize,
        max_chunk_size: usize,
    ) -> Result<O, Box<dyn Error>> {
        let injector = Injector::new();
        let mut stealers = Vec::new();
        let finished = AtomicBool::new(false);

        thread::scope(|s| {
            let mut worker_queues = Vec::new();
            for _ in 0..num_workers {
                let wq = Worker::new_fifo();
                stealers.push(wq.stealer());
                worker_queues.push(wq);
            }

            for wq in worker_queues {
                let stealers = &stealers;
                let injector = &injector;
                let sink = &self.sink;
                let xforms = &self.transforms;
                let finished = &finished;
                s.spawn(move || {
                    while !finished.load(Ordering::Relaxed) {
                        while let Some(task) = find_task(&wq, injector, stealers) {
                            let mut quivers = vec![task];
                            for t in xforms {
                                quivers = quivers
                                    .into_iter()
                                    .flat_map(|quiver| t.transform(ctx, quiver))
                                    .collect_vec();
                            }

                            for quiver in quivers {
                                sink.sink(ctx, quiver);
                            }
                        }
                    }
                });
            }

            while let Some(quiver) = self.source.produce(ctx) {
                match quiver {
                    Ok(q) => {
                        if q.num_rows() > max_chunk_size {
                            for start in (0..q.num_rows()).step_by(max_chunk_size) {
                                injector.push(q.slice(
                                    start..usize::min(start + max_chunk_size, q.num_rows()),
                                ));
                            }
                        } else {
                            injector.push(q);
                        }
                    }
                    Err(e) => return Err(e),
                };
            }
            finished.store(true, Ordering::Relaxed);

            Ok(())
        })?;

        Ok(self.sink.finish(ctx))
    }
}

// https://docs.rs/crossbeam/latest/crossbeam/deque/index.html
fn find_task<T>(local: &Worker<T>, global: &Injector<T>, stealers: &[Stealer<T>]) -> Option<T> {
    // Pop a task from the local queue, if not empty.
    local.pop().or_else(|| {
        // Otherwise, we need to look for a task elsewhere.
        iter::repeat_with(|| {
            // Try stealing a batch of tasks from the global queue.
            global
                .steal_batch_and_pop(local)
                // Or try stealing a task from one of the other threads.
                .or_else(|| stealers.iter().map(|s| s.steal()).collect())
        })
        // Loop while no task was stolen and any steal operation needs to be retried.
        .find(|s| !s.is_retry())
        // Extract the stolen task, if there is one.
        .and_then(|s| s.success())
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::Int32Array;

    use crate::{
        expression::AQExpr,
        parquet::{tests::create_test_parquet, ParquetSource},
        predicate::Predicate,
        spool::Spool,
        AQExecutorContext,
    };

    use super::Pipeline;

    #[test]
    fn test_serial_pipeline() {
        let ctx = AQExecutorContext::default();
        let p = create_test_parquet();

        let source = ParquetSource::new(p).unwrap();
        let expr = AQExpr::Materialize(Box::new(AQExpr::Filter(
            Predicate::LessThanConst(0.into(), Arc::new(Int32Array::new_scalar(500))),
            Box::new(AQExpr::Input),
        )));
        let sink = Spool::new();

        let pipeline = Pipeline {
            source: Box::new(source),
            transforms: vec![Box::new(expr)],
            sink: Box::new(sink),
        };

        let results = pipeline.execute_serial(&ctx).unwrap();
        assert_eq!(results.iter().map(|aq| aq.num_rows()).sum::<usize>(), 500);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_parallel_pipeline() {
        let ctx = AQExecutorContext::default();
        let p = create_test_parquet();

        let source = ParquetSource::new(p).unwrap();
        let expr = AQExpr::Materialize(Box::new(AQExpr::Filter(
            Predicate::LessThanConst(0.into(), Arc::new(Int32Array::new_scalar(500))),
            Box::new(AQExpr::Input),
        )));
        let sink = Spool::new();

        let pipeline = Pipeline {
            source: Box::new(source),
            transforms: vec![Box::new(expr)],
            sink: Box::new(sink),
        };

        let results = pipeline.execute_parallel(&ctx, 4, 256).unwrap();
        assert_eq!(results.iter().map(|aq| aq.num_rows()).sum::<usize>(), 500);
    }
}
