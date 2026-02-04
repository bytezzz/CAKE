use crate::configuration::get_metric_type;
use arrow_schema::ArrowError;
use cpu_time::ProcessTime;

/// Helper trait for converting metric values to f64
pub trait MetricValue: Copy + Send + Sync {
    fn to_f64(self) -> f64;
}

impl MetricValue for f64 {
    fn to_f64(self) -> f64 {
        self
    }
}

impl MetricValue for u64 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl MetricValue for u32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl MetricValue for i64 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}

/// Trait for measuring arbitrary scalar metrics during operation execution
pub trait MetricCollector: Send + Sync {
    /// The type of scalar value being measured (e.g., f64 for time, u64 for cycles)
    type Value: MetricValue;

    /// Start a new measurement
    fn start_measurement(&self) -> Box<dyn Measurement<Value = Self::Value>>;

    /// Get the name of this metric for logging/debugging
    fn metric_name(&self) -> &str;
}

/// Represents an ongoing measurement
pub trait Measurement: Send {
    type Value;

    /// Finalize the measurement and return the result
    fn finish(self: Box<Self>) -> Self::Value;
}

// ============== Time-based metric collector ==============

#[derive(Clone, Copy)]
pub struct TimeMetricCollector;

impl MetricCollector for TimeMetricCollector {
    type Value = f64; // nanoseconds as f64

    fn start_measurement(&self) -> Box<dyn Measurement<Value = Self::Value>> {
        Box::new(TimeMeasurement {
            start_time: ProcessTime::now(),
        })
    }

    fn metric_name(&self) -> &str {
        "time_ns"
    }
}

struct TimeMeasurement {
    start_time: ProcessTime,
}

impl Measurement for TimeMeasurement {
    type Value = f64;

    fn finish(self: Box<Self>) -> Self::Value {
        self.start_time.elapsed().as_nanos() as f64
    }
}

// ============== Performance counter collectors (perfcnt-backed) ==============

#[cfg(all(target_os = "linux", any(target_arch = "x86", target_arch = "x86_64")))]
mod perf_counter_collectors {
    use super::{Measurement, MetricCollector};
    use perfcnt::linux::{HardwareEventType, PerfCounterBuilderLinux};
    use perfcnt::{AbstractPerfCounter, PerfCounter};

    #[derive(Clone, Copy)]
    pub struct CycleMetricCollector;

    impl MetricCollector for CycleMetricCollector {
        type Value = u64;

        fn start_measurement(&self) -> Box<dyn Measurement<Value = Self::Value>> {
            let counter =
                PerfCounterBuilderLinux::from_hardware_event(HardwareEventType::CPUCycles)
                    .finish()
                    .expect("Failed to create CPU cycle counter");

            counter.start().expect("Failed to start counter");

            Box::new(CycleMeasurement {
                counter: Box::new(counter),
            })
        }

        fn metric_name(&self) -> &str {
            "cpu_cycles"
        }
    }

    struct CycleMeasurement {
        counter: Box<PerfCounter>,
    }

    impl Measurement for CycleMeasurement {
        type Value = u64;

        fn finish(mut self: Box<Self>) -> Self::Value {
            self.counter.stop().expect("Failed to stop counter");
            self.counter.read().expect("Failed to read counter")
        }
    }

    #[derive(Clone, Copy)]
    pub struct InstructionMetricCollector;

    impl MetricCollector for InstructionMetricCollector {
        type Value = u64;

        fn start_measurement(&self) -> Box<dyn Measurement<Value = Self::Value>> {
            let counter =
                PerfCounterBuilderLinux::from_hardware_event(HardwareEventType::Instructions)
                    .finish()
                    .expect("Failed to create instruction counter");

            counter.start().expect("Failed to start counter");

            Box::new(InstructionMeasurement {
                counter: Box::new(counter),
            })
        }

        fn metric_name(&self) -> &str {
            "instructions"
        }
    }

    struct InstructionMeasurement {
        counter: Box<PerfCounter>,
    }

    impl Measurement for InstructionMeasurement {
        type Value = u64;

        fn finish(mut self: Box<Self>) -> Self::Value {
            self.counter.stop().expect("Failed to stop counter");
            self.counter.read().expect("Failed to read counter")
        }
    }

    #[derive(Clone, Copy)]
    pub struct CacheMissMetricCollector;

    impl MetricCollector for CacheMissMetricCollector {
        type Value = u64;

        fn start_measurement(&self) -> Box<dyn Measurement<Value = Self::Value>> {
            let counter =
                PerfCounterBuilderLinux::from_hardware_event(HardwareEventType::CacheMisses)
                    .finish()
                    .expect("Failed to create cache miss counter");

            counter.start().expect("Failed to start counter");

            Box::new(CacheMissMeasurement {
                counter: Box::new(counter),
            })
        }

        fn metric_name(&self) -> &str {
            "cache_misses"
        }
    }

    struct CacheMissMeasurement {
        counter: Box<PerfCounter>,
    }

    impl Measurement for CacheMissMeasurement {
        type Value = u64;

        fn finish(mut self: Box<Self>) -> Self::Value {
            self.counter.stop().expect("Failed to stop counter");
            self.counter.read().expect("Failed to read counter")
        }
    }
}

#[cfg(not(all(target_os = "linux", any(target_arch = "x86", target_arch = "x86_64"))))]
mod perf_counter_collectors {
    use super::{Measurement, MetricCollector};

    fn unsupported_metric(metric_name: &str) -> ! {
        panic!(
            "{metric_name} metric collector requires Linux on x86/x86_64 for perfcnt; \
             current target: {}/{}",
            std::env::consts::OS,
            std::env::consts::ARCH
        );
    }

    macro_rules! unsupported_collector {
        ($name:ident, $metric_label:literal) => {
            #[derive(Clone, Copy)]
            pub struct $name;

            impl MetricCollector for $name {
                type Value = u64;

                fn start_measurement(&self) -> Box<dyn Measurement<Value = Self::Value>> {
                    unsupported_metric($metric_label);
                }

                fn metric_name(&self) -> &str {
                    $metric_label
                }
            }
        };
    }

    unsupported_collector!(CycleMetricCollector, "cpu_cycles");
    unsupported_collector!(InstructionMetricCollector, "instructions");
    unsupported_collector!(CacheMissMetricCollector, "cache_misses");
}

use perf_counter_collectors::{
    CacheMissMetricCollector, CycleMetricCollector, InstructionMetricCollector,
};

// ============== Generic measured petition ==============

pub trait MeasuredPetition<'a>: Sized {
    type Output;
    type Collector: MetricCollector;

    fn op_type(&self) -> &'static str;
    fn to_features(&self) -> Vec<f32>;
    fn metric_collector(&self) -> &Self::Collector;

    /// Take ownership of operations with measurement capability
    #[allow(clippy::type_complexity)]
    fn take_measured_operations(
        self,
    ) -> Vec<
        Box<
            dyn FnOnce() -> Result<
                    (Self::Output, <Self::Collector as MetricCollector>::Value),
                    ArrowError,
                > + 'a,
        >,
    >;

    /// Execute a specific operation and return both result and metric
    fn operate(
        self,
        operation_id: usize,
    ) -> Result<(Self::Output, <Self::Collector as MetricCollector>::Value), ArrowError> {
        let mut operations = self.take_measured_operations();
        if operation_id >= operations.len() {
            return Err(ArrowError::InvalidArgumentError(format!(
                "Invalid operation ID: {}, out of bounds for {} operations",
                operation_id,
                operations.len()
            )));
        }
        let operation_fn = operations.remove(operation_id);
        operation_fn()
    }

    /// Execute all operations and collect all metrics (converted to f64)
    fn discover(self) -> Result<(Self::Output, Vec<f64>), ArrowError> {
        let ops = self.take_measured_operations();
        let mut metrics = Vec::new();
        let mut iter = ops.into_iter();

        let first_op = iter.next().ok_or_else(|| {
            ArrowError::InvalidArgumentError("No operations provided".to_string())
        })?;

        let (output, first_metric) = first_op()?;
        metrics.push(first_metric.to_f64());

        for op in iter {
            let (_result, metric) = op()?;
            metrics.push(metric.to_f64());
        }

        Ok((output, metrics))
    }
}

// ============== Factory functions for creating collectors based on config ==============

/// Enum wrapper for different metric collectors
pub enum MetricCollectorWrapper {
    Time(TimeMetricCollector),
    Cycles(CycleMetricCollector),
    Instructions(InstructionMetricCollector),
    CacheMisses(CacheMissMetricCollector),
}

impl Clone for MetricCollectorWrapper {
    fn clone(&self) -> Self {
        match self {
            MetricCollectorWrapper::Time(c) => MetricCollectorWrapper::Time(*c),
            MetricCollectorWrapper::Cycles(c) => MetricCollectorWrapper::Cycles(*c),
            MetricCollectorWrapper::Instructions(c) => MetricCollectorWrapper::Instructions(*c),
            MetricCollectorWrapper::CacheMisses(c) => MetricCollectorWrapper::CacheMisses(*c),
        }
    }
}

impl MetricCollectorWrapper {
    pub fn start_measurement(&self) -> Box<dyn MetricMeasurement> {
        match self {
            MetricCollectorWrapper::Time(c) => Box::new(TimeMeasurementWrapper {
                inner: c.start_measurement(),
            }),
            MetricCollectorWrapper::Cycles(c) => Box::new(CycleMeasurementWrapper {
                inner: c.start_measurement(),
            }),
            MetricCollectorWrapper::Instructions(c) => Box::new(InstructionMeasurementWrapper {
                inner: c.start_measurement(),
            }),
            MetricCollectorWrapper::CacheMisses(c) => Box::new(CacheMissMeasurementWrapper {
                inner: c.start_measurement(),
            }),
        }
    }
}

/// Unified measurement trait that returns f64
pub trait MetricMeasurement: Send {
    fn finish(self: Box<Self>) -> f64;
}

struct TimeMeasurementWrapper {
    inner: Box<dyn Measurement<Value = f64>>,
}

impl MetricMeasurement for TimeMeasurementWrapper {
    fn finish(self: Box<Self>) -> f64 {
        self.inner.finish()
    }
}

struct CycleMeasurementWrapper {
    inner: Box<dyn Measurement<Value = u64>>,
}

impl MetricMeasurement for CycleMeasurementWrapper {
    fn finish(self: Box<Self>) -> f64 {
        self.inner.finish() as f64
    }
}

struct InstructionMeasurementWrapper {
    inner: Box<dyn Measurement<Value = u64>>,
}

impl MetricMeasurement for InstructionMeasurementWrapper {
    fn finish(self: Box<Self>) -> f64 {
        self.inner.finish() as f64
    }
}

struct CacheMissMeasurementWrapper {
    inner: Box<dyn Measurement<Value = u64>>,
}

impl MetricMeasurement for CacheMissMeasurementWrapper {
    fn finish(self: Box<Self>) -> f64 {
        self.inner.finish() as f64
    }
}

/// Create a metric collector based on configuration
pub fn create_metric_collector(op_type: &str) -> Result<MetricCollectorWrapper, ArrowError> {
    let metric_type = get_metric_type(op_type)?;

    match metric_type.as_str() {
        "time" => Ok(MetricCollectorWrapper::Time(TimeMetricCollector)),
        "cycles" => Ok(MetricCollectorWrapper::Cycles(CycleMetricCollector)),
        "instructions" => Ok(MetricCollectorWrapper::Instructions(
            InstructionMetricCollector,
        )),
        "cache_misses" => Ok(MetricCollectorWrapper::CacheMisses(
            CacheMissMetricCollector,
        )),
        _ => Err(ArrowError::InvalidArgumentError(format!(
            "Unknown metric type: {}. Valid options: time, cycles, instructions, cache_misses",
            metric_type
        ))),
    }
}
