//! Performance profiling integration module
//!
//! This module provides integration with Linux `perf` tool for performance profiling.
//! It uses file descriptors to communicate with a perf process that has been started
//! with the `--control` option.
//!
//! # Usage
//!
//! ## Starting perf with control file descriptors
//!
//! Before running your Rust application, you need to set up perf with control FIFOs.
//! Here's an example from `run.sh`:
//!
//! ```bash
//! #!/bin/bash
//!
//! # Build the Rust application first
//! cargo build --release
//!
//! # Name of the FIFOs
//! FIFO_PREFIX="perf_fd"
//!
//! # Remove dangling files if any
//! rm -rf ${FIFO_PREFIX}.*
//!
//! # Create two FIFOs for control and acknowledgment
//! mkfifo ${FIFO_PREFIX}.ctl
//! mkfifo ${FIFO_PREFIX}.ack
//!
//! # Associate file descriptors
//! exec {perf_ctl_fd}<>${FIFO_PREFIX}.ctl
//! exec {perf_ack_fd}<>${FIFO_PREFIX}.ack
//!
//! # Set environment variables for the Rust application
//! export PERF_CTL_FD=${perf_ctl_fd}
//! export PERF_ACK_FD=${perf_ack_fd}
//!
//! # Start perf with the associated file descriptors
//! # The --delay=-1 option starts perf in a paused state
//! # The --control option specifies the control and ack file descriptors
//! perf stat \
//!     --delay=-1 \
//!     --control fd:${perf_ctl_fd},${perf_ack_fd} \
//!     -- ./target/release/your_application
//!
//! # Alternative: Use perf record for call graph profiling
//! # perf record -g \
//! #     --delay=-1 \
//! #     --control fd:${perf_ctl_fd},${perf_ack_fd} \
//! #     -- ./target/release/your_application
//!
//! # Clean up FIFOs after execution
//! rm -rf ${FIFO_PREFIX}.*
//! ```
//!
//! ## Using in Rust code
//!
//! ```rust
//! use perf_manager::{perf_pause, perf_resume};
//!
//! // Start profiling for a specific section
//! perf_resume();
//!
//! // Your performance-critical code here
//! expensive_computation();
//!
//! // Pause profiling to exclude non-critical sections
//! perf_pause();
//!
//! // Non-critical code (e.g., initialization, cleanup)
//! setup_or_cleanup();
//!
//! // Resume profiling for another critical section
//! perf_resume();
//!
//! // More performance-critical code
//! another_expensive_computation();
//!
//! perf_pause();
//! ```
//!
//! ## Environment Variables
//!
//! The module expects two environment variables to be set:
//! - `PERF_CTL_FD`: File descriptor for sending control commands to perf
//! - `PERF_ACK_FD`: File descriptor for receiving acknowledgments from perf
//!
//! If these environment variables are not set, the perf_pause() and perf_resume()
//! functions will be no-ops, allowing the code to run normally without profiling.

use std::env;
use std::fs::File;
use std::io::{Read, Write};
use std::os::unix::io::FromRawFd;
use std::sync::{Mutex, OnceLock};

static PERF_MANAGER: OnceLock<Mutex<PerfManager>> = OnceLock::new();

struct PerfManager {
    ctl_fd: Option<File>,
    ack_fd: Option<File>,
    enabled: bool,
}

impl PerfManager {
    const ENABLE_CMD: &'static [u8] = b"enable";
    const DISABLE_CMD: &'static [u8] = b"disable";
    const ACK_CMD: &'static str = "ack\n\0";

    fn new() -> Self {
        let (ctl_fd, ack_fd, enabled) = match (env::var("PERF_CTL_FD"), env::var("PERF_ACK_FD")) {
            (Ok(ctl_fd_str), Ok(ack_fd_str)) => {
                if let (Ok(ctl_fd_num), Ok(ack_fd_num)) =
                    (ctl_fd_str.parse::<i32>(), ack_fd_str.parse::<i32>())
                {
                    unsafe {
                        let ctl_file = File::from_raw_fd(ctl_fd_num);
                        let ack_file = File::from_raw_fd(ack_fd_num);
                        (Some(ctl_file), Some(ack_file), true)
                    }
                } else {
                    (None, None, false)
                }
            }
            _ => (None, None, false),
        };

        PerfManager {
            ctl_fd,
            ack_fd,
            enabled,
        }
    }

    fn send_command(&mut self, command: &[u8]) {
        if !self.enabled {
            return;
        }

        if let (Some(ref mut ctl_fd), Some(ref mut ack_fd)) = (&mut self.ctl_fd, &mut self.ack_fd) {
            if let Err(e) = ctl_fd.write_all(command) {
                eprintln!("Error writing to control FIFO: {}", e);
                return;
            }

            if let Err(e) = ctl_fd.flush() {
                eprintln!("Error flushing control FIFO: {}", e);
                return;
            }

            let mut ack_buf = vec![0u8; 10];
            match ack_fd.read(&mut ack_buf) {
                Ok(n) => {
                    ack_buf.truncate(n);
                    match std::str::from_utf8(&ack_buf) {
                        Ok(ack_str) => {
                            if ack_str != Self::ACK_CMD {
                                eprintln!(
                                    "Warning: Unexpected acknowledgment from perf: {:?}",
                                    ack_str
                                );
                            }
                        }
                        Err(e) => {
                            eprintln!("Error: Invalid UTF-8 in acknowledgment: {:?}", e);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Error reading acknowledgment: {}", e);
                }
            }
        }
    }

    fn pause(&mut self) {
        self.send_command(Self::DISABLE_CMD);
    }

    fn resume(&mut self) {
        self.send_command(Self::ENABLE_CMD);
    }
}

fn get_perf_manager() -> &'static Mutex<PerfManager> {
    PERF_MANAGER.get_or_init(|| Mutex::new(PerfManager::new()))
}

/// Pause performance profiling.
///
/// This sends a "disable" command to the perf process, causing it to stop
/// collecting performance data. This is useful for excluding non-critical
/// sections of code from profiling results.
///
/// # Example
/// ```rust
/// // Pause profiling during initialization
/// perf_pause();
/// let config = load_configuration();
/// let data = load_initial_data();
///
/// // Resume profiling for the actual computation
/// perf_resume();
/// process_data(&data);
/// ```
pub fn perf_pause() {
    if let Ok(mut manager) = get_perf_manager().lock() {
        manager.pause();
    }
}

/// Resume performance profiling.
///
/// This sends an "enable" command to the perf process, causing it to start
/// or resume collecting performance data. Use this to profile specific
/// sections of code that are performance-critical.
///
/// # Example
/// ```rust
/// // Profile only the critical path
/// perf_resume();
/// let result = expensive_algorithm(&input);
/// perf_pause();
///
/// // Non-critical logging excluded from profiling
/// log_results(&result);
/// ```
pub fn perf_resume() {
    if let Ok(mut manager) = get_perf_manager().lock() {
        manager.resume();
    }
}
