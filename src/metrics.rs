//! Metrics describing the evolution of the training process.

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Reason a training run terminated.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum StopReason {
    /// The configured target vocabulary size was reached.
    TargetVocabReached,
    /// The configured maximum merge iterations was reached.
    MaxIterationsReached,
    /// No candidate pairs exceeded the minimum frequency.
    NoEligiblePairs,
    /// Plateau-based early stopping aborted the run.
    PlateauReached,
}

/// Metrics captured for each merge iteration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IterationMetrics {
    /// Sequential iteration number (1-indexed).
    pub iteration: usize,
    /// Highest pair frequency observed during the iteration.
    pub best_frequency: usize,
    /// Total number of merges applied within the corpus.
    pub merges_applied: usize,
    /// Count of distinct pairs remaining after the iteration.
    pub distinct_pairs: usize,
    /// Execution time for the iteration.
    pub elapsed_iteration: Duration,
    /// Total time elapsed since training started.
    pub elapsed_total: Duration,
    /// Resident set size sample captured from `/proc/self/status` on Linux.
    pub rss_kb: Option<usize>,
}

/// Aggregate metrics produced by a training session.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrainingMetrics {
    /// Per-iteration snapshots accrued during training.
    pub iterations: Vec<IterationMetrics>,
    /// Total duration of the training session.
    pub total_duration: Duration,
    /// Reason training terminated.
    pub stop_reason: StopReason,
}

impl TrainingMetrics {
    /// Creates an empty metrics container with pre-allocated capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            iterations: Vec::with_capacity(capacity),
            total_duration: Duration::ZERO,
            stop_reason: StopReason::TargetVocabReached,
        }
    }
}

#[cfg(target_os = "linux")]
fn current_rss_kb() -> Option<usize> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let file = File::open("/proc/self/status").ok()?;
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let value = rest
                .split_whitespace()
                .find_map(|part| part.parse::<usize>().ok());
            return value;
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
fn current_rss_kb() -> Option<usize> {
    None
}

/// Samples the current resident set size (RSS) on supported platforms.
pub fn sample_rss_kb() -> Option<usize> {
    current_rss_kb()
}
