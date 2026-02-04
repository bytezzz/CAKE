use crate::dispatchers::{
    DecisionModel, ExplorationRequest, Heuristic, CakeHyperParams, ModelDecision,
    SmartWareConfiguration,
};
use ndarray::linalg::general_mat_mul;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::distributions::{Distribution, WeightedIndex};
use rand::prelude::*;
use rand::rngs::SmallRng;
use std::cmp::Ordering;

type Float = f32;

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct ConditionalOrderingResult {
    valid: bool,
    all_different_strictly: bool,
    effective_sample_size: Float,
    weighted_means: Vec<Float>,
}

#[derive(Default)]
struct Workspace {
    means: Array2<Float>,
    diffs: Array2<Float>,
    x_norms_sq: Array1<Float>,
}

fn sample_multinomial_counts(
    n_samples: usize,
    prob_weights: &[Float],
    n_bootstrap: usize,
    rng: &mut SmallRng,
) -> Array2<Float> {
    let dist = WeightedIndex::new(prob_weights.to_vec())
        .expect("probability weights must contain at least one positive entry");
    let mut counts = Array2::<Float>::zeros((n_bootstrap, n_samples));
    for mut row in counts.outer_iter_mut() {
        for _ in 0..n_samples {
            let idx = dist.sample(rng);
            row[idx] += 1.0;
        }
    }
    counts
}

fn test_conditional_ordering(
    x_feat: ArrayView2<'_, Float>,
    y_lat: ArrayView2<'_, Float>,
    query_x: ArrayView1<'_, Float>,
    alpha: Float,
    n_bootstrap: usize,
    params: &CakeHyperParams,
    rng: &mut SmallRng,
    workspace: &mut Workspace,
) -> ConditionalOrderingResult {
    let n_samples = y_lat.nrows();
    let n_methods = y_lat.ncols();
    if n_samples == 0 || n_methods == 0 || n_bootstrap == 0 {
        return ConditionalOrderingResult {
            valid: false,
            all_different_strictly: false,
            effective_sample_size: 0.0,
            weighted_means: Vec::new(),
        };
    }

    // 1) Distance weights via GEMV: ||x-q||^2 = ||x||^2 + ||q||^2 - 2 x·q
    if workspace.x_norms_sq.len() != n_samples {
        workspace.x_norms_sq = Array1::zeros(n_samples);
    }
    for (slot, row) in workspace.x_norms_sq.iter_mut().zip(x_feat.rows()) {
        *slot = row.dot(&row);
    }
    let q_norm_sq = query_x.dot(&query_x);
    let dot_prod = x_feat.dot(&query_x); // GEMV
    let mut dists_sq = workspace.x_norms_sq.clone();
    dists_sq += q_norm_sq;
    dists_sq.iter_mut().zip(dot_prod.iter()).for_each(|(a, b)| {
        *a -= 2.0 * *b;
    });
    dists_sq.mapv_inplace(|v| v.max(0.0));

    let bw = params.bandwidth as Float;
    let bw2 = bw * bw;
    let mut weights = dists_sq.mapv(|d| (-d / bw2).exp());
    let wsum: Float = weights.sum();
    if wsum < 1e-10 {
        return ConditionalOrderingResult {
            valid: false,
            all_different_strictly: false,
            effective_sample_size: 0.0,
            weighted_means: Vec::new(),
        };
    }

    weights.mapv_inplace(|w| w / wsum);
    let weight_sum_sq: Float = weights.dot(&weights);
    let neff = if weight_sum_sq > 0.0 {
        1.0 / weight_sum_sq
    } else {
        0.0
    };
    if neff < params.n_min as Float {
        return ConditionalOrderingResult {
            valid: false,
            all_different_strictly: false,
            effective_sample_size: neff,
            weighted_means: Vec::new(),
        };
    }

    // 2) Weighted bootstrapping: counts @ Y_lat
    let counts =
        sample_multinomial_counts(n_samples, weights.as_slice().unwrap(), n_bootstrap, rng);
    if workspace.means.dim() != (n_bootstrap, n_methods) {
        workspace.means = Array2::zeros((n_bootstrap, n_methods));
    } else {
        workspace.means.fill(0.0);
    }
    general_mat_mul(
        1.0 / n_samples as Float,
        &counts,
        &y_lat,
        0.0,
        &mut workspace.means,
    );

    // Weighted means reused for decision reporting
    let weighted_means_arr = y_lat.t().dot(&weights);
    let weighted_means = weighted_means_arr.to_vec();

    // 3) Pairwise comparisons: best vs. others only
    let best_idx = weighted_means
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    let comparison_count = n_methods.saturating_sub(1);
    if comparison_count == 0 {
        return ConditionalOrderingResult {
            valid: true,
            all_different_strictly: true,
            effective_sample_size: neff,
            weighted_means,
        };
    }

    let b_minus_1 = (n_bootstrap - 1) as Float;
    let adjusted_alpha = alpha / comparison_count as Float;
    let q_lo = adjusted_alpha * 0.5;
    let q_hi = 1.0 - q_lo;

    // Diff matrix: (comparison_count, n_bootstrap) row-major for cache-friendly sorts
    if workspace.diffs.dim() != (comparison_count, n_bootstrap) {
        workspace.diffs = Array2::zeros((comparison_count, n_bootstrap));
    } else {
        workspace.diffs.fill(0.0);
    }
    let mut confident_best = true;
    let mut row_idx = 0;
    for other in 0..n_methods {
        if other == best_idx {
            continue;
        }
        let (i, j) = if best_idx < other {
            (best_idx, other)
        } else {
            (other, best_idx)
        };
        let mut row_mut = workspace.diffs.row_mut(row_idx);
        for (dst, (a, b)) in row_mut.iter_mut().zip(
            workspace
                .means
                .column(i)
                .iter()
                .zip(workspace.means.column(j).iter()),
        ) {
            *dst = *a - *b;
        }

        // row is contiguous because row-major layout
        let slice = row_mut.as_slice_mut().unwrap();
        slice.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        let idx_lo = ((q_lo * b_minus_1).round() as usize).min(slice.len() - 1);
        let idx_hi = ((q_hi * b_minus_1).round() as usize).min(slice.len() - 1);
        let v_lo = slice[idx_lo];
        let v_hi = slice[idx_hi];

        let winner = if v_hi < 0.0 {
            Some(i)
        } else if v_lo > 0.0 {
            Some(j)
        } else {
            None
        };
        if winner != Some(best_idx) {
            confident_best = false;
        }
        row_idx += 1;
    }

    ConditionalOrderingResult {
        valid: true,
        all_different_strictly: confident_best,
        effective_sample_size: neff,
        weighted_means,
    }
}

pub struct CakeBS<H: Heuristic> {
    heuristic: H,
    config: SmartWareConfiguration,
    pending: Option<ModelDecision>,
    force_explore: bool,
    last_context: Option<Vec<Float>>,
    flat_x: Vec<Float>,
    flat_y: Vec<Float>,
    num_samples: usize,
    rng: SmallRng,
    workspace: Workspace,
}

impl<H: Heuristic> CakeBS<H> {
    pub fn new(heuristic: H, config: SmartWareConfiguration) -> Self {
        let rng = SmallRng::seed_from_u64(42);
        Self {
            heuristic,
            config,
            pending: None,
            force_explore: false,
            last_context: None,
            flat_x: Vec::new(),
            flat_y: Vec::new(),
            num_samples: 0,
            rng,
            workspace: Workspace::default(),
        }
    }

    pub fn set_rng_seed(&mut self, seed: u64) {
        self.rng = SmallRng::seed_from_u64(seed);
    }

    fn observe_inner(&mut self, features: &[f32]) -> ModelDecision {
        assert_eq!(
            features.len(),
            self.config.num_features,
            "feature length {} does not match expected {}",
            features.len(),
            self.config.num_features
        );
        let mut x = Vec::with_capacity(features.len());
        for f in features {
            x.push(*f as Float);
        }
        self.last_context = Some(x.clone());

        let params = self.config.cake_params();
        let alpha = params
            .alpha
            .expect("cake.alpha is required for cake_bs policy") as Float;

        if self.num_samples == 0 {
            self.force_explore = true;
            return ModelDecision::Initialize;
        }

        let x_feat =
            ArrayView2::from_shape((self.num_samples, self.config.num_features), &self.flat_x)
                .expect("flat_x shape mismatch");
        let y_lat =
            ArrayView2::from_shape((self.num_samples, self.config.num_policies), &self.flat_y)
                .expect("flat_y shape mismatch");
        let result = test_conditional_ordering(
            x_feat.view(),
            y_lat.view(),
            ArrayView1::from(&x),
            alpha,
            1000,
            params,
            &mut self.rng,
            &mut self.workspace,
        );

        if !result.valid {
            self.force_explore = true;
            return ModelDecision::Initialize;
        }

        let best_idx = result
            .weighted_means
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        self.force_explore = !result.all_different_strictly;
        let predictive_mean: f64 =
            result.weighted_means.get(best_idx).copied().unwrap_or(0.0) as f64;

        ModelDecision::Predict {
            selected_policy: best_idx,
            predictive_policy: best_idx,
            optimisitc_policy: best_idx,
            predictive_mean,
            optimisitc_mean: predictive_mean,
        }
    }

    fn add_observation(&mut self, features: &[Float], latencies: &[Float]) {
        let mut feat_row = Vec::with_capacity(features.len());
        for f in features {
            feat_row.push(*f);
        }
        let mut lat_row = Vec::with_capacity(latencies.len());
        lat_row.extend_from_slice(latencies);
        self.flat_x.extend_from_slice(&feat_row);
        self.flat_y.extend_from_slice(&lat_row);
        self.num_samples += 1;
        self.last_context = None;
    }
}

impl<H: Heuristic> DecisionModel for CakeBS<H> {
    fn start_decision(&mut self, features: &[f32]) -> &ModelDecision {
        debug_assert!(self.pending.is_none(), "pending decision leaked");
        let decision = self.observe_inner(features);
        self.pending = Some(decision);
        self.pending.as_ref().unwrap()
    }

    fn after_selected(&mut self, _latency: f32) -> ExplorationRequest {
        let pending = self.pending.take().unwrap();
        match pending {
            ModelDecision::Predict { .. } => {
                if self.force_explore {
                    ExplorationRequest::RequireAll
                } else {
                    ExplorationRequest::Skip
                }
            }
            ModelDecision::Initialize => ExplorationRequest::RequireAll,
        }
    }

    fn finalize(&mut self, features: &[f32], latencies: &[Option<f64>]) {
        // Only update when we received full information (RequireAll case).
        if latencies.len() != self.config.num_policies {
            self.last_context = None;
            return;
        }
        if latencies.iter().any(|l| l.is_none()) {
            self.last_context = None;
            return;
        }
        let mut dense = Vec::with_capacity(latencies.len());
        for l in latencies {
            dense.push(l.unwrap() as Float);
        }

        let feature_source: Vec<Float> = self
            .last_context
            .as_ref()
            .map(|ctx| ctx.clone())
            .unwrap_or_else(|| features.to_vec());

        self.add_observation(&feature_source, &dense);
    }

    fn heuristic(&self, features: &[f32]) -> usize {
        (self.heuristic)(features)
    }
}
