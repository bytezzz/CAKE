use crate::dispatchers::regret_tree::RegretTree;
use crate::dispatchers::{
    CakeHyperParams, DecisionModel, ExplorationRequest, Heuristic, ModelDecision,
    SmartWareConfiguration,
};
use argminmax::ArgMinMax;
use ndarray::linalg::general_mat_vec_mul;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

type Float = f32;

// Approximate inverse CDF (probit) for the standard normal distribution (Acklam 2003).
fn ndtri(p: Float) -> Float {
    debug_assert!((0.0..=1.0).contains(&p));

    let a1 = -3.969_683_028_665_376e+01;
    let a2 = 2.209_460_984_245_205e+02;
    let a3 = -2.759_285_104_469_687e+02;
    let a4 = 1.383_577_518_672_690e+02;
    let a5 = -3.066_479_806_614_716e+01;
    let a6 = 2.506_628_277_459_239e+00;

    let b1 = -5.447_609_879_822_406e+01;
    let b2 = 1.615_858_368_580_409e+02;
    let b3 = -1.556_989_798_598_866e+02;
    let b4 = 6.680_131_188_771_972e+01;
    let b5 = -1.328_068_155_288_572e+01;

    let c1 = -7.784_894_002_430_293e-03;
    let c2 = -3.223_964_580_411_365e-01;
    let c3 = -2.400_758_277_161_838e+00;
    let c4 = -2.549_732_539_343_734e+00;
    let c5 = 4.374_664_141_464_968e+00;
    let c6 = 2.938_163_982_698_783e+00;

    let d1 = 7.784_695_709_041_462e-03;
    let d2 = 3.224_671_290_700_398e-01;
    let d3 = 2.445_134_137_142_996e+00;
    let d4 = 3.754_408_661_907_416e+00;

    let p_low = 0.024_25;
    let p_high = 1.0 - p_low;

    if p <= 0.0 {
        return Float::NEG_INFINITY;
    }
    if p >= 1.0 {
        return Float::INFINITY;
    }

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
            / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    } else if p > p_high {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
            / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    } else {
        let q = p - 0.5;
        let r = q * q;
        (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
            / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct ConditionalOrderingResult {
    valid: bool,
    confident_best: bool,
    effective_sample_size: Float,
    weighted_means: Vec<Float>,
}

#[derive(Default)]
struct Workspace {
    x_norms_sq: Array1<Float>,
    method_means: Array1<Float>,
    method_second_moments: Array1<Float>,
    method_std_errs: Array1<Float>,
}

fn fill_weighted_stats(
    weights: ArrayView1<'_, Float>,
    y_lat: ArrayView2<'_, Float>,
    y_lat_sq: ArrayView2<'_, Float>,
    method_means: &mut Array1<Float>,
    method_second_moments: &mut Array1<Float>,
    method_std_errs: &mut Array1<Float>,
) {
    let n_samples = y_lat.nrows();
    if n_samples == 0 {
        return;
    }
    let n_methods = y_lat.ncols();
    let inv_n_samples: Float = 1.0 / n_samples as Float;
    if method_means.len() != n_methods {
        *method_means = Array1::zeros(n_methods);
        *method_second_moments = Array1::zeros(n_methods);
        *method_std_errs = Array1::zeros(n_methods);
    }
    general_mat_vec_mul(1.0, &y_lat.t(), &weights, 0.0, method_means);
    general_mat_vec_mul(1.0, &y_lat_sq.t(), &weights, 0.0, method_second_moments);

    for m in 0..n_methods {
        let mu = method_means[m];
        let var = (method_second_moments[m] - mu * mu) * inv_n_samples;
        let clamped = if var < 0.0 { 0.0 } else { var };
        method_std_errs[m] = clamped.sqrt();
    }
}

fn test_conditional_ordering(
    x_feat: ArrayView2<'_, Float>,
    y_lat: ArrayView2<'_, Float>,
    y_lat_sq: ArrayView2<'_, Float>,
    query_x: ArrayView1<'_, Float>,
    cached_norms: &[Float],
    alpha: Float,
    params: &CakeHyperParams,
    workspace: &mut Workspace,
) -> ConditionalOrderingResult {
    let n_samples = y_lat.nrows();
    let n_methods = y_lat.ncols();
    if n_samples == 0 || n_methods == 0 || cached_norms.len() != n_samples {
        return ConditionalOrderingResult {
            valid: false,
            confident_best: false,
            effective_sample_size: 0.0,
            weighted_means: Vec::new(),
        };
    }
    if !(alpha > 0.0 && alpha < 1.0) {
        return ConditionalOrderingResult {
            valid: false,
            confident_best: false,
            effective_sample_size: 0.0,
            weighted_means: Vec::new(),
        };
    }

    if workspace.x_norms_sq.len() != n_samples {
        workspace.x_norms_sq = Array1::zeros(n_samples);
    }
    general_mat_vec_mul(1.0, &x_feat, &query_x, 0.0, &mut workspace.x_norms_sq);
    let q_norm_sq = query_x.dot(&query_x);
    for (dot_entry, &x_norm_sq) in workspace.x_norms_sq.iter_mut().zip(cached_norms.iter()) {
        let dist = x_norm_sq + q_norm_sq - 2.0 * *dot_entry;
        *dot_entry = dist.max(0.0);
    }

    let bw = params.bandwidth as Float;
    let bw2 = bw * bw;

    // REUSE MEMORY: Reuse x_norms_sq buffer to store weights to avoid allocation.
    // From this point on, x_norms_sq contains the weights.
    workspace.x_norms_sq.mapv_inplace(|d| (-d / bw2).exp());
    let wsum: Float = workspace.x_norms_sq.sum();
    if wsum < 1e-10_f32 {
        return ConditionalOrderingResult {
            valid: false,
            confident_best: false,
            effective_sample_size: 0.0,
            weighted_means: Vec::new(),
        };
    }

    let inv_wsum: Float = 1.0 / wsum;
    workspace.x_norms_sq.mapv_inplace(|w| w * inv_wsum);
    let weight_sum_sq: Float = workspace.x_norms_sq.dot(&workspace.x_norms_sq);
    let neff: Float = if weight_sum_sq > 0.0 {
        1.0 / weight_sum_sq
    } else {
        0.0
    };
    if neff < params.n_min as Float {
        return ConditionalOrderingResult {
            valid: false,
            confident_best: false,
            effective_sample_size: neff,
            weighted_means: Vec::new(),
        };
    }

    // 2) Weighted mean and CLT standard error (matches the previous Normal approximation path).
    fill_weighted_stats(
        workspace.x_norms_sq.view(),
        y_lat.view(),
        y_lat_sq.view(),
        &mut workspace.method_means,
        &mut workspace.method_second_moments,
        &mut workspace.method_std_errs,
    );
    let weighted_means = workspace.method_means.to_vec();

    // 3) Pairwise comparisons
    let best_idx = weighted_means.argmin();

    let comparison_count = n_methods.saturating_sub(1);
    if comparison_count == 0 {
        return ConditionalOrderingResult {
            valid: true,
            confident_best: true,
            effective_sample_size: neff,
            weighted_means,
        };
    }

    let adjusted_alpha = alpha / comparison_count as Float;
    let z_crit = ndtri(1.0 - adjusted_alpha * 0.5) as Float;

    let mut confident_best = true;
    for other in 0..n_methods {
        if other == best_idx {
            continue;
        }
        let (i, j) = if best_idx < other {
            (best_idx, other)
        } else {
            (other, best_idx)
        };
        let diff_mu = workspace.method_means[i] - workspace.method_means[j];
        let se_i = workspace.method_std_errs[i];
        let se_j = workspace.method_std_errs[j];
        let diff_se = (se_i * se_i + se_j * se_j).sqrt();

        let (ci_lo, ci_hi) = if diff_se == 0.0 {
            (diff_mu, diff_mu)
        } else {
            let margin = z_crit * diff_se;
            let ci_lo = diff_mu - margin;
            let ci_hi = diff_mu + margin;
            (ci_lo, ci_hi)
        };

        let winner = if ci_hi < 0.0 {
            Some(i)
        } else if ci_lo > 0.0 {
            Some(j)
        } else {
            None
        };
        if winner != Some(best_idx) {
            confident_best = false;
        }
    }

    ConditionalOrderingResult {
        valid: true,
        confident_best,
        effective_sample_size: neff,
        weighted_means,
    }
}

pub struct CakeCLT<H: Heuristic> {
    heuristic: H,
    config: SmartWareConfiguration,
    regret_tree_after_decisions: Option<usize>,
    regret_tree: Option<RegretTree>,
    pending: Option<ModelDecision>,
    force_explore: bool,
    last_context: Option<Vec<Float>>,
    flat_x: Vec<Float>,
    flat_y: Vec<Float>,
    flat_y_sq: Vec<Float>,
    cached_x_norms: Vec<Float>,
    num_samples: usize,
    decision_count: usize,
    workspace: Workspace,
}

impl<H: Heuristic> CakeCLT<H> {
    pub fn new(heuristic: H, config: SmartWareConfiguration) -> Self {
        let regret_tree_after_decisions = config.cake_params().regret_tree_after_decisions;
        Self {
            heuristic,
            config,
            regret_tree_after_decisions,
            regret_tree: None,
            pending: None,
            force_explore: false,
            last_context: None,
            flat_x: Vec::new(),
            flat_y: Vec::new(),
            flat_y_sq: Vec::new(),
            cached_x_norms: Vec::new(),
            num_samples: 0,
            decision_count: 0,
            workspace: Workspace::default(),
        }
    }

    fn buffer_size(&self) -> Option<usize> {
        self.config
            .cake
            .as_ref()
            .and_then(|params| params.buffer_size)
    }

    fn buffer_limit_reached(&self) -> bool {
        self.buffer_size()
            .map_or(false, |limit| self.num_samples >= limit)
    }

    fn maybe_train_regret_tree(&mut self, current_features: &[f32]) {
        let Some(after) = self.regret_tree_after_decisions else {
            return;
        };
        if self.regret_tree.is_some() {
            return;
        }

        let train_at_decision = after.saturating_add(1);
        if self.decision_count < train_at_decision {
            return;
        }

        let tree = if self.num_samples == 0 {
            let policy = (self.heuristic)(current_features);
            RegretTree::stub(policy)
        } else {
            let num_features = self.config.num_features;
            let num_policies = self.config.num_policies;
            let n_rows = self.num_samples;
            let n_cols = num_features + num_policies;

            let mut flat = Vec::with_capacity(n_rows * n_cols);
            for row in 0..n_rows {
                let feat_offset = row * num_features;
                flat.extend_from_slice(&self.flat_x[feat_offset..feat_offset + num_features]);

                let lat_offset = row * num_policies;
                flat.extend_from_slice(&self.flat_y[lat_offset..lat_offset + num_policies]);
            }
            let data = Array2::from_shape_vec((n_rows, n_cols), flat)
                .expect("regret tree training data shape mismatch");
            let max_depth = self
                .config
                .cake_params()
                .max_depth
                .expect("cake.max_depth is required for cake policy");
            RegretTree::fit(data, num_features, max_depth)
        };

        self.regret_tree = Some(tree);
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
            .expect("cake.alpha is required for cake policy") as Float;

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
        let y_lat_sq = ArrayView2::from_shape(
            (self.num_samples, self.config.num_policies),
            &self.flat_y_sq,
        )
        .expect("flat_y_sq shape mismatch");
        let result = test_conditional_ordering(
            x_feat.view(),
            y_lat.view(),
            y_lat_sq.view(),
            ArrayView1::from(&x),
            &self.cached_x_norms,
            alpha,
            params,
            &mut self.workspace,
        );

        if !result.valid {
            self.force_explore = true;
            return ModelDecision::Initialize;
        }

        let best_idx = if result.weighted_means.is_empty() {
            0
        } else {
            result.weighted_means.argmin()
        };

        self.force_explore = !result.confident_best;

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

    fn add_observation(&mut self, features: &[f32], latencies: &[Float]) {
        self.flat_x.reserve(features.len());

        for &f in features {
            self.flat_x.push(f as Float);
        }
        let feat_norm_sq: Float = features.iter().map(|&v| (v as Float).powi(2)).sum();
        self.cached_x_norms.push(feat_norm_sq);

        self.flat_y.reserve(latencies.len());
        self.flat_y_sq.reserve(latencies.len());

        for &y in latencies {
            self.flat_y.push(y);
            self.flat_y_sq.push(y * y);
        }

        self.num_samples += 1;
        self.last_context = None;
    }
}

impl<H: Heuristic> DecisionModel for CakeCLT<H> {
    fn start_decision(&mut self, features: &[f32]) -> &ModelDecision {
        debug_assert!(self.pending.is_none(), "pending decision leaked");
        self.decision_count += 1;
        self.maybe_train_regret_tree(features);

        let decision = if let Some(tree) = &self.regret_tree {
            assert_eq!(
                features.len(),
                self.config.num_features,
                "feature length {} does not match expected {}",
                features.len(),
                self.config.num_features
            );

            let selected_policy = tree.predict(ArrayView1::from(features));
            self.force_explore = false;

            ModelDecision::Predict {
                selected_policy,
                predictive_policy: selected_policy,
                optimisitc_policy: selected_policy,
                predictive_mean: 0.0,
                optimisitc_mean: 0.0,
            }
        } else {
            self.observe_inner(features)
        };
        self.pending = Some(decision);
        self.pending.as_ref().unwrap()
    }

    fn after_selected(&mut self, _latency: f32) -> ExplorationRequest {
        let pending = self.pending.take().unwrap();
        if self.buffer_limit_reached() {
            self.force_explore = false;
            return ExplorationRequest::Skip;
        }
        if self.regret_tree.is_some() {
            self.force_explore = false;
            return ExplorationRequest::Skip;
        }
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

        if self.buffer_limit_reached() {
            self.last_context = None;
            return;
        }

        let latencies_f64: Vec<f64> = latencies
            .iter()
            .map(|l| l.expect("checked above"))
            .collect();
        let mut dense: Vec<Float> = Vec::with_capacity(latencies.len());
        for &l in &latencies_f64 {
            dense.push(l as Float);
        }

        let feature_source: Vec<f32> = self
            .last_context
            .as_ref()
            .map(|ctx| ctx.iter().map(|v| *v as f32).collect())
            .unwrap_or_else(|| features.to_vec());

        self.add_observation(&feature_source, &dense);
    }

    fn heuristic(&self, features: &[f32]) -> usize {
        (self.heuristic)(features)
    }
}
