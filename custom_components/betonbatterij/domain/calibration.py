"""Self-learning parameter calibration, Box-Huber-LM solver, and seasonal memory.

Owns pure-Python parameter identification for `wk` and `kt`:
- Box-projected Huber Levenberg-Marquardt MAP solver (`solve_b2_box_huber_lm`)
  with Fisher covariance, correlation matrix, and collinearity diagnostics,
- 52-week Bayesian precision-weighted seasonal memory (`compute_aged_prior`,
  `update_posterior_memory`) with Fisher information gating,
- Stratified 365-day `SeasonalEpisodeBank` retaining top-K clean excitation
  episodes per regime with 24-hour Kalman spin-up,
- Robust steady-state floor pipe conductance calibration
  (`calibrate_k_hx_robust`) and autonomous seasonal/cooling gate
  (`check_seasonal_calibration_gate`), and
- Joint multi-horizon trajectory MAP calibration with out-of-sample holdout
  acceptance gating (`calibrate_zone_a2`).

Does not own RingBuffer ETL (`preprocessing.py`), unrolled Kalman tapes
(`kalman_tape.py`), COP curve fitting (`cop_model.py`), or Home Assistant
locking/persistence (`engine.py`, `ha_bridge/`).

Units: temperatures in °C, thermal capacities in Wh/K, resistances in K/kW,
conductances in W/K, apertures in m², powers in W, time in s.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import math
from typing import Any, Literal

from custom_components.betonbatterij import const
from custom_components.betonbatterij.domain import contracts
from custom_components.betonbatterij.domain import evaluation
from custom_components.betonbatterij.domain import hydraulics
from custom_components.betonbatterij.domain import kalman_tape
from custom_components.betonbatterij.domain import preprocessing
from custom_components.betonbatterij.domain import priors
from custom_components.betonbatterij.domain import solar_weather

# Seasonal memory & calibration constants
MEMORY_WEEKLY_GAMMA: float = preprocessing.MEMORY_WEEKLY_GAMMA
FISHER_GATE_REL_THRESHOLD: float = 0.05  # Min Fisher info vs Day-1 precision
EPISODE_SPINUP_STEPS: int = 288  # 24h Kalman spin-up per stored episode
EPISODE_WINDOW_STEPS: int = 72  # 6h rollout window per stored episode
EPISODE_MAX_PER_REGIME: int = 8
EPISODE_RETENTION_S: int = 365 * 86400
DEFROST_PAD_STEPS: int = 3  # ±15 min around defrost cycles
RESIDUAL_SIGMA_C: float = 0.08  # Trajectory measurement noise scale [K]
MIN_REGIME_VETO_WINDOWS: int = 5  # Min windows before sub-regime can veto


def _hist_matches_buffer_offsets(
    hist: preprocessing.PreprocessedHistory | None,
    buffer_offsets: contracts.BufferSensorOffsets,
) -> bool:
    """Returns True if `hist` matches `buffer_offsets` within `1e-9 °C`."""
    if hist is None:
        return False
    top_h = float(
        getattr(hist, "buf_top_offset_c", priors.BUF_TOP_SENSOR_OFFSET_C)
    )
    bot_h = float(
        getattr(hist, "buf_bot_offset_c", priors.BUF_BOT_SENSOR_OFFSET_C)
    )
    return (
        abs(top_h - buffer_offsets.buf_top_offset_c) <= 1e-9
        and abs(bot_h - buffer_offsets.buf_bot_offset_c) <= 1e-9
    )


# ============================================================================
# 1. Pure-Python Linear Algebra & Box-Projected Huber-LM Solver
# ============================================================================
@dataclasses.dataclass(frozen=True, slots=True)
class SolverResult:
    """Result of `solve_b2_box_huber_lm` with Fisher uncertainty & correlations.

    Attributes:
        theta: Final parameter vector clamped within `[lower, upper]`.
        success: True when converged or completed at least one valid iteration.
        n_iter: Number of Levenberg-Marquardt outer iterations executed.
        n_fev: Total number of `residual_fn` evaluations.
        cost: Final Huber + Gaussian prior MAP objective value.
        posterior_std: Posterior 1-sigma parameter standard deviations.
        posterior_cov: Posterior parameter covariance matrix (`p x p`).
        fisher_diag: Data-only Fisher information diagonal (`J^T W J / sigma^2`).
        correlation_matrix: Posterior parameter correlation matrix (`p x p`).
        collinearity_pairs: Pairs `(j, k, rho)` where `|rho| > 0.90`.
        message: Human-readable solver termination summary.
    """

    theta: list[float]
    success: bool
    n_iter: int
    n_fev: int
    cost: float
    posterior_std: list[float]
    posterior_cov: list[list[float]]
    fisher_diag: list[float]
    correlation_matrix: list[list[float]]
    collinearity_pairs: list[tuple[int, int, float]]
    message: str


def _solve_linear_system(
    mat: Sequence[Sequence[float]],
    rhs: Sequence[float],
) -> list[float]:
    """Solves `mat * x = rhs` via Gaussian elimination with partial pivoting."""
    n = len(rhs)
    if n == 0:
        return []
    aug = [
        [float(mat[i][j]) for j in range(n)] + [float(rhs[i])]
        for i in range(n)
    ]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1e-15:
            aug[col][col] += 1e-12
            pivot_row = col
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        piv = aug[col][col]
        for row in range(col + 1, n):
            factor = aug[row][col] / piv
            if factor == 0.0:
                continue
            for k in range(col, n + 1):
                aug[row][k] -= factor * aug[col][k]

    sol = [0.0] * n
    for row in range(n - 1, -1, -1):
        acc = aug[row][n]
        for col in range(row + 1, n):
            acc -= aug[row][col] * sol[col]
        piv = aug[row][row]
        sol[row] = acc / piv if abs(piv) > 1e-18 else 0.0
    return sol


def _invert_spd_matrix(mat: Sequence[Sequence[float]]) -> list[list[float]]:
    """Inverts a small (`p <= 6`) symmetric positive-definite matrix."""
    n = len(mat)
    cols: list[list[float]] = []
    for j in range(n):
        e_j = [1.0 if i == j else 0.0 for i in range(n)]
        cols.append(_solve_linear_system(mat, e_j))
    return [
        [0.5 * (cols[j][i] + cols[i][j]) for j in range(n)] for i in range(n)
    ]


def _solve_box_kkt_step(
    h_damped: Sequence[Sequence[float]],
    grad: Sequence[float],
    theta: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> list[float]:
    """Computes KKT active-set box-projected step `delta` for `lb <= th+d <= ub`."""
    p = len(theta)
    fixed: dict[int, float] = {}
    for j in range(p):
        if theta[j] <= lower[j] + 1e-11 and grad[j] > 0.0:
            fixed[j] = 0.0
        elif theta[j] >= upper[j] - 1e-11 and grad[j] < 0.0:
            fixed[j] = 0.0

    delta = [0.0] * p
    for _ in range(p + 1):
        free = [j for j in range(p) if j not in fixed]
        if not free:
            break
        m = len(free)
        sub_h = [
            [h_damped[free[a]][free[b]] for b in range(m)] for a in range(m)
        ]
        sub_rhs = [
            -grad[free[a]]
            - sum(h_damped[free[a]][c] * dc for c, dc in fixed.items())
            for a in range(m)
        ]
        sub_sol = _solve_linear_system(sub_h, sub_rhs)

        worst_j: int | None = None
        worst_alpha = 1.0
        clamped_val = 0.0
        for a, j in enumerate(free):
            dj = sub_sol[a]
            cand = theta[j] + dj
            if cand < lower[j] - 1e-12 and dj < 0.0:
                alpha = max(0.0, (lower[j] - theta[j]) / dj)
                if alpha < worst_alpha:
                    worst_alpha = alpha
                    worst_j = j
                    clamped_val = lower[j] - theta[j]
            elif cand > upper[j] + 1e-12 and dj > 0.0:
                alpha = max(0.0, (upper[j] - theta[j]) / dj)
                if alpha < worst_alpha:
                    worst_alpha = alpha
                    worst_j = j
                    clamped_val = upper[j] - theta[j]

        if worst_j is None:
            for a, j in enumerate(free):
                delta[j] = sub_sol[a]
            break

        fixed[worst_j] = clamped_val
        for a, j in enumerate(free):
            delta[j] = (
                min(upper[j], max(lower[j], theta[j] + sub_sol[a])) - theta[j]
            )

    for c, dc in fixed.items():
        delta[c] = dc

    return [
        min(upper[j], max(lower[j], theta[j] + delta[j])) - theta[j]
        for j in range(p)
    ]


def _huber_map_cost(
    theta: Sequence[float],
    residuals: Sequence[float],
    prior_mean: Sequence[float],
    prior_prec_diag: Sequence[float],
    huber_delta: float,
) -> float:
    """Evaluates Huber data loss plus diagonal Gaussian MAP prior penalty."""
    total = 0.0
    for rv in residuals:
        abs_rv = abs(rv)
        if abs_rv <= huber_delta:
            total += 0.5 * rv * rv
        else:
            total += huber_delta * (abs_rv - 0.5 * huber_delta)
    for j, th_j in enumerate(theta):
        diff = th_j - float(prior_mean[j])
        total += 0.5 * float(prior_prec_diag[j]) * diff * diff
    return total


def solve_b2_box_huber_lm(
    residual_fn: Callable[[Sequence[float]], Sequence[float]],
    theta0: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    prior_mean: Sequence[float],
    prior_prec_diag: Sequence[float],
    huber_delta: float = 2.5,
    max_iter: int = 18,
    tol: float = 1e-4,
) -> SolverResult:
    """Solves Box-Projected Huber Levenberg-Marquardt MAP optimization.

    Args:
        residual_fn: Callable mapping parameter vector `theta` to standardized
            residual sequence `r(theta)`.
        theta0: Initial parameter vector (`len == p`).
        lower: Lower physical box bounds (`len == p`).
        upper: Upper physical box bounds (`len == p`).
        prior_mean: Gaussian prior mean vector (`len == p`).
        prior_prec_diag: Diagonal prior precisions `1 / sigma_prior^2` (`len == p`).
        huber_delta: Huber loss transition threshold in standardized units.
        max_iter: Maximum Levenberg-Marquardt outer iterations.
        tol: Relative box-span step convergence tolerance.

    Returns:
        `SolverResult` containing the MAP estimate, Fisher diagonal, posterior
        standard deviations, correlation matrix, and collinearity diagnostics.
    """
    p = len(theta0)
    theta = [
        min(float(upper[j]), max(float(lower[j]), float(theta0[j])))
        for j in range(p)
    ]
    spans = [max(1e-6, float(upper[j]) - float(lower[j])) for j in range(p)]

    r = [float(v) for v in residual_fn(theta)]
    n_fev = 1
    m = len(r)
    cost = _huber_map_cost(theta, r, prior_mean, prior_prec_diag, huber_delta)

    lam = 1e-2
    converged = False
    msg = "Maximum iterations reached"
    n_iter = 0
    jt_w_j = [[0.0] * p for _ in range(p)]

    for it in range(1, max_iter + 1):
        n_iter = it
        weights = [
            1.0 if abs(rv) <= huber_delta else (huber_delta / abs(rv))
            for rv in r
        ]

        # Forward finite-difference Jacobian J (m x p) with backward fallback
        jac_cols: list[list[float]] = []
        for j in range(p):
            hj = max(1e-6, 1e-4 * spans[j])
            ub_j = float(upper[j])
            lb_j = float(lower[j])
            if theta[j] + hj <= ub_j:
                tj = theta[j] + hj
            else:
                tj = max(lb_j, theta[j] - hj)
            denom = tj - theta[j]
            if abs(denom) <= 1e-12:
                jac_cols.append([0.0] * m)
                continue
            th_p = list(theta)
            th_p[j] = tj
            rp = residual_fn(th_p)
            n_fev += 1
            inv_denom = 1.0 / denom
            jac_cols.append(
                [(float(rp[i]) - r[i]) * inv_denom for i in range(m)]
            )

        # Normal matrix J^T W J and MAP gradient
        jt_w_j = [[0.0] * p for _ in range(p)]
        grad = [0.0] * p
        for j in range(p):
            cj = jac_cols[j]
            g_data = 0.0
            for i in range(m):
                wi_cj = weights[i] * cj[i]
                g_data += wi_cj * r[i]
            grad[j] = g_data + float(prior_prec_diag[j]) * (
                theta[j] - float(prior_mean[j])
            )
            for k in range(j, p):
                ck = jac_cols[k]
                s_jk = 0.0
                for i in range(m):
                    s_jk += weights[i] * cj[i] * ck[i]
                jt_w_j[j][k] = s_jk
                jt_w_j[k][j] = s_jk

        # Check projected KKT gradient norm
        proj_grad = 0.0
        for j in range(p):
            if theta[j] <= float(lower[j]) + 1e-11 and grad[j] >= 0.0:
                continue
            if theta[j] >= float(upper[j]) - 1e-11 and grad[j] <= 0.0:
                continue
            proj_grad = max(proj_grad, abs(grad[j]) * spans[j])
        if proj_grad < 1e-6:
            converged = True
            msg = "Converged (KKT projected gradient norm)"
            break

        scale_diag = [
            max(
                jt_w_j[j][j] + float(prior_prec_diag[j]),
                1.0 / (spans[j] * spans[j]),
            )
            for j in range(p)
        ]

        accepted_step = False
        max_rel_step = 0.0
        for _ in range(7):
            h_damped = [
                [
                    jt_w_j[j][k]
                    + (
                        float(prior_prec_diag[j]) + lam * scale_diag[j]
                        if j == k
                        else 0.0
                    )
                    for k in range(p)
                ]
                for j in range(p)
            ]
            delta = _solve_box_kkt_step(h_damped, grad, theta, lower, upper)
            max_rel_step = max(
                (abs(delta[j]) / spans[j] for j in range(p)), default=0.0
            )
            if max_rel_step < 1e-12:
                accepted_step = True
                break

            th_trial = [
                min(float(upper[j]), max(float(lower[j]), theta[j] + delta[j]))
                for j in range(p)
            ]
            r_trial = [float(v) for v in residual_fn(th_trial)]
            n_fev += 1
            c_trial = _huber_map_cost(
                th_trial, r_trial, prior_mean, prior_prec_diag, huber_delta
            )

            if c_trial <= cost - 1e-12 or (
                max_rel_step < tol and c_trial <= cost + 1e-9
            ):
                rel_cost_drop = (cost - c_trial) / max(1.0, cost)
                theta = th_trial
                r = r_trial
                cost = c_trial
                lam = max(1e-7, lam * 0.3)
                accepted_step = True
                if max_rel_step < tol or (
                    0.0 <= rel_cost_drop < 1e-6 and max_rel_step < 5.0 * tol
                ):
                    converged = True
                    msg = "Converged (step & cost tolerance)"
                break
            lam = min(1e8, lam * 4.0)

        if converged:
            break
        if not accepted_step:
            if max_rel_step < tol:
                converged = True
                msg = "Converged at trust-region boundary"
            else:
                msg = "Stalled on damping increase"
            break

    # Posterior Fisher covariance & correlation matrix at final theta
    if m > p:
        abs_r = sorted(abs(rv) for rv in r)
        mid = m // 2
        med_abs = (
            abs_r[mid]
            if (m % 2 == 1)
            else 0.5 * (abs_r[mid - 1] + abs_r[mid])
        )
        sigma_mad = min(4.0, max(0.5, 1.4826 * med_abs))
    else:
        sigma_mad = 1.0
    inv_var = 1.0 / (sigma_mad * sigma_mad)

    fisher_diag = [jt_w_j[j][j] * inv_var for j in range(p)]
    post_prec = [
        [
            jt_w_j[j][k] * inv_var
            + (max(1e-12, float(prior_prec_diag[j])) if j == k else 0.0)
            for k in range(p)
        ]
        for j in range(p)
    ]
    post_cov = _invert_spd_matrix(post_prec)
    post_std = [math.sqrt(max(1e-16, post_cov[j][j])) for j in range(p)]

    corr_mat = [[0.0] * p for _ in range(p)]
    collinear: list[tuple[int, int, float]] = []
    for j in range(p):
        corr_mat[j][j] = 1.0
        for k in range(j + 1, p):
            rho = post_cov[j][k] / (post_std[j] * post_std[k])
            rho = min(1.0, max(-1.0, rho))
            corr_mat[j][k] = rho
            corr_mat[k][j] = rho
            if abs(rho) > 0.90:
                collinear.append((j, k, rho))

    return SolverResult(
        theta=theta,
        success=converged or n_iter >= 1,
        n_iter=n_iter,
        n_fev=n_fev,
        cost=cost,
        posterior_std=post_std,
        posterior_cov=post_cov,
        fisher_diag=fisher_diag,
        correlation_matrix=corr_mat,
        collinearity_pairs=collinear,
        message=msg,
    )


# ============================================================================
# 2. Precision-Weighted Posterior Memory & Fisher Information Gate
# ============================================================================
def compute_aged_prior(
    old_memory: contracts.ParameterPosteriorMemory | None,
    bounds: Mapping[str, contracts.ParameterBounds],
    gamma: float = MEMORY_WEEKLY_GAMMA,
) -> tuple[dict[str, float], dict[str, float]]:
    """Ages previous seasonal posterior memory 1 week toward Day-1 priors.

    Args:
        old_memory: Previously persisted seasonal posterior memory, or `None`.
        bounds: Parameter box bounds and Day-1 Gaussian priors per parameter.
        gamma: Weekly precision retention factor (`2**(-1/52)` for 52w half-life).

    Returns:
        Tuple `(prior_mean, prior_precision)` keyed by parameter name.
    """
    prior_mean, prior_prec, _ = preprocessing.age_precision_memory(
        old_memory, bounds, gamma
    )
    return prior_mean, prior_prec


def update_posterior_memory(
    old_memory: contracts.ParameterPosteriorMemory | None,
    bounds: Mapping[str, contracts.ParameterBounds],
    fitted_mean: Mapping[str, float],
    fisher_diag: Mapping[str, float],
) -> tuple[
    dict[str, float],
    dict[str, float],
    contracts.ParameterPosteriorMemory,
]:
    """Updates seasonal memory with 52-week aging and Fisher information gate.

    Unobserved seasonal parameters (`fisher_diag[j] < 0.05 * tau_day1`, e.g.,
    `a_sol_m2` during dark winter weeks) retain their learned
    `old_memory.mean[j]` instead of drifting back toward the Day-1 prior.

    Args:
        old_memory: Previous posterior memory or `None`.
        bounds: Parameter bounds and Day-1 Gaussian priors.
        fitted_mean: Newly estimated MAP parameter dictionary.
        fisher_diag: Data-only Fisher information diagonal per parameter.

    Returns:
        Tuple `(new_mean, new_precision, updated_memory)`.
    """
    aged_mean, aged_prec = compute_aged_prior(
        old_memory, bounds, MEMORY_WEEKLY_GAMMA
    )
    new_mean: dict[str, float] = {}
    new_prec: dict[str, float] = {}

    for name, b in bounds.items():
        tau_0 = 1.0 / max(1e-12, b.prior_std * b.prior_std)
        info_j = max(0.0, float(fisher_diag.get(name, 0.0)))

        if info_j < FISHER_GATE_REL_THRESHOLD * tau_0:
            # Information gate active: preserve learned seasonal posterior mean
            if old_memory is not None and name in old_memory.mean:
                new_mean[name] = min(
                    b.upper, max(b.lower, float(old_memory.mean[name]))
                )
                new_prec[name] = aged_prec[name]
            else:
                new_mean[name] = b.prior_mean
                new_prec[name] = tau_0
        else:
            val = float(fitted_mean.get(name, aged_mean[name]))
            new_mean[name] = min(b.upper, max(b.lower, val))
            # Envelope precision so repeated manual fits never explode
            new_prec[name] = min(
                100.0 * tau_0, max(aged_prec[name], tau_0 + info_j)
            )

    mem = contracts.ParameterPosteriorMemory(
        mean=dict(new_mean), precision=dict(new_prec)
    )
    return new_mean, new_prec, mem


# ============================================================================
# 3. Clean Regime Window Extraction & 365-Day `SeasonalEpisodeBank`
# ============================================================================
@dataclasses.dataclass(slots=True)
class TrainingWindow:
    """Clean excitation window `[k0, k0 + length_steps]` in a history slice.

    Attributes:
        regime: Regime key (`"night"`, `"heat"`, `"solar"`, or `"ac"`).
        k0: Start step index in `PreprocessedHistory`.
        length_steps: Number of 5-minute rollout steps in the window.
        quality_score: Regime-specific signal-to-noise quality metric.
    """

    regime: str
    k0: int
    length_steps: int
    quality_score: float


@dataclasses.dataclass(slots=True)
class StoredEpisode:
    """Curated `24h` spin-up + `6h` window seasonal episode in `SeasonalEpisodeBank`.

    Attributes:
        zone: Zone identifier (`"wk"` or `"kt"`).
        regime: Regime key (`"night"`, `"heat"`, `"solar"`, or `"ac"`).
        start_ts_utc: Rollout start timestamp `ts_utc[k0]` [s].
        end_ts_utc: Rollout end timestamp [s].
        spinup_steps: Preceding Kalman spin-up steps stored in `columns`.
        window_steps: Open-loop evaluation steps stored after `spinup_steps`.
        quality_score: Excitation quality score used for top-K ranking.
        columns: Sliced RingBuffer column dictionary covering spin-up + window.
    """

    zone: contracts.ZoneId
    regime: str
    start_ts_utc: int
    end_ts_utc: int
    spinup_steps: int
    window_steps: int
    quality_score: float
    columns: dict[str, list[float | int]]
    clean_verified: bool = False


def _build_step_hard_mask(
    hist: preprocessing.PreprocessedHistory,
    zone: contracts.ZoneId,
) -> list[bool]:
    """Returns per-step exclusion mask (`True` = dropout/cooling/defrost)."""
    n = hist.n_rows
    valid_flag = const.FLAG_VALID_WK if zone == "wk" else const.FLAG_VALID_KT
    open_win_flag = (
        const.FLAG_OPEN_WIN_WK if zone == "wk" else const.FLAG_OPEN_WIN_KT
    )
    t_meas = hist.t_air_wk_c if zone == "wk" else hist.t_air_kt_c
    i_sol = hist.i_sol_wk_wm2 if zone == "wk" else hist.i_sol_kt_wm2

    masked = [False] * n
    defrost_mask = const.FLAG_DEFROST_WP | const.FLAG_DEFROST_AC
    defrost_padded = [False] * n
    for k in range(n):
        if hist.flags[k] & defrost_mask:
            lo = max(0, k - DEFROST_PAD_STEPS)
            hi = min(n, k + DEFROST_PAD_STEPS + 1)
            for m in range(lo, hi):
                defrost_padded[m] = True

    is_kt = zone == "kt"
    p_ac_arr = hist.p_el_ac_kt_w
    for k in range(n):
        fl = hist.flags[k]
        if (
            not (fl & valid_flag)
            or (fl & const.FLAG_GAP_BRIDGED)
            or hist.dt_s[k] > 600.0
            or hist.cooling_or_tail[k]
            or (fl & open_win_flag)
            or defrost_padded[k]
        ):
            masked[k] = True
            continue
        if k >= 3:
            slope_k_h = (t_meas[k] - t_meas[k - 3]) * 4.0
            if slope_k_h < -0.8:
                lo_in = max(0, k - 6)
                recent_high = False
                for m in range(lo_in, k + 1):
                    if i_sol[m] > 250.0 or (is_kt and p_ac_arr[m] > 80.0):
                        recent_high = True
                        break
                if slope_k_h < (-2.5 if recent_high else -0.8):
                    masked[k] = True

    return masked


def extract_clean_regime_windows(
    hist: preprocessing.PreprocessedHistory,
    zone: contracts.ZoneId,
    d_init: Sequence[float],
    min_k0: int = 144,
    max_per_regime: int = 24,
) -> list[TrainingWindow]:
    """Extracts clean high-excitation windows for `night`, `heat`, `solar`, `ac`.

    Args:
        hist: Preprocessed telemetry slice.
        zone: Zone identifier (`"wk"` or `"kt"`).
        d_init: Reconstructed disturbance sequence `d_est_w` [W] used to guard
            against starting rollouts during active unmodeled disturbances.
        min_k0: Minimum Kalman spin-up index before the first candidate `k0`.
        max_per_regime: Maximum non-overlapping windows retained per regime.

    Returns:
        List of non-overlapping `TrainingWindow` instances across all regimes.
    """
    n = hist.n_rows
    if n <= min_k0 + 24:
        return []

    masked = _build_step_hard_mask(hist, zone)
    t_meas = hist.t_air_wk_c if zone == "wk" else hist.t_air_kt_c
    i_sol = hist.i_sol_wk_wm2 if zone == "wk" else hist.i_sol_kt_wm2
    q_fh = hist.q_fh_wk_hyd_w if zone == "wk" else hist.q_fh_kt_hyd_w

    is_lts = (
        hist.is_lts_row if len(hist.is_lts_row) == n else [False] * n
    )
    missing_prim = (
        hist.missing_prim_th if len(hist.missing_prim_th) == n else [False] * n
    )
    lts_or_missing = [is_lts[m] or missing_prim[m] for m in range(n)]
    has_p_wp_raw = len(hist.p_wp_raw_el_w) == n
    active_off_ok = [
        hist.f_comp_hz[m] <= 0.5
        and (hist.p_wp_raw_el_w[m] <= 50.0 if has_p_wp_raw else True)
        and hist.p_prim_th_w[m] <= 50.0
        and hist.p_el_ac_kt_w[m] <= 40.0
        for m in range(n)
    ]

    candidates_by_reg: dict[str, list[TrainingWindow]] = {
        "night": [],
        "heat": [],
        "solar": [],
        "ac": [],
    }

    hrs_arr: list[int] = [0] * n
    mins_arr: list[int] = [0] * n
    cook_slot_wk: list[bool] = [False] * n
    is_wk = zone == "wk"
    aanwezig_mask = const.FLAG_AANWEZIG
    ts_arr = hist.ts_utc
    fl_arr = hist.flags
    last_hr_key = -1
    cached_hr_base = 0
    for m in range(n):
        ts_m = ts_arr[m]
        hr_key = ts_m // 3600
        if hr_key != last_hr_key:
            cached_hr_base, _ = preprocessing.local_hour_minute(hr_key * 3600)
            last_hr_key = hr_key
        hr_m = cached_hr_base
        mn_m = (ts_m % 3600) // 60
        mins_m = hr_m * 60 + mn_m
        hrs_arr[m] = hr_m
        mins_arr[m] = mins_m
        if is_wk and (fl_arr[m] & aanwezig_mask) and (1050 <= mins_m <= 1230):
            cook_slot_wk[m] = True

    baseline_cache: dict[int, float] = {}

    def _local_d_baseline(center_idx: int) -> float:
        bucket = center_idx // 6
        cached = baseline_cache.get(bucket)
        if cached is not None:
            return cached
        center_aligned = bucket * 6
        lo_b = max(min_k0, center_aligned - 144)
        hi_b = min(n, center_aligned + 144)
        vals = sorted(
            float(d_init[m])
            for m in range(lo_b, hi_b, 6)
            if not masked[m]
        )
        res = (
            0.0
            if not vals
            else min(220.0, max(-220.0, vals[len(vals) // 2]))
        )
        baseline_cache[bucket] = res
        return res

    def _is_calm_pre_excitation(pre_idx: int) -> bool:
        d_val = float(d_init[pre_idx])
        if abs(d_val) > 360.0:
            return False
        return abs(d_val - _local_d_baseline(pre_idx)) <= 140.0

    # 1. Passive cold night windows (`00:30–06:30`, Q_fh == 0, I_sol == 0)
    for k0 in range(min_k0, n - 30, 6):
        if masked[k0] or not (30 <= mins_arr[k0] <= 240):
            continue
        if not _is_calm_pre_excitation(k0):
            continue

        pre_off_ok = all(
            active_off_ok[m] for m in range(max(0, k0 - 24), k0 + 1)
        )
        touches_lts = lts_or_missing[k0]
        if touches_lts and not pre_off_ok:
            continue
        all_off = pre_off_ok

        lw = 0
        max_lw = min(evaluation.HORIZON_6H_STEPS, n - 1 - k0)
        for s in range(1, max_lw + 1):
            idx = k0 + s
            if mins_arr[idx] > 390 and hrs_arr[idx] < 12:
                break
            if masked[idx] or q_fh[idx] > 1.0 or i_sol[idx] > 2.0:
                break
            if not is_wk and (
                hist.p_el_ac_kt_w[idx] > 15.0
                or hist.p_desk_kt_w[idx] > 25.0
                or hist.pir_occ_kt[idx] >= 0.5
            ):
                break
            next_touches = touches_lts or lts_or_missing[idx]
            next_off = all_off and active_off_ok[idx]
            if next_touches and not next_off:
                break
            touches_lts = next_touches
            all_off = next_off
            lw = s

        if lw >= 30:
            mean_dt = sum(
                t_meas[k0 + s] - hist.t_out_c[k0 + s] for s in range(1, lw + 1)
            ) / float(lw)
            if mean_dt >= 5.0:
                score = mean_dt * math.sqrt(float(lw) / 72.0)
                candidates_by_reg["night"].append(
                    TrainingWindow("night", k0, lw, score)
                )

    # 2. Active floor heating + coast windows (up to 6.0h)
    heat_strict: list[TrainingWindow] = []
    heat_relaxed: list[TrainingWindow] = []
    strict_wh = 3800.0 if is_wk else 950.0
    relaxed_wh = 2000.0 if is_wk else 550.0

    for k0 in range(min_k0, n - 36, 6):
        if masked[k0] or is_lts[k0] or missing_prim[k0] or cook_slot_wk[k0]:
            continue
        k_pre_h = k0
        while k_pre_h > max(min_k0, k0 - 12) and q_fh[k_pre_h] > 50.0:
            k_pre_h -= 1
        if not _is_calm_pre_excitation(k_pre_h):
            continue

        lw = 0
        max_lw = min(evaluation.HORIZON_6H_STEPS, n - 1 - k0)
        for s in range(1, max_lw + 1):
            idx = k0 + s
            if (
                masked[idx]
                or is_lts[idx]
                or missing_prim[idx]
                or cook_slot_wk[idx]
            ):
                break
            if not is_wk and hist.p_el_ac_kt_w[idx] > 25.0:
                break
            lw = s

        if lw < 36:
            continue
        active_steps = sum(1 for s in range(1, lw + 1) if q_fh[k0 + s] > 50.0)
        e_wh = sum(
            q_fh[k0 + s] * (hist.dt_s[k0 + s] / 3600.0)
            for s in range(1, lw + 1)
        )
        if active_steps >= 24 and e_wh >= strict_wh:
            heat_strict.append(TrainingWindow("heat", k0, lw, e_wh))
        elif active_steps >= 16 and e_wh >= relaxed_wh:
            heat_relaxed.append(TrainingWindow("heat", k0, lw, e_wh))

    candidates_by_reg["heat"] = (
        heat_strict if len(heat_strict) >= 2 else (heat_strict + heat_relaxed)
    )

    # 3. Sunny daytime windows (`08:00–12:00` start, zero pv_fallback)
    min_sol_wh = 550.0 if is_wk else 350.0
    for k0 in range(min_k0, n - 36, 6):
        if masked[k0] or not (480 <= mins_arr[k0] <= 720):
            continue
        k_pre_s = k0
        while k_pre_s > max(min_k0, k0 - 48) and i_sol[k_pre_s] > 25.0:
            k_pre_s -= 1
        if not _is_calm_pre_excitation(k_pre_s):
            continue

        pre_off_s = all(
            active_off_ok[m] for m in range(max(0, k0 - 24), k0 + 1)
        )
        touches_lts_s = lts_or_missing[k0]
        if touches_lts_s and not pre_off_s:
            continue
        all_off_s = pre_off_s

        lw = 0
        max_lw = min(evaluation.HORIZON_6H_STEPS, n - 1 - k0)
        for s in range(1, max_lw + 1):
            idx = k0 + s
            if (
                masked[idx]
                or hist.pv_fallback[idx]
                or bool(fl_arr[idx] & const.FLAG_PV_FALLBACK)
                or cook_slot_wk[idx]
            ):
                break
            next_touches_s = touches_lts_s or lts_or_missing[idx]
            next_off_s = all_off_s and active_off_ok[idx]
            if next_touches_s and not next_off_s:
                break
            touches_lts_s = next_touches_s
            all_off_s = next_off_s
            lw = s

        if lw < 36:
            continue
        sol_wh_m2 = sum(
            i_sol[k0 + s] * (hist.dt_s[k0 + s] / 3600.0)
            for s in range(1, lw + 1)
        )
        if sol_wh_m2 >= min_sol_wh:
            candidates_by_reg["solar"].append(
                TrainingWindow("solar", k0, lw, sol_wh_m2)
            )

    # 4. Split-AC heating windows on `kt` (1.5h..2.0h = 18..24 steps)
    if zone == "kt":
        for k0 in range(min_k0, n - 18, 2):
            if masked[k0] or is_lts[k0]:
                continue
            k_pre_ac = k0
            while (
                k_pre_ac > max(min_k0, k0 - 12)
                and hist.p_el_ac_kt_w[k_pre_ac] > 25.0
            ):
                k_pre_ac -= 1
            if not _is_calm_pre_excitation(k_pre_ac):
                continue
            lw = 0
            max_lw = min(24, n - 1 - k0)
            for s in range(1, max_lw + 1):
                idx = k0 + s
                if masked[idx] or is_lts[idx] or q_fh[idx] > 250.0:
                    break
                if hist.p_el_ac_kt_w[idx] > 25.0 and round(
                    hist.ac_mode_code[idx]
                ) != int(const.AC_MODE_HEAT):
                    break
                lw = s
            if lw < 18:
                continue
            high_steps = sum(
                1
                for s in range(1, lw + 1)
                if round(hist.ac_mode_code[k0 + s]) == int(const.AC_MODE_HEAT)
                and hist.p_el_ac_kt_w[k0 + s] > 120.0
            )
            if high_steps >= 3:
                ac_wh = sum(
                    hist.p_el_ac_kt_w[k0 + s] * (hist.dt_s[k0 + s] / 3600.0)
                    for s in range(1, lw + 1)
                )
                candidates_by_reg["ac"].append(
                    TrainingWindow("ac", k0, lw, ac_wh)
                )

    # Greedy non-overlapping top-K selection per regime
    selected: list[TrainingWindow] = []
    for reg_name, cands in candidates_by_reg.items():
        if not cands:
            continue
        cands_sorted = sorted(
            cands, key=lambda w: w.quality_score, reverse=True
        )
        picked: list[TrainingWindow] = []
        min_sep = 12 if reg_name == "ac" else 36
        for w in cands_sorted:
            if all(abs(w.k0 - p.k0) >= min_sep for p in picked):
                picked.append(w)
                if len(picked) >= max_per_regime:
                    break
        selected.extend(picked)

    return selected


class SeasonalEpisodeBank:
    """Stratified 365-day reservoir of top-8 cleanest episodes per regime.

    Retains high-excitation seasonal windows (sunny afternoons, cold winter
    nights, active floor heating runs, and split-AC boosts) across the full
    annual cycle even after aging out of the 90-day live RingBuffer.
    """

    def __init__(self, max_per_regime: int = EPISODE_MAX_PER_REGIME) -> None:
        """Initializes an empty `SeasonalEpisodeBank`.

        Args:
            max_per_regime: Maximum retained episodes per `(zone, regime)` slot.
        """
        self._max_per_regime: int = int(max_per_regime)
        self._buckets: dict[contracts.ZoneId, dict[str, list[StoredEpisode]]] = {
            "wk": {"night": [], "heat": [], "solar": []},
            "kt": {"night": [], "heat": [], "solar": [], "ac": []},
        }
        self._cached_dict: dict[str, Any] | None = None

    def clone(self) -> SeasonalEpisodeBank:
        """Returns an isolated shallow copy of all regime episode buckets."""
        copy_bank = SeasonalEpisodeBank(max_per_regime=self._max_per_regime)
        for zone in ("wk", "kt"):
            z_id: contracts.ZoneId = "wk" if zone == "wk" else "kt"
            copy_bank._buckets[z_id] = {
                reg: list(ep_list)
                for reg, ep_list in self._buckets[z_id].items()
            }
        return copy_bank

    def clear(self) -> None:
        """Empties all episode buckets and invalidates the serialized cache."""
        for reg_map in self._buckets.values():
            for ep_list in reg_map.values():
                ep_list.clear()
        self._cached_dict = None

    @staticmethod
    def _is_episode_clean_for_regime(ep: StoredEpisode) -> bool:
        """Filters out active-heating episodes contaminated by LTS/missing prim."""
        if ep.clean_verified or ep.regime not in ("heat", "ac", "airco_heat"):
            return True
        is_lts, missing_prim = preprocessing.detect_lts_and_missing_prim_rows(
            ep.columns
        )
        if any(is_lts):
            return False
        if ep.regime == "heat" and any(missing_prim):
            return False
        ep.clean_verified = True
        return True

    def get_episodes(
        self,
        zone: contracts.ZoneId,
        regime: str | None = None,
    ) -> list[StoredEpisode]:
        """Returns clean stored episodes for `zone` (and optional `regime`)."""
        z_map = self._buckets[zone]
        if regime is not None:
            reg_key = "ac" if regime == "airco_heat" else regime
            return [
                ep
                for ep in z_map.get(reg_key, ())
                if self._is_episode_clean_for_regime(ep)
            ]
        out: list[StoredEpisode] = []
        for ep_list in z_map.values():
            out.extend(
                ep for ep in ep_list if self._is_episode_clean_for_regime(ep)
            )
        return out

    def update_from_columns(
        self,
        columns: Mapping[str, Sequence[float | int]],
        flow_rates_l_min: Mapping[str, float] | None = None,
        k_hx_by_group: Mapping[str, float] | None = None,
        lat_deg: float = priors.LATITUDE_DEG,
        lon_deg: float = priors.LONGITUDE_DEG,
        *,
        precomputed_hist: preprocessing.PreprocessedHistory | None = None,
        buffer_offsets: contracts.BufferSensorOffsets = (
            priors.DAY1_BUFFER_OFFSETS
        ),
    ) -> int:
        """Scans `columns` and retains top-K cleanest episodes over 365 days.

        Args:
            columns: RingBuffer training-split column dictionary.
            flow_rates_l_min: Optional manifold rotameter flow rates [L/min].
            k_hx_by_group: Optional `K_hx` [W/K] per hydraulic group.
            lat_deg: Observer latitude in degrees North.
            lon_deg: Observer longitude in degrees East.
            precomputed_hist: Optional precomputed `PreprocessedHistory`.
            buffer_offsets: Active 1-Wire buffer tank sensor offsets.

        Returns:
            Number of candidate episodes merged into the reservoir buckets.
        """
        ts_seq = columns.get("ts_utc", ())
        n = len(ts_seq)
        if n < 180:
            return 0

        latest_ts = int(ts_seq[-1])
        cutoff_ts = latest_ts - EPISODE_RETENTION_S

        if (
            precomputed_hist is not None
            and precomputed_hist.n_rows == n
            and _hist_matches_buffer_offsets(precomputed_hist, buffer_offsets)
        ):
            hist = precomputed_hist
        else:
            hist = preprocessing.preprocess_history_columns(
                columns,
                flow_rates_l_min=flow_rates_l_min,
                k_hx_by_group=k_hx_by_group,
                lat_deg=lat_deg,
                lon_deg=lon_deg,
                buffer_offsets=buffer_offsets,
            )
        min_spinup = (
            EPISODE_SPINUP_STEPS
            if n >= EPISODE_SPINUP_STEPS + 72
            else max(96, n - 96)
        )

        added_or_kept = 0
        for zone in ("wk", "kt"):
            z_id: contracts.ZoneId = "wk" if zone == "wk" else "kt"
            day1 = (
                priors.DAY1_PRIORS_WK
                if z_id == "wk"
                else priors.DAY1_PRIORS_KT
            )
            states_0 = kalman_tape.reconstruct_states_tuples(hist, z_id, day1)
            d_init = [st[2] for st in states_0]

            wins = extract_clean_regime_windows(
                hist,
                z_id,
                d_init,
                min_k0=min_spinup,
                max_per_regime=self._max_per_regime,
            )
            by_reg = self._buckets[z_id]

            # Prune expired (>365d) or legacy LTS-contaminated episodes
            for reg_key in list(by_reg.keys()):
                old_list = by_reg[reg_key]
                pruned = [
                    ep
                    for ep in old_list
                    if ep.start_ts_utc >= cutoff_ts
                    and self._is_episode_clean_for_regime(ep)
                ]
                if len(pruned) != len(old_list):
                    self._cached_dict = None
                by_reg[reg_key] = pruned

            has_hist_lts = len(hist.is_lts_row) == n
            has_hist_miss = len(hist.missing_prim_th) == n

            for w in wins:
                if w.regime not in by_reg:
                    continue
                spin = min(EPISODE_SPINUP_STEPS, w.k0)
                s_idx = w.k0 - spin
                if w.regime in ("heat", "ac"):
                    for m in range(w.k0 - 1, s_idx - 1, -1):
                        if hist.is_lts_row[m] or (
                            w.regime == "heat" and hist.missing_prim_th[m]
                        ):
                            s_idx = m + 1
                            spin = w.k0 - s_idx
                            break
                    if spin < 96:
                        continue
                w_len = (
                    min(EPISODE_WINDOW_STEPS, n - 1 - w.k0)
                    if w.regime != "ac"
                    else w.length_steps
                )
                e_idx = min(n, w.k0 + w_len + 1)
                if w.regime in ("heat", "ac") and has_hist_lts and has_hist_miss:
                    if any(hist.is_lts_row[s_idx:e_idx]) or (
                        w.regime == "heat"
                        and any(hist.missing_prim_th[s_idx:e_idx])
                    ):
                        continue
                ep_cols = preprocessing.slice_columns(columns, s_idx, e_idx)
                ep = StoredEpisode(
                    zone=z_id,
                    regime=w.regime,
                    start_ts_utc=int(hist.ts_utc[w.k0]),
                    end_ts_utc=int(hist.ts_utc[e_idx - 1]),
                    spinup_steps=spin,
                    window_steps=e_idx - 1 - w.k0,
                    quality_score=w.quality_score,
                    columns=ep_cols,
                    clean_verified=(has_hist_lts and has_hist_miss),
                )
                if not (has_hist_lts and has_hist_miss) and not (
                    self._is_episode_clean_for_regime(ep)
                ):
                    continue
                merged = list(by_reg[w.regime]) + [ep]
                merged.sort(key=lambda item: item.quality_score, reverse=True)

                deduped: list[StoredEpisode] = []
                for cand in merged:
                    if all(
                        abs(cand.start_ts_utc - ex.start_ts_utc) >= 43200
                        for ex in deduped
                    ):
                        deduped.append(cand)
                        if len(deduped) >= self._max_per_regime:
                            break
                by_reg[w.regime] = deduped
                self._cached_dict = None
                added_or_kept += 1

        return added_or_kept

    @staticmethod
    def _serialize_episode_columns(
        columns: Mapping[str, Sequence[float | int | None]],
    ) -> dict[str, list[float | int | None]]:
        """Sanitizes `NaN`/`Inf` floats to `None` for strict `orjson` Store."""
        out: dict[str, list[float | int | None]] = {}
        pos_inf = math.inf
        neg_inf = -math.inf
        for col_name, seq in columns.items():
            if col_name in const.RINGBUFFER_INT_COLS:
                out[col_name] = [0 if v is None else int(v) for v in seq]
            else:
                out[col_name] = [
                    None
                    if (
                        v is None
                        or v != v
                        or v == pos_inf
                        or v == neg_inf
                    )
                    else float(v)
                    for v in seq
                ]
        return out

    @staticmethod
    def _deserialize_episode_columns(
        raw_columns: Mapping[str, Any],
    ) -> dict[str, list[float | int]]:
        """Restores `None` float sentinels back to `math.nan` for replay."""
        out: dict[str, list[float | int]] = {}
        nan_val = math.nan
        pos_inf = math.inf
        neg_inf = -math.inf
        for col_name, seq in raw_columns.items():
            if not isinstance(seq, Sequence):
                continue
            if col_name in const.RINGBUFFER_INT_COLS:
                out[col_name] = [0 if v is None else int(v) for v in seq]
            else:
                out[col_name] = [
                    nan_val
                    if (
                        v is None
                        or v != v
                        or v == pos_inf
                        or v == neg_inf
                    )
                    else float(v)
                    for v in seq
                ]
        return out

    def to_dict(self) -> dict[str, Any]:
        """Serializes episode bank for `betonbatterij.episode_bank` HA Store."""
        if self._cached_dict is None:
            self._cached_dict = {
                "max_per_regime": self._max_per_regime,
                "buckets": {
                    zone: {
                        reg: [
                            {
                                "zone": ep.zone,
                                "regime": ep.regime,
                                "start_ts_utc": int(ep.start_ts_utc),
                                "end_ts_utc": int(ep.end_ts_utc),
                                "spinup_steps": int(ep.spinup_steps),
                                "window_steps": int(ep.window_steps),
                                "quality_score": (
                                    float(ep.quality_score)
                                    if math.isfinite(ep.quality_score)
                                    else 0.0
                                ),
                                "columns": self._serialize_episode_columns(
                                    ep.columns
                                ),
                            }
                            for ep in ep_list
                        ]
                        for reg, ep_list in reg_map.items()
                    }
                    for zone, reg_map in self._buckets.items()
                },
            }
        return self._cached_dict

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SeasonalEpisodeBank:
        """Restores `SeasonalEpisodeBank` from persisted dictionary."""
        bank = cls(
            max_per_regime=int(
                data.get("max_per_regime", EPISODE_MAX_PER_REGIME)
            )
        )
        raw_buckets = data.get("buckets", {})
        for zone in ("wk", "kt"):
            z_id: contracts.ZoneId = "wk" if zone == "wk" else "kt"
            z_raw = raw_buckets.get(z_id, {})
            if not isinstance(z_raw, Mapping):
                continue
            for reg, ep_list in z_raw.items():
                reg_bucket = "ac" if reg == "airco_heat" else reg
                if reg_bucket not in bank._buckets[z_id] or not isinstance(
                    ep_list, Sequence
                ):
                    continue
                restored: list[StoredEpisode] = []
                for item in ep_list:
                    if (
                        not isinstance(item, Mapping)
                        or not isinstance(item.get("columns"), Mapping)
                    ):
                        continue
                    raw_qs = item.get("quality_score", 0.0)
                    qs_val = (
                        float(raw_qs)
                        if raw_qs is not None and math.isfinite(float(raw_qs))
                        else 0.0
                    )
                    cand_ep = StoredEpisode(
                        zone=z_id,
                        regime=reg_bucket,
                        start_ts_utc=int(item["start_ts_utc"]),
                        end_ts_utc=int(item["end_ts_utc"]),
                        spinup_steps=int(item["spinup_steps"]),
                        window_steps=int(item["window_steps"]),
                        quality_score=qs_val,
                        columns=cls._deserialize_episode_columns(
                            item["columns"]
                        ),
                    )
                    if cls._is_episode_clean_for_regime(cand_ep):
                        restored.append(cand_ep)
                bank._buckets[z_id][reg_bucket] = (
                    list(bank._buckets[z_id][reg_bucket]) + restored
                )[: bank._max_per_regime]
        return bank


# ============================================================================
# 4. Robust Pipe Heat Transfer (`K_hx`) Calibration & Seasonal Gate
# ============================================================================
def _weighted_median_and_trimmed_mean(
    values: Sequence[float],
    weights: Sequence[float],
    trim_lo: float = 0.15,
    trim_hi: float = 0.85,
) -> float:
    """Returns `0.5 * weighted_median + 0.5 * trimmed_mean_[15%, 85%]`."""
    pairs = sorted(
        ((float(v), max(1e-9, float(w))) for v, w in zip(values, weights)),
        key=lambda x: x[0],
    )
    tot_w = sum(w for _, w in pairs)
    if tot_w <= 0.0:
        return float(values[0])

    # Weighted median
    half_w = 0.5 * tot_w
    cum = 0.0
    w_med = pairs[-1][0]
    for val, w in pairs:
        cum += w
        if cum >= half_w:
            w_med = val
            break

    # Weighted 15%..85% trimmed mean
    lo_w = trim_lo * tot_w
    hi_w = trim_hi * tot_w
    trim_num = 0.0
    trim_den = 0.0
    c0 = 0.0
    for val, w in pairs:
        c1 = c0 + w
        ov = min(c1, hi_w) - max(c0, lo_w)
        if ov > 0.0:
            trim_num += val * ov
            trim_den += ov
        c0 = c1

    w_trim = (trim_num / trim_den) if trim_den > 1e-12 else w_med
    return 0.5 * w_med + 0.5 * w_trim


def _finite_col_val(
    columns: Mapping[str, Sequence[float | int]],
    key: str,
    idx: int,
    default: float,
) -> float:
    """Reads a finite float from `columns[key][idx]`, falling back cleanly."""
    seq = columns.get(key)
    if seq is None or idx >= len(seq) or seq[idx] is None:
        return default
    fval = float(seq[idx])
    return fval if math.isfinite(fval) else default


def _extract_group_valves(
    hist: preprocessing.PreprocessedHistory,
    columns: Mapping[str, Sequence[float | int]],
    idx: int,
) -> dict[str, float]:
    """Returns effective valve openings `u_eff` across all 10 hydraulic groups."""
    return {
        "wk": hist.u_eff_wk[idx],
        "gang": hist.u_eff_wk[idx],
        "slaapkamer": min(
            1.0, max(0.0, _finite_col_val(columns, "u_eff_slaapkamer", idx, 0.0))
        ),
        "badkamer": min(
            1.0, max(0.0, _finite_col_val(columns, "u_eff_badkamer", idx, 0.0))
        ),
        "werkkamer_1e": min(
            1.0,
            max(0.0, _finite_col_val(columns, "u_eff_werkkamer_1e", idx, 0.0)),
        ),
        "kamer_1e_midden": min(
            1.0,
            max(
                0.0,
                _finite_col_val(columns, "u_eff_kamer_1e_midden", idx, 0.0),
            ),
        ),
        "kamer_1e_zuid": min(
            1.0,
            max(
                0.0, _finite_col_val(columns, "u_eff_kamer_1e_zuid", idx, 0.0)
            ),
        ),
        "kantoor": hist.u_eff_kt[idx],
        "badkamer_boven": min(
            1.0,
            max(0.0, _finite_col_val(columns, "u_eff_badkamer_boven", idx, 0.0)),
        ),
        "logeerkamer": min(
            1.0,
            max(0.0, _finite_col_val(columns, "u_eff_logeerkamer", idx, 0.0)),
        ),
    }


def _estimate_group_floor_temp(
    group_name: str,
    idx: int,
    tf_wk: Sequence[float],
    tf_kt: Sequence[float],
    columns: Mapping[str, Sequence[float | int]],
) -> float:
    """Returns slab temperature estimate [°C] for hydraulic group `group_name`."""
    if group_name == "wk":
        return tf_wk[idx]
    if group_name == "gang":
        return tf_wk[idx] - 0.5
    if group_name == "kantoor":
        return tf_kt[idx]
    col_name = f"t_air_{group_name}_c"
    t_room = _finite_col_val(columns, col_name, idx, 18.0)
    return t_room + 0.5


def calibrate_k_hx_robust(
    columns: Mapping[str, Sequence[float | int]],
    flow_rates_l_min: Mapping[str, float],
    current_k_hx: Mapping[str, float],
    lat_deg: float = priors.LATITUDE_DEG,
    lon_deg: float = priors.LONGITUDE_DEG,
    *,
    precomputed_hist: preprocessing.PreprocessedHistory | None = None,
    buffer_offsets: contracts.BufferSensorOffsets = priors.DAY1_BUFFER_OFFSETS,
) -> dict[str, float]:
    """Calibrates `K_hx` [W/K] on steady `18..25 Hz` runs (`>= 40 min`, `dT >= 2 K`).

    Args:
        columns: RingBuffer training-split column dictionary.
        flow_rates_l_min: Manifold rotameter flow rates [L/min] per group.
        current_k_hx: Active `K_hx` [W/K] dictionary per hydraulic group.
        lat_deg: Observer latitude in degrees North.
        lon_deg: Observer longitude in degrees East.
        precomputed_hist: Optional precomputed `PreprocessedHistory`.
        buffer_offsets: Active 1-Wire buffer tank sensor offsets.

    Returns:
        Updated dictionary mapping all 10 hydraulic groups to `K_hx` [W/K].
    """
    flows = dict(const.DEFAULT_FLOW_RATES_L_MIN)
    flows.update({k: float(v) for k, v in flow_rates_l_min.items()})

    out_khx: dict[str, float] = {
        g: hydraulics.scale_k_hx_eps_ntu(
            priors.DAY1_K_HX_BY_GROUP_W_K[g],
            const.DEFAULT_FLOW_RATES_L_MIN[g],
            flows[g],
        )
        for g in const.ALL_HYDRAULIC_GROUPS
    }
    for g, v in current_k_hx.items():
        out_khx[preprocessing.zone_to_hydraulic_group(g)] = float(v)

    ts_seq = columns.get("ts_utc", ())
    n = len(ts_seq)
    if n < 16:
        return out_khx

    if (
        precomputed_hist is not None
        and precomputed_hist.n_rows == n
        and abs(precomputed_hist.hyd_k_hx_wk - out_khx["wk"]) <= 1e-9
        and abs(precomputed_hist.hyd_k_hx_kt - out_khx["kantoor"]) <= 1e-9
        and _hist_matches_buffer_offsets(precomputed_hist, buffer_offsets)
    ):
        hist = precomputed_hist
    else:
        hist = preprocessing.preprocess_history_columns(
            columns,
            flow_rates_l_min=flows,
            k_hx_by_group=out_khx,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            buffer_offsets=buffer_offsets,
        )

    st_wk = kalman_tape.reconstruct_states_tuples(
        hist,
        "wk",
        dataclasses.replace(
            priors.DAY1_PRIORS_WK, k_hx_w_k=out_khx["wk"]
        ),
    )
    tf_wk = [s[1] for s in st_wk]

    st_kt = kalman_tape.reconstruct_states_tuples(
        hist,
        "kt",
        dataclasses.replace(
            priors.DAY1_PRIORS_KT, k_hx_w_k=out_khx["kantoor"]
        ),
    )
    tf_kt = [s[1] for s in st_kt]

    steady_indices: list[int] = []
    q_sec_cal_map: dict[int, float] = {}
    run_len = 0
    is_lts = hist.is_lts_row if len(hist.is_lts_row) == n else [False] * n
    missing_prim = (
        hist.missing_prim_th if len(hist.missing_prim_th) == n else [False] * n
    )
    top_off = float(buffer_offsets.buf_top_offset_c)
    bot_off = float(buffer_offsets.buf_bot_offset_c)

    for k in range(n):
        if is_lts[k] or missing_prim[k]:
            run_len = 0
            continue
        fl = hist.flags[k]
        room_active = (
            (fl & const.FLAG_DHW_TANK) == 0
            and (fl & const.FLAG_Z1_OFF_HARVEST) == 0
            and (fl & const.FLAG_DEFROST_WP) == 0
            and not hist.cooling_or_tail[k]
            and hist.f_comp_hz[k] >= 10.0
        )
        if room_active:
            run_len += 1
        else:
            run_len = 0
            continue

        if run_len < 8:
            continue
        if not (18.0 <= hist.f_comp_hz[k] <= 25.0):
            continue
        if hist.u_pump_sec[k] < 0.99:
            continue

        top_s_raw = _opt_col_temp(columns, "t_buf_top_start_c", k)
        top_e_raw = _opt_col_temp(columns, "t_buf_top_end_c", k)
        top_s = (
            top_s_raw + top_off
            if top_s_raw is not None
            else hist.t_buf_top_c[k]
        )
        top_e = (
            top_e_raw + top_off
            if top_e_raw is not None
            else hist.t_buf_top_c[k]
        )
        if abs(top_e - top_s) > 0.25:
            continue

        bot_s_raw = _opt_col_temp(columns, "t_buf_bot_start_c", k)
        bot_e_raw = _opt_col_temp(columns, "t_buf_bot_end_c", k)
        bot_s = (
            bot_s_raw + bot_off
            if bot_s_raw is not None
            else hist.t_buf_bot_cal_c[k]
        )
        bot_e = (
            bot_e_raw + bot_off
            if bot_e_raw is not None
            else hist.t_buf_bot_cal_c[k]
        )
        t_mean_s = 0.5 * (top_s + bot_s)
        t_mean_e = 0.5 * (top_e + bot_e)
        q_cal = hist.p_prim_th_w[k] - priors.C_BUF_J_K * (
            t_mean_e - t_mean_s
        ) / max(60.0, hist.dt_s[k])
        if not math.isfinite(q_cal) or q_cal <= 400.0:
            continue

        steady_indices.append(k)
        q_sec_cal_map[k] = q_cal

    if len(steady_indices) < 6:
        return out_khx

    def _scaled_day1_khx(group_name: str) -> float:
        return hydraulics.scale_k_hx_eps_ntu(
            priors.DAY1_K_HX_BY_GROUP_W_K[group_name],
            const.DEFAULT_FLOW_RATES_L_MIN[group_name],
            flows[group_name],
        )

    gang_to_wk_ratio = flows["gang"] / max(0.2, flows["wk"])
    scaled_day1_wk = _scaled_day1_khx("wk")

    # Pass 1: Calibrate `wk` (+ `gang`) preferring BG-solo steady 20 Hz steps
    wk_solo_vals: list[float] = []
    wk_solo_wts: list[float] = []
    wk_mix_vals: list[float] = []
    wk_mix_wts: list[float] = []

    for k in steady_indices:
        if hist.u_eff_wk[k] < 0.98:
            continue
        dt_wk = hist.t_sup_sec_c[k] - tf_wk[k]
        if dt_wk < 2.0:
            continue
        uv = _extract_group_valves(hist, columns, k)
        other_u = sum(
            uv[g]
            for g in const.ALL_HYDRAULIC_GROUPS
            if g not in ("wk", "gang")
        )
        t_fl_gang = _estimate_group_floor_temp(
            "gang", k, tf_wk, tf_kt, columns
        )
        dt_gang = max(0.5, hist.t_sup_sec_c[k] - t_fl_gang)
        q_cal = q_sec_cal_map[k]

        if other_u < 0.05:
            denom = (
                hist.u_pump_sec[k]
                * hist.u_eff_wk[k]
                * (dt_wk + gang_to_wk_ratio * dt_gang)
            )
            cand = q_cal / max(0.1, denom)
            if 0.35 * scaled_day1_wk <= cand <= 2.3 * scaled_day1_wk:
                wk_solo_vals.append(cand)
                wk_solo_wts.append(dt_wk * dt_wk)
        elif other_u <= 1.5:
            q_mod_tot = sum(
                hist.u_pump_sec[k]
                * uv[g]
                * out_khx[g]
                * max(
                    0.0,
                    hist.t_sup_sec_c[k]
                    - _estimate_group_floor_temp(g, k, tf_wk, tf_kt, columns),
                )
                for g in const.ALL_HYDRAULIC_GROUPS
            )
            if q_mod_tot > 200.0:
                cand = out_khx["wk"] * (q_cal / q_mod_tot)
                if 0.35 * scaled_day1_wk <= cand <= 2.3 * scaled_day1_wk:
                    wk_mix_vals.append(cand)
                    wk_mix_wts.append(dt_wk * dt_wk)

    def _clamp_khx(group_name: str, raw_val: float) -> float:
        scaled_day1_g = _scaled_day1_khx(group_name)
        cap_g = min(
            1.65 * scaled_day1_g,
            0.75 * flows[group_name] * priors.WATER_POWER_FACTOR_W_PER_LMIN_K,
        )
        return max(1.0, min(cap_g, max(0.4 * scaled_day1_g, raw_val)))

    if len(wk_solo_vals) >= 6:
        est_wk = _weighted_median_and_trimmed_mean(wk_solo_vals, wk_solo_wts)
        out_khx["wk"] = _clamp_khx("wk", est_wk)
        out_khx["gang"] = _clamp_khx("gang", out_khx["wk"] * gang_to_wk_ratio)
    elif len(wk_solo_vals) + len(wk_mix_vals) >= 6:
        est_wk = _weighted_median_and_trimmed_mean(
            wk_solo_vals + wk_mix_vals, wk_solo_wts + wk_mix_wts
        )
        out_khx["wk"] = _clamp_khx("wk", est_wk)
        out_khx["gang"] = _clamp_khx("gang", out_khx["wk"] * gang_to_wk_ratio)

    # Pass 2: Calibrate `kantoor` (`kt`) using updated BG `K_hx`
    kt_vals: list[float] = []
    kt_wts: list[float] = []
    scaled_day1_kt = _scaled_day1_khx("kantoor")

    for k in steady_indices:
        if hist.u_eff_kt[k] < 0.98:
            continue
        dt_kt = hist.t_sup_sec_c[k] - tf_kt[k]
        if dt_kt < 2.0:
            continue
        uv = _extract_group_valves(hist, columns, k)
        q_cal = q_sec_cal_map[k]
        q_others = sum(
            hist.u_pump_sec[k]
            * uv[g]
            * out_khx[g]
            * max(
                0.0,
                hist.t_sup_sec_c[k]
                - _estimate_group_floor_temp(g, k, tf_wk, tf_kt, columns),
            )
            for g in const.ALL_HYDRAULIC_GROUPS
            if g != "kantoor"
        )
        q_rem = q_cal - q_others
        denom_kt = hist.u_pump_sec[k] * hist.u_eff_kt[k] * dt_kt
        cand_sub = q_rem / max(0.1, denom_kt)

        if 0.35 * scaled_day1_kt <= cand_sub <= 2.3 * scaled_day1_kt:
            kt_vals.append(cand_sub)
            kt_wts.append(dt_kt * dt_kt)
        else:
            q_mod_tot = q_others + denom_kt * out_khx["kantoor"]
            if q_mod_tot > 200.0:
                cand_rat = out_khx["kantoor"] * (q_cal / q_mod_tot)
                if 0.35 * scaled_day1_kt <= cand_rat <= 2.3 * scaled_day1_kt:
                    kt_vals.append(cand_rat)
                    kt_wts.append(dt_kt * dt_kt)

    if len(kt_vals) >= 6:
        est_kt = _weighted_median_and_trimmed_mean(kt_vals, kt_wts)
        out_khx["kantoor"] = _clamp_khx("kantoor", est_kt)

    return out_khx


def _opt_col_temp(
    columns: Mapping[str, Sequence[float | int]],
    key: str,
    idx: int,
) -> float | None:
    """Reads an optional finite temperature from `columns[key][idx]`."""
    seq = columns.get(key)
    if seq is None or idx >= len(seq):
        return None
    fval = float(seq[idx])
    if not math.isfinite(fval) or fval < -100.0:
        return None
    return fval


def check_seasonal_calibration_gate(
    columns: Mapping[str, Sequence[float | int]],
    lat_deg: float = priors.LATITUDE_DEG,
    lon_deg: float = priors.LONGITUDE_DEG,
) -> tuple[bool, str]:
    """Evaluates autonomous Layer-1 seasonal & cooling gate (`mean_7d < 14 °C`).

    Args:
        columns: RingBuffer column dictionary.

    Returns:
        Tuple `(gate_open, reason_message)`.
    """
    ts_seq = columns.get("ts_utc", ())
    n = len(ts_seq)
    if n < const.STEPS_PER_DAY:
        return False, "Insufficient history (< 24h) for seasonal gate check"

    flags_seq = columns.get("flags")
    ac_mode_seq = columns.get("ac_mode_code")

    # 1. Check last 48h (`576` steps) for any floor or split-AC cooling
    start_48h = max(0, n - evaluation.SPINUP_48H_STEPS)
    for k in range(start_48h, n):
        fl = (
            int(flags_seq[k])
            if flags_seq is not None and k < len(flags_seq)
            else 0
        )
        if fl & (
            const.FLAG_FLOOR_COOL | const.FLAG_AC_COOL_KT | const.FLAG_SUMMER
        ):
            return (
                False,
                "Cooling or summer mode active within the last 48 hours",
            )
        if (
            ac_mode_seq is not None
            and k < len(ac_mode_seq)
            and ac_mode_seq[k] is not None
        ):
            ac_m = float(ac_mode_seq[k])
            if math.isfinite(ac_m) and round(ac_m) in (
                int(const.AC_MODE_COOL),
                int(const.AC_MODE_DRY),
            ):
                return (
                    False,
                    "Split-AC cooling/dry mode active within the last 48 hours",
                )

    # 2. Check 7-day rolling mean outdoor temperature < 14.0 °C
    start_7d = max(0, n - 7 * const.STEPS_PER_DAY)
    hm_ema: float | None = None
    t_sum = 0.0
    cnt = 0
    for k in range(start_7d, n):
        ts_k = int(ts_seq[k])
        t_dak = _opt_col_temp(columns, "t_out_dak_c", k)
        t_wea = _opt_col_temp(columns, "t_out_weather_c", k)
        t_hm = _opt_col_temp(columns, "t_out_hm_c", k)
        if t_dak is None and t_wea is None and t_hm is None:
            t_dak = _opt_col_temp(columns, "t_out_c", k)
        if t_hm is not None:
            hm_ema = (
                t_hm if hm_ema is None else (t_hm / 6.0) + (5.0 / 6.0) * hm_ema
            )
        t_out_k = solar_weather.select_outdoor_temp(
            ts_utc=ts_k,
            t_dak_c=t_dak,
            t_weather_c=t_wea,
            t_hm_ema_c=hm_ema,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
        )
        t_sum += t_out_k
        cnt += 1

    mean_t_out_7d = t_sum / max(1, cnt)
    if mean_t_out_7d >= 14.0:
        return (
            False,
            f"7-day mean outdoor temperature {mean_t_out_7d:.2f} °C "
            f">= 14.0 °C threshold",
        )

    return (
        True,
        f"Seasonal gate open (mean_t_out_7d={mean_t_out_7d:.2f} °C, "
        f"no cooling in 48h)",
    )


# ============================================================================
# 5. Joint Multi-Horizon Trajectory MAP Calibration + Out-of-Sample Gate
# ============================================================================
def _active_bounds_for_zone(
    zone: contracts.ZoneId,
) -> dict[str, contracts.ParameterBounds]:
    """Returns active parameter box bounds and priors for `zone`."""
    raw = (
        priors.PARAM_BOUNDS_WK
        if zone == "wk"
        else priors.PARAM_BOUNDS_KT
    )
    if priors.FIT_INTERFLOOR_CONDUCTANCE:
        return dict(raw)
    skip = "h_up_w_k" if zone == "wk" else "h_down_w_k"
    return {k: v for k, v in raw.items() if k != skip}


def _apply_theta_to_params(
    base: contracts.ZoneParameters,
    param_names: Sequence[str],
    theta_vals: Sequence[float],
    k_hx_val: float,
) -> contracts.ZoneParameters:
    """Applies optimization vector `theta_vals` and `k_hx_val` onto `base`."""
    updates: dict[str, float] = {
        name: float(val) for name, val in zip(param_names, theta_vals)
    }
    updates["k_hx_w_k"] = float(k_hx_val)
    return dataclasses.replace(base, **updates)


def _blend_zone_parameters(
    old_p: contracts.ZoneParameters,
    new_p: contracts.ZoneParameters,
    alpha_new: float,
    *,
    keep_new_khx: bool = False,
) -> contracts.ZoneParameters:
    """Linearly blends `old_p` and `new_p` with weight `alpha_new`.

    Args:
        old_p: Active `ZoneParameters` prior to calibration.
        new_p: Newly estimated MAP `ZoneParameters`.
        alpha_new: Weight on `new_p` in `[0.0, 1.0]`.
        keep_new_khx: When True, preserves `new_p.k_hx_w_k` directly because
            `k_hx_w_k` was already damped upstream.
    """
    a = min(1.0, max(0.0, alpha_new))
    b = 1.0 - a
    k_hx_blended = (
        new_p.k_hx_w_k
        if keep_new_khx
        else (b * old_p.k_hx_w_k + a * new_p.k_hx_w_k)
    )
    return contracts.ZoneParameters(
        c_air_wh_k=old_p.c_air_wh_k,
        c_floor_wh_k=b * old_p.c_floor_wh_k + a * new_p.c_floor_wh_k,
        r_fh_k_kw=b * old_p.r_fh_k_kw + a * new_p.r_fh_k_kw,
        h_loss_w_k=b * old_p.h_loss_w_k + a * new_p.h_loss_w_k,
        h_down_w_k=b * old_p.h_down_w_k + a * new_p.h_down_w_k,
        alpha_sol=old_p.alpha_sol,
        a_sol_m2=b * old_p.a_sol_m2 + a * new_p.a_sol_m2,
        tau_d_h=old_p.tau_d_h,
        eta_ac=b * old_p.eta_ac + a * new_p.eta_ac,
        k_hx_w_k=k_hx_blended,
        h_up_w_k=b * old_p.h_up_w_k + a * new_p.h_up_w_k,
        c_wind=b * old_p.c_wind + a * new_p.c_wind,
    )


def _sample_rollout_steps(regime: str, length_steps: int) -> list[int]:
    """Returns subsampled trajectory evaluation steps within a window."""
    stride = 2 if regime == "ac" else 6
    pts = list(range(stride, length_steps + 1, stride))
    if not pts or pts[-1] != length_steps:
        pts.append(length_steps)
    return pts


def calibrate_zone_a2(
    columns: Mapping[str, Sequence[float | int]],
    zone: contracts.ZoneId,
    current_params: contracts.ZoneParameters,
    mode: Literal["manual", "auto_weekly"],
    episode_bank: SeasonalEpisodeBank | None = None,
    memory: contracts.ParameterPosteriorMemory | None = None,
    flow_rates_l_min: Mapping[str, float] | None = None,
    k_hx_by_group: Mapping[str, float] | None = None,
    lat_deg: float = priors.LATITUDE_DEG,
    lon_deg: float = priors.LONGITUDE_DEG,
    *,
    old_k_hx_by_group: Mapping[str, float] | None = None,
    buffer_offsets: contracts.BufferSensorOffsets = priors.DAY1_BUFFER_OFFSETS,
    old_buffer_offsets: contracts.BufferSensorOffsets | None = None,
    precomputed_train_hist: preprocessing.PreprocessedHistory | None = None,
    precomputed_holdout_hist: preprocessing.PreprocessedHistory | None = None,
    precomputed_holdout_before_hist: (
        preprocessing.PreprocessedHistory | None
    ) = None,
    precomputed_holdout_prior_hist: (
        preprocessing.PreprocessedHistory | None
    ) = None,
    is_lts_row: Sequence[bool] | None = None,
    khx_already_damped: bool = False,
) -> tuple[contracts.CalibrationZoneOutcome, contracts.ParameterPosteriorMemory]:
    """Runs Joint Multi-Horizon Trajectory MAP + Out-of-Sample Holdout Gate.

    Args:
        columns: Full RingBuffer telemetry columns (up to 90 days).
        zone: Zone identifier (`"wk"` or `"kt"`).
        current_params: Active `ZoneParameters` prior to calibration.
        mode: `"manual"` (direct MAP on acceptance) or `"auto_weekly"`
            (`0.80 * old + 0.20 * map` damped update with seasonal gate).
        episode_bank: Optional 365-day `SeasonalEpisodeBank`.
        memory: Optional seasonal `ParameterPosteriorMemory`.
        flow_rates_l_min: Optional manifold rotameter flow rates [L/min].
        k_hx_by_group: Optional candidate `K_hx` [W/K] per hydraulic group.
        lat_deg: Observer latitude in degrees North.
        lon_deg: Observer longitude in degrees East.
        old_k_hx_by_group: Optional pre-calibration `K_hx` [W/K] per group for
            evaluating `holdout_before`.
        buffer_offsets: Candidate 1-Wire buffer tank sensor offsets.
        old_buffer_offsets: Pre-calibration 1-Wire buffer tank sensor offsets
            for evaluating `holdout_before`.
        precomputed_train_hist: Optional precomputed training history.
        precomputed_holdout_hist: Optional precomputed candidate holdout hist.
        precomputed_holdout_before_hist: Optional precomputed baseline holdout
            history for `holdout_before`.
        precomputed_holdout_prior_hist: Optional precomputed Day-1 holdout hist.
        is_lts_row: Optional precomputed LTS backfill mask.
        khx_already_damped: True when `k_hx_by_group` was already damped by
            `0.20` upstream (prevents double-damping in `auto_weekly` mode).

    Returns:
        Tuple `(outcome, updated_memory)`.
    """
    ts_all = columns.get("ts_utc", ())
    n_total = len(ts_all)
    ts_now = int(ts_all[-1]) if n_total > 0 else 0
    day1_prior = (
        priors.DAY1_PRIORS_WK
        if zone == "wk"
        else priors.DAY1_PRIORS_KT
    )
    eff_old_khx = (
        old_k_hx_by_group if old_k_hx_by_group is not None else k_hx_by_group
    )
    eff_old_buf_offsets = (
        old_buffer_offsets
        if old_buffer_offsets is not None
        else buffer_offsets
    )
    active_bounds = _active_bounds_for_zone(zone)
    param_names = list(active_bounds.keys())
    prior_mean_init, prior_prec_init = compute_aged_prior(memory, active_bounds)
    fallback_mem = (
        memory
        if memory is not None
        else contracts.ParameterPosteriorMemory(
            mean=dict(prior_mean_init), precision=dict(prior_prec_init)
        )
    )

    def _eval_fallback(
        eval_steps_arg: int = evaluation.EVAL_14D_STEPS,
        pre_eval: preprocessing.PreprocessedHistory | None = None,
        pre_prior: preprocessing.PreprocessedHistory | None = None,
    ) -> contracts.ZoneEvaluationReport:
        return evaluation.evaluate_14d_hindcast(
            columns,
            zone,
            current_params,
            day1_prior,
            flow_rates_l_min,
            eff_old_khx,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            buffer_offsets=eff_old_buf_offsets,
            precomputed_hist_eval=pre_eval,
            precomputed_hist_prior=pre_prior,
            eval_steps=eval_steps_arg,
        )

    if n_total < 4 * const.STEPS_PER_DAY:
        rep = _eval_fallback()
        return (
            contracts.CalibrationZoneOutcome(
                zone=zone,
                accepted=False,
                reason=(
                    "Insufficient telemetry history (< 4 days) for "
                    "train/holdout split"
                ),
                mode=mode,
                ts_utc=ts_now,
                params_old=current_params,
                params_map_raw=current_params,
                params_candidate=current_params,
                params_applied=current_params,
                posterior_std={
                    k: active_bounds[k].prior_std for k in param_names
                },
                correlation_matrix={
                    k: {m: (1.0 if k == m else 0.0) for m in param_names}
                    for k in param_names
                },
                collinearity_warnings=[],
                holdout_before=rep,
                holdout_after=rep,
            ),
            fallback_mem,
        )

    # Autonomous seasonal gate for scheduled Sunday 03:15 updates
    if mode == "auto_weekly":
        gate_ok, gate_reason = check_seasonal_calibration_gate(
            columns, lat_deg=lat_deg, lon_deg=lon_deg
        )
        if not gate_ok:
            rep = _eval_fallback()
            return (
                contracts.CalibrationZoneOutcome(
                    zone=zone,
                    accepted=False,
                    reason=f"Seasonal gate blocked: {gate_reason}",
                    mode=mode,
                    ts_utc=ts_now,
                    params_old=current_params,
                    params_map_raw=current_params,
                    params_candidate=current_params,
                    params_applied=current_params,
                    posterior_std={
                        k: active_bounds[k].prior_std for k in param_names
                    },
                    correlation_matrix={
                        k: {m: (1.0 if k == m else 0.0) for m in param_names}
                        for k in param_names
                    },
                    collinearity_warnings=[],
                    holdout_before=rep,
                    holdout_after=rep,
                ),
                fallback_mem,
            )

    # Shared raw-aware chronological Train / Holdout split
    train_end = preprocessing.compute_train_holdout_split(
        columns, is_lts_row=is_lts_row
    )
    holdout_steps = n_total - train_end
    holdout_slice_start = max(0, train_end - evaluation.SPINUP_48H_STEPS)
    n_hold = n_total - holdout_slice_start
    holdout_start_ts = int(ts_all[train_end])
    train_start_ts = int(ts_all[0])

    holdout_cols_cache: Mapping[str, Sequence[float | int]] | None = None

    def _get_holdout_cols() -> Mapping[str, Sequence[float | int]]:
        nonlocal holdout_cols_cache
        if holdout_cols_cache is None:
            holdout_cols_cache = preprocessing.slice_columns(
                columns, holdout_slice_start, n_total
            )
        return holdout_cols_cache

    cand_khx = current_params.k_hx_w_k
    if k_hx_by_group:
        key_khx = (
            "wk"
            if zone == "wk"
            else ("kantoor" if "kantoor" in k_hx_by_group else "kt")
        )
        if key_khx in k_hx_by_group:
            cand_khx = float(k_hx_by_group[key_khx])

    if (
        precomputed_train_hist is not None
        and precomputed_train_hist.n_rows == train_end
        and _hist_matches_buffer_offsets(precomputed_train_hist, buffer_offsets)
    ):
        hist_train = precomputed_train_hist
    else:
        train_cols = preprocessing.slice_columns(columns, 0, train_end)
        hist_train = preprocessing.preprocess_history_columns(
            train_cols,
            flow_rates_l_min=flow_rates_l_min,
            k_hx_by_group=k_hx_by_group,
            eta_ac_ref=current_params.eta_ac,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            buffer_offsets=buffer_offsets,
        )
    states_init = kalman_tape.reconstruct_states_tuples(
        hist_train, zone, current_params
    )
    d_init = [st[2] for st in states_init]

    min_spin_train = min(EPISODE_SPINUP_STEPS, max(96, hist_train.n_rows // 5))
    train_windows = extract_clean_regime_windows(
        hist_train, zone, d_init, min_k0=min_spin_train, max_per_regime=20
    )

    # External seasonal episodes from EpisodeBank (zero holdout leak)
    ext_episodes: list[
        tuple[
            StoredEpisode,
            kalman_tape.PrecomputedZoneTape,
        ]
    ] = []
    if episode_bank is not None:
        for ep in episode_bank.get_episodes(zone):
            if ep.end_ts_utc >= holdout_start_ts:
                continue
            overlaps_train = (
                ep.end_ts_utc >= train_start_ts
                and ep.start_ts_utc < holdout_start_ts
            )
            if overlaps_train:
                continue
            ep_hist = preprocessing.preprocess_history_columns(
                ep.columns,
                flow_rates_l_min=flow_rates_l_min,
                k_hx_by_group=k_hx_by_group,
                eta_ac_ref=current_params.eta_ac,
                lat_deg=lat_deg,
                lon_deg=lon_deg,
                buffer_offsets=buffer_offsets,
            )
            if ep_hist.n_rows > ep.spinup_steps + 12:
                ext_episodes.append(
                    (ep, kalman_tape.get_zone_tape(ep_hist, zone))
                )

    if not train_windows and not ext_episodes:
        rep = _eval_fallback(
            eval_steps_arg=holdout_steps,
            pre_eval=(
                precomputed_holdout_before_hist
                if precomputed_holdout_before_hist is not None
                else precomputed_holdout_hist
            ),
            pre_prior=precomputed_holdout_prior_hist,
        )
        return (
            contracts.CalibrationZoneOutcome(
                zone=zone,
                accepted=False,
                reason="No clean excitation windows found in training history",
                mode=mode,
                ts_utc=ts_now,
                params_old=current_params,
                params_map_raw=current_params,
                params_candidate=current_params,
                params_applied=current_params,
                posterior_std={
                    k: active_bounds[k].prior_std for k in param_names
                },
                correlation_matrix={
                    k: {m: (1.0 if k == m else 0.0) for m in param_names}
                    for k in param_names
                },
                collinearity_warnings=[],
                holdout_before=rep,
                holdout_after=rep,
            ),
            fallback_mem,
        )

    # Balance regime weights so abundant regimes do not drown sparse regimes
    reg_counts: dict[str, int] = {"night": 0, "heat": 0, "solar": 0, "ac": 0}
    for w in train_windows:
        reg_counts[w.regime] = reg_counts.get(w.regime, 0) + 1
    for ep, _ in ext_episodes:
        reg_counts[ep.regime] = reg_counts.get(ep.regime, 0) + 1

    # F4.1: Freeze weakly excited seasonal/actuator parameters at aged prior
    frozen_params: dict[str, float] = {}
    fit_param_names: list[str] = []
    for name in param_names:
        if name == "eta_ac" and reg_counts.get("ac", 0) < 4:
            frozen_params[name] = prior_mean_init[name]
        elif name == "a_sol_m2" and reg_counts.get("solar", 0) < 2:
            frozen_params[name] = prior_mean_init[name]
        else:
            fit_param_names.append(name)

    reg_scale: dict[str, float] = {
        reg: (1.0 / math.sqrt(max(1.0, cnt / 6.0))) if cnt > 0 else 1.0
        for reg, cnt in reg_counts.items()
    }

    # Precompute 1D parameter-independent tapes & window specs once before LM
    train_tape = (
        kalman_tape.get_zone_tape(hist_train, zone) if train_windows else None
    )
    train_specs: list[tuple[int, int, Sequence[int], float]] = [
        (
            w.k0,
            w.length_steps,
            _sample_rollout_steps(w.regime, w.length_steps),
            reg_scale[w.regime] / RESIDUAL_SIGMA_C,
        )
        for w in train_windows
    ]
    ext_specs: list[
        tuple[
            kalman_tape.PrecomputedZoneTape,
            list[tuple[int, int, Sequence[int], float]],
        ]
    ] = []
    for ep, ep_tape in ext_episodes:
        k0_ep = ep.spinup_steps
        max_s = min(ep.window_steps, ep_tape.n_rows - 1 - k0_ep)
        s_pts = [
            pt
            for pt in _sample_rollout_steps(ep.regime, ep.window_steps)
            if pt <= max_s
        ]
        ext_specs.append(
            (
                ep_tape,
                [
                    (
                        k0_ep,
                        max_s,
                        s_pts,
                        reg_scale[ep.regime] / RESIDUAL_SIGMA_C,
                    )
                ],
            )
        )

    base_for_lm = (
        dataclasses.replace(current_params, **frozen_params)
        if frozen_params
        else current_params
    )

    def _residual_fn(theta_vec: Sequence[float]) -> list[float]:
        cand_p = _apply_theta_to_params(
            base_for_lm, fit_param_names, theta_vec, cand_khx
        )
        residuals: list[float] = []
        if train_tape is not None and train_specs:
            residuals.extend(
                kalman_tape.evaluate_windows_on_tape(
                    train_tape, cand_p, train_specs
                )
            )
        for ep_tape, ep_spec in ext_specs:
            residuals.extend(
                kalman_tape.evaluate_windows_on_tape(ep_tape, cand_p, ep_spec)
            )
        return residuals

    theta0 = [float(getattr(current_params, k)) for k in fit_param_names]
    lower = [active_bounds[k].lower for k in fit_param_names]
    upper = [active_bounds[k].upper for k in fit_param_names]
    p_mean = [prior_mean_init[k] for k in fit_param_names]
    p_prec = [prior_prec_init[k] for k in fit_param_names]

    sol = solve_b2_box_huber_lm(
        residual_fn=_residual_fn,
        theta0=theta0,
        lower=lower,
        upper=upper,
        prior_mean=p_mean,
        prior_prec_diag=p_prec,
        huber_delta=2.5,
        max_iter=18,
        tol=1e-4,
    )

    raw_fitted: dict[str, float] = dict(frozen_params)
    fisher_map: dict[str, float] = {name: 0.0 for name in frozen_params}
    for i, name in enumerate(fit_param_names):
        raw_fitted[name] = sol.theta[i]
        fisher_map[name] = sol.fisher_diag[i]

    gated_mean, _, cand_memory = update_posterior_memory(
        memory, active_bounds, raw_fitted, fisher_map
    )

    gated_vec = [gated_mean[name] for name in param_names]
    params_map_raw = _apply_theta_to_params(
        current_params, param_names, gated_vec, cand_khx
    )

    if mode == "manual":
        params_candidate = params_map_raw
    else:
        params_candidate = _blend_zone_parameters(
            current_params,
            params_map_raw,
            alpha_new=0.20,
            keep_new_khx=khx_already_damped,
        )

    # Out-of-sample evaluation on unseen holdout slice with shared ref d_act
    if (
        precomputed_holdout_hist is not None
        and precomputed_holdout_hist.n_rows == n_hold
        and _hist_matches_buffer_offsets(
            precomputed_holdout_hist, buffer_offsets
        )
    ):
        hist_hold = precomputed_holdout_hist
    else:
        hist_hold = preprocessing.preprocess_history_columns(
            _get_holdout_cols(),
            flow_rates_l_min=flow_rates_l_min,
            k_hx_by_group=k_hx_by_group,
            eta_ac_ref=current_params.eta_ac,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            buffer_offsets=buffer_offsets,
        )
    states_hold_ref = kalman_tape.reconstruct_states_tuples(
        hist_hold, zone, params_map_raw
    )
    ref_d_hold = [s[2] for s in states_hold_ref]

    zone_hyd_group = preprocessing.zone_to_hydraulic_group(zone)
    old_khx_eval_map = dict(priors.DAY1_K_HX_BY_GROUP_W_K)
    if eff_old_khx is not None:
        for g_name, val in eff_old_khx.items():
            old_khx_eval_map[preprocessing.zone_to_hydraulic_group(g_name)] = (
                float(val)
            )
    old_khx_eval_map[zone_hyd_group] = current_params.k_hx_w_k

    cand_khx_eval_map = dict(priors.DAY1_K_HX_BY_GROUP_W_K)
    if k_hx_by_group is not None:
        for g_name, val in k_hx_by_group.items():
            cand_khx_eval_map[
                preprocessing.zone_to_hydraulic_group(g_name)
            ] = float(val)

    if (
        precomputed_holdout_before_hist is not None
        and precomputed_holdout_before_hist.n_rows == n_hold
        and _hist_matches_buffer_offsets(
            precomputed_holdout_before_hist, eff_old_buf_offsets
        )
    ):
        hist_hold_before = precomputed_holdout_before_hist
    elif _hist_matches_buffer_offsets(hist_hold, eff_old_buf_offsets):
        if old_khx_eval_map == cand_khx_eval_map:
            hist_hold_before = hist_hold
        else:
            hist_hold_before = preprocessing.recompute_history_for_khx(
                hist_hold,
                _get_holdout_cols(),
                flow_rates_l_min=flow_rates_l_min,
                k_hx_by_group=old_khx_eval_map,
            )
    else:
        hist_hold_before = preprocessing.preprocess_history_columns(
            _get_holdout_cols(),
            flow_rates_l_min=flow_rates_l_min,
            k_hx_by_group=old_khx_eval_map,
            eta_ac_ref=current_params.eta_ac,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            buffer_offsets=eff_old_buf_offsets,
        )

    k_hx_prior_hold = dict(priors.DAY1_K_HX_BY_GROUP_W_K)
    k_hx_prior_hold[zone_hyd_group] = day1_prior.k_hx_w_k
    if (
        precomputed_holdout_prior_hist is not None
        and precomputed_holdout_prior_hist.n_rows == n_hold
        and _hist_matches_buffer_offsets(
            precomputed_holdout_prior_hist, priors.DAY1_BUFFER_OFFSETS
        )
    ):
        hist_hold_prior = precomputed_holdout_prior_hist
    elif _hist_matches_buffer_offsets(hist_hold, priors.DAY1_BUFFER_OFFSETS):
        hist_hold_prior = preprocessing.recompute_history_for_khx(
            hist_hold,
            _get_holdout_cols(),
            flow_rates_l_min=flow_rates_l_min,
            k_hx_by_group=k_hx_prior_hold,
        )
    elif _hist_matches_buffer_offsets(
        hist_hold_before, priors.DAY1_BUFFER_OFFSETS
    ):
        hist_hold_prior = preprocessing.recompute_history_for_khx(
            hist_hold_before,
            _get_holdout_cols(),
            flow_rates_l_min=flow_rates_l_min,
            k_hx_by_group=k_hx_prior_hold,
        )
    else:
        hist_hold_prior = preprocessing.preprocess_history_columns(
            _get_holdout_cols(),
            flow_rates_l_min=flow_rates_l_min,
            k_hx_by_group=k_hx_prior_hold,
            eta_ac_ref=day1_prior.eta_ac,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            buffer_offsets=priors.DAY1_BUFFER_OFFSETS,
        )

    eval_cols_arg = (
        holdout_cols_cache if holdout_cols_cache is not None else columns
    )
    holdout_before = evaluation.evaluate_14d_hindcast(
        eval_cols_arg,
        zone,
        current_params,
        day1_prior,
        flow_rates_l_min=flow_rates_l_min,
        k_hx_by_group=eff_old_khx,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        buffer_offsets=eff_old_buf_offsets,
        _ref_d_act=ref_d_hold,
        precomputed_hist_eval=hist_hold_before,
        precomputed_hist_prior=hist_hold_prior,
        eval_steps=holdout_steps,
    )
    holdout_after = evaluation.evaluate_14d_hindcast(
        eval_cols_arg,
        zone,
        params_candidate,
        day1_prior,
        flow_rates_l_min=flow_rates_l_min,
        k_hx_by_group=k_hx_by_group,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        buffer_offsets=buffer_offsets,
        _ref_d_act=ref_d_hold,
        precomputed_hist_eval=hist_hold,
        precomputed_hist_prior=hist_hold_prior,
        eval_steps=holdout_steps,
    )

    accepted = True
    reject_reasons: list[str] = []

    j_before = holdout_before.overall_6h.j_eval_c
    j_after = holdout_after.overall_6h.j_eval_c
    if j_before is not None and j_after is not None:
        if j_after > j_before + 1e-6:
            accepted = False
            reject_reasons.append(
                f"Holdout J_eval degraded ({j_after:.4f} °C > "
                f"{j_before:.4f} °C)"
            )

    for reg_key, m_before in holdout_before.by_regime.items():
        m_after = holdout_after.by_regime.get(reg_key)
        if (
            m_after is None
            or m_before.n_windows < MIN_REGIME_VETO_WINDOWS
            or m_after.n_windows < MIN_REGIME_VETO_WINDOWS
        ):
            continue
        if reg_key == "heating_and_coast":
            if (
                m_before.j_eval_c is not None
                and m_after.j_eval_c is not None
                and m_after.j_eval_c > m_before.j_eval_c + 0.010
            ):
                diff_j = m_after.j_eval_c - m_before.j_eval_c
                accepted = False
                reject_reasons.append(
                    f"Regime 'heating_and_coast' J_eval degraded by "
                    f"+{diff_j:.4f} °C (> +0.010 °C)"
                )
            elif (
                m_before.rmse_end_c is not None
                and m_after.rmse_end_c is not None
                and m_after.rmse_end_c > m_before.rmse_end_c + 0.015
            ):
                diff_r = m_after.rmse_end_c - m_before.rmse_end_c
                accepted = False
                reject_reasons.append(
                    f"Regime 'heating_and_coast' RMSE_end degraded by "
                    f"+{diff_r:.4f} °C (> +0.015 °C)"
                )
            elif (
                m_before.bias_end_c is not None
                and m_after.bias_end_c is not None
                and abs(m_after.bias_end_c) > 0.15
                and abs(m_after.bias_end_c) > abs(m_before.bias_end_c) + 0.03
            ):
                accepted = False
                reject_reasons.append(
                    f"Regime 'heating_and_coast' |bias_end| degraded "
                    f"({abs(m_before.bias_end_c):.4f} -> "
                    f"{abs(m_after.bias_end_c):.4f} °C)"
                )
            continue
        if (
            m_before.rmse_end_c is not None
            and m_after.rmse_end_c is not None
            and (m_after.rmse_end_c - m_before.rmse_end_c) > 0.03
        ):
            diff_r = m_after.rmse_end_c - m_before.rmse_end_c
            accepted = False
            reject_reasons.append(
                f"Regime '{reg_key}' RMSE_end degraded by +{diff_r:.4f} °C "
                f"(> +0.03 °C)"
            )
        elif (
            m_before.j_eval_c is not None
            and m_after.j_eval_c is not None
            and (m_after.j_eval_c - m_before.j_eval_c) > 0.03
        ):
            diff_j = m_after.j_eval_c - m_before.j_eval_c
            accepted = False
            reject_reasons.append(
                f"Regime '{reg_key}' J_eval degraded by +{diff_j:.4f} °C "
                f"(> +0.03 °C)"
            )

    if accepted:
        params_applied = params_candidate
        final_mem = cand_memory
        if j_before is not None and j_after is not None:
            reason = (
                f"Accepted ({mode}): holdout J_eval "
                f"{j_before:.3f} -> {j_after:.3f} °C"
            )
        else:
            reason = f"Accepted ({mode}): MAP converged in {sol.n_iter} steps"
    else:
        params_applied = current_params
        final_mem = fallback_mem
        reason = "; ".join(reject_reasons)

    fit_idx_map = {name: i for i, name in enumerate(fit_param_names)}
    post_std_dict: dict[str, float] = {}
    for name in param_names:
        if name in fit_idx_map:
            post_std_dict[name] = round(
                sol.posterior_std[fit_idx_map[name]], 6
            )
        else:
            post_std_dict[name] = round(
                1.0 / math.sqrt(max(1e-12, prior_prec_init[name])), 6
            )

    corr_dict: dict[str, dict[str, float]] = {}
    for name_i in param_names:
        row_corr: dict[str, float] = {}
        for name_j in param_names:
            if name_i == name_j:
                row_corr[name_j] = 1.0
            elif name_i in fit_idx_map and name_j in fit_idx_map:
                row_corr[name_j] = round(
                    sol.correlation_matrix[fit_idx_map[name_i]][
                        fit_idx_map[name_j]
                    ],
                    4,
                )
            else:
                row_corr[name_j] = 0.0
        corr_dict[name_i] = row_corr

    collin_warns = [
        f"High collinearity between {fit_param_names[i]} and "
        f"{fit_param_names[j]}: rho={rho:.3f}"
        for i, j, rho in sol.collinearity_pairs
    ]

    bound_warns: list[str] = []
    for name in param_names:
        b = active_bounds[name]
        val_applied = float(getattr(params_applied, name))
        tol_b = 1e-3 * (b.upper - b.lower)
        if val_applied <= b.lower + tol_b:
            bound_warns.append(
                f"{name} at lower bound ({val_applied:.4g} <= {b.lower:.4g})"
            )
        elif val_applied >= b.upper - tol_b:
            bound_warns.append(
                f"{name} at upper bound ({val_applied:.4g} >= {b.upper:.4g})"
            )

    outcome = contracts.CalibrationZoneOutcome(
        zone=zone,
        accepted=accepted,
        reason=reason,
        mode=mode,
        ts_utc=ts_now,
        params_old=current_params,
        params_map_raw=params_map_raw,
        params_candidate=params_candidate,
        params_applied=params_applied,
        posterior_std=post_std_dict,
        correlation_matrix=corr_dict,
        collinearity_warnings=collin_warns,
        holdout_before=holdout_before,
        holdout_after=holdout_after,
        bound_warnings=bound_warns,
    )
    return outcome, final_mem


def posterior_memory_to_dict(
    mem: contracts.ParameterPosteriorMemory | None,
) -> dict[str, Any] | None:
    """Serializes `ParameterPosteriorMemory` to a JSON-compatible dict."""
    if mem is None:
        return None
    return {"mean": dict(mem.mean), "precision": dict(mem.precision)}


def posterior_memory_from_dict(
    data: Mapping[str, Any] | None,
) -> contracts.ParameterPosteriorMemory | None:
    """Deserializes `ParameterPosteriorMemory` from a persisted dict."""
    if not isinstance(data, Mapping):
        return None
    raw_mean = data.get("mean")
    raw_prec = data.get("precision")
    if not isinstance(raw_mean, Mapping) or not isinstance(raw_prec, Mapping):
        return None
    mean: dict[str, float] = {}
    precision: dict[str, float] = {}
    for k, mean_val in raw_mean.items():
        key = str(k)
        prec_val = raw_prec.get(key)
        if preprocessing.is_finite_number(
            mean_val
        ) and preprocessing.is_finite_number(prec_val):
            mean[key] = float(mean_val)
            precision[key] = float(prec_val)
    return contracts.ParameterPosteriorMemory(
        mean=mean,
        precision=precision,
    )
