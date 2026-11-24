"""Multi-regime 14-day open-loop hindcast evaluation and live 6h FIFO tracker.

Owns deterministic 14-day multi-regime open-loop hindcast evaluation
(`evaluate_14d_hindcast`), reference object-level Kalman state reconstruction
(`reconstruct_kalman_states`), live 72-step (6-hour) open-loop FIFO hindcast
tracking (`LiveHindcastTracker`), and `ZoneEvaluationReport` dictionary
serialization (`zone_eval_report_to_dict`, `zone_eval_report_from_dict`).

Does not own RingBuffer ETL (`preprocessing.py`), unrolled calibration tapes
(`kalman_tape.py`), or CSV/JSON file export (`reporting.py`).

Units: temperatures in °C, thermal powers in W, errors/RMSE in °C, skill score
in %, time in s (or 5-minute steps).
"""

from __future__ import annotations

import collections
from collections.abc import Mapping, Sequence
import dataclasses
import math
from typing import Any

from custom_components.betonbatterij import const
from custom_components.betonbatterij.domain import contracts
from custom_components.betonbatterij.domain import kalman_tape
from custom_components.betonbatterij.domain import preprocessing
from custom_components.betonbatterij.domain import priors
from custom_components.betonbatterij.domain import zoh_kalman

# Evaluation horizon, spin-up, and stride constants.
SPINUP_48H_STEPS: int = 576  # 48h * 12 steps/h
EVAL_14D_STEPS: int = preprocessing.EVAL_14D_STEPS  # 4,032 steps
MAX_EVAL_SLICE_STEPS: int = SPINUP_48H_STEPS + EVAL_14D_STEPS  # 4,608 steps
HORIZON_6H_STEPS: int = preprocessing.HORIZON_6H_STEPS  # 6.0 hours
HORIZON_AC_1H_STEPS: int = 12  # 1.0 hour
MIN_CLEAN_6H_STEPS: int = 48  # >= 4.0 hours clean before mid-window truncation
STRIDE_6H_STEPS: int = 6  # 30 minutes
STRIDE_AC_1H_STEPS: int = 2  # 10 minutes
COOLING_TAIL_STEPS: int = preprocessing.COOLING_TAIL_STEPS  # 2.0 hours


def reconstruct_kalman_states(
    hist: preprocessing.PreprocessedHistory,
    zone: contracts.ZoneId,
    params: contracts.ZoneParameters,
    exog_list: Sequence[contracts.ExogenousStepInput] | None = None,
) -> tuple[
    list[contracts.ThermalState],
    list[float],
    list[float],
    zoh_kalman.ZoneKalmanEstimator,
]:
    """Runs a deterministic forward Kalman reconstruction pass under `params`.

    Args:
        hist: Preprocessed telemetry slice.
        zone: Zone identifier (`"wk"` or `"kt"`).
        params: Candidate `ZoneParameters` to evaluate.
        exog_list: Optional pre-built exogenous step inputs for `zone`.

    Returns:
        A tuple `(states, q_fh_series, q_ac_series, estimator)` containing the
        posterior `ThermalState` per step, realized `Q_fh` and `Q_ac` series,
        and the final `ZoneKalmanEstimator` instance.
    """
    n = hist.n_rows
    exogs = (
        exog_list
        if exog_list is not None
        else preprocessing.build_zone_exog_list(hist, zone)
    )
    est = zoh_kalman.ZoneKalmanEstimator(zone, params)
    if n == 0:
        return [], [], [], est

    valid_flag = const.FLAG_VALID_WK if zone == "wk" else const.FLAG_VALID_KT
    t_meas_arr = hist.t_air_wk_c if zone == "wk" else hist.t_air_kt_c

    states: list[contracts.ThermalState] = []
    q_fh_series: list[float] = [0.0] * n
    q_ac_series: list[float] = [0.0] * n

    q_ac_0 = preprocessing.step_q_ac_for_params(hist, zone, 0, params)
    q_fh_0 = hist.q_fh_wk_hyd_w[0] if zone == "wk" else hist.q_fh_kt_hyd_w[0]
    est.cold_start(
        t_air_c=t_meas_arr[0],
        exog=exogs[0],
        q_fh_w=q_fh_0,
        q_ac_w=q_ac_0,
        u_pump=hist.u_pump_sec[0],
        t_buf_top_c=hist.t_buf_top_c[0],
    )
    states.append(est.state)
    q_fh_series[0] = q_fh_0
    q_ac_series[0] = q_ac_0

    for k in range(1, n):
        fl_k = hist.flags[k]
        valid_k = bool(fl_k & valid_flag)
        aanwezig_k = bool(fl_k & const.FLAG_AANWEZIG)
        dt_k = hist.dt_s[k]

        q_fh_k = preprocessing.step_q_fh_for_params(
            hist, zone, k, est.state.t_floor_c, params
        )
        q_ac_k = preprocessing.step_q_ac_for_params(hist, zone, k, params)
        q_fh_series[k] = q_fh_k
        q_ac_series[k] = q_ac_k

        if dt_k > const.MAX_RESTART_BRIDGE_GAP_S:
            est.cold_start(
                t_air_c=t_meas_arr[k],
                exog=exogs[k],
                q_fh_w=q_fh_k,
                q_ac_w=q_ac_k,
                u_pump=hist.u_pump_sec[k],
                t_buf_top_c=hist.t_buf_top_c[k],
            )
            states.append(est.state)
            continue

        temp_changed = abs(t_meas_arr[k] - t_meas_arr[k - 1]) >= 1e-4
        hum_changed = valid_k and (not temp_changed) and (k % 6 == 0)
        t_surf = hist.t_surf_meas_wk_c[k] if zone == "wk" else None

        st = est.step(
            ts_utc=hist.ts_utc[k],
            dt_s=dt_k,
            exog=exogs[k],
            q_fh_w=q_fh_k,
            q_ac_w=q_ac_k,
            u_pump=hist.u_pump_sec[k],
            t_air_meas_c=t_meas_arr[k] if valid_k else None,
            temp_changed=temp_changed,
            hum_changed=hum_changed,
            sensor_stale=not valid_k,
            aanwezig=aanwezig_k,
            t_surf_meas_c=t_surf,
        )
        states.append(st)

    return states, q_fh_series, q_ac_series, est


class LiveHindcastTracker:
    """FIFO tracker of in-flight 72-step (6-hour) open-loop predictions."""

    def __init__(self, horizon_steps: int = HORIZON_6H_STEPS) -> None:
        """Initializes the tracker with a target rollout horizon.

        Args:
            horizon_steps: Number of 5-minute ZOH steps to track (`72` = 6h).
        """
        self._horizon_steps: int = int(horizon_steps)
        self._queue: collections.deque[list[float | int]] = collections.deque()
        self._last_published_c: float | None = None
        self._last_published_exact_c: float | None = None
        self._warmup_remaining_min: int = self._horizon_steps * 5

    @property
    def horizon_steps(self) -> int:
        """Number of 5-minute steps in the hindcast horizon."""
        return self._horizon_steps

    @property
    def queue(self) -> collections.deque[list[float | int]]:
        """FIFO queue of `[age_steps, t_air_ol, t_floor_ol, d_ol]` entries."""
        return self._queue

    @property
    def last_published_c(self) -> float | None:
        """Most recently published 6h hindcast temperature rounded to 0.01 °C."""
        return self._last_published_c

    @property
    def last_published_exact_c(self) -> float | None:
        """Unrounded float of the most recently popped 72-step prediction."""
        return self._last_published_exact_c

    @property
    def warmup_remaining_min(self) -> int:
        """Minutes remaining before the oldest seed reaches `horizon_steps`."""
        return self._warmup_remaining_min

    def clear(self) -> None:
        """Clears the FIFO queue on cold start or outages exceeding 3 hours."""
        self._queue.clear()
        self._last_published_c = None
        self._last_published_exact_c = None
        self._warmup_remaining_min = self._horizon_steps * 5

    def step(
        self,
        posterior_state: contracts.ThermalState,
        params: contracts.ZoneParameters,
        exog_realized: contracts.ExogenousStepInput,
        q_fh_realized_w: float,
        q_ac_realized_w: float = 0.0,
        dt_s: float = 300.0,
    ) -> tuple[float | None, int]:
        """Advances queued states 1 ZOH step, pops at age 72, and seeds posterior.

        Args:
            posterior_state: Fresh closed-loop Kalman posterior at window end.
            params: Active `ZoneParameters` for open-loop propagation.
            exog_realized: Realized exogenous boundary inputs over the step.
            q_fh_realized_w: Realized floor heating power over the step [W].
            q_ac_realized_w: Realized split-AC thermal power over the step [W].
            dt_s: Step duration [s].

        Returns:
            A tuple `(published_hindcast_c, warmup_remaining_min)`, where
            `published_hindcast_c` is `round(t_air_ol, 2)` once warmed up (or
            `None` during the initial 6-hour spin-up).
        """
        for entry in self._queue:
            st_prev = contracts.ThermalState(
                t_air_c=float(entry[1]),
                t_floor_c=float(entry[2]),
                d_est_w=float(entry[3]),
            )
            st_next = zoh_kalman.propagate_open_loop_step(
                state=st_prev,
                params=params,
                exog=exog_realized,
                q_fh_w=q_fh_realized_w,
                q_ac_w=q_ac_realized_w,
                dt_s=dt_s,
            )
            entry[0] = int(entry[0]) + 1
            entry[1] = st_next.t_air_c
            entry[2] = st_next.t_floor_c
            entry[3] = st_next.d_est_w

        published_c: float | None = None
        while self._queue and int(self._queue[0][0]) >= self._horizon_steps:
            oldest = self._queue.popleft()
            if int(oldest[0]) == self._horizon_steps or published_c is None:
                exact_val = float(oldest[1])
                self._last_published_exact_c = exact_val
                published_c = round(exact_val, 2)

        self._queue.append(
            [
                0,
                posterior_state.t_air_c,
                posterior_state.t_floor_c,
                posterior_state.d_est_w,
            ]
        )

        if published_c is not None:
            self._last_published_c = published_c
            self._warmup_remaining_min = 0
            return published_c, 0

        oldest_age = int(self._queue[0][0]) if self._queue else 0
        rem_min = max(5, (self._horizon_steps - oldest_age) * 5)
        self._warmup_remaining_min = rem_min
        return None, rem_min

    def bridge_gap(
        self,
        dt_gap_s: float,
        params: contracts.ZoneParameters,
        exog_avg: contracts.ExogenousStepInput,
        q_fh_avg_w: float,
        q_ac_avg_w: float = 0.0,
    ) -> None:
        """Bridges a Home Assistant restart gap (`<= 3h`) via variable-`dt` ZOH.

        Args:
            dt_gap_s: Duration of the downtime gap [s]. Gaps exceeding
                `MAX_RESTART_BRIDGE_GAP_S` clear the queue.
            params: Active `ZoneParameters`.
            exog_avg: Average exogenous inputs across the restart gap.
            q_fh_avg_w: Average floor heating power across the gap [W].
            q_ac_avg_w: Average split-AC thermal power across the gap [W].
        """
        if dt_gap_s <= 0.0:
            return
        if dt_gap_s > const.MAX_RESTART_BRIDGE_GAP_S:
            self.clear()
            return

        gap_steps = max(1, int(round(dt_gap_s / float(const.DT_STEP_S))))
        for entry in self._queue:
            st_prev = contracts.ThermalState(
                t_air_c=float(entry[1]),
                t_floor_c=float(entry[2]),
                d_est_w=float(entry[3]),
            )
            st_next = zoh_kalman.propagate_open_loop_step(
                state=st_prev,
                params=params,
                exog=exog_avg,
                q_fh_w=q_fh_avg_w,
                q_ac_w=q_ac_avg_w,
                dt_s=dt_gap_s,
            )
            entry[0] = int(entry[0]) + gap_steps
            entry[1] = st_next.t_air_c
            entry[2] = st_next.t_floor_c
            entry[3] = st_next.d_est_w

        while self._queue and int(self._queue[0][0]) >= self._horizon_steps:
            popped = self._queue.popleft()
            if int(popped[0]) == self._horizon_steps:
                self._last_published_exact_c = float(popped[1])
                self._last_published_c = round(float(popped[1]), 2)
                self._warmup_remaining_min = 0

        if self._queue and self._last_published_c is None:
            oldest_age = int(self._queue[0][0])
            self._warmup_remaining_min = max(
                5, (self._horizon_steps - oldest_age) * 5
            )

    def reseed_from_history(
        self,
        steps_data: Sequence[dict[str, Any]],
        params: contracts.ZoneParameters,
    ) -> float | None:
        """Rebuilds the 72-step FIFO queue after a parameter update or rollback.

        Args:
            steps_data: Chronological sequence of step dictionaries containing
                `posterior_state`, `exog`, `q_fh_w`, `q_ac_w`, and `dt_s`.
            params: Newly applied `ZoneParameters`.

        Returns:
            Immediate 6h hindcast temperature [°C] if `>= 73` history steps
            were replayed, otherwise `None`.
        """
        self.clear()
        if not steps_data:
            return None

        tail = steps_data[-(self._horizon_steps + 1) :]

        first = tail[0]
        init_st = first.get("initial_state") or first.get("prior_state")
        if len(tail) == self._horizon_steps and isinstance(
            init_st, contracts.ThermalState
        ):
            self._queue.append(
                [0, init_st.t_air_c, init_st.t_floor_c, init_st.d_est_w]
            )

        last_pub: float | None = None
        for item in tail:
            post = item.get("posterior_state") or item.get("state")
            if not isinstance(post, contracts.ThermalState):
                post = contracts.ThermalState(
                    t_air_c=float(item["t_air_c"]),
                    t_floor_c=float(item["t_floor_c"]),
                    d_est_w=float(item.get("d_est_w", 0.0)),
                )
            exog = item.get("exog") or item.get("exog_realized")
            if not isinstance(exog, contracts.ExogenousStepInput):
                exog = contracts.ExogenousStepInput(
                    t_out_c=float(item.get("t_out_c", 10.0)),
                    v_wind_ms=float(item.get("v_wind_ms", 3.0)),
                    i_sol_wm2=float(item.get("i_sol_wm2", 0.0)),
                    t_down_c=float(
                        item.get("t_down_c", priors.T_GROUND_MEAN_C)
                    ),
                    q_int_known_w=float(
                        item.get("q_int_known_w", priors.BASELOAD_WK_W)
                    ),
                    t_up_c=float(item.get("t_up_c", 18.0)),
                )
            q_fh = float(item.get("q_fh_w", item.get("q_fh_realized_w", 0.0)))
            q_ac = float(item.get("q_ac_w", item.get("q_ac_realized_w", 0.0)))
            dt_step = float(item.get("dt_s", const.DT_STEP_S))
            last_pub, _ = self.step(post, params, exog, q_fh, q_ac, dt_step)

        return last_pub

    def to_dict(self) -> dict[str, Any]:
        """Serializes tracker state for Home Assistant Store persistence."""
        return {
            "horizon_steps": self._horizon_steps,
            "queue": [
                [int(it[0]), float(it[1]), float(it[2]), float(it[3])]
                for it in self._queue
            ],
            "last_published_c": self._last_published_c,
            "warmup_remaining_min": self._warmup_remaining_min,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LiveHindcastTracker:
        """Restores a `LiveHindcastTracker` from a persisted dictionary."""
        horizon = int(data.get("horizon_steps", HORIZON_6H_STEPS))
        tracker = cls(horizon_steps=horizon)
        for row in data.get("queue", ()):
            if len(row) >= 4:
                tracker._queue.append(
                    [int(row[0]), float(row[1]), float(row[2]), float(row[3])]
                )
        last_pub = data.get("last_published_c")
        tracker._last_published_c = (
            float(last_pub) if last_pub is not None else None
        )
        tracker._last_published_exact_c = tracker._last_published_c
        tracker._warmup_remaining_min = int(
            data.get(
                "warmup_remaining_min",
                horizon * 5 if not tracker._queue else 0,
            )
        )
        return tracker


def _empty_regime_metrics(
    regime: str,
    horizon_steps: int,
) -> contracts.RegimeHindcastMetrics:
    """Returns an empty `RegimeHindcastMetrics` record when `n_windows == 0`."""
    return contracts.RegimeHindcastMetrics(
        regime=regime,
        horizon_steps=horizon_steps,
        n_windows=0,
        rmse_end_c=None,
        rmse_path_c=None,
        mae_end_c=None,
        bias_end_c=None,
        j_eval_c=None,
        prior_rmse_end_c=None,
        skill_score_pct=None,
    )


def _compute_regime_metrics(
    regime: str,
    horizon_steps: int,
    eval_err_paths: Sequence[Sequence[float]],
    prior_err_paths: Sequence[Sequence[float]],
) -> contracts.RegimeHindcastMetrics:
    """Computes `RMSE_end`, `RMSE_path`, `MAE`, `Bias`, `J_eval`, and Skill."""
    w_cnt = len(eval_err_paths)
    if w_cnt == 0:
        return _empty_regime_metrics(regime, horizon_steps)

    sum_sq_end = 0.0
    sum_abs_end = 0.0
    sum_bias_end = 0.0
    sum_path_mse = 0.0
    sum_sq_prior_end = 0.0

    for e_eval, e_prior in zip(eval_err_paths, prior_err_paths):
        lw = len(e_eval)
        end_err = e_eval[-1]
        prior_end_err = e_prior[-1]
        sum_sq_end += end_err * end_err
        sum_abs_end += abs(end_err)
        sum_bias_end += end_err
        sum_path_mse += sum(x * x for x in e_eval) / float(lw)
        sum_sq_prior_end += prior_end_err * prior_end_err

    rmse_end = math.sqrt(sum_sq_end / float(w_cnt))
    rmse_path = math.sqrt(sum_path_mse / float(w_cnt))
    mae_end = sum_abs_end / float(w_cnt)
    bias_end = sum_bias_end / float(w_cnt)
    j_eval = 0.65 * rmse_end + 0.35 * rmse_path
    prior_rmse_end = math.sqrt(sum_sq_prior_end / float(w_cnt))
    skill_score_pct = 100.0 * (1.0 - rmse_end / max(prior_rmse_end, 0.02))

    return contracts.RegimeHindcastMetrics(
        regime=regime,
        horizon_steps=horizon_steps,
        n_windows=w_cnt,
        rmse_end_c=rmse_end,
        rmse_path_c=rmse_path,
        mae_end_c=mae_end,
        bias_end_c=bias_end,
        j_eval_c=j_eval,
        prior_rmse_end_c=prior_rmse_end,
        skill_score_pct=skill_score_pct,
    )


def _is_valid_eval_start(
    hist: preprocessing.PreprocessedHistory,
    zone: contracts.ZoneId,
    k0: int,
    t_meas_arr: Sequence[float],
    d_act: Sequence[float],
) -> bool:
    """Checks causal evaluation start-point validity at step `k0`."""
    valid_flag = const.FLAG_VALID_WK if zone == "wk" else const.FLAG_VALID_KT
    open_win_flag = (
        const.FLAG_OPEN_WIN_WK if zone == "wk" else const.FLAG_OPEN_WIN_KT
    )
    fl0 = hist.flags[k0]
    if not (fl0 & valid_flag) or hist.cooling_or_tail[k0]:
        return False
    if fl0 & open_win_flag:
        return False
    if k0 >= 3 and (t_meas_arr[k0] - t_meas_arr[k0 - 3]) / 0.25 < -0.8:
        return False
    if abs(d_act[k0]) > 200.0:
        return False
    return True


def _truncate_on_unmodeled_event(
    hist: preprocessing.PreprocessedHistory,
    zone: contracts.ZoneId,
    k0: int,
    horizon_steps: int,
    d_act: Sequence[float],
    tau_d_s: float,
    t_meas_arr: Sequence[float],
    is_ac_window: bool = False,
) -> int:
    """Returns clean causal rollout length `L_w <= horizon_steps`."""
    valid_flag = const.FLAG_VALID_WK if zone == "wk" else const.FLAG_VALID_KT
    open_win_flag = (
        const.FLAG_OPEN_WIN_WK if zone == "wk" else const.FLAG_OPEN_WIN_KT
    )
    d0 = d_act[k0]

    for s in range(1, horizon_steps + 1):
        idx = k0 + s
        fl = hist.flags[idx]
        if (
            not (fl & valid_flag)
            or (fl & const.FLAG_GAP_BRIDGED)
            or hist.dt_s[idx] > 600.0
        ):
            return s - 1
        if hist.cooling_or_tail[idx]:
            return s - 1
        if fl & open_win_flag:
            return s - 1
        if idx >= 3 and (t_meas_arr[idx] - t_meas_arr[idx - 3]) / 0.25 < -0.8:
            return s - 1

        active_ac_step = (
            is_ac_window
            and round(hist.ac_mode_code[idx]) == int(const.AC_MODE_HEAT)
            and hist.p_el_ac_kt_w[idx] > 80.0
        )
        if not active_ac_step:
            expected_d = d0 * math.exp(-(s * float(const.DT_STEP_S)) / tau_d_s)
            if abs(d_act[idx] - expected_d) > 250.0:
                return s - 1

    return horizon_steps


def evaluate_14d_hindcast(
    columns: Mapping[str, Sequence[float | int]],
    zone: contracts.ZoneId,
    params_eval: contracts.ZoneParameters,
    params_prior: contracts.ZoneParameters,
    flow_rates_l_min: Mapping[str, float] | None = None,
    k_hx_by_group: Mapping[str, float] | None = None,
    *,
    lat_deg: float = priors.LATITUDE_DEG,
    lon_deg: float = priors.LONGITUDE_DEG,
    buffer_offsets: contracts.BufferSensorOffsets = (
        priors.DAY1_BUFFER_OFFSETS
    ),
    _ref_d_act: Sequence[float] | None = None,
    precomputed_hist_eval: preprocessing.PreprocessedHistory | None = None,
    precomputed_hist_prior: preprocessing.PreprocessedHistory | None = None,
    eval_steps: int = EVAL_14D_STEPS,
) -> contracts.ZoneEvaluationReport:
    """Evaluates 14-day (or custom tail) 6h & `kt` 1h AC open-loop hindcast.

    Runs dual parameter-consistent forward Kalman reconstructions under
    `params_eval` and `params_prior` over up to `48h` spin-up + `eval_steps`
    evaluation so comparisons against Day-1 priors suffer zero initial-state
    enthalpy jump.

    Args:
        columns: RingBuffer column sequences.
        zone: Target zone (`"wk"` or `"kt"`).
        params_eval: Active or candidate `ZoneParameters` to score.
        params_prior: Reference Day-1 prior `ZoneParameters` for Skill Score.
        flow_rates_l_min: Optional manifold flow rates [L/min].
        k_hx_by_group: Optional `K_hx` [W/K] mapping per group.
        lat_deg: Observer latitude in degrees North.
        lon_deg: Observer longitude in degrees East.
        buffer_offsets: Active or candidate 1-Wire `BufferSensorOffsets` for
            `params_eval` (`params_prior` always uses `DAY1_BUFFER_OFFSETS`).
        _ref_d_act: Optional shared disturbance series used by the holdout gate
            to guarantee identical window selection between two parameter sets.
        precomputed_hist_eval: Optional precomputed history for `params_eval`.
        precomputed_hist_prior: Optional precomputed history for `params_prior`.
        eval_steps: Number of tail evaluation steps after 48h spin-up
            (default `EVAL_14D_STEPS = 4032`).

    Returns:
        Complete `ZoneEvaluationReport` across `overall_6h` and all physical
        regimes.
    """
    ts_all = columns.get("ts_utc", ())
    n_total = len(ts_all)

    regime_names = (
        ("night_cooling", "heating_and_coast", "sunny_afternoon")
        if zone == "wk"
        else (
            "night_cooling",
            "heating_and_coast",
            "sunny_afternoon",
            "airco_heating_1h",
        )
    )

    if n_total < SPINUP_48H_STEPS:
        ts_eval = int(ts_all[-1]) if n_total > 0 else 0
        by_reg = {
            r: _empty_regime_metrics(
                r,
                HORIZON_AC_1H_STEPS
                if r == "airco_heating_1h"
                else HORIZON_6H_STEPS,
            )
            for r in regime_names
        }
        return contracts.ZoneEvaluationReport(
            zone=zone,
            ts_evaluated_utc=ts_eval,
            overall_6h=_empty_regime_metrics("overall_6h", HORIZON_6H_STEPS),
            by_regime=by_reg,
        )

    eff_eval_steps = max(HORIZON_6H_STEPS + 1, int(eval_steps))
    max_slice_steps = SPINUP_48H_STEPS + eff_eval_steps
    start_slice = max(0, n_total - max_slice_steps)
    n_slice = n_total - start_slice
    sub_cols_cache: Mapping[str, Sequence[float | int]] | None = None

    def _get_sub_cols() -> Mapping[str, Sequence[float | int]]:
        nonlocal sub_cols_cache
        if sub_cols_cache is None:
            sub_cols_cache = (
                columns
                if start_slice == 0
                else preprocessing.slice_columns(columns, start_slice, n_total)
            )
        return sub_cols_cache

    if n_slice >= SPINUP_48H_STEPS + HORIZON_6H_STEPS:
        spinup_steps = SPINUP_48H_STEPS
    else:
        spinup_steps = max(288, n_slice - 2 * HORIZON_6H_STEPS)

    k_hx_eval_map = dict(priors.DAY1_K_HX_BY_GROUP_W_K)
    if k_hx_by_group is not None:
        for group_name, val in k_hx_by_group.items():
            k_hx_eval_map[preprocessing.zone_to_hydraulic_group(group_name)] = (
                float(val)
            )
    zone_khx_key = preprocessing.zone_to_hydraulic_group(zone)
    k_hx_eval_map[zone_khx_key] = params_eval.k_hx_w_k

    k_hx_prior_map = dict(priors.DAY1_K_HX_BY_GROUP_W_K)
    k_hx_prior_map[zone_khx_key] = params_prior.k_hx_w_k

    def _hist_matches_khx_and_offsets(
        cand_h: preprocessing.PreprocessedHistory | None,
        target_khx: float,
        target_offsets: contracts.BufferSensorOffsets,
    ) -> bool:
        if cand_h is None or cand_h.n_rows != n_slice:
            return False
        h_khx = cand_h.hyd_k_hx_wk if zone == "wk" else cand_h.hyd_k_hx_kt
        top_h = float(
            getattr(cand_h, "buf_top_offset_c", priors.BUF_TOP_SENSOR_OFFSET_C)
        )
        bot_h = float(
            getattr(cand_h, "buf_bot_offset_c", priors.BUF_BOT_SENSOR_OFFSET_C)
        )
        return (
            abs(h_khx - target_khx) <= 1e-9
            and abs(top_h - target_offsets.buf_top_offset_c) <= 1e-9
            and abs(bot_h - target_offsets.buf_bot_offset_c) <= 1e-9
        )

    if _hist_matches_khx_and_offsets(
        precomputed_hist_eval, k_hx_eval_map[zone_khx_key], buffer_offsets
    ):
        assert precomputed_hist_eval is not None
        hist_eval = precomputed_hist_eval
    elif _hist_matches_khx_and_offsets(
        precomputed_hist_prior, k_hx_eval_map[zone_khx_key], buffer_offsets
    ):
        assert precomputed_hist_prior is not None
        hist_eval = precomputed_hist_prior
    elif (
        precomputed_hist_eval is not None
        and precomputed_hist_eval.n_rows == n_slice
    ):
        hist_eval = preprocessing.recompute_history_for_khx(
            precomputed_hist_eval,
            _get_sub_cols(),
            flow_rates_l_min=flow_rates_l_min,
            k_hx_by_group=k_hx_eval_map,
            buffer_offsets=buffer_offsets,
        )
    else:
        hist_eval = preprocessing.preprocess_history_columns(
            _get_sub_cols(),
            flow_rates_l_min=flow_rates_l_min,
            k_hx_by_group=k_hx_eval_map,
            eta_ac_ref=params_eval.eta_ac,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            buffer_offsets=buffer_offsets,
        )

    day1_offsets = priors.DAY1_BUFFER_OFFSETS
    if _hist_matches_khx_and_offsets(
        precomputed_hist_prior, k_hx_prior_map[zone_khx_key], day1_offsets
    ):
        assert precomputed_hist_prior is not None
        hist_prior = precomputed_hist_prior
    elif _hist_matches_khx_and_offsets(
        hist_eval, k_hx_prior_map[zone_khx_key], day1_offsets
    ):
        hist_prior = hist_eval
    else:
        base_for_prior = (
            precomputed_hist_prior
            if (
                precomputed_hist_prior is not None
                and precomputed_hist_prior.n_rows == n_slice
            )
            else hist_eval
        )
        hist_prior = preprocessing.recompute_history_for_khx(
            base_for_prior,
            _get_sub_cols(),
            flow_rates_l_min=flow_rates_l_min,
            k_hx_by_group=k_hx_prior_map,
            buffer_offsets=day1_offsets,
        )

    tape_eval = kalman_tape.get_zone_tape(hist_eval, zone)
    tape_prior = (
        tape_eval
        if hist_prior is hist_eval
        else kalman_tape.get_zone_tape(hist_prior, zone)
    )

    states_eval = kalman_tape.reconstruct_states_on_tape(tape_eval, params_eval)
    states_prior = kalman_tape.reconstruct_states_on_tape(
        tape_prior, params_prior
    )

    ad_eval, bd_eval, _, _, _ = zoh_kalman.discretize_ab_cached(
        params_eval, 300.0, 0.0
    )
    ad_prior, bd_prior, _, _, _ = zoh_kalman.discretize_ab_cached(
        params_prior, 300.0, 0.0
    )
    eval_khx_ratio = (
        (params_eval.k_hx_w_k / tape_eval.hyd_k_hx)
        if tape_eval.hyd_k_hx > 1e-6
        else 1.0
    )
    eval_khx_act = (
        eval_khx_ratio
        if (
            tape_eval.hyd_k_hx > 1e-6
            and abs(params_eval.k_hx_w_k - tape_eval.hyd_k_hx) > 1e-6
        )
        else 1.0
    )
    prior_khx_ratio = (
        (params_prior.k_hx_w_k / tape_prior.hyd_k_hx)
        if tape_prior.hyd_k_hx > 1e-6
        else 1.0
    )
    prior_khx_act = (
        prior_khx_ratio
        if (
            tape_prior.hyd_k_hx > 1e-6
            and abs(params_prior.k_hx_w_k - tape_prior.hyd_k_hx) > 1e-6
        )
        else 1.0
    )

    if _ref_d_act is not None and len(_ref_d_act) == n_slice:
        d_act = list(_ref_d_act)
    else:
        d_act = [st[2] for st in states_eval]

    t_meas = hist_eval.t_air_wk_c if zone == "wk" else hist_eval.t_air_kt_c
    i_sol_arr = (
        hist_eval.i_sol_wk_wm2 if zone == "wk" else hist_eval.i_sol_kt_wm2
    )
    tau_d_s = max(300.0, params_eval.tau_d_h * 3600.0)

    overall_eval: list[list[float]] = []
    overall_prior: list[list[float]] = []
    reg_eval: dict[str, list[list[float]]] = {r: [] for r in regime_names}
    reg_prior: dict[str, list[list[float]]] = {r: [] for r in regime_names}

    max_k0_6h = n_slice - 1 - HORIZON_6H_STEPS
    for k0 in range(spinup_steps, max_k0_6h + 1, STRIDE_6H_STEPS):
        if not _is_valid_eval_start(hist_eval, zone, k0, t_meas, d_act):
            continue

        lw = _truncate_on_unmodeled_event(
            hist_eval,
            zone,
            k0,
            HORIZON_6H_STEPS,
            d_act,
            tau_d_s,
            t_meas,
            is_ac_window=False,
        )
        if lw < MIN_CLEAN_6H_STEPS:
            continue

        e_eval, q_fh_ol = kalman_tape.rollout_open_loop_window_fast(
            tape_eval,
            k0,
            lw,
            states_eval[k0],
            params_eval,
            ad_eval,
            bd_eval,
            eval_khx_ratio,
            eval_khx_act,
        )
        e_prior, _ = kalman_tape.rollout_open_loop_window_fast(
            tape_prior,
            k0,
            lw,
            states_prior[k0],
            params_prior,
            ad_prior,
            bd_prior,
            prior_khx_ratio,
            prior_khx_act,
        )

        overall_eval.append(e_eval)
        overall_prior.append(e_prior)

        hr0, min0 = preprocessing.local_hour_minute(hist_eval.ts_utc[k0])
        mins_of_day = hr0 * 60 + min0

        e_fh_wh = sum(
            q_fh_ol[s - 1] * (hist_eval.dt_s[k0 + s] / 3600.0)
            for s in range(1, lw + 1)
        )
        sol_slice = [i_sol_arr[k0 + s] for s in range(1, lw + 1)]
        max_sol = max(sol_slice)
        mean_sol = sum(sol_slice) / float(lw)
        max_ac = max(hist_eval.p_el_ac_kt_w[k0 + s] for s in range(1, lw + 1))
        max_desk = max(hist_eval.p_desk_kt_w[k0 + s] for s in range(1, lw + 1))

        if (
            30 <= mins_of_day <= 120
            and e_fh_wh < 100.0
            and max_sol < 10.0
            and (zone != "kt" or (max_ac <= 25.0 and max_desk <= 30.0))
        ):
            reg_eval["night_cooling"].append(e_eval)
            reg_prior["night_cooling"].append(e_prior)

        min_heat_wh = 3000.0 if zone == "wk" else 800.0
        if (
            e_fh_wh >= min_heat_wh
            and mean_sol < 150.0
            and (zone != "kt" or max_ac <= 25.0)
        ):
            reg_eval["heating_and_coast"].append(e_eval)
            reg_prior["heating_and_coast"].append(e_prior)

        if 570 <= mins_of_day <= 810 and (
            mean_sol >= 120.0 or max_sol >= 250.0
        ):
            reg_eval["sunny_afternoon"].append(e_eval)
            reg_prior["sunny_afternoon"].append(e_prior)

    if zone == "kt":
        max_k0_ac = n_slice - 1 - HORIZON_AC_1H_STEPS
        cand_k0_set: set[int] = set(
            range(spinup_steps, max_k0_ac + 1, STRIDE_AC_1H_STEPS)
        )
        for k in range(spinup_steps, max_k0_ac + 1):
            if (
                hist_eval.p_el_ac_kt_w[k] <= 80.0
                and hist_eval.p_el_ac_kt_w[k + 1] > 80.0
            ):
                cand_k0_set.add(k)

        for k0 in sorted(cand_k0_set):
            if not _is_valid_eval_start(hist_eval, zone, k0, t_meas, d_act):
                continue

            first_9_high = sum(
                1
                for s in range(1, 10)
                if round(hist_eval.ac_mode_code[k0 + s])
                == int(const.AC_MODE_HEAT)
                and hist_eval.p_el_ac_kt_w[k0 + s] > 80.0
            )
            if first_9_high < 3:
                continue

            ac_wh_9 = sum(
                hist_eval.p_el_ac_kt_w[k0 + s]
                * (hist_eval.dt_s[k0 + s] / 3600.0)
                for s in range(1, 10)
                if round(hist_eval.ac_mode_code[k0 + s])
                == int(const.AC_MODE_HEAT)
            )
            if ac_wh_9 < 40.0:
                continue

            lw_ac = _truncate_on_unmodeled_event(
                hist_eval,
                zone,
                k0,
                HORIZON_AC_1H_STEPS,
                d_act,
                tau_d_s,
                t_meas,
                is_ac_window=True,
            )
            if lw_ac < HORIZON_AC_1H_STEPS:
                continue

            e_eval_ac, q_fh_ac = kalman_tape.rollout_open_loop_window_fast(
                tape_eval,
                k0,
                HORIZON_AC_1H_STEPS,
                states_eval[k0],
                params_eval,
                ad_eval,
                bd_eval,
                eval_khx_ratio,
                eval_khx_act,
            )
            if max(q_fh_ac) >= 300.0:
                continue

            e_prior_ac, _ = kalman_tape.rollout_open_loop_window_fast(
                tape_prior,
                k0,
                HORIZON_AC_1H_STEPS,
                states_prior[k0],
                params_prior,
                ad_prior,
                bd_prior,
                prior_khx_ratio,
                prior_khx_act,
            )
            reg_eval["airco_heating_1h"].append(e_eval_ac)
            reg_prior["airco_heating_1h"].append(e_prior_ac)

    overall_metrics = _compute_regime_metrics(
        "overall_6h", HORIZON_6H_STEPS, overall_eval, overall_prior
    )
    by_regime_metrics = {
        r: _compute_regime_metrics(
            r,
            HORIZON_AC_1H_STEPS
            if r == "airco_heating_1h"
            else HORIZON_6H_STEPS,
            reg_eval[r],
            reg_prior[r],
        )
        for r in regime_names
    }

    return contracts.ZoneEvaluationReport(
        zone=zone,
        ts_evaluated_utc=int(hist_eval.ts_utc[-1]),
        overall_6h=overall_metrics,
        by_regime=by_regime_metrics,
    )


def zone_eval_report_to_dict(
    report: contracts.ZoneEvaluationReport | None,
) -> dict[str, Any] | None:
    """Converts a `ZoneEvaluationReport` into a JSON-serializable dictionary."""
    if report is None:
        return None
    return {
        "zone": report.zone,
        "ts_evaluated_utc": report.ts_evaluated_utc,
        "overall_6h": dataclasses.asdict(report.overall_6h),
        "by_regime": {
            k: dataclasses.asdict(v) for k, v in report.by_regime.items()
        },
    }


def zone_eval_report_from_dict(
    data: Mapping[str, Any] | None,
) -> contracts.ZoneEvaluationReport | None:
    """Reconstructs a `ZoneEvaluationReport` from a dictionary."""
    if not data:
        return None
    overall_raw = data["overall_6h"]
    overall = contracts.RegimeHindcastMetrics(**overall_raw)
    by_reg = {
        k: contracts.RegimeHindcastMetrics(**v)
        for k, v in data.get("by_regime", {}).items()
    }
    return contracts.ZoneEvaluationReport(
        zone="wk" if data["zone"] == "wk" else "kt",
        ts_evaluated_utc=int(data["ts_evaluated_utc"]),
        overall_6h=overall,
        by_regime=by_reg,
    )
