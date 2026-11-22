"""Exact 3x3 ZOH state estimator, Lyapunov-Taylor fast-doubling, and guards.

Pure Python 3.13 standard library implementation (zero `homeassistant.*`
imports) for Component 2 (`zoh_kalman.py`, Choice E2).
"""

from __future__ import annotations

import collections
from collections.abc import Mapping
import dataclasses
import functools
import math
from typing import Any, TypeAlias, TypeGuard

from custom_components.betonbatterij import const
from custom_components.betonbatterij import house_config
from custom_components.betonbatterij.domain import contracts
from custom_components.betonbatterij.domain import solar_weather

# Physical continuous-time process noise constants (Component 2 Section 1.1).
Q_AIR_BASE_K2_S: float = 2.5e-7
Q_FLOOR_IDLE_K2_S: float = 2.8e-8
Q_FLOOR_PUMP_K2_S: float = 2.5e-7
TAU_HYDR_S: float = 900.0
Q_AIR_OPEN_WINDOW_K2_S: float = 2.0e-5

# Measurement noise and guard constants (Component 2 Sections 3-5).
R_MEAS_FINE_K2: float = 0.03 * 0.03  # 0.01 °C ZHA sensor.
R_MEAS_COARSE_K2: float = 0.08 * 0.08  # 0.1 °C ZHA sensor.
R_MEAS_HEARTBEAT_FINE_K2: float = 0.02 * 0.02
R_MEAS_SURF_K2: float = 0.15 * 0.15

OPEN_WINDOW_SLOPE_K_PER_H: float = -0.8
OPEN_WINDOW_HOLD_S: float = 2100.0  # 35 min hold (K_floor = 0).
TAU_D_VENT_S: float = 1200.0  # 20 min accelerated decay when d_est < 0.
COLD_START_FLOOR_CAP_K: float = 0.06  # Max |dT_floor| per step in 6h start.
COLD_START_CAP_DURATION_S: float = 21_600.0
MAX_RESTART_BRIDGE_GAP_S: float = 10_800.0

_SQRT2: float = math.sqrt(2.0)
_INV_SQRT2PI: float = 1.0 / math.sqrt(2.0 * math.pi)

_Mat3: TypeAlias = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]
_Mat3x7: TypeAlias = tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
]


def _is_finite(val: float | int | None) -> TypeGuard[float | int]:
    """Checks whether `val` is a finite `float` or `int`."""
    return val is not None and math.isfinite(val)


def _matmul3(a: _Mat3, b: _Mat3) -> _Mat3:
    """Computes out-of-place 3x3 matrix product `A @ B`."""
    return (
        (
            a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1],
            a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2],
        ),
        (
            a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1],
            a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2],
        ),
        (
            a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0],
            a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1],
            a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2],
        ),
    )


def _matmul3_bt(a: _Mat3, b: _Mat3) -> _Mat3:
    """Computes out-of-place 3x3 matrix product `A @ B^T`."""
    return (
        (
            a[0][0] * b[0][0] + a[0][1] * b[0][1] + a[0][2] * b[0][2],
            a[0][0] * b[1][0] + a[0][1] * b[1][1] + a[0][2] * b[1][2],
            a[0][0] * b[2][0] + a[0][1] * b[2][1] + a[0][2] * b[2][2],
        ),
        (
            a[1][0] * b[0][0] + a[1][1] * b[0][1] + a[1][2] * b[0][2],
            a[1][0] * b[1][0] + a[1][1] * b[1][1] + a[1][2] * b[1][2],
            a[1][0] * b[2][0] + a[1][1] * b[2][1] + a[1][2] * b[2][2],
        ),
        (
            a[2][0] * b[0][0] + a[2][1] * b[0][1] + a[2][2] * b[0][2],
            a[2][0] * b[1][0] + a[2][1] * b[1][1] + a[2][2] * b[1][2],
            a[2][0] * b[2][0] + a[2][1] * b[2][1] + a[2][2] * b[2][2],
        ),
    )


def _add3(a: _Mat3, b: _Mat3) -> _Mat3:
    """Computes elementwise 3x3 sum `A + B`."""
    return (
        (a[0][0] + b[0][0], a[0][1] + b[0][1], a[0][2] + b[0][2]),
        (a[1][0] + b[1][0], a[1][1] + b[1][1], a[1][2] + b[1][2]),
        (a[2][0] + b[2][0], a[2][1] + b[2][1], a[2][2] + b[2][2]),
    )


def _scale3(a: _Mat3, c: float) -> _Mat3:
    """Scales a 3x3 matrix by scalar `c`."""
    return (
        (a[0][0] * c, a[0][1] * c, a[0][2] * c),
        (a[1][0] * c, a[1][1] * c, a[1][2] * c),
        (a[2][0] * c, a[2][1] * c, a[2][2] * c),
    )


def _sym_lyapunov_step(
    a_tilde: _Mat3,
    c_prev: _Mat3,
    scale: float,
) -> _Mat3:
    """Computes `scale * (A_tilde @ C_prev + (A_tilde @ C_prev)^T)`."""
    ac = _matmul3(a_tilde, c_prev)
    p00 = 2.0 * ac[0][0] * scale
    p11 = 2.0 * ac[1][1] * scale
    p22 = 2.0 * ac[2][2] * scale
    p01 = (ac[0][1] + ac[1][0]) * scale
    p02 = (ac[0][2] + ac[2][0]) * scale
    p12 = (ac[1][2] + ac[2][1]) * scale
    return (
        (p00, p01, p02),
        (p01, p11, p12),
        (p02, p12, p22),
    )


def _sym3(m: _Mat3) -> _Mat3:
    """Symmetrizes a 3x3 matrix `0.5 * (M + M^T)`."""
    m01 = 0.5 * (m[0][1] + m[1][0])
    m02 = 0.5 * (m[0][2] + m[2][0])
    m12 = 0.5 * (m[1][2] + m[2][1])
    return (
        (m[0][0], m01, m02),
        (m01, m[1][1], m12),
        (m02, m12, m[2][2]),
    )


def _effective_wind_excess(
    params: contracts.ZoneParameters,
    v_wind_ms: float,
) -> float:
    """Buckets wind speeds `<= 3.0 m/s` (or `c_wind == 0`) onto `0.0`."""
    if params.c_wind == 0.0 or v_wind_ms <= 3.0:
        return 0.0
    return float(v_wind_ms) - 3.0


@functools.lru_cache(maxsize=512)
def _discretize_ab_cached(
    params: contracts.ZoneParameters,
    dt_s: float,
    v_wind_excess_ms: float,
) -> tuple[_Mat3, _Mat3x7, _Mat3, int, float]:
    """Computes cached deterministic ZOH `(A_d, B_d, A_tilde_c, s, h)`."""
    c_air = 3600.0 * params.c_air_wh_k
    c_floor = 3600.0 * params.c_floor_wh_k
    h_fh = 1000.0 / params.r_fh_k_kw
    tau_d = 3600.0 * params.tau_d_h
    h_oa = params.h_loss_w_k * (1.0 + params.c_wind * v_wind_excess_ms)
    h_up = params.h_up_w_k
    h_down = params.h_down_w_k

    # Scaled A_tilde_c = S * A_c * S^-1 with S = diag(1, 1, 1e-3).
    ta00 = -(h_fh + h_oa + h_up) / c_air
    ta01 = h_fh / c_air
    ta02 = 1000.0 / c_air
    ta10 = h_fh / c_floor
    ta11 = -(h_fh + h_down) / c_floor
    ta22 = -1.0 / tau_d

    a_tilde_c: _Mat3 = (
        (ta00, ta01, ta02),
        (ta10, ta11, 0.0),
        (0.0, 0.0, ta22),
    )

    norm_inf = dt_s * max(
        abs(ta00) + abs(ta01) + abs(ta02),
        abs(ta10) + abs(ta11),
        abs(ta22),
    )
    s_steps = (
        0 if norm_inf <= 0.30 else max(0, math.ceil(math.log2(norm_inf / 0.30)))
    )
    h = dt_s / float(1 << s_steps)

    m_h = _scale3(a_tilde_c, h)
    ident: _Mat3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    a_h: _Mat3 = ident
    psi_h: _Mat3 = _scale3(ident, h)
    term: _Mat3 = ident

    for k in range(1, 13):
        term = _scale3(_matmul3(m_h, term), 1.0 / k)
        a_h = _add3(a_h, term)
        psi_h = _add3(psi_h, _scale3(term, h / (k + 1)))

    for _ in range(s_steps):
        psi_next = _add3(psi_h, _matmul3(a_h, psi_h))
        a_next = _matmul3(a_h, a_h)
        a_h, psi_h = a_next, psi_next

    # Unscale S^-1 * X * S.
    a_d: _Mat3 = (
        (a_h[0][0], a_h[0][1], a_h[0][2] * 1.0e-3),
        (a_h[1][0], a_h[1][1], a_h[1][2] * 1.0e-3),
        (0.0, 0.0, a_h[2][2]),
    )
    psi_d: _Mat3 = (
        (psi_h[0][0], psi_h[0][1], psi_h[0][2] * 1.0e-3),
        (psi_h[1][0], psi_h[1][1], psi_h[1][2] * 1.0e-3),
        (0.0, 0.0, psi_h[2][2]),
    )

    # Continuous B_c (3x7) for u = [T_out, T_up, T_down, I_sol, Q_fh, Q_int, Q_ac].
    bc0 = (
        h_oa / c_air,
        h_up / c_air,
        0.0,
        (params.alpha_sol * params.a_sol_m2) / c_air,
        0.0,
        1.0 / c_air,
        1.0 / c_air,
    )
    bc1 = (
        0.0,
        0.0,
        h_down / c_floor,
        ((1.0 - params.alpha_sol) * params.a_sol_m2) / c_floor,
        1.0 / c_floor,
        0.0,
        0.0,
    )

    b_d: _Mat3x7 = (
        tuple(psi_d[0][0] * bc0[j] + psi_d[0][1] * bc1[j] for j in range(7)),
        tuple(psi_d[1][0] * bc0[j] + psi_d[1][1] * bc1[j] for j in range(7)),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    return a_d, b_d, a_tilde_c, s_steps, h


def _lyapunov_van_loan_3x3(
    a_tilde_c: _Mat3,
    s_steps: int,
    h: float,
    c_diag_scaled: tuple[float, float, float],
    a_h_init: _Mat3 | None = None,
) -> _Mat3:
    """Integrates `Q_d` for scaled diagonal spectral density `c_diag_scaled`."""
    c_term: _Mat3 = (
        (h * c_diag_scaled[0], 0.0, 0.0),
        (0.0, h * c_diag_scaled[1], 0.0),
        (0.0, 0.0, h * c_diag_scaled[2]),
    )
    q_h: _Mat3 = c_term

    for p in range(1, 13):
        c_term = _sym_lyapunov_step(a_tilde_c, c_term, h / (p + 1))
        q_h = _add3(q_h, c_term)

    if s_steps > 0:
        if a_h_init is not None:
            a_h = a_h_init
        else:
            m_h = _scale3(a_tilde_c, h)
            ident: _Mat3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
            a_h = ident
            term: _Mat3 = ident
            for k in range(1, 13):
                term = _scale3(_matmul3(m_h, term), 1.0 / k)
                a_h = _add3(a_h, term)

        for _ in range(s_steps):
            aq = _matmul3(a_h, q_h)
            q_next = _sym3(_add3(q_h, _matmul3_bt(aq, a_h)))
            a_next = _matmul3(a_h, a_h)
            a_h, q_h = a_next, q_next

    q01 = 0.5 * (q_h[0][1] + q_h[1][0])
    q02 = 0.5 * (q_h[0][2] + q_h[2][0]) * 1.0e3
    q12 = 0.5 * (q_h[1][2] + q_h[2][1]) * 1.0e3
    return (
        (q_h[0][0], q01, q02),
        (q01, q_h[1][1], q12),
        (q02, q12, q_h[2][2] * 1.0e6),
    )


@functools.lru_cache(maxsize=512)
def _discretize_qd_basis_cached(
    params: contracts.ZoneParameters,
    dt_s: float,
    v_wind_excess_ms: float,
) -> tuple[_Mat3, _Mat3, _Mat3]:
    """Computes cached unit-basis discrete covariance matrices `(G_0, G_1, G_2)`."""
    _, _, a_tilde_c, s_steps, h = _discretize_ab_cached(
        params, dt_s, v_wind_excess_ms
    )
    a_h_base: _Mat3 | None = None
    if s_steps > 0:
        m_h = _scale3(a_tilde_c, h)
        ident: _Mat3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        a_h_base = ident
        term: _Mat3 = ident
        for k in range(1, 13):
            term = _scale3(_matmul3(m_h, term), 1.0 / k)
            a_h_base = _add3(a_h_base, term)

    g0 = _lyapunov_van_loan_3x3(
        a_tilde_c, s_steps, h, (1.0, 0.0, 0.0), a_h_base
    )
    g1 = _lyapunov_van_loan_3x3(
        a_tilde_c, s_steps, h, (0.0, 1.0, 0.0), a_h_base
    )
    g2 = _lyapunov_van_loan_3x3(
        a_tilde_c, s_steps, h, (0.0, 0.0, 1.0e-6), a_h_base
    )
    return g0, g1, g2


def _combine_qd_from_basis(
    g0: _Mat3,
    g1: _Mat3,
    g2: _Mat3,
    qc00: float,
    qc11: float,
    qc22: float,
) -> _Mat3:
    """Linearly combines `(G_0, G_1, G_2)` into symmetric `Q_d`."""
    q00 = qc00 * g0[0][0] + qc11 * g1[0][0] + qc22 * g2[0][0]
    q01 = qc00 * g0[0][1] + qc11 * g1[0][1] + qc22 * g2[0][1]
    q02 = qc00 * g0[0][2] + qc11 * g1[0][2] + qc22 * g2[0][2]
    q11 = qc00 * g0[1][1] + qc11 * g1[1][1] + qc22 * g2[1][1]
    q12 = qc00 * g0[1][2] + qc11 * g1[1][2] + qc22 * g2[1][2]
    q22 = qc00 * g0[2][2] + qc11 * g1[2][2] + qc22 * g2[2][2]
    return (
        (q00, q01, q02),
        (q01, q11, q12),
        (q02, q12, q22),
    )


def discretize_zoh_3x3(
    params: contracts.ZoneParameters,
    dt_s: float = float(const.DT_STEP_S),
    v_wind_ms: float = 0.0,
    sigma_d_w: float = 110.0,
    u_pump: float = 0.0,
    q_fh_w: float = 0.0,
    q_air_vent: float = 0.0,
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[tuple[float, ...], ...],
    tuple[tuple[float, float, float], ...],
]:
    """Computes exact simultaneous ZOH matrices `(A_d, B_d, Q_d)` for `dt_s > 0`.

    Balances coordinates via `S = diag(1, 1, 1e-3)` and combines cached unit
    Lyapunov-Taylor basis integrals `(G_0, G_1, G_2)` linearly in scalar floats.

    Args:
        params: Physical `ZoneParameters` for `wk` or `kt`.
        dt_s: Integration step in seconds (`> 0`).
        v_wind_ms: Outdoor wind speed [m/s].
        sigma_d_w: Ornstein-Uhlenbeck stationary disturbance std-dev [W].
        u_pump: Secondary floor circulation pump duty `[0..1]`.
        q_fh_w: Active floor thermal power [W] (scales hydraulic uncertainty).
        q_air_vent: Additional air process-noise spectral density [K²/s].

    Returns:
        A tuple `(a_d, b_d, q_d)` where `a_d` is `3x3`, `b_d` is `3x7`, and
        `q_d` is the symmetric `3x3` discrete process-noise covariance.
    """
    dt_clean = float(dt_s)
    if dt_clean <= 0.0:
        raise ValueError(f"Expected positive dt_s, got {dt_s=}")
    v_excess = _effective_wind_excess(params, float(v_wind_ms))
    a_d, b_d, _, _, _ = _discretize_ab_cached(params, dt_clean, v_excess)
    g0, g1, g2 = _discretize_qd_basis_cached(params, dt_clean, v_excess)

    c_floor = 3600.0 * params.c_floor_wh_k
    tau_d = 3600.0 * params.tau_d_h
    qc00 = Q_AIR_BASE_K2_S + max(0.0, float(q_air_vent))
    u_p = max(0.0, min(1.0, float(u_pump)))
    qf = float(q_fh_w)
    qc11 = (
        Q_FLOOR_IDLE_K2_S
        + u_p * (Q_FLOOR_PUMP_K2_S - Q_FLOOR_IDLE_K2_S)
        + ((0.08 * abs(qf)) ** 2) * TAU_HYDR_S / (c_floor * c_floor)
    )
    sd = float(sigma_d_w)
    qc22 = (2.0 * sd * sd) / tau_d
    q_d = _combine_qd_from_basis(g0, g1, g2, qc00, qc11, qc22)
    return a_d, b_d, q_d


discretize_zoh_step = discretize_zoh_3x3


def propagate_open_loop_step(
    state: contracts.ThermalState,
    params: contracts.ZoneParameters,
    exog: contracts.ExogenousStepInput,
    q_fh_w: float,
    q_ac_w: float = 0.0,
    dt_s: float = float(const.DT_STEP_S),
) -> contracts.ThermalState:
    """Advances `state` deterministically by one cached ZOH step.

    Args:
        state: Current `ThermalState` `[T_air, T_floor, d_est]`.
        params: Physical `ZoneParameters`.
        exog: Exogenous boundary conditions for `[t, t + dt_s)`.
        q_fh_w: Applied waterside floor thermal power [W].
        q_ac_w: Applied split-AC net thermal power [W].
        dt_s: Step duration in seconds.

    Returns:
        Updated `ThermalState` at `t + dt_s`.
    """
    v_excess = _effective_wind_excess(params, exog.v_wind_ms)
    a_d, b_d, _, _, _ = _discretize_ab_cached(params, float(dt_s), v_excess)

    u0 = exog.t_out_c
    u1 = exog.t_up_c
    u2 = exog.t_down_c
    u3 = exog.i_sol_wm2
    u4 = q_fh_w
    u5 = exog.q_int_known_w
    u6 = q_ac_w

    b0 = b_d[0]
    b1 = b_d[1]

    x0 = (
        a_d[0][0] * state.t_air_c
        + a_d[0][1] * state.t_floor_c
        + a_d[0][2] * state.d_est_w
        + b0[0] * u0
        + b0[1] * u1
        + b0[2] * u2
        + b0[3] * u3
        + b0[4] * u4
        + b0[5] * u5
        + b0[6] * u6
    )
    x1 = (
        a_d[1][0] * state.t_air_c
        + a_d[1][1] * state.t_floor_c
        + a_d[1][2] * state.d_est_w
        + b1[0] * u0
        + b1[1] * u1
        + b1[2] * u2
        + b1[3] * u3
        + b1[4] * u4
        + b1[5] * u5
        + b1[6] * u6
    )
    x2 = a_d[2][2] * state.d_est_w
    return contracts.ThermalState(t_air_c=x0, t_floor_c=x1, d_est_w=x2)


class ResolutionDetector:
    """Change-conditioned `0.01 °C` vs `0.1 °C` ZHA sensor resolution detector.

    Attributes:
        flags: Bounded history (`maxlen=12`) of distinct-sample precision flags.
        last_distinct_value: Most recent distinct temperature observation [°C].
    """

    def __init__(self) -> None:
        """Initializes an empty resolution detector."""
        self.flags: collections.deque[bool] = collections.deque(maxlen=12)
        self.last_distinct_value: float | None = None

    @property
    def is_fine_resolution(self) -> bool:
        """Whether recent distinct samples exhibit `0.01 °C` resolution."""
        return any(self.flags)

    @property
    def r_meas_k2(self) -> float:
        """Measurement noise variance `R_v` [K²] for the detected resolution."""
        return R_MEAS_FINE_K2 if self.is_fine_resolution else R_MEAS_COARSE_K2

    def observe(self, y_c: float) -> bool:
        """Records a reported temperature sample and updates resolution state.

        Args:
            y_c: Reported room air temperature [°C].

        Returns:
            Updated `is_fine_resolution` status.
        """
        val = float(y_c)
        is_fine = abs(10.0 * val - round(10.0 * val)) > 0.05
        is_distinct = (
            self.last_distinct_value is None
            or abs(val - self.last_distinct_value) >= 0.005
        )
        if is_fine or is_distinct:
            self.flags.append(is_fine)
            self.last_distinct_value = val
        return self.is_fine_resolution

    def update(self, y_c: float) -> bool:
        """Alias for `observe(y_c)`."""
        return self.observe(y_c)

    def reset(self) -> None:
        """Clears the stored resolution history."""
        self.flags.clear()
        self.last_distinct_value = None

    def to_dict(self) -> dict[str, Any]:
        """Serializes detector state to a JSON-compatible dictionary."""
        return {
            "flags": list(self.flags),
            "last_distinct_value": self.last_distinct_value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ResolutionDetector:
        """Restores a `ResolutionDetector` from a persisted dictionary."""
        det = cls()
        raw_flags = data.get("flags")
        if isinstance(raw_flags, list):
            for fl in raw_flags[-12:]:
                det.flags.append(bool(fl))
        ldv = data.get("last_distinct_value")
        if _is_finite(ldv):
            det.last_distinct_value = float(ldv)
        return det


def _norm_pdf(z: float) -> float:
    """Evaluates the standard normal probability density `phi(z)`."""
    return _INV_SQRT2PI * math.exp(-0.5 * z * z)


def _norm_cdf(z: float) -> float:
    """Evaluates the standard normal cumulative distribution `Phi(z)`."""
    return 0.5 * (1.0 + math.erf(z / _SQRT2))


def _joseph_scalar_update(
    x_minus: tuple[float, float, float],
    p_minus: list[list[float]],
    h_row: tuple[float, float, float],
    innovation: float,
    r_v: float,
    gain_override: tuple[float, float, float] | None = None,
) -> tuple[tuple[float, float, float], list[list[float]]]:
    """Applies an unconditionally SPD Joseph-form scalar Kalman update."""
    if gain_override is not None:
        k0, k1, k2 = gain_override
    else:
        ph0 = (
            p_minus[0][0] * h_row[0]
            + p_minus[0][1] * h_row[1]
            + p_minus[0][2] * h_row[2]
        )
        ph1 = (
            p_minus[1][0] * h_row[0]
            + p_minus[1][1] * h_row[1]
            + p_minus[1][2] * h_row[2]
        )
        ph2 = (
            p_minus[2][0] * h_row[0]
            + p_minus[2][1] * h_row[1]
            + p_minus[2][2] * h_row[2]
        )
        s_val = max(
            1e-12, h_row[0] * ph0 + h_row[1] * ph1 + h_row[2] * ph2 + r_v
        )
        k0 = ph0 / s_val
        k1 = ph1 / s_val
        k2 = ph2 / s_val

    x_plus = (
        x_minus[0] + k0 * innovation,
        x_minus[1] + k1 * innovation,
        x_minus[2] + k2 * innovation,
    )

    k_vec = (k0, k1, k2)
    w_mat: _Mat3 = (
        (1.0 - k0 * h_row[0], -k0 * h_row[1], -k0 * h_row[2]),
        (-k1 * h_row[0], 1.0 - k1 * h_row[1], -k1 * h_row[2]),
        (-k2 * h_row[0], -k2 * h_row[1], 1.0 - k2 * h_row[2]),
    )
    p_tup: _Mat3 = (
        (p_minus[0][0], p_minus[0][1], p_minus[0][2]),
        (p_minus[1][0], p_minus[1][1], p_minus[1][2]),
        (p_minus[2][0], p_minus[2][1], p_minus[2][2]),
    )
    wp = _matmul3(w_mat, p_tup)
    wpwt = _matmul3_bt(wp, w_mat)

    p_plus: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(3)]
    for i in range(3):
        for j in range(i, 3):
            val = 0.5 * (wpwt[i][j] + wpwt[j][i]) + r_v * k_vec[i] * k_vec[j]
            if i == j and val < 1e-14:
                val = 1e-14
            p_plus[i][j] = val
            p_plus[j][i] = val

    return x_plus, p_plus


def _tobit_equivalent_innovation(
    y_center: float,
    half_width_k: float,
    mu_minus: float,
    p00_minus: float,
    sigma_eps_k: float = 0.03,
) -> tuple[float, float] | None:
    """Computes moment-matched `(e_eff, r_eff)` for `y` in deadband interval."""
    var_y = max(1e-10, p00_minus + sigma_eps_k * sigma_eps_k)
    sigma_y = math.sqrt(var_y)
    alpha = ((y_center - half_width_k) - mu_minus) / sigma_y
    beta = ((y_center + half_width_k) - mu_minus) / sigma_y

    z_mass = _norm_cdf(beta) - _norm_cdf(alpha)
    if z_mass >= 1e-11:
        phi_a = _norm_pdf(alpha)
        phi_b = _norm_pdf(beta)
        delta_mu = (phi_a - phi_b) / z_mass
        gamma_var = (
            1.0
            + (alpha * phi_a - beta * phi_b) / z_mass
            - delta_mu * delta_mu
        )
        gamma_var = max(0.02, min(0.999, gamma_var))
        one_minus_g = 1.0 - gamma_var
        s_eff = var_y / one_minus_g
        r_eff = max(sigma_eps_k * sigma_eps_k, s_eff - p00_minus)
        e_eff = (sigma_y * delta_mu) / one_minus_g
        return e_eff, r_eff

    if mu_minus > y_center + half_width_k:
        return (y_center + half_width_k) - mu_minus, 0.05 * 0.05
    if mu_minus < y_center - half_width_k:
        return (y_center - half_width_k) - mu_minus, 0.05 * 0.05
    return None


class ZoneKalmanEstimator:
    """Choice E2 regime-adaptive 3x3 ZOH Kalman estimator for `wk` or `kt`.

    Attributes:
        zone: Modeled zone identifier (`"wk"` or `"kt"`).
        params: Active physical `ZoneParameters`.
        state: Posterior `ThermalState` `[T_air, T_floor, d_est]`.
        cov_p: Posterior `3x3` SPD error covariance matrix.
        initialized: True once `cold_start()` or first `step()` has run.
        open_window_active: True while the `K_floor = 0` open-window gate holds.
        sensor_stale: True when ZHA room sensor liveness exceeded threshold.
        resolution_detector: ZHA `0.01 °C` vs `0.1 °C` detector instance.
    """

    def __init__(
        self,
        zone: contracts.ZoneId,
        params: contracts.ZoneParameters,
    ) -> None:
        """Initializes an unseeded `ZoneKalmanEstimator` for `zone`.

        Args:
            zone: Target zone identifier (`"wk"` or `"kt"`).
            params: Initial physical `ZoneParameters`.
        """
        self.zone: contracts.ZoneId = zone
        self.params: contracts.ZoneParameters = params
        self.state: contracts.ThermalState = contracts.ThermalState(
            t_air_c=19.5, t_floor_c=20.5, d_est_w=0.0
        )
        self.cov_p: list[list[float]] = [
            [R_MEAS_COARSE_K2, 0.0, 0.0],
            [0.0, 0.25, 0.0],
            [0.0, 0.0, 6400.0],
        ]
        self.initialized: bool = False
        self.open_window_active: bool = False
        self.sensor_stale: bool = False
        self.resolution_detector: ResolutionDetector = ResolutionDetector()

        self._open_window_hold_rem_s: float = 0.0
        self._cold_start_cap_rem_s: float = 0.0
        self._time_since_temp_change_s: float = 0.0
        self._last_meas_temp_c: float | None = None
        self._air_obs_history: collections.deque[tuple[int, float]] = (
            collections.deque(maxlen=12)
        )

    @property
    def t_floor_std_c(self) -> float:
        """One-sigma uncertainty `sqrt(P_11)` [°C] of the concrete slab core."""
        return math.sqrt(max(0.0, self.cov_p[1][1]))

    def _select_sigma_d_w(self, ts_utc: int, aanwezig: bool) -> float:
        """Selects context-adaptive disturbance scale `sigma_d(t)` [W]."""
        hr, mn = solar_weather.local_hour_minute(
            ts_utc, house_config.TIMEZONE_NAME
        )
        mins = hr * 60 + mn
        if not aanwezig or 30 <= mins < 390:  # 00:30..06:30 local time.
            return 20.0
        if self.zone == "wk" and 1020 <= mins < 1230:  # 17:00..20:30 local.
            return 280.0
        return 110.0 if self.zone == "wk" else 75.0

    def cold_start(
        self,
        t_air_c: float,
        exog: contracts.ExogenousStepInput,
        q_fh_w: float = 0.0,
        q_ac_w: float = 0.0,
        u_pump: float = 0.0,
        t_buf_top_c: float | None = None,
    ) -> None:
        """Initializes latent slab temperature on the slow eigenmode manifold.

        Args:
            t_air_c: Initial measured room air temperature [°C].
            exog: Current exogenous boundary conditions.
            q_fh_w: Current waterside floor power [W].
            q_ac_w: Current split-AC thermal power [W].
            u_pump: Secondary circulation pump duty `[0..1]`.
            t_buf_top_c: Optional 200 L buffer top temperature [°C] used when
                starting during an active floor-pump run.
        """
        t_air = float(t_air_c)
        h_fh = 1000.0 / self.params.r_fh_k_kw
        v_excess = _effective_wind_excess(self.params, exog.v_wind_ms)
        h_oa = self.params.h_loss_w_k * (1.0 + self.params.c_wind * v_excess)
        h_up = self.params.h_up_w_k
        h_down = self.params.h_down_w_k

        q_air_loss = (
            h_oa * (t_air - exog.t_out_c)
            + h_up * (t_air - exog.t_up_c)
            - exog.q_int_known_w
            - q_ac_w
            - self.params.alpha_sol * self.params.a_sol_m2 * exog.i_sol_wm2
        )
        c_tot_wh_k = self.params.c_air_wh_k + self.params.c_floor_wh_k
        q_tot_net = (
            q_air_loss
            + h_down * (t_air - exog.t_down_c)
            - (1.0 - self.params.alpha_sol)
            * self.params.a_sol_m2
            * exog.i_sol_wm2
        )
        c_air_dot_t_slow = -(self.params.c_air_wh_k / c_tot_wh_k) * q_tot_net
        t_floor_passive = t_air + (q_air_loss + c_air_dot_t_slow) / h_fh

        if u_pump <= 0.5:
            t_floor_0 = t_floor_passive
            p11_0 = 0.50 * 0.50
            self._cold_start_cap_rem_s = 0.0
        else:
            q_sol_floor = (
                (1.0 - self.params.alpha_sol)
                * self.params.a_sol_m2
                * exog.i_sol_wm2
            )
            if _is_finite(t_buf_top_c):
                k_hx = self.params.k_hx_w_k * min(1.0, float(u_pump))
                t_floor_pipe = (
                    k_hx * float(t_buf_top_c)
                    + h_fh * t_air
                    + h_down * exog.t_down_c
                    + q_sol_floor
                ) / (k_hx + h_fh + h_down)
            else:
                t_floor_pipe = (
                    float(q_fh_w)
                    + h_fh * t_air
                    + h_down * exog.t_down_c
                    + q_sol_floor
                ) / (h_fh + h_down)
            t_floor_0 = 0.5 * (t_floor_passive + t_floor_pipe)
            p11_0 = 1.00 * 1.00
            self._cold_start_cap_rem_s = COLD_START_CAP_DURATION_S

        self.resolution_detector.observe(t_air)
        r_v0 = self.resolution_detector.r_meas_k2
        self.state = contracts.ThermalState(
            t_air_c=t_air, t_floor_c=t_floor_0, d_est_w=0.0
        )
        self.cov_p = [
            [r_v0, 0.0, 0.0],
            [0.0, p11_0, 0.0],
            [0.0, 0.0, 80.0 * 80.0],
        ]
        self.initialized = True
        self.open_window_active = False
        self.sensor_stale = False
        self._open_window_hold_rem_s = 0.0
        self._time_since_temp_change_s = 0.0
        self._last_meas_temp_c = t_air
        self._air_obs_history.clear()

    def _observed_cooling_rate_k_per_h(
        self,
        ts_utc: int,
        t_meas: float,
    ) -> float:
        """Computes 15-min (`900 s`) finite-difference air cooling rate [K/h]."""
        if not self._air_obs_history:
            return 0.0
        target_ts = ts_utc - 900
        best_ts, best_temp = self._air_obs_history[0]
        best_diff = abs(best_ts - target_ts)
        for h_ts, h_temp in self._air_obs_history:
            if h_ts >= ts_utc:
                continue
            diff = abs(h_ts - target_ts)
            if diff <= best_diff:
                best_diff = diff
                best_ts, best_temp = h_ts, h_temp

        dt_hist_s = ts_utc - best_ts
        if dt_hist_s <= 0 or dt_hist_s > 1800:
            return 0.0
        denom_h = max(0.25, dt_hist_s / 3600.0)
        return (t_meas - best_temp) / denom_h

    def _constrained_air_gain(
        self,
        p_minus: list[list[float]],
        r_v: float,
        innovation: float,
    ) -> tuple[float, float, float]:
        """Computes gain vector `K` for `H = [1, 0, 0]` with safety guards."""
        s_val = max(1e-12, p_minus[0][0] + r_v)
        k0 = p_minus[0][0] / s_val
        k1 = 0.0 if self.open_window_active else p_minus[1][0] / s_val
        k2 = p_minus[2][0] / s_val

        if (
            self._cold_start_cap_rem_s > 0.0
            and abs(innovation) > 1e-12
            and abs(k1 * innovation) > COLD_START_FLOOR_CAP_K
        ):
            k1 = math.copysign(COLD_START_FLOOR_CAP_K / abs(innovation), k1)

        return k0, k1, k2

    def step(
        self,
        ts_utc: int,
        dt_s: float,
        exog: contracts.ExogenousStepInput,
        q_fh_w: float,
        q_ac_w: float,
        u_pump: float,
        t_air_meas_c: float | None,
        temp_changed: bool = True,
        hum_changed: bool = False,
        sensor_stale: bool = False,
        aanwezig: bool = True,
        t_surf_meas_c: float | None = None,
    ) -> contracts.ThermalState:
        """Advances state and covariance by one Choice E2 ZOH Kalman step.

        Args:
            ts_utc: Window-end Unix timestamp in seconds (UTC).
            dt_s: Step duration in seconds.
            exog: Exogenous boundary conditions over the window.
            q_fh_w: Applied waterside floor thermal power [W].
            q_ac_w: Applied split-AC thermal power [W].
            u_pump: Secondary circulation pump duty `[0..1]`.
            t_air_meas_c: Window-end room air sensor value [°C], or `None`.
            temp_changed: True if the ZHA temperature entity changed value.
            hum_changed: True if the paired ZHA humidity entity changed value.
            sensor_stale: True if device-wide ZHA liveness exceeded 7200 s.
            aanwezig: Household presence flag (`input_boolean.aanwezig`).
            t_surf_meas_c: Optional PVC floor surface sensor reading [°C].

        Returns:
            Updated posterior `ThermalState`.
        """
        step_s = max(1.0, float(dt_s))

        if not self.initialized:
            if not sensor_stale and _is_finite(t_air_meas_c):
                t_init = float(t_air_meas_c)
            else:
                t_init = self.state.t_air_c
            self.cold_start(
                t_air_c=t_init,
                exog=exog,
                q_fh_w=q_fh_w,
                q_ac_w=q_ac_w,
                u_pump=u_pump,
            )

        if not self._air_obs_history and self._last_meas_temp_c is not None:
            self._air_obs_history.append(
                (ts_utc - int(round(step_s)), self._last_meas_temp_c)
            )

        self.sensor_stale = bool(sensor_stale)

        if self._open_window_hold_rem_s > 0.0:
            self._open_window_hold_rem_s = max(
                0.0, self._open_window_hold_rem_s - step_s
            )
        self.open_window_active = self._open_window_hold_rem_s > 0.0

        use_fast_vent_decay = self.state.d_est_w < -1.0 and (
            self.open_window_active or self.state.d_est_w < -150.0
        )
        eff_params = (
            dataclasses.replace(self.params, tau_d_h=TAU_D_VENT_S / 3600.0)
            if use_fast_vent_decay
            else self.params
        )

        sigma_d = self._select_sigma_d_w(ts_utc, aanwezig)
        q_air_vent = Q_AIR_OPEN_WINDOW_K2_S if self.open_window_active else 0.0

        a_d, b_d, q_d = discretize_zoh_3x3(
            params=eff_params,
            dt_s=step_s,
            v_wind_ms=exog.v_wind_ms,
            sigma_d_w=sigma_d,
            u_pump=u_pump,
            q_fh_w=q_fh_w,
            q_air_vent=q_air_vent,
        )

        # 1. Prior state and covariance prediction.
        if use_fast_vent_decay:
            v_excess = _effective_wind_excess(self.params, exog.v_wind_ms)
            a_d_nom, b_d_nom, _, _, _ = _discretize_ab_cached(
                self.params, step_s, v_excess
            )
            a_d_floor_row = a_d_nom[1]
            b_d_floor_row = b_d_nom[1]
        else:
            a_d_floor_row = a_d[1]
            b_d_floor_row = b_d[1]

        u0 = exog.t_out_c
        u1 = exog.t_up_c
        u2 = exog.t_down_c
        u3 = exog.i_sol_wm2
        u4 = q_fh_w
        u5 = exog.q_int_known_w
        u6 = q_ac_w
        b0 = b_d[0]
        bf = b_d_floor_row
        x0_old = self.state.t_air_c
        x1_old = self.state.t_floor_c
        x2_old = self.state.d_est_w
        x_minus = (
            a_d[0][0] * x0_old
            + a_d[0][1] * x1_old
            + a_d[0][2] * x2_old
            + b0[0] * u0
            + b0[1] * u1
            + b0[2] * u2
            + b0[3] * u3
            + b0[4] * u4
            + b0[5] * u5
            + b0[6] * u6,
            a_d_floor_row[0] * x0_old
            + a_d_floor_row[1] * x1_old
            + a_d_floor_row[2] * x2_old
            + bf[0] * u0
            + bf[1] * u1
            + bf[2] * u2
            + bf[3] * u3
            + bf[4] * u4
            + bf[5] * u5
            + bf[6] * u6,
            a_d[2][2] * x2_old,
        )

        p_tup: _Mat3 = (
            (self.cov_p[0][0], self.cov_p[0][1], self.cov_p[0][2]),
            (self.cov_p[1][0], self.cov_p[1][1], self.cov_p[1][2]),
            (self.cov_p[2][0], self.cov_p[2][1], self.cov_p[2][2]),
        )
        ap = _matmul3(a_d, p_tup)
        apat = _matmul3_bt(ap, a_d)
        p_minus_tup = _sym3(_add3(apat, q_d))
        p_minus = [
            list(p_minus_tup[0]),
            list(p_minus_tup[1]),
            list(p_minus_tup[2]),
        ]

        # 2. Primary room air measurement or deadband pseudo-update.
        x_plus = x_minus
        p_plus = p_minus

        if not sensor_stale and _is_finite(t_air_meas_c):
            y_meas = float(t_air_meas_c)
            if temp_changed:
                self._time_since_temp_change_s = 0.0
                self._last_meas_temp_c = y_meas
                self.resolution_detector.observe(y_meas)
                r_v = self.resolution_detector.r_meas_k2

                e_k = y_meas - x_minus[0]
                s_k = p_minus[0][0] + r_v
                obs_slope = self._observed_cooling_rate_k_per_h(ts_utc, y_meas)
                gate_thresh = max(0.20, 2.5 * math.sqrt(max(1e-12, s_k)))

                if obs_slope < OPEN_WINDOW_SLOPE_K_PER_H and e_k < -gate_thresh:
                    self._open_window_hold_rem_s = OPEN_WINDOW_HOLD_S
                    self.open_window_active = True

                gain = self._constrained_air_gain(p_minus, r_v, e_k)
                x_plus, p_plus = _joseph_scalar_update(
                    x_minus,
                    p_minus,
                    h_row=(1.0, 0.0, 0.0),
                    innovation=e_k,
                    r_v=r_v,
                    gain_override=gain,
                )
            else:
                self._time_since_temp_change_s += step_s
                y_ref = (
                    self._last_meas_temp_c
                    if self._last_meas_temp_c is not None
                    else y_meas
                )

                if not self.resolution_detector.is_fine_resolution:
                    tobit = _tobit_equivalent_innovation(
                        y_center=y_ref,
                        half_width_k=0.10,
                        mu_minus=x_minus[0],
                        p00_minus=p_minus[0][0],
                        sigma_eps_k=0.03,
                    )
                    if tobit is not None:
                        e_eff, r_eff = tobit
                        gain = self._constrained_air_gain(
                            p_minus, r_eff, e_eff
                        )
                        x_plus, p_plus = _joseph_scalar_update(
                            x_minus,
                            p_minus,
                            h_row=(1.0, 0.0, 0.0),
                            innovation=e_eff,
                            r_v=r_eff,
                            gain_override=gain,
                        )
                elif hum_changed and self._time_since_temp_change_s >= 1800.0:
                    e_hb = y_ref - x_minus[0]
                    gain = self._constrained_air_gain(
                        p_minus, R_MEAS_HEARTBEAT_FINE_K2, e_hb
                    )
                    x_plus, p_plus = _joseph_scalar_update(
                        x_minus,
                        p_minus,
                        h_row=(1.0, 0.0, 0.0),
                        innovation=e_hb,
                        r_v=R_MEAS_HEARTBEAT_FINE_K2,
                        gain_override=gain,
                    )

            self._air_obs_history.append((ts_utc, y_meas))
        else:
            self._time_since_temp_change_s += step_s

        # 3. Optional 2nd sequential Joseph update for PVC surface sensor.
        if _is_finite(t_surf_meas_c):
            y_surf = float(t_surf_meas_c)
            area_m2 = (
                house_config.AREA_WK_M2
                if self.zone == "wk"
                else house_config.AREA_KT_M2
            )
            r_surf_k_kw = 1000.0 / (
                house_config.H_SURF_UPWARD_W_M2K * area_m2
            )
            gamma_surf = max(
                0.10, min(0.95, r_surf_k_kw / self.params.r_fh_k_kw)
            )
            h_surf = (1.0 - gamma_surf, gamma_surf, 0.0)
            y_surf_hat = h_surf[0] * x_plus[0] + h_surf[1] * x_plus[1]
            e_surf = y_surf - y_surf_hat
            x_plus, p_plus = _joseph_scalar_update(
                x_plus,
                p_plus,
                h_row=h_surf,
                innovation=e_surf,
                r_v=R_MEAS_SURF_K2,
            )

        if self._cold_start_cap_rem_s > 0.0:
            self._cold_start_cap_rem_s = max(
                0.0, self._cold_start_cap_rem_s - step_s
            )

        self.state = contracts.ThermalState(
            t_air_c=x_plus[0], t_floor_c=x_plus[1], d_est_w=x_plus[2]
        )
        self.cov_p = p_plus
        return self.state

    def bridge_restart_gap(
        self,
        ts_utc: int,
        dt_gap_s: float,
        exog_saved: contracts.ExogenousStepInput,
        exog_now: contracts.ExogenousStepInput,
        q_fh_saved_w: float,
        q_fh_now_w: float,
        q_ac_saved_w: float,
        q_ac_now_w: float,
        u_pump_avg: float,
        t_air_meas_c: float | None,
    ) -> contracts.ThermalState:
        """Bridges a Home Assistant restart gap (`<= 3 h`) via variable-dt ZOH.

        Uses trapezoidal averages of saved and current exogenous/actuator inputs
        for gaps up to `10800 s`, and falls back to `cold_start()` on longer
        outages.

        Args:
            ts_utc: Current Unix timestamp in seconds (UTC).
            dt_gap_s: Elapsed outage duration in seconds.
            exog_saved: Exogenous inputs saved prior to shutdown.
            exog_now: Exogenous inputs measured upon startup.
            q_fh_saved_w: Waterside floor power prior to shutdown [W].
            q_fh_now_w: Waterside floor power upon startup [W].
            q_ac_saved_w: Split-AC thermal power prior to shutdown [W].
            q_ac_now_w: Split-AC thermal power upon startup [W].
            u_pump_avg: Estimated average pump duty across the gap `[0..1]`.
            t_air_meas_c: Current room air temperature reading [°C], or `None`.

        Returns:
            Updated posterior `ThermalState` after bridging the gap.
        """
        gap_s = float(dt_gap_s)
        if gap_s <= 0.0:
            return self.state

        if not self.initialized or gap_s > MAX_RESTART_BRIDGE_GAP_S:
            t_init = (
                float(t_air_meas_c)
                if _is_finite(t_air_meas_c)
                else self.state.t_air_c
            )
            self.cold_start(
                t_air_c=t_init,
                exog=exog_now,
                q_fh_w=q_fh_now_w,
                q_ac_w=q_ac_now_w,
                u_pump=u_pump_avg,
            )
            return self.state

        exog_mid = contracts.ExogenousStepInput(
            t_out_c=0.5 * (exog_saved.t_out_c + exog_now.t_out_c),
            v_wind_ms=0.5 * (exog_saved.v_wind_ms + exog_now.v_wind_ms),
            i_sol_wm2=0.5 * (exog_saved.i_sol_wm2 + exog_now.i_sol_wm2),
            t_down_c=0.5 * (exog_saved.t_down_c + exog_now.t_down_c),
            q_int_known_w=0.5
            * (exog_saved.q_int_known_w + exog_now.q_int_known_w),
            t_up_c=0.5 * (exog_saved.t_up_c + exog_now.t_up_c),
        )
        q_fh_mid = 0.5 * (float(q_fh_saved_w) + float(q_fh_now_w))
        q_ac_mid = 0.5 * (float(q_ac_saved_w) + float(q_ac_now_w))

        sigma_d = self._select_sigma_d_w(ts_utc, aanwezig=True)
        a_d, b_d, q_d = discretize_zoh_3x3(
            params=self.params,
            dt_s=gap_s,
            v_wind_ms=exog_mid.v_wind_ms,
            sigma_d_w=sigma_d,
            u_pump=u_pump_avg,
            q_fh_w=q_fh_mid,
            q_air_vent=0.0,
        )

        u0 = exog_mid.t_out_c
        u1 = exog_mid.t_up_c
        u2 = exog_mid.t_down_c
        u3 = exog_mid.i_sol_wm2
        u4 = q_fh_mid
        u5 = exog_mid.q_int_known_w
        u6 = q_ac_mid
        b0 = b_d[0]
        b1 = b_d[1]
        x0_old = self.state.t_air_c
        x1_old = self.state.t_floor_c
        x2_old = self.state.d_est_w
        x_minus = (
            a_d[0][0] * x0_old
            + a_d[0][1] * x1_old
            + a_d[0][2] * x2_old
            + b0[0] * u0
            + b0[1] * u1
            + b0[2] * u2
            + b0[3] * u3
            + b0[4] * u4
            + b0[5] * u5
            + b0[6] * u6,
            a_d[1][0] * x0_old
            + a_d[1][1] * x1_old
            + a_d[1][2] * x2_old
            + b1[0] * u0
            + b1[1] * u1
            + b1[2] * u2
            + b1[3] * u3
            + b1[4] * u4
            + b1[5] * u5
            + b1[6] * u6,
            a_d[2][2] * x2_old,
        )

        p_tup: _Mat3 = (
            (self.cov_p[0][0], self.cov_p[0][1], self.cov_p[0][2]),
            (self.cov_p[1][0], self.cov_p[1][1], self.cov_p[1][2]),
            (self.cov_p[2][0], self.cov_p[2][1], self.cov_p[2][2]),
        )
        p_minus_tup = _sym3(_add3(_matmul3_bt(_matmul3(a_d, p_tup), a_d), q_d))
        p_minus = [
            list(p_minus_tup[0]),
            list(p_minus_tup[1]),
            list(p_minus_tup[2]),
        ]

        self._open_window_hold_rem_s = max(
            0.0, self._open_window_hold_rem_s - gap_s
        )
        self.open_window_active = self._open_window_hold_rem_s > 0.0
        self._cold_start_cap_rem_s = max(
            0.0, self._cold_start_cap_rem_s - gap_s
        )
        self._air_obs_history.clear()

        if _is_finite(t_air_meas_c):
            y_meas = float(t_air_meas_c)
            self.resolution_detector.observe(y_meas)
            r_v = self.resolution_detector.r_meas_k2
            e_k = y_meas - x_minus[0]
            x_plus, p_plus = _joseph_scalar_update(
                x_minus,
                p_minus,
                h_row=(1.0, 0.0, 0.0),
                innovation=e_k,
                r_v=r_v,
            )
            self._last_meas_temp_c = y_meas
            self._time_since_temp_change_s = 0.0
            self._air_obs_history.append((ts_utc, y_meas))
        else:
            x_plus, p_plus = x_minus, p_minus
            self._time_since_temp_change_s += gap_s

        self.state = contracts.ThermalState(
            t_air_c=x_plus[0], t_floor_c=x_plus[1], d_est_w=x_plus[2]
        )
        self.cov_p = p_plus
        return self.state

    def to_dict(self) -> dict[str, Any]:
        """Serializes estimator state and covariance for `Store` persistence."""
        return {
            "zone": self.zone,
            "params": dataclasses.asdict(self.params),
            "state": dataclasses.asdict(self.state),
            "cov_p": [
                [float(self.cov_p[i][j]) for j in range(3)] for i in range(3)
            ],
            "initialized": self.initialized,
            "open_window_active": self.open_window_active,
            "sensor_stale": self.sensor_stale,
            "open_window_hold_rem_s": self._open_window_hold_rem_s,
            "cold_start_cap_rem_s": self._cold_start_cap_rem_s,
            "time_since_temp_change_s": self._time_since_temp_change_s,
            "last_meas_temp_c": self._last_meas_temp_c,
            "air_obs_history": [
                [int(ts), float(val)] for ts, val in self._air_obs_history
            ],
            "resolution_detector": self.resolution_detector.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ZoneKalmanEstimator:
        """Restores a `ZoneKalmanEstimator` from a persisted dictionary.

        Args:
            data: Dictionary produced by `to_dict()`.

        Returns:
            Restored `ZoneKalmanEstimator` instance.

        Raises:
            ValueError: If `data` does not contain a valid `params` mapping.
        """
        zone_raw = str(data.get("zone", "wk"))
        zone: contracts.ZoneId = "kt" if zone_raw == "kt" else "wk"

        p_raw = data.get("params")
        if isinstance(p_raw, Mapping):
            params = contracts.ZoneParameters(
                c_air_wh_k=float(p_raw["c_air_wh_k"]),
                c_floor_wh_k=float(p_raw["c_floor_wh_k"]),
                r_fh_k_kw=float(p_raw["r_fh_k_kw"]),
                h_loss_w_k=float(p_raw["h_loss_w_k"]),
                h_down_w_k=float(p_raw["h_down_w_k"]),
                alpha_sol=float(p_raw["alpha_sol"]),
                a_sol_m2=float(p_raw["a_sol_m2"]),
                tau_d_h=float(p_raw["tau_d_h"]),
                eta_ac=float(p_raw["eta_ac"]),
                k_hx_w_k=float(p_raw["k_hx_w_k"]),
                h_up_w_k=float(p_raw.get("h_up_w_k", 0.0)),
                c_wind=float(p_raw.get("c_wind", 0.0)),
            )
        else:
            raise ValueError(
                "Missing 'params' mapping in ZoneKalmanEstimator.from_dict."
            )

        est = cls(zone=zone, params=params)

        s_raw = data.get("state")
        if isinstance(s_raw, Mapping):
            est.state = contracts.ThermalState(
                t_air_c=float(s_raw["t_air_c"]),
                t_floor_c=float(s_raw["t_floor_c"]),
                d_est_w=float(s_raw["d_est_w"]),
            )

        cov_raw = data.get("cov_p")
        if isinstance(cov_raw, list) and len(cov_raw) == 3:
            est.cov_p = [
                [float(cov_raw[i][j]) for j in range(3)] for i in range(3)
            ]

        est.initialized = bool(data.get("initialized", False))
        est.open_window_active = bool(data.get("open_window_active", False))
        est.sensor_stale = bool(data.get("sensor_stale", False))
        est._open_window_hold_rem_s = float(
            data.get("open_window_hold_rem_s", 0.0)
        )
        est._cold_start_cap_rem_s = float(
            data.get("cold_start_cap_rem_s", 0.0)
        )
        est._time_since_temp_change_s = float(
            data.get("time_since_temp_change_s", 0.0)
        )

        last_m = data.get("last_meas_temp_c")
        if _is_finite(last_m):
            est._last_meas_temp_c = float(last_m)

        hist_raw = data.get("air_obs_history")
        if isinstance(hist_raw, list):
            for pair in hist_raw[-12:]:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    est._air_obs_history.append((int(pair[0]), float(pair[1])))

        det_raw = data.get("resolution_detector")
        if isinstance(det_raw, Mapping):
            est.resolution_detector = ResolutionDetector.from_dict(det_raw)

        return est
