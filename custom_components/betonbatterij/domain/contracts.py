"""Immutable domain contracts for Layer 1 Digital Twin and Layer 2 API.

Zero `homeassistant.*` imports are allowed inside `domain/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

ZoneId: TypeAlias = Literal["wk", "kt"]


@dataclass(frozen=True, slots=True)
class ParameterBounds:
    """Box bounds and Day-1 Gaussian prior for a calibrated scalar parameter.

    Attributes:
        lower: Hard lower box bound.
        upper: Hard upper box bound.
        prior_mean: Day-1 physical prior mean.
        prior_std: Day-1 Gaussian 1-sigma uncertainty.
    """

    lower: float
    upper: float
    prior_mean: float
    prior_std: float


@dataclass(frozen=True, slots=True)
class ThermalState:
    """Three-state thermal vector `x_i = [T_air, T_floor, d_est]^T`.

    Attributes:
        t_air_c: Estimated or measured room air temperature [°C].
        t_floor_c: Estimated concrete slab core temperature [°C].
        d_est_w: Unmodeled air heat disturbance [W] (OU decay `tau_d = 1.8 h`).
    """

    t_air_c: float
    t_floor_c: float
    d_est_w: float


@dataclass(frozen=True, slots=True)
class ZoneParameters:
    """Physical 2R2C, interfloor, and actuator parameters for a modeled zone.

    Attributes:
        c_air_wh_k: Fixed air plus furniture capacity [Wh/K].
        c_floor_wh_k: Calibrated concrete slab capacity [Wh/K].
        r_fh_k_kw: Calibrated slab-to-air thermal resistance [K/kW].
        h_loss_w_k: Calibrated exterior shell conductance at <= 3 m/s [W/K].
        h_down_w_k: Downward conductance [W/K].
        alpha_sol: Direct solar fraction entering the air node [-].
        a_sol_m2: Calibrated effective solar aperture [m²].
        tau_d_h: Ornstein-Uhlenbeck disturbance decay time constant [h].
        eta_ac: Net split-AC thermal efficiency at desk height @ 7 °C [-].
        k_hx_w_k: Calibrated floor pipe heat transfer coefficient [W/K].
        h_up_w_k: Upward ceiling conductance [W/K].
        c_wind: Wind sensitivity factor above 3.0 m/s [s/m].
    """

    c_air_wh_k: float
    c_floor_wh_k: float
    r_fh_k_kw: float
    h_loss_w_k: float
    h_down_w_k: float
    alpha_sol: float
    a_sol_m2: float
    tau_d_h: float
    eta_ac: float
    k_hx_w_k: float
    h_up_w_k: float = 0.0
    c_wind: float = 0.0


@dataclass(frozen=True, slots=True)
class CopParameters:
    """Calibrated Panasonic 9 kW T-CAP COP curve parameters (Component 6).

    Attributes:
        eta_carnot_cv: Space-heating Carnot efficiency factor [-].
        beta_defrost: Defrost COP penalty amplitude near +1 °C [-].
        eta_carnot_dhw: Domestic hot water Carnot efficiency factor [-].
        delta_p_wp_bias_w: Additive bias correction on WP electrical power [W].
    """

    eta_carnot_cv: float = 0.48
    beta_defrost: float = 0.18
    eta_carnot_dhw: float = 0.34
    delta_p_wp_bias_w: float = 145.0


@dataclass(frozen=True, slots=True)
class ZoneSnapshot:
    """Real-time O(1) readout of a modeled zone for HA sensors and Layer 2."""

    zone: ZoneId
    ts_utc: int
    state: ThermalState
    t_surface_est_c: float
    t_air_meas_c: float | None
    t_floor_std_c: float
    q_fh_now_w: float
    t_hindcast_6h_c: float | None
    sensor_stale: bool
    open_window_active: bool
    stored_slab_energy_wh: float
    params: ZoneParameters
    rmse_6h_c: float | None


@dataclass(frozen=True, slots=True)
class ExogenousStepInput:
    """Exogenous environmental boundary conditions for a 5-min ZOH step."""

    t_out_c: float
    v_wind_ms: float = 3.0
    i_sol_wm2: float = 0.0
    t_down_c: float = 10.5
    q_int_known_w: float = 50.0
    t_up_c: float = 18.0


@dataclass(frozen=True, slots=True)
class ActuatorStepInput:
    """Actuator inputs for side-effect-free forward scenario simulation."""

    q_fh_w: float | None = None
    t_supply_sec_c: float | None = None
    u_eff: float = 1.0
    p_el_ac_w: float = 0.0


@dataclass(frozen=True, slots=True)
class SimulationStepResult:
    """Result of one side-effect-free forward ZOH simulation step."""

    step_index: int
    state: ThermalState
    t_surface_est_c: float
    q_fh_applied_w: float
    q_ac_applied_w: float
    q_floor_to_air_w: float
    q_shell_loss_w: float


@dataclass(frozen=True, slots=True)
class PreheatEstimate:
    """Lead-time estimate for scheduled comfort targets (Layer 2)."""

    reachable: bool
    lead_time_min: int
    predicted_t_air_end_c: float
    predicted_t_floor_end_c: float
    total_thermal_energy_kwh: float


@dataclass(frozen=True, slots=True)
class CoastEstimate:
    """Passive coasting and slab overshoot estimate at `Q_fh = 0` (Layer 2)."""

    coast_duration_min: int
    min_t_air_horizon_c: float
    peak_t_air_overshoot_c: float


@dataclass(frozen=True, slots=True)
class LinearizedDiscreteModel:
    """Discrete 2x2 state-space `x_{k+1} = Phi*x_k + Gamma_fh*Q_fh + Gamma_ac*P_ac`."""

    zone: ZoneId
    dt_s: float
    phi_2x2: tuple[tuple[float, float], tuple[float, float]]
    gamma_fh: tuple[float, float]
    gamma_ac_el_ref: tuple[float, float]
    steady_loss_total_w_k: float
    effective_tau_coast_h: float


@dataclass(frozen=True, slots=True)
class RegimeHindcastMetrics:
    """Open-loop hindcast metrics for a single physical regime."""

    regime: str
    horizon_steps: int
    n_windows: int
    rmse_end_c: float | None
    rmse_path_c: float | None
    mae_end_c: float | None
    bias_end_c: float | None
    j_eval_c: float | None
    prior_rmse_end_c: float | None
    skill_score_pct: float | None


@dataclass(frozen=True, slots=True)
class ZoneEvaluationReport:
    """Complete 14-day hindcast evaluation across all regimes for one zone."""

    zone: ZoneId
    ts_evaluated_utc: int
    overall_6h: RegimeHindcastMetrics
    by_regime: dict[str, RegimeHindcastMetrics] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ParameterPosteriorMemory:
    """Bayesian precision-weighted seasonal memory (Choice D2)."""

    mean: dict[str, float]
    precision: dict[str, float]


@dataclass(frozen=True, slots=True)
class CalibrationZoneOutcome:
    """Outcome of a manual or weekly batch calibration for a single zone."""

    zone: ZoneId
    accepted: bool
    reason: str
    mode: Literal["manual", "auto_weekly"]
    ts_utc: int
    params_old: ZoneParameters
    params_map_raw: ZoneParameters
    params_candidate: ZoneParameters
    params_applied: ZoneParameters
    posterior_std: dict[str, float]
    correlation_matrix: dict[str, dict[str, float]]
    collinearity_warnings: list[str]
    holdout_before: ZoneEvaluationReport
    holdout_after: ZoneEvaluationReport
    bound_warnings: list[str] = field(default_factory=list)
