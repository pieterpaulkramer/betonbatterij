"""Immutable domain contracts and Layer-2 protocol for BetonBatterij.

This is a pure leaf module with zero internal package imports and zero
`homeassistant.*`, `numpy`, or `scipy` dependencies. All state, parameter,
forecast, and evaluation records are immutable (`frozen=True, slots=True`)
dataclasses, and `ThermalTwinProtocol` defines the read-only Layer-2 boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import MISSING, dataclass, field, fields
import math
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

ZoneId: TypeAlias = Literal["wk", "kt"]


def _parse_finite_float(raw: Any) -> float | None:
    """Returns `float(raw)` if `raw` is a finite non-boolean number, else None."""
    if raw is None or isinstance(raw, bool):
        return None
    try:
        fval = float(raw)
    except (TypeError, ValueError):
        return None
    return fval if math.isfinite(fval) else None


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

    def to_dict(self) -> dict[str, float]:
        """Serializes all scalar parameters to a plain `dict[str, float]`."""
        return {f.name: float(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, Any] | None,
        defaults: ZoneParameters | None = None,
    ) -> ZoneParameters:
        """Deserializes `ZoneParameters` with strict finite-float checks.

        Args:
            raw: Serialized parameter mapping (or `None`).
            defaults: Optional fallback `ZoneParameters` supplying missing or
                non-finite fields.

        Returns:
            Constructed immutable `ZoneParameters`.

        Raises:
            ValueError: If `defaults` is `None` and `raw` omits or provides a
                non-finite value for any required field.
        """
        if not isinstance(raw, Mapping):
            if defaults is not None:
                return defaults
            raise ValueError(
                "Cannot deserialize ZoneParameters from non-mapping without "
                "defaults"
            )
        kwargs: dict[str, float] = {}
        for f in fields(cls):
            parsed = _parse_finite_float(raw.get(f.name))
            if parsed is not None:
                kwargs[f.name] = parsed
            elif defaults is not None:
                kwargs[f.name] = float(getattr(defaults, f.name))
            elif f.default is not MISSING:
                kwargs[f.name] = float(f.default)
            else:
                raise ValueError(
                    f"Missing or non-finite ZoneParameters field '{f.name}'"
                )
        return cls(**kwargs)


@dataclass(frozen=True, slots=True)
class CopParameters:
    """Calibrated Panasonic 9 kW T-CAP Carnot COP curve parameters.

    Attributes:
        eta_carnot_cv: Space-heating Carnot efficiency factor [-].
        beta_defrost: Defrost COP penalty amplitude near 0 °C [-].
        eta_carnot_dhw: Domestic hot water Carnot efficiency factor [-].
        delta_p_wp_bias_w: Additive bias correction on WP electrical power [W].
    """

    eta_carnot_cv: float = 0.48
    beta_defrost: float = 0.18
    eta_carnot_dhw: float = 0.34
    delta_p_wp_bias_w: float = 145.0

    def to_dict(self) -> dict[str, float]:
        """Serializes all COP parameters to a plain `dict[str, float]`."""
        return {f.name: float(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, Any] | None,
        defaults: CopParameters | None = None,
    ) -> CopParameters:
        """Deserializes `CopParameters` with finite-float checks and fallback.

        Args:
            raw: Serialized COP parameter mapping (or `None`).
            defaults: Optional fallback `CopParameters` (defaults to `cls()`).

        Returns:
            Constructed immutable `CopParameters`.
        """
        base = defaults if defaults is not None else cls()
        if not isinstance(raw, Mapping):
            return base
        kwargs: dict[str, float] = {}
        for f in fields(cls):
            parsed = _parse_finite_float(raw.get(f.name))
            kwargs[f.name] = (
                parsed if parsed is not None else float(getattr(base, f.name))
            )
        return cls(**kwargs)


@dataclass(frozen=True, slots=True)
class BufferSensorOffsets:
    """Additive temperature corrections applied to raw 1-Wire buffer sensors.

    Calibrated sensor temperature equals `t_raw_c + offset_c` [°C], anchored to
    stationary `0 Hz` primary sniffing equilibrium (`wp_outlet == wp_inlet`).

    Attributes:
        buf_top_offset_c: Additive correction on raw top 1-Wire sensor [°C].
        buf_bot_offset_c: Additive correction on raw bottom 1-Wire sensor [°C].
    """

    buf_top_offset_c: float = 0.0
    buf_bot_offset_c: float = -0.33

    def to_dict(self) -> dict[str, float]:
        """Serializes buffer sensor offsets to a plain `dict[str, float]`."""
        return {f.name: float(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, Any] | None,
        defaults: BufferSensorOffsets | None = None,
    ) -> BufferSensorOffsets:
        """Deserializes `BufferSensorOffsets` with finite-float checks."""
        base = defaults if defaults is not None else cls()
        if not isinstance(raw, Mapping):
            return base
        kwargs: dict[str, float] = {}
        for f in fields(cls):
            parsed = _parse_finite_float(raw.get(f.name))
            kwargs[f.name] = (
                parsed if parsed is not None else float(getattr(base, f.name))
            )
        return cls(**kwargs)


@dataclass(frozen=True, slots=True)
class ZoneSnapshot:
    """Real-time O(1) readout of a modeled zone for HA sensors and Layer 2.

    Attributes:
        zone: Zone identifier (`"wk"` or `"kt"`).
        ts_utc: UTC epoch timestamp of the latest processed step [s].
        state: Latest posterior 3-state vector `[T_air, T_floor, d_est]`.
        t_surface_est_c: Estimated PVC floor surface temperature [°C].
        t_air_meas_c: Latest valid measured air temperature [°C], or `None`.
        t_floor_std_c: Kalman posterior 1-sigma uncertainty on `T_floor` [°C].
        q_fh_now_w: Latest estimated floor heating power [W].
        t_hindcast_6h_c: Latest published 6-hour open-loop hindcast [°C].
        sensor_stale: True when the room ZHA thermometer is stale.
        open_window_active: True when the open-window Kalman guard is active.
        stored_slab_energy_wh: Thermal energy stored in slab above `T_air` [Wh].
        params: Active physical `ZoneParameters` for the zone.
        rmse_6h_c: Latest 14-day overall 6h hindcast RMSE [°C], or `None`.
        warmup_remaining_min: Remaining 6h hindcast FIFO warmup duration [min].
    """

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
    warmup_remaining_min: int = 0


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
    gamma_cal: float = 1.0
    k_hx_eff_w_k: float = 0.0


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
    """Bayesian precision-weighted seasonal posterior parameter memory."""

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


@runtime_checkable
class ThermalTwinProtocol(Protocol):
    """Read-only structural protocol defining the Layer-1 -> Layer-2 contract.

    Layer-2 optimization and MPC consumers interact with `DigitalTwinEngine`
    exclusively through these side-effect-free query, rollout, and solver
    export methods.
    """

    @property
    def cop_params(self) -> CopParameters:
        """Active calibrated heat-pump `CopParameters`."""
        ...

    def get_zone_snapshot(self, zone: ZoneId) -> ZoneSnapshot:
        """Returns the latest O(1) immutable `ZoneSnapshot` for `zone`."""
        ...

    def simulate_step(
        self,
        zone: ZoneId,
        state: ThermalState | None,
        exog: ExogenousStepInput,
        actuator: ActuatorStepInput | None = None,
        dt_s: float = 300.0,
        step_index: int = 0,
    ) -> SimulationStepResult:
        """Executes one side-effect-free forward ZOH step from `state`."""
        ...

    def simulate_horizon(
        self,
        zone: ZoneId,
        exog_forecast: Sequence[ExogenousStepInput],
        actuators: (
            Sequence[ActuatorStepInput] | ActuatorStepInput | None
        ) = None,
        initial_state: ThermalState | None = None,
        dt_s: float = 300.0,
    ) -> list[SimulationStepResult]:
        """Runs a side-effect-free multi-step ZOH rollout over `exog_forecast`."""
        ...

    def estimate_coast_duration(
        self,
        zone: ZoneId,
        floor_temp_min_air_c: float,
        exog_forecast: Sequence[ExogenousStepInput],
        initial_state: ThermalState | None = None,
        dt_s: float = 300.0,
    ) -> CoastEstimate:
        """Estimates passive coasting duration and slab overshoot at `Q_fh=0`."""
        ...

    def estimate_preheat_lead_time(
        self,
        zone: ZoneId,
        target_temp_c: float,
        exog_forecast: Sequence[ExogenousStepInput],
        t_supply_sec_c: float = 30.0,
        q_fh_max_w: float | None = None,
        p_el_ac_w: float = 0.0,
        initial_state: ThermalState | None = None,
        max_surface_temp_c: float = 27.5,
        dt_s: float = 300.0,
    ) -> PreheatEstimate:
        """Estimates preheat lead time to reach `target_temp_c`."""
        ...

    def estimate_total_sink_capacity_w(
        self,
        t_supply_sec_c: float,
        open_zones: Sequence[str] | Mapping[str, float],
    ) -> float:
        """Estimates total secondary floor heat uptake [W] across `open_zones`."""
        ...

    def predict_cop_cv(
        self,
        t_out_c: float,
        t_sup_c: float,
        rh_out_pct: float = 80.0,
    ) -> float:
        """Predicts calibrated space-heating COP for `(t_out_c, t_sup_c, rh)`."""
        ...

    def predict_cop_dhw(
        self,
        t_out_c: float,
        t_dhw_start_c: float,
        t_dhw_target_c: float,
        rh_out_pct: float = 80.0,
    ) -> float:
        """Predicts calibrated 2-stage DHW COP across `[start_c -> target_c]`."""
        ...

    def export_solver_model(
        self,
        zone: ZoneId,
        dt_s: float = 1800.0,
    ) -> LinearizedDiscreteModel:
        """Exports linearized 2x2 ZOH matrices for external MILP/MPC solvers."""
        ...
