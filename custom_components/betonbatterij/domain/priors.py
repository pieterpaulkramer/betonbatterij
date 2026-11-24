"""Pure physical building geometry, Day-1 priors, and parameter bounds.

Single source of truth for all building thermal constants, hydraulic group
properties, and Bayesian calibration box bounds used inside `domain/`. Contains
zero Home Assistant entity IDs, zero disk/JSON loading, and zero external
dependencies.
"""

from __future__ import annotations

from custom_components.betonbatterij.domain import contracts

LATITUDE_DEG: float = 51.4
LONGITUDE_DEG: float = 5.5
TIMEZONE_NAME: str = "Europe/Amsterdam"

PV_PEAK_KWP: float = 5.74
PV_TILT_DEG: float = 30.0
PV_AZIMUTH_DEG: float = 175.0  # 180 deg = South, 175 deg = 5 deg East of South
WINDOW_WK_TILT_DEG: float = 90.0
WINDOW_WK_AZIMUTH_DEG: float = 175.0
ROOF_KT_TILT_DEG: float = 38.0

SCREEN_SMALL_WEIGHT: float = 0.35
SCREEN_LARGE_WEIGHT: float = 0.65

AREA_WK_M2: float = 50.0
AREA_KT_M2: float = 23.0
H_SURF_UPWARD_W_M2K: float = 10.8  # EN 1264 upward heat transfer floor -> air

T_GROUND_MEAN_C: float = 10.5
BETA_DOWN_1E_LEAK: float = 0.25  # 25% downward leak from active 1st-floor loops
BASELOAD_WK_W: float = 50.0

WATER_POWER_FACTOR_W_PER_LMIN_K: float = 69.73  # rho * c_p / 60 [W/(L/min*K)]
BUFFER_VOLUME_L: float = 200.0
C_BUF_WH_K: float = 232.44  # 200 L * 1.1622 Wh/(L*K)
C_BUF_J_K: float = C_BUF_WH_K * 3600.0  # 836,784 J/K
BUF_TOP_SENSOR_OFFSET_C: float = 0.0  # Day-1 prior before 0 Hz sniffing cal
BUF_BOT_SENSOR_OFFSET_C: float = -0.33  # Raw bottom reads +0.33 deg C high
BUF_SENSOR_OFFSET_MIN_C: float = -3.0
BUF_SENSOR_OFFSET_MAX_C: float = 3.0
DAY1_BUFFER_OFFSETS: contracts.BufferSensorOffsets = (
    contracts.BufferSensorOffsets(
        buf_top_offset_c=BUF_TOP_SENSOR_OFFSET_C,
        buf_bot_offset_c=BUF_BOT_SENSOR_OFFSET_C,
    )
)
WAX_VALVE_OPEN_TIME_S: float = 228.0
WAX_VALVE_CLOSE_TIME_S: float = 228.0

DAY1_PRIORS_WK: contracts.ZoneParameters = contracts.ZoneParameters(
    c_air_wh_k=450.0,
    c_floor_wh_k=5500.0,
    r_fh_k_kw=2.60,
    h_loss_w_k=65.0,
    h_up_w_k=120.0,
    h_down_w_k=9.0,
    alpha_sol=0.35,
    a_sol_m2=3.20,
    tau_d_h=1.8,
    eta_ac=0.0,
    k_hx_w_k=245.0,
)

DAY1_PRIORS_KT: contracts.ZoneParameters = contracts.ZoneParameters(
    c_air_wh_k=220.0,
    c_floor_wh_k=2400.0,
    r_fh_k_kw=5.00,
    h_loss_w_k=25.0,
    h_up_w_k=0.0,
    h_down_w_k=40.0,
    alpha_sol=0.80,
    a_sol_m2=0.40,
    tau_d_h=1.8,
    eta_ac=3.60,
    k_hx_w_k=65.0,
)

PARAM_BOUNDS_WK: dict[str, contracts.ParameterBounds] = {
    "c_floor_wh_k": contracts.ParameterBounds(
        lower=3500.0, upper=8000.0, prior_mean=5500.0, prior_std=1100.0
    ),
    "r_fh_k_kw": contracts.ParameterBounds(
        lower=1.80, upper=3.60, prior_mean=2.60, prior_std=0.50
    ),
    "h_loss_w_k": contracts.ParameterBounds(
        lower=35.0, upper=105.0, prior_mean=65.0, prior_std=18.0
    ),
    "a_sol_m2": contracts.ParameterBounds(
        lower=1.00, upper=6.00, prior_mean=3.20, prior_std=1.20
    ),
    "h_up_w_k": contracts.ParameterBounds(
        lower=80.0, upper=170.0, prior_mean=120.0, prior_std=20.0
    ),
}

PARAM_BOUNDS_KT: dict[str, contracts.ParameterBounds] = {
    "c_floor_wh_k": contracts.ParameterBounds(
        lower=1500.0, upper=3500.0, prior_mean=2400.0, prior_std=500.0
    ),
    "r_fh_k_kw": contracts.ParameterBounds(
        lower=3.50, upper=6.50, prior_mean=5.00, prior_std=0.80
    ),
    "h_loss_w_k": contracts.ParameterBounds(
        lower=12.0, upper=65.0, prior_mean=25.0, prior_std=10.0
    ),
    "a_sol_m2": contracts.ParameterBounds(
        lower=0.10, upper=2.00, prior_mean=0.40, prior_std=0.40
    ),
    "eta_ac": contracts.ParameterBounds(
        lower=2.00, upper=5.50, prior_mean=3.60, prior_std=0.70
    ),
    "h_down_w_k": contracts.ParameterBounds(
        lower=20.0, upper=75.0, prior_mean=40.0, prior_std=12.0
    ),
}

FIT_INTERFLOOR_CONDUCTANCE: bool = False

DAY1_K_HX_BY_GROUP_W_K: dict[str, float] = {
    "wk": 245.0,
    "gang": 98.0,
    "slaapkamer": 110.0,
    "badkamer": 55.0,
    "werkkamer_1e": 60.0,
    "kamer_1e_midden": 60.0,
    "kamer_1e_zuid": 60.0,
    "kantoor": 65.0,
    "badkamer_boven": 55.0,
    "logeerkamer": 60.0,
}

UNMODELED_GROUP_THERMAL_PROPS: dict[str, tuple[float, float]] = {
    "gang": (1500.0, 2.80),
    "slaapkamer": (2600.0, 4.80),
    "badkamer": (900.0, 4.50),
    "werkkamer_1e": (1300.0, 5.00),
    "kamer_1e_midden": (1300.0, 5.00),
    "kamer_1e_zuid": (1300.0, 5.00),
    "badkamer_boven": (900.0, 4.80),
    "logeerkamer": (1800.0, 5.00),
}

DAY1_COP_PARAMS: contracts.CopParameters = contracts.CopParameters(
    eta_carnot_cv=0.48,
    beta_defrost=0.18,
    eta_carnot_dhw=0.34,
    delta_p_wp_bias_w=145.0,
)

COP_PARAM_BOUNDS: dict[str, contracts.ParameterBounds] = {
    "eta_carnot_cv": contracts.ParameterBounds(
        lower=0.34, upper=0.62, prior_mean=0.48, prior_std=0.06
    ),
    "beta_defrost": contracts.ParameterBounds(
        lower=0.05, upper=0.35, prior_mean=0.18, prior_std=0.07
    ),
    "eta_carnot_dhw": contracts.ParameterBounds(
        lower=0.22, upper=0.48, prior_mean=0.34, prior_std=0.06
    ),
    "delta_p_wp_bias_w": contracts.ParameterBounds(
        lower=80.0, upper=220.0, prior_mean=145.0, prior_std=35.0
    ),
}
