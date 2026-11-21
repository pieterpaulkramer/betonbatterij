"""Shared constants for the BetonBatterij integration (Layer 1 Digital Twin).

Defines domain identifiers, storage keys, manifold rotameter option keys,
bitmask flags, and the canonical 58-column RingBuffer schema. Contains zero
`homeassistant.*` imports so offline tests can import it directly.
"""

from __future__ import annotations

DOMAIN: str = "betonbatterij"
VERSION: str = "0.1.0"

DT_STEP_S: int = 300
STEPS_PER_HOUR: int = 12
STEPS_PER_DAY: int = 288
RINGBUFFER_DAYS: int = 90
RINGBUFFER_ROWS: int = RINGBUFFER_DAYS * STEPS_PER_DAY  # 25,920 rows
RINGBUFFER_CHUNK_ROWS: int = 7 * STEPS_PER_DAY  # 2,016 rows per weekly chunk
RINGBUFFER_NUM_CHUNKS: int = 14  # 14 rotating week slots (90d + active week)

STORAGE_VERSION: int = 1
STORAGE_KEY_STATE: str = "betonbatterij.state"
STORAGE_KEY_EPISODE_BANK: str = "betonbatterij.episode_bank"
STORAGE_KEY_RING_PREFIX: str = "betonbatterij.ringbuffer_w"
EXPORT_DIR_DEFAULT: str = "/share/betonbatterij_export"

OPT_FLOW_WK_L_MIN: str = "flow_wk_l_min"
OPT_FLOW_GANG_L_MIN: str = "flow_gang_l_min"
OPT_FLOW_SLAAPKAMER_L_MIN: str = "flow_slaapkamer_l_min"
OPT_FLOW_BADKAMER_L_MIN: str = "flow_badkamer_l_min"
OPT_FLOW_WERKKAMER_1E_L_MIN: str = "flow_werkkamer_1e_l_min"
OPT_FLOW_KAMER_1E_MIDDEN_L_MIN: str = "flow_kamer_1e_midden_l_min"
OPT_FLOW_KAMER_1E_ZUID_L_MIN: str = "flow_kamer_1e_zuid_l_min"
OPT_FLOW_KANTOOR_L_MIN: str = "flow_kantoor_l_min"
OPT_FLOW_BADKAMER_BOVEN_L_MIN: str = "flow_badkamer_boven_l_min"
OPT_FLOW_LOGEERKAMER_L_MIN: str = "flow_logeerkamer_l_min"

ALL_HYDRAULIC_GROUPS: tuple[str, ...] = (
    "wk",
    "gang",
    "slaapkamer",
    "badkamer",
    "werkkamer_1e",
    "kamer_1e_midden",
    "kamer_1e_zuid",
    "kantoor",
    "badkamer_boven",
    "logeerkamer",
)

FLOW_OPTION_KEYS: dict[str, str] = {
    "wk": OPT_FLOW_WK_L_MIN,
    "gang": OPT_FLOW_GANG_L_MIN,
    "slaapkamer": OPT_FLOW_SLAAPKAMER_L_MIN,
    "badkamer": OPT_FLOW_BADKAMER_L_MIN,
    "werkkamer_1e": OPT_FLOW_WERKKAMER_1E_L_MIN,
    "kamer_1e_midden": OPT_FLOW_KAMER_1E_MIDDEN_L_MIN,
    "kamer_1e_zuid": OPT_FLOW_KAMER_1E_ZUID_L_MIN,
    "kantoor": OPT_FLOW_KANTOOR_L_MIN,
    "badkamer_boven": OPT_FLOW_BADKAMER_BOVEN_L_MIN,
    "logeerkamer": OPT_FLOW_LOGEERKAMER_L_MIN,
}

DEFAULT_FLOW_RATES_L_MIN: dict[str, float] = {
    "wk": 5.0,
    "gang": 2.0,
    "slaapkamer": 2.0,
    "badkamer": 1.0,
    "werkkamer_1e": 1.0,
    "kamer_1e_midden": 1.0,
    "kamer_1e_zuid": 1.0,
    "kantoor": 1.0,
    "badkamer_boven": 1.0,
    "logeerkamer": 1.0,
}

FLAG_VALID_WK: int = 1 << 0
FLAG_VALID_KT: int = 1 << 1
FLAG_AANWEZIG: int = 1 << 2
FLAG_SUMMER: int = 1 << 3
FLAG_FLOOR_COOL: int = 1 << 4
FLAG_AC_COOL_KT: int = 1 << 5
FLAG_DHW_TANK: int = 1 << 6
FLAG_Z1_OFF_HARVEST: int = 1 << 7
FLAG_PUMP_SEC_ON: int = 1 << 8
FLAG_OPEN_WIN_WK: int = 1 << 9
FLAG_OPEN_WIN_KT: int = 1 << 10
FLAG_GAP_BRIDGED: int = 1 << 11
FLAG_PV_FALLBACK: int = 1 << 12
FLAG_DEFROST_WP: int = 1 << 13
FLAG_DEFROST_AC: int = 1 << 14
FLAG_INTERNAL_HEATER: int = 1 << 15

AC_MODE_OFF: float = 0.0
AC_MODE_HEAT: float = 1.0
AC_MODE_COOL: float = 2.0
AC_MODE_DRY: float = 3.0
AC_MODE_OTHER: float = 4.0

RINGBUFFER_INT_COLS: tuple[str, ...] = ("ts_utc", "flags")

RINGBUFFER_FLOAT_COLS: tuple[str, ...] = (
    "u_eff_bg",
    "u_eff_slaapkamer",
    "u_eff_badkamer",
    "u_eff_werkkamer_1e",
    "u_eff_kamer_1e_midden",
    "u_eff_kamer_1e_zuid",
    "u_eff_kantoor",
    "u_eff_badkamer_boven",
    "u_eff_logeerkamer",
    "u_pump_sec",
    "t_air_wk_c",
    "t_air_slaapkamer_c",
    "t_air_badkamer_c",
    "t_air_werkkamer_1e_c",
    "t_air_kamer_1e_midden_c",
    "t_air_kamer_1e_zuid_c",
    "t_air_kantoor_c",
    "t_air_badkamer_boven_c",
    "t_air_logeerkamer_c",
    "v_prim_l_min",
    "t_wp_out_c",
    "t_wp_in_c",
    "p_prim_th_w",  # Sub-5m integrated 69.73 * V_prim * (T_out - T_in) [W]
    "f_comp_hz",
    "p_wp_raw_el_w",
    "t_buf_top_start_c",
    "t_buf_top_end_c",
    "t_buf_bot_start_c",  # Raw uncalibrated bottom sensor at t0 [°C]
    "t_buf_bot_end_c",  # Raw uncalibrated bottom sensor at t1 [°C]
    "t_dhw_top_c",
    "t_dhw_bot_c",
    "p_p1_net_w",
    "t_out_dak_c",
    "t_out_weather_c",
    "t_out_hm_c",
    "rh_out_pct",
    "v_wind_ms",
    "p_pv_w",
    "p_pv_forecast_w",
    "epex_eur_kwh",
    "screen_small_pos",
    "screen_large_pos",
    "p_el_ac_kt_w",
    "ac_mode_code",
    "p_desk_kt_w",
    "pir_occ_kt",
    "p_growlights_kt_w",
    "t_bg_sup_c",
    "t_bg_ret_c",
    "t_floor_surf_meas_wk_c",
    "t_floor_wk_c",
    "d_est_wk_w",
    "q_fh_wk_w",
    "t_hind6h_wk_c",
    "t_floor_kt_c",
    "d_est_kt_w",
    "q_fh_kt_w",
    "t_hind6h_kt_c",
)

RINGBUFFER_ALL_COLS: tuple[str, ...] = RINGBUFFER_INT_COLS + RINGBUFFER_FLOAT_COLS
