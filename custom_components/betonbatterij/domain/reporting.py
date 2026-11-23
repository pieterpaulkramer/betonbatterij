"""Diagnostic CSV/JSON export and strict JSON payload sanitization.

Owns atomic Syncthing-safe disk export of the 31-column `telemetry_5m.csv` and
`fit_report.json` diagnostic artifacts (`CSV_EXPORT_COLUMNS`,
`export_csv_and_report`) plus recursive non-finite float sanitization
(`sanitize_json_payload`) for strict `orjson` / JSON persistence.

Does not own mathematical hindcast evaluation (`evaluation.py`), Kalman state
estimation (`zoh_kalman.py`), or Home Assistant storage adapters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import dataclasses
import datetime
import json
import math
import os
import pathlib
from typing import Any

from custom_components.betonbatterij import const
from custom_components.betonbatterij.domain import contracts
from custom_components.betonbatterij.domain import preprocessing
from custom_components.betonbatterij.domain import priors
from custom_components.betonbatterij.domain import zoh_kalman

CSV_EXPORT_COLUMNS: tuple[str, ...] = (
    "ts_utc",
    "iso_utc",
    "flags",
    "t_out_c",
    "v_wind_ms",
    "i_sol_wk_wm2",
    "i_sol_kt_wm2",
    "tau_screen_wk",
    "t_buf_top_c",
    "t_buf_bot_c",
    "q_sec_tot_w",
    "u_eff_wk",
    "f_flow_wk",
    "t_air_wk_c",
    "t_floor_wk_c",
    "d_est_wk_w",
    "q_fh_wk_w",
    "t_hind6h_wk_c",
    "u_eff_kt",
    "f_flow_kt",
    "t_air_kt_c",
    "t_down_kt_c",
    "q_int_kt_w",
    "p_el_ac_kt_w",
    "t_floor_kt_c",
    "d_est_kt_w",
    "t_hind6h_kt_c",
    "t_surf_wk_c",
    "t_surf_kt_c",
    "p_wp_raw_el_w",
    "f_comp_hz",
)


def sanitize_json_payload(obj: Any) -> Any:
    """Recursively replaces non-finite floats (`NaN`, `Inf`) with `None`."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return sanitize_json_payload(dataclasses.asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): sanitize_json_payload(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json_payload(item) for item in obj]
    return obj


def _fmt_csv_float(val: float | None, decimals: int = 3) -> str:
    """Formats a finite float to `decimals` places, or `""` when missing."""
    if val is None or not math.isfinite(val):
        return ""
    return f"{val:.{decimals}f}"


def export_csv_and_report(
    export_dir: str | pathlib.Path,
    columns: Mapping[str, Sequence[float | int]],
    fit_report: Mapping[str, Any],
    lat_deg: float = priors.LATITUDE_DEG,
    lon_deg: float = priors.LONGITUDE_DEG,
) -> tuple[str, str]:
    """Atomically writes 31-column `telemetry_5m.csv` and `fit_report.json`.

    Uses atomic `.tmp` -> `flush()` -> `os.fsync()` -> `os.replace()` semantics
    so external sync tools never read half-written files.

    Args:
        export_dir: Target directory path.
        columns: RingBuffer telemetry column sequences.
        fit_report: Diagnostic fit report dictionary.
        lat_deg: Observer latitude in degrees North.
        lon_deg: Observer longitude in degrees East.

    Returns:
        A tuple `(csv_path, json_path)` of written file paths.
    """
    out_dir = pathlib.Path(export_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_final = out_dir / "telemetry_5m.csv"
    csv_tmp = out_dir / "telemetry_5m.csv.tmp"
    json_final = out_dir / "fit_report.json"
    json_tmp = out_dir / "fit_report.json.tmp"

    flows = fit_report.get("flow_weights_l_min")
    flow_map = (
        flows if isinstance(flows, Mapping) else const.DEFAULT_FLOW_RATES_L_MIN
    )

    zones_info = fit_report.get("zones", {})
    wk_raw = (
        zones_info.get("wk", {}).get("parameters_active")
        if isinstance(zones_info, Mapping)
        else None
    )
    kt_raw = (
        zones_info.get("kt", {}).get("parameters_active")
        if isinstance(zones_info, Mapping)
        else None
    )
    params_wk = contracts.ZoneParameters.from_dict(
        wk_raw, defaults=priors.DAY1_PRIORS_WK
    )
    params_kt = contracts.ZoneParameters.from_dict(
        kt_raw, defaults=priors.DAY1_PRIORS_KT
    )

    raw_khx = fit_report.get("k_hx_by_group_w_k")
    k_hx_map: dict[str, float] = {}
    if isinstance(raw_khx, Mapping):
        for group_name, val in raw_khx.items():
            if preprocessing.is_finite_number(val):
                k_hx_map[
                    preprocessing.zone_to_hydraulic_group(str(group_name))
                ] = float(val)
    k_hx_map.setdefault("wk", params_wk.k_hx_w_k)
    k_hx_map.setdefault("kantoor", params_kt.k_hx_w_k)

    hist = preprocessing.preprocess_history_columns(
        columns,
        flow_rates_l_min=flow_map,
        k_hx_by_group=k_hx_map,
        eta_ac_ref=params_kt.eta_ac,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
    )

    n = hist.n_rows
    t_floor_wk_seq = columns.get("t_floor_wk_c", ())
    t_floor_kt_seq = columns.get("t_floor_kt_c", ())
    q_fh_wk_seq = columns.get("q_fh_wk_w", ())
    t_hind_wk_seq = columns.get("t_hind6h_wk_c", ())
    t_hind_kt_seq = columns.get("t_hind6h_kt_c", ())
    p_wp_raw_seq = columns.get("p_wp_raw_el_w", ())

    with open(csv_tmp, "w", encoding="utf-8", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(CSV_EXPORT_COLUMNS)
        for k in range(n):
            ts_k = hist.ts_utc[k]
            iso_k = datetime.datetime.fromtimestamp(
                ts_k, tz=datetime.timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
            tf_wk = (
                float(t_floor_wk_seq[k])
                if k < len(t_floor_wk_seq)
                else hist.t_air_wk_c[k]
            )
            tf_kt = (
                float(t_floor_kt_seq[k])
                if k < len(t_floor_kt_seq)
                else hist.t_air_kt_c[k]
            )
            q_wk = (
                float(q_fh_wk_seq[k])
                if k < len(q_fh_wk_seq)
                else hist.q_fh_wk_hyd_w[k]
            )
            t_hind_wk = (
                float(t_hind_wk_seq[k])
                if k < len(t_hind_wk_seq)
                else math.nan
            )
            t_hind_kt = (
                float(t_hind_kt_seq[k])
                if k < len(t_hind_kt_seq)
                else math.nan
            )
            p_wp_raw = (
                preprocessing.safe_float(p_wp_raw_seq[k], 0.0)
                if k < len(p_wp_raw_seq)
                else 0.0
            )

            t_surf_wk = zoh_kalman.estimate_zone_surface_temp(
                "wk", hist.t_air_wk_c[k], tf_wk, params_wk
            )
            t_surf_kt = zoh_kalman.estimate_zone_surface_temp(
                "kt", hist.t_air_kt_c[k], tf_kt, params_kt
            )

            writer.writerow(
                [
                    ts_k,
                    iso_k,
                    hist.flags[k],
                    _fmt_csv_float(hist.t_out_c[k], 2),
                    _fmt_csv_float(hist.v_wind_ms[k], 2),
                    _fmt_csv_float(hist.i_sol_wk_wm2[k], 1),
                    _fmt_csv_float(hist.i_sol_kt_wm2[k], 1),
                    _fmt_csv_float(hist.tau_screen_wk[k], 3),
                    _fmt_csv_float(hist.t_buf_top_c[k], 2),
                    _fmt_csv_float(hist.t_buf_bot_cal_c[k], 2),
                    _fmt_csv_float(hist.q_sec_tot_w[k], 1),
                    _fmt_csv_float(hist.u_eff_wk[k], 3),
                    _fmt_csv_float(hist.f_flow_wk[k], 3),
                    _fmt_csv_float(hist.t_air_wk_c[k], 2),
                    _fmt_csv_float(tf_wk, 3),
                    _fmt_csv_float(hist.d_est_col_wk_w[k], 1),
                    _fmt_csv_float(q_wk, 1),
                    _fmt_csv_float(t_hind_wk, 2),
                    _fmt_csv_float(hist.u_eff_kt[k], 3),
                    _fmt_csv_float(hist.f_flow_kt[k], 3),
                    _fmt_csv_float(hist.t_air_kt_c[k], 2),
                    _fmt_csv_float(hist.t_down_kt_c[k], 2),
                    _fmt_csv_float(hist.q_int_kt_w[k], 1),
                    _fmt_csv_float(hist.p_el_ac_kt_w[k], 1),
                    _fmt_csv_float(tf_kt, 3),
                    _fmt_csv_float(hist.d_est_col_kt_w[k], 1),
                    _fmt_csv_float(t_hind_kt, 2),
                    _fmt_csv_float(t_surf_wk, 2),
                    _fmt_csv_float(t_surf_kt, 2),
                    _fmt_csv_float(p_wp_raw, 1),
                    _fmt_csv_float(hist.f_comp_hz[k], 1),
                ]
            )
        f_csv.flush()
        os.fsync(f_csv.fileno())
    os.replace(csv_tmp, csv_final)

    with open(json_tmp, "w", encoding="utf-8") as f_json:
        json.dump(
            sanitize_json_payload(dict(fit_report)),
            f_json,
            indent=2,
            sort_keys=False,
            allow_nan=False,
        )
        f_json.flush()
        os.fsync(f_json.fileno())
    os.replace(json_tmp, json_final)

    return str(csv_final), str(json_final)
