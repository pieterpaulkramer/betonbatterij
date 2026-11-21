"""Version-controlled house geometry, generic entity IDs, and Day-1 priors.

Pure Python with zero `homeassistant.*` imports so domain and bridge unit tests
can import all constants, entity mappings, and local JSON overrides offline.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
import json
import logging
from pathlib import Path
from typing import Any

from .domain.contracts import CopParameters, ParameterBounds, ZoneParameters

_LOGGER = logging.getLogger(__name__)

LOCAL_CONFIG_FILENAME: str = "betonbatterij_local.json"

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
BUF_BOT_SENSOR_OFFSET_C: float = -0.33  # Raw bottom reads +0.33 deg C high
WAX_VALVE_OPEN_TIME_S: float = 228.0
WAX_VALVE_CLOSE_TIME_S: float = 228.0

DAY1_PRIORS_WK: ZoneParameters = ZoneParameters(
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

DAY1_PRIORS_KT: ZoneParameters = ZoneParameters(
    c_air_wh_k=220.0,
    c_floor_wh_k=2400.0,
    r_fh_k_kw=5.00,
    h_loss_w_k=45.0,
    h_up_w_k=0.0,
    h_down_w_k=75.0,
    alpha_sol=0.80,
    a_sol_m2=0.40,
    tau_d_h=1.8,
    eta_ac=3.60,
    k_hx_w_k=65.0,
)

PARAM_BOUNDS_WK: dict[str, ParameterBounds] = {
    "c_floor_wh_k": ParameterBounds(
        lower=3500.0, upper=8000.0, prior_mean=5500.0, prior_std=1100.0
    ),
    "r_fh_k_kw": ParameterBounds(
        lower=1.80, upper=3.60, prior_mean=2.60, prior_std=0.50
    ),
    "h_loss_w_k": ParameterBounds(
        lower=35.0, upper=105.0, prior_mean=65.0, prior_std=18.0
    ),
    "a_sol_m2": ParameterBounds(
        lower=1.00, upper=6.00, prior_mean=3.20, prior_std=1.20
    ),
    "h_up_w_k": ParameterBounds(
        lower=80.0, upper=170.0, prior_mean=120.0, prior_std=20.0
    ),
}

PARAM_BOUNDS_KT: dict[str, ParameterBounds] = {
    "c_floor_wh_k": ParameterBounds(
        lower=1500.0, upper=3500.0, prior_mean=2400.0, prior_std=500.0
    ),
    "r_fh_k_kw": ParameterBounds(
        lower=3.50, upper=6.50, prior_mean=5.00, prior_std=0.80
    ),
    "h_loss_w_k": ParameterBounds(
        lower=25.0, upper=80.0, prior_mean=45.0, prior_std=14.0
    ),
    "a_sol_m2": ParameterBounds(
        lower=0.10, upper=2.00, prior_mean=0.40, prior_std=0.40
    ),
    "eta_ac": ParameterBounds(
        lower=2.00, upper=5.50, prior_mean=3.60, prior_std=0.70
    ),
    "h_down_w_k": ParameterBounds(
        lower=50.0, upper=100.0, prior_mean=75.0, prior_std=20.0
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

DAY1_COP_PARAMS: CopParameters = CopParameters(
    eta_carnot_cv=0.48,
    beta_defrost=0.18,
    eta_carnot_dhw=0.34,
    delta_p_wp_bias_w=145.0,
)

COP_PARAM_BOUNDS: dict[str, ParameterBounds] = {
    "eta_carnot_cv": ParameterBounds(
        lower=0.34, upper=0.62, prior_mean=0.48, prior_std=0.06
    ),
    "beta_defrost": ParameterBounds(
        lower=0.05, upper=0.35, prior_mean=0.18, prior_std=0.07
    ),
    "eta_carnot_dhw": ParameterBounds(
        lower=0.22, upper=0.48, prior_mean=0.34, prior_std=0.06
    ),
    "delta_p_wp_bias_w": ParameterBounds(
        lower=80.0, upper=220.0, prior_mean=145.0, prior_std=35.0
    ),
}


@dataclass(frozen=True, slots=True)
class RoomSensorConfig:
    """Room temperature and humidity ZHA sensor pair with optional swap history.

    Attributes:
        temp_entity_id: Current Home Assistant temperature entity ID.
        humidity_entity_id: Companion ZHA humidity entity ID on the same device.
        pre_swap_temp_entity_id: Previous temperature entity ID prior to swap.
        swap_timestamp_utc: UTC epoch timestamp when sensors were swapped.
    """

    temp_entity_id: str
    humidity_entity_id: str | None = None
    pre_swap_temp_entity_id: str | None = None
    swap_timestamp_utc: int | None = None


@dataclass(frozen=True, slots=True)
class HouseEntityConfig:
    """Home Assistant entity IDs consumed by BetonBatterij Layer 1."""

    presence_boolean: str = "input_boolean.aanwezig"
    wp_z1_climate: str = "climate.panasonic_heat_pump_main_z1_temp"
    wp_z1_cooling_climate: str = "climate.panasonic_heat_pump_main_z1_temp_cooling"
    wp_operating_mode: str = "select.panasonic_heat_pump_main_operating_mode_state"
    wp_threeway_valve: str = "sensor.panasonic_heat_pump_main_threeway_valve_state"
    wp_defrosting: str = "binary_sensor.panasonic_heat_pump_main_defrosting_state"
    wp_internal_heater: str = (
        "binary_sensor.panasonic_heat_pump_main_internal_heater_state"
    )

    wp_pump_flow: str = "sensor.panasonic_heat_pump_main_pump_flow"
    wp_outlet_temp: str = "sensor.panasonic_heat_pump_main_main_outlet_temp"
    wp_inlet_temp: str = "sensor.panasonic_heat_pump_main_main_inlet_temp"
    wp_compressor_freq: str = "sensor.panasonic_heat_pump_main_compressor_freq"
    wp_power_consumption: str = (
        "sensor.panasonic_heat_pump_consumption"  # 200 W quantized [W]
    )
    wp_outside_temp: str = "sensor.panasonic_heat_pump_main_outside_temp"

    buf_top_temp: str = "sensor.buffer_vat_boven_temperatuur"
    buf_bot_temp: str = "sensor.buffer_vat_onder_temperatuur"
    dhw_top_temp: str = "sensor.sww_boiler_boven_temperatuur"
    dhw_bot_temp: str = "sensor.sww_boiler_onder_temperatuur"

    pump_secondary_switch: str = "switch.panasonic_heat_pump_gpio_relay_two"
    valve_bg: str = "switch.panasonic_heat_pump_gpio_relay_one"
    valve_slaapkamer: str = "switch.vloerverwarming_slaapkamer"
    valve_slaapkamer_voor: str = "switch.vloerverwarming_slaapkamer_voor"
    valve_slaapkamer_achter: str = "switch.vloerverwarming_slaapkamer_achter"
    valve_badkamer: str = "switch.vloerverwarming_badkamer"
    valve_werkkamer_1e: str = "switch.vloerverwarming_werkkamer_1e"
    valve_kamer_1e_midden: str = "switch.vloerverwarming_kamer_1e_midden"
    valve_kamer_1e_zuid: str = "switch.vloerverwarming_kamer_1e_zuid"
    valve_kantoor: str = "switch.vloerverwarming_kantoor"
    valve_badkamer_boven: str = "switch.vloerverwarming_badkamer_boven"
    valve_logeerkamer: str = "switch.vloerverwarming_logeerkamer"

    outdoor_temp_dak: str = "sensor.buitenthermometer_temperatuur"
    outdoor_humidity_dak: str = "sensor.buitenthermometer_luchtvochtigheid"
    weather_entity: str = "weather.home"
    sun_entity: str = "sun.sun"
    pv_power_w: str = "sensor.pv_omvormer_vermogen"
    pv_forecast_now_w: str = "sensor.power_production_now"
    epex_price_eur_kwh: str = "sensor.epex_huidige_uurprijs"
    screen_small_cover: str = "cover.zonwering_woonkamer_klein"
    screen_large_cover: str = "cover.zonwering_woonkamer_groot"
    p1_power_consumed_kw: str = "sensor.p1_meter_power_consumed"  # [kW]
    p1_power_produced_kw: str = "sensor.p1_meter_power_produced"  # [kW]

    ac_kantoor_climate: str = "climate.airco_kantoor"
    ac_kantoor_power_w: str = "sensor.airco_kantoor_power"
    ac_kantoor_defrosting: str = "binary_sensor.airco_kantoor_defrosting"
    ac_kantoor_occupancy: str = "binary_sensor.airco_kantoor_occupancy"
    growlights_kantoor_power_w: str = "sensor.groeilampen_kantoor_power"

    desk_kantoor_power_w: str | None = None
    bg_manifold_supply_temp: str | None = None
    bg_manifold_return_temp: str | None = None
    bg_manifold_return_offset_c: float = 0.0
    floor_surface_sensor_wk: str | None = None


ROOM_SENSORS: dict[str, RoomSensorConfig] = {
    "wk": RoomSensorConfig(
        temp_entity_id="sensor.thermostaat_woonkamer_temperatuur",
        humidity_entity_id="sensor.thermostaat_woonkamer_luchtvochtigheid",
    ),
    "slaapkamer": RoomSensorConfig(
        temp_entity_id="sensor.thermostaat_slaapkamer_temperatuur",
        humidity_entity_id="sensor.thermostaat_slaapkamer_luchtvochtigheid",
    ),
    "badkamer": RoomSensorConfig(
        temp_entity_id="sensor.thermometer_badkamer_temperatuur",
        humidity_entity_id="sensor.thermometer_badkamer_luchtvochtigheid",
    ),
    "werkkamer_1e": RoomSensorConfig(
        temp_entity_id="sensor.thermometer_werkkamer_1e_temperatuur",
        humidity_entity_id="sensor.thermometer_werkkamer_1e_luchtvochtigheid",
    ),
    "kamer_1e_midden": RoomSensorConfig(
        temp_entity_id="sensor.thermometer_kamer_1e_midden_temperatuur",
        humidity_entity_id=(
            "sensor.thermometer_kamer_1e_midden_luchtvochtigheid"
        ),
    ),
    "kamer_1e_zuid": RoomSensorConfig(
        temp_entity_id="sensor.thermometer_kamer_1e_zuid_temperatuur",
        humidity_entity_id="sensor.thermometer_kamer_1e_zuid_luchtvochtigheid",
    ),
    "kantoor": RoomSensorConfig(
        temp_entity_id="sensor.thermometer_kantoor_temperatuur",
        humidity_entity_id="sensor.thermometer_kantoor_luchtvochtigheid",
    ),
    "badkamer_boven": RoomSensorConfig(
        temp_entity_id="sensor.thermometer_badkamer_boven_temperatuur",
        humidity_entity_id=(
            "sensor.thermometer_badkamer_boven_luchtvochtigheid"
        ),
    ),
    "logeerkamer": RoomSensorConfig(
        temp_entity_id="sensor.thermometer_logeerkamer_temperatuur",
        humidity_entity_id="sensor.thermometer_logeerkamer_luchtvochtigheid",
    ),
}

HOUSE_ENTITIES: HouseEntityConfig = HouseEntityConfig()

_OPTIONAL_ENTITY_FIELDS: frozenset[str] = frozenset(
    {
        "desk_kantoor_power_w",
        "bg_manifold_supply_temp",
        "bg_manifold_return_temp",
        "floor_surface_sensor_wk",
    }
)
_VALID_ENTITY_FIELDS: frozenset[str] = frozenset(
    f.name for f in fields(HouseEntityConfig)
)


def load_house_config(
    path: str | Path | None = None,
) -> tuple[HouseEntityConfig, dict[str, RoomSensorConfig]]:
    """Load optional JSON entity overrides onto `HOUSE_ENTITIES` & `ROOM_SENSORS`.

    Args:
        path: Path to `/config/betonbatterij_local.json` (or `None`). When
            `path` is `None` or does not exist on disk, generic placeholder
            defaults are returned without error.

    Returns:
        A tuple `(entities, room_sensors)` with any overrides merged cleanly.

    Raises:
        ValueError: If `path` exists and contains non-object JSON or invalid
            override field types.
    """
    if path is None:
        return HOUSE_ENTITIES, dict(ROOM_SENSORS)

    cfg_path = Path(path)
    if not cfg_path.is_file():
        return HOUSE_ENTITIES, dict(ROOM_SENSORS)

    with cfg_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if not isinstance(payload, Mapping):
        raise ValueError(
            f"Expected JSON object in {cfg_path}, got {type(payload).__name__}"
        )

    entity_overrides: dict[str, Any] = {}
    raw_entities = payload.get("entities")
    if raw_entities is not None:
        if not isinstance(raw_entities, Mapping):
            raise ValueError(f"'entities' in {cfg_path} must be a JSON object")
        for key, raw_val in raw_entities.items():
            field_name = str(key)
            if field_name.startswith("_"):
                continue
            if field_name not in _VALID_ENTITY_FIELDS:
                _LOGGER.warning(
                    "Ignoring unknown entity key '%s' in %s",
                    field_name,
                    cfg_path,
                )
                continue
            if field_name == "bg_manifold_return_offset_c":
                entity_overrides[field_name] = (
                    0.0 if raw_val is None else float(raw_val)
                )
            elif field_name in _OPTIONAL_ENTITY_FIELDS:
                if raw_val is None:
                    entity_overrides[field_name] = None
                elif isinstance(raw_val, str):
                    cleaned = raw_val.strip()
                    entity_overrides[field_name] = cleaned if cleaned else None
                else:
                    raise ValueError(
                        f"Expected string or null for '{field_name}' in {cfg_path}"
                    )
            else:
                if raw_val is None:
                    continue
                if not isinstance(raw_val, str) or not raw_val.strip():
                    raise ValueError(
                        f"Expected non-empty entity_id string for '{field_name}'"
                    )
                entity_overrides[field_name] = raw_val.strip()

    entities = (
        replace(HOUSE_ENTITIES, **entity_overrides)
        if entity_overrides
        else HOUSE_ENTITIES
    )

    merged_rooms: dict[str, RoomSensorConfig] = dict(ROOM_SENSORS)
    raw_rooms = payload.get("room_sensors")
    if raw_rooms is not None:
        if not isinstance(raw_rooms, Mapping):
            raise ValueError(
                f"'room_sensors' in {cfg_path} must be a JSON object"
            )
        for key, room_val in raw_rooms.items():
            room_key = str(key)
            if room_key.startswith("_"):
                continue
            base_cfg = merged_rooms.get(room_key)
            if isinstance(room_val, str) and room_val.strip():
                merged_rooms[room_key] = RoomSensorConfig(
                    temp_entity_id=room_val.strip(),
                    humidity_entity_id=(
                        base_cfg.humidity_entity_id if base_cfg else None
                    ),
                    pre_swap_temp_entity_id=(
                        base_cfg.pre_swap_temp_entity_id if base_cfg else None
                    ),
                    swap_timestamp_utc=(
                        base_cfg.swap_timestamp_utc if base_cfg else None
                    ),
                )
            elif isinstance(room_val, Mapping):
                temp_raw = room_val.get(
                    "temp_entity_id",
                    base_cfg.temp_entity_id if base_cfg else None,
                )
                if not isinstance(temp_raw, str) or not temp_raw.strip():
                    raise ValueError(
                        f"Room '{room_key}' in {cfg_path} requires a non-empty "
                        f"'temp_entity_id'"
                    )
                if "humidity_entity_id" in room_val:
                    hum_raw = room_val["humidity_entity_id"]
                    if hum_raw is not None and not isinstance(hum_raw, str):
                        raise ValueError(
                            f"Expected string or null for "
                            f"room_sensors.{room_key}.humidity_entity_id"
                        )
                    hum_id = (
                        hum_raw.strip()
                        if isinstance(hum_raw, str) and hum_raw.strip()
                        else None
                    )
                else:
                    hum_id = base_cfg.humidity_entity_id if base_cfg else None

                if "pre_swap_temp_entity_id" in room_val:
                    pre_raw = room_val["pre_swap_temp_entity_id"]
                    if pre_raw is not None and not isinstance(pre_raw, str):
                        raise ValueError(
                            f"Expected string or null for "
                            f"room_sensors.{room_key}.pre_swap_temp_entity_id"
                        )
                    pre_id = (
                        pre_raw.strip()
                        if isinstance(pre_raw, str) and pre_raw.strip()
                        else None
                    )
                else:
                    pre_id = (
                        base_cfg.pre_swap_temp_entity_id if base_cfg else None
                    )

                if "swap_timestamp_utc" in room_val:
                    swap_raw = room_val["swap_timestamp_utc"]
                    if isinstance(swap_raw, bool):
                        raise ValueError(
                            f"Expected integer or null for "
                            f"room_sensors.{room_key}.swap_timestamp_utc"
                        )
                    swap_ts = int(swap_raw) if swap_raw is not None else None
                else:
                    swap_ts = base_cfg.swap_timestamp_utc if base_cfg else None

                merged_rooms[room_key] = RoomSensorConfig(
                    temp_entity_id=temp_raw.strip(),
                    humidity_entity_id=hum_id,
                    pre_swap_temp_entity_id=pre_id,
                    swap_timestamp_utc=swap_ts,
                )
            else:
                raise ValueError(
                    f"Invalid room_sensors entry for '{room_key}' in {cfg_path}"
                )

    return entities, merged_rooms
