"""Home Assistant entity ID defaults and local JSON override loading.

Owns generic placeholder `HouseEntityConfig` / `ROOM_SENSORS` mappings and
`load_house_config()` for optional `/homeassistant/betonbatterij_local.json`
overrides. Pure physical building constants, Day-1 priors, and calibration
bounds live exclusively in `domain/priors.py`.

Note: `HouseEntityConfig.sun_entity` is reserved/unused because solar elevation
and azimuth are computed analytically in `domain/solar_weather.py`.

Pure Python with zero `homeassistant.*` or `domain.*` imports so bridge and
config tests can run offline.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
import json
import logging
from pathlib import Path
from typing import Any

__all__ = [
    "HOUSE_ENTITIES",
    "HouseEntityConfig",
    "LOCAL_CONFIG_FILENAME",
    "ROOM_SENSORS",
    "RoomSensorConfig",
    "load_house_config",
]

_LOGGER = logging.getLogger(__name__)

LOCAL_CONFIG_FILENAME: str = "betonbatterij_local.json"


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
    """Home Assistant entity IDs consumed by BetonBatterij Layer 1.

    Note:
        `sun_entity` is reserved/unused because solar geometry is computed
        analytically in `domain/solar_weather.py`.
    """

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
    sun_entity: str = "sun.sun"  # Reserved/unused; analytical solar geometry used
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


def _validate_optional_entity_field(
    raw_val: Any, context_label: str
) -> str | None:
    """Validate and normalize an optional string-or-null entity ID field."""
    if raw_val is None:
        return None
    if isinstance(raw_val, str):
        cleaned = raw_val.strip()
        return cleaned if cleaned else None
    raise ValueError(f"Expected string or null for {context_label}")


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
                entity_overrides[field_name] = _validate_optional_entity_field(
                    raw_val, f"'{field_name}' in {cfg_path}"
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
                hum_id = (
                    _validate_optional_entity_field(
                        room_val["humidity_entity_id"],
                        f"room_sensors.{room_key}.humidity_entity_id",
                    )
                    if "humidity_entity_id" in room_val
                    else (base_cfg.humidity_entity_id if base_cfg else None)
                )
                pre_id = (
                    _validate_optional_entity_field(
                        room_val["pre_swap_temp_entity_id"],
                        f"room_sensors.{room_key}.pre_swap_temp_entity_id",
                    )
                    if "pre_swap_temp_entity_id" in room_val
                    else (
                        base_cfg.pre_swap_temp_entity_id if base_cfg else None
                    )
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
