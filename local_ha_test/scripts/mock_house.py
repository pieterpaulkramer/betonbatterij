"""Virtual house state injector for local Docker Home Assistant testing.

Populates every entity defined in `HouseEntityConfig` and `ROOM_SENSORS` with
physically consistent Home Assistant states and attributes so BetonBatterij
can be exercised across all heating, DHW, defrost, solar, cooling, and fault
regimes without touching real hardware.
"""

from __future__ import annotations

import argparse
import logging
import math
import pathlib
import sys
import time
from collections.abc import Mapping
from typing import Any

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_LOCAL_HA_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(_LOCAL_HA_DIR) not in sys.path:
    sys.path.insert(0, str(_LOCAL_HA_DIR))

from custom_components.betonbatterij import house_config  # noqa: E402
from scripts import ha_client  # noqa: E402

_LOGGER = logging.getLogger(__name__)

SCENARIO_NAMES: tuple[str, ...] = (
    "idle",
    "cv_heating",
    "dhw_tank",
    "defrost",
    "buffer_harvest",
    "ac_kantoor_heat",
    "solar_day",
    "summer_cooling",
    "sensor_stale",
)


def _temp_attrs(friendly_name: str) -> dict[str, Any]:
    return {
        "friendly_name": friendly_name,
        "unit_of_measurement": "°C",
        "device_class": "temperature",
        "state_class": "measurement",
    }


def _hum_attrs(friendly_name: str) -> dict[str, Any]:
    return {
        "friendly_name": friendly_name,
        "unit_of_measurement": "%",
        "device_class": "humidity",
        "state_class": "measurement",
    }


def _power_attrs(friendly_name: str, unit: str = "W") -> dict[str, Any]:
    return {
        "friendly_name": friendly_name,
        "unit_of_measurement": unit,
        "device_class": "power",
        "state_class": "measurement",
    }


def build_scenario_states(
    scenario: str,
    entities: house_config.HouseEntityConfig,
    room_sensors: Mapping[str, house_config.RoomSensorConfig],
) -> dict[str, tuple[str | int | float, dict[str, Any]]]:
    """Build a complete map of `entity_id -> (state, attributes)` for a scenario.

    Args:
        scenario: One of `SCENARIO_NAMES`.
        entities: Resolved `HouseEntityConfig` entity mapping.
        room_sensors: Resolved `ROOM_SENSORS` mapping.

    Returns:
        Dictionary mapping each HA `entity_id` to `(state_value, attributes)`.

    Raises:
        ValueError: If `scenario` is not recognized.
    """
    if scenario not in SCENARIO_NAMES:
        raise ValueError(
            f"Unknown scenario '{scenario}'. Valid: {', '.join(SCENARIO_NAMES)}"
        )

    e = entities
    # Baseline winter idle state.
    states: dict[str, tuple[str | int | float, dict[str, Any]]] = {
        e.presence_boolean: ("on", {"friendly_name": "Aanwezig (Test)"}),
        e.wp_z1_climate: (
            "off",
            {"friendly_name": "WP Z1 Verwarming", "temperature": 0.0},
        ),
        e.wp_z1_cooling_climate: (
            "off",
            {"friendly_name": "WP Z1 Koeling", "temperature": 0.0},
        ),
        e.wp_operating_mode: (
            "Heat",
            {
                "friendly_name": "WP Bedrijfsmodus",
                "options": ["Heat", "Cool", "Tank", "Heat+Tank", "Cool+Tank"],
            },
        ),
        e.wp_threeway_valve: ("Room", {"friendly_name": "WP 3-wegklep"}),
        e.wp_defrosting: ("off", {"friendly_name": "WP Ontdooien"}),
        e.wp_internal_heater: ("off", {"friendly_name": "WP Back-up Heater"}),
        e.wp_pump_flow: (
            0.0,
            {
                "friendly_name": "WP Primair Debiet",
                "unit_of_measurement": "L/min",
                "state_class": "measurement",
            },
        ),
        e.wp_outlet_temp: (23.0, _temp_attrs("WP Aanvoer")),
        e.wp_inlet_temp: (22.8, _temp_attrs("WP Retour")),
        e.wp_compressor_freq: (
            0,
            {
                "friendly_name": "WP Compressor Frequentie",
                "unit_of_measurement": "Hz",
                "state_class": "measurement",
            },
        ),
        e.wp_power_consumption: (0, _power_attrs("WP Vermogen", "W")),
        e.wp_outside_temp: (6.0, _temp_attrs("WP Buitentemperatuur")),
        e.buf_top_temp: (24.5, _temp_attrs("Buffervat Boven")),
        e.buf_bot_temp: (22.5, _temp_attrs("Buffervat Onder")),
        e.dhw_top_temp: (48.0, _temp_attrs("SWW Boiler Boven")),
        e.dhw_bot_temp: (44.0, _temp_attrs("SWW Boiler Onder")),
        e.pump_secondary_switch: ("off", {"friendly_name": "Secundaire Pomp"}),
        e.valve_bg: ("off", {"friendly_name": "Klep BG"}),
        e.valve_slaapkamer: ("off", {"friendly_name": "Klep Slaapkamer"}),
        e.valve_slaapkamer_voor: (
            "off",
            {"friendly_name": "Klep Slaapkamer Voor"},
        ),
        e.valve_slaapkamer_achter: (
            "off",
            {"friendly_name": "Klep Slaapkamer Achter"},
        ),
        e.valve_badkamer: ("off", {"friendly_name": "Klep Badkamer"}),
        e.valve_werkkamer_1e: ("off", {"friendly_name": "Klep Werkkamer 1e"}),
        e.valve_kamer_1e_midden: (
            "off",
            {"friendly_name": "Klep Kamer 1e Midden"},
        ),
        e.valve_kamer_1e_zuid: ("off", {"friendly_name": "Klep Kamer 1e Zuid"}),
        e.valve_kantoor: ("off", {"friendly_name": "Klep Kantoor"}),
        e.valve_badkamer_boven: (
            "off",
            {"friendly_name": "Klep Badkamer Boven"},
        ),
        e.valve_logeerkamer: ("off", {"friendly_name": "Klep Logeerkamer"}),
        e.outdoor_temp_dak: (6.2, _temp_attrs("Buitentemperatuur Dak")),
        e.outdoor_humidity_dak: (78.0, _hum_attrs("Buitenluchtvochtigheid Dak")),
        e.weather_entity: (
            "cloudy",
            {
                "friendly_name": "Weer Thuis (Test)",
                "temperature": 6.1,
                "humidity": 80,
                "wind_speed": 14.4,  # km/h -> 4.0 m/s
                "wind_speed_unit": "km/h",
            },
        ),
        e.pv_power_w: (120.0, _power_attrs("PV Vermogen", "W")),
        e.pv_forecast_now_w: (150.0, _power_attrs("PV Verwachting Nu", "W")),
        e.epex_price_eur_kwh: (
            0.245,
            {
                "friendly_name": "EPEX Uurprijs",
                "unit_of_measurement": "EUR/kWh",
                "state_class": "measurement",
            },
        ),
        e.screen_small_cover: (
            "open",
            {"friendly_name": "Zonwering Klein", "current_position": 100},
        ),
        e.screen_large_cover: (
            "open",
            {"friendly_name": "Zonwering Groot", "current_position": 100},
        ),
        # P1 smart meter reports in kW (matching verified HA quirk).
        e.p1_power_consumed_kw: (0.45, _power_attrs("P1 Afname", "kW")),
        e.p1_power_produced_kw: (0.0, _power_attrs("P1 Teruglevering", "kW")),
        e.ac_kantoor_climate: (
            "off",
            {"friendly_name": "Airco Kantoor", "temperature": 20.0},
        ),
        e.ac_kantoor_power_w: (2.0, _power_attrs("Airco Kantoor Vermogen")),
        e.ac_kantoor_defrosting: ("off", {"friendly_name": "Airco Ontdooien"}),
        e.ac_kantoor_occupancy: ("off", {"friendly_name": "Kantoor Bezetting"}),
        e.growlights_kantoor_power_w: (
            0.0,
            _power_attrs("Groeilampen Kantoor"),
        ),
    }

    # Optional hardware sensors if configured in local JSON.
    if e.desk_kantoor_power_w:
        states[e.desk_kantoor_power_w] = (
            15.0,
            _power_attrs("Bureau Kantoor Vermogen"),
        )
    if e.bg_manifold_supply_temp:
        states[e.bg_manifold_supply_temp] = (
            24.0,
            _temp_attrs("BG Verdeler Aanvoer"),
        )
    if e.bg_manifold_return_temp:
        states[e.bg_manifold_return_temp] = (
            22.5,
            _temp_attrs("BG Verdeler Retour"),
        )
    if e.floor_surface_sensor_wk:
        states[e.floor_surface_sensor_wk] = (
            21.2,
            _temp_attrs("Vloeroppervlak Woonkamer"),
        )

    # Default room temperatures & humidities.
    room_defaults: dict[str, tuple[float, float]] = {
        "wk": (20.4, 52.0),
        "slaapkamer": (17.8, 56.0),
        "badkamer": (19.6, 62.0),
        "werkkamer_1e": (18.5, 54.0),
        "kamer_1e_midden": (18.4, 55.0),
        "kamer_1e_zuid": (18.7, 54.0),
        "kantoor": (18.9, 51.0),
        "badkamer_boven": (18.6, 58.0),
        "logeerkamer": (17.9, 55.0),
    }
    for room_key, cfg in room_sensors.items():
        temp_c, rh_pct = room_defaults.get(room_key, (19.0, 55.0))
        states[cfg.temp_entity_id] = (
            temp_c,
            _temp_attrs(f"Temperatuur {room_key}"),
        )
        if cfg.humidity_entity_id:
            states[cfg.humidity_entity_id] = (
                rh_pct,
                _hum_attrs(f"Luchtvochtigheid {room_key}"),
            )

    # Apply scenario-specific overrides.
    if scenario == "cv_heating":
        states[e.wp_z1_climate] = (
            "heat",
            {"friendly_name": "WP Z1 Verwarming", "temperature": 2.0},
        )
        states[e.wp_operating_mode] = ("Heat+Tank", states[e.wp_operating_mode][1])
        states[e.wp_threeway_valve] = ("Room", states[e.wp_threeway_valve][1])
        states[e.wp_pump_flow] = (15.0, states[e.wp_pump_flow][1])
        states[e.wp_outlet_temp] = (28.5, states[e.wp_outlet_temp][1])
        states[e.wp_inlet_temp] = (24.5, states[e.wp_inlet_temp][1])
        states[e.wp_compressor_freq] = (24, states[e.wp_compressor_freq][1])
        states[e.wp_power_consumption] = (800, states[e.wp_power_consumption][1])
        states[e.buf_top_temp] = (28.2, states[e.buf_top_temp][1])
        states[e.buf_bot_temp] = (24.6, states[e.buf_bot_temp][1])
        states[e.pump_secondary_switch] = ("on", states[e.pump_secondary_switch][1])
        states[e.valve_bg] = ("on", states[e.valve_bg][1])
        states[e.p1_power_consumed_kw] = (1.35, states[e.p1_power_consumed_kw][1])
        if e.bg_manifold_supply_temp:
            states[e.bg_manifold_supply_temp] = (
                27.9,
                states[e.bg_manifold_supply_temp][1],
            )
        if e.bg_manifold_return_temp:
            states[e.bg_manifold_return_temp] = (
                24.4,
                states[e.bg_manifold_return_temp][1],
            )

    elif scenario == "dhw_tank":
        # Deliberately keep wp_z1_climate='heat' to verify that BetonBatterij
        # detects DHW via threeway_valve=='Tank' (not z1_temp).
        states[e.wp_z1_climate] = (
            "heat",
            {"friendly_name": "WP Z1 Verwarming", "temperature": 0.0},
        )
        states[e.wp_operating_mode] = ("Heat+Tank", states[e.wp_operating_mode][1])
        states[e.wp_threeway_valve] = ("Tank", states[e.wp_threeway_valve][1])
        states[e.wp_pump_flow] = (16.5, states[e.wp_pump_flow][1])
        states[e.wp_outlet_temp] = (58.0, states[e.wp_outlet_temp][1])
        states[e.wp_inlet_temp] = (51.0, states[e.wp_inlet_temp][1])
        states[e.wp_compressor_freq] = (48, states[e.wp_compressor_freq][1])
        states[e.wp_power_consumption] = (2200, states[e.wp_power_consumption][1])
        states[e.buf_top_temp] = (27.2, states[e.buf_top_temp][1])
        states[e.buf_bot_temp] = (24.0, states[e.buf_bot_temp][1])
        states[e.dhw_top_temp] = (52.0, states[e.dhw_top_temp][1])
        states[e.dhw_bot_temp] = (46.0, states[e.dhw_bot_temp][1])
        states[e.p1_power_consumed_kw] = (2.65, states[e.p1_power_consumed_kw][1])

    elif scenario == "defrost":
        states[e.wp_z1_climate] = (
            "heat",
            {"friendly_name": "WP Z1 Verwarming", "temperature": 0.0},
        )
        states[e.wp_threeway_valve] = ("Room", states[e.wp_threeway_valve][1])
        states[e.wp_defrosting] = ("on", states[e.wp_defrosting][1])
        states[e.wp_outside_temp] = (1.0, states[e.wp_outside_temp][1])
        states[e.outdoor_temp_dak] = (1.2, states[e.outdoor_temp_dak][1])
        states[e.wp_pump_flow] = (19.0, states[e.wp_pump_flow][1])
        states[e.wp_outlet_temp] = (18.5, states[e.wp_outlet_temp][1])
        states[e.wp_inlet_temp] = (23.0, states[e.wp_inlet_temp][1])
        states[e.wp_compressor_freq] = (54, states[e.wp_compressor_freq][1])
        states[e.wp_power_consumption] = (1800, states[e.wp_power_consumption][1])
        states[e.pump_secondary_switch] = ("on", states[e.pump_secondary_switch][1])
        states[e.valve_bg] = ("on", states[e.valve_bg][1])

    elif scenario == "buffer_harvest":
        states[e.wp_z1_climate] = ("off", states[e.wp_z1_climate][1])
        states[e.wp_threeway_valve] = ("Room", states[e.wp_threeway_valve][1])
        states[e.wp_pump_flow] = (0.0, states[e.wp_pump_flow][1])
        states[e.wp_compressor_freq] = (0, states[e.wp_compressor_freq][1])
        states[e.buf_top_temp] = (29.5, states[e.buf_top_temp][1])
        states[e.buf_bot_temp] = (24.0, states[e.buf_bot_temp][1])
        states[e.pump_secondary_switch] = ("on", states[e.pump_secondary_switch][1])
        states[e.valve_bg] = ("on", states[e.valve_bg][1])

    elif scenario == "ac_kantoor_heat":
        states[e.ac_kantoor_climate] = (
            "heat",
            {"friendly_name": "Airco Kantoor", "temperature": 21.0},
        )
        states[e.ac_kantoor_power_w] = (480.0, states[e.ac_kantoor_power_w][1])
        states[e.ac_kantoor_occupancy] = ("on", states[e.ac_kantoor_occupancy][1])
        states[e.growlights_kantoor_power_w] = (
            65.0,
            states[e.growlights_kantoor_power_w][1],
        )
        if e.desk_kantoor_power_w:
            states[e.desk_kantoor_power_w] = (
                125.0,
                states[e.desk_kantoor_power_w][1],
            )
        kt_cfg = room_sensors["kantoor"]
        states[kt_cfg.temp_entity_id] = (
            20.3,
            states[kt_cfg.temp_entity_id][1],
        )

    elif scenario == "solar_day":
        states[e.outdoor_temp_dak] = (14.5, states[e.outdoor_temp_dak][1])
        states[e.wp_outside_temp] = (14.2, states[e.wp_outside_temp][1])
        states[e.weather_entity] = (
            "sunny",
            {
                "friendly_name": "Weer Thuis (Test)",
                "temperature": 14.3,
                "humidity": 55,
                "wind_speed": 10.8,  # 3.0 m/s
                "wind_speed_unit": "km/h",
            },
        )
        states[e.pv_power_w] = (3600.0, states[e.pv_power_w][1])
        states[e.pv_forecast_now_w] = (3800.0, states[e.pv_forecast_now_w][1])
        states[e.p1_power_consumed_kw] = (0.0, states[e.p1_power_consumed_kw][1])
        states[e.p1_power_produced_kw] = (2.95, states[e.p1_power_produced_kw][1])
        states[e.epex_price_eur_kwh] = (0.04, states[e.epex_price_eur_kwh][1])
        wk_cfg = room_sensors["wk"]
        states[wk_cfg.temp_entity_id] = (21.6, states[wk_cfg.temp_entity_id][1])

    elif scenario == "summer_cooling":
        states[e.outdoor_temp_dak] = (28.5, states[e.outdoor_temp_dak][1])
        states[e.wp_outside_temp] = (28.0, states[e.wp_outside_temp][1])
        states[e.weather_entity] = (
            "sunny",
            {
                "friendly_name": "Weer Thuis (Test)",
                "temperature": 28.2,
                "humidity": 48,
                "wind_speed": 7.2,
                "wind_speed_unit": "km/h",
            },
        )
        states[e.wp_operating_mode] = ("Cool", states[e.wp_operating_mode][1])
        states[e.wp_z1_cooling_climate] = (
            "cool",
            {"friendly_name": "WP Z1 Koeling", "temperature": 19.0},
        )
        states[e.ac_kantoor_climate] = (
            "cool",
            {"friendly_name": "Airco Kantoor", "temperature": 22.0},
        )
        states[e.ac_kantoor_power_w] = (650.0, states[e.ac_kantoor_power_w][1])
        states[e.screen_small_cover] = (
            "closed",
            {"friendly_name": "Zonwering Klein", "current_position": 0},
        )
        states[e.screen_large_cover] = (
            "closed",
            {"friendly_name": "Zonwering Groot", "current_position": 0},
        )

    elif scenario == "sensor_stale":
        wk_cfg = room_sensors["wk"]
        states[wk_cfg.temp_entity_id] = (
            "unavailable",
            states[wk_cfg.temp_entity_id][1],
        )
        states[e.epex_price_eur_kwh] = (
            "unavailable",
            states[e.epex_price_eur_kwh][1],
        )
        states[e.weather_entity] = ("unavailable", {"friendly_name": "Weer"})

    return states


def load_effective_house_config(
    config_dir: pathlib.Path,
) -> tuple[house_config.HouseEntityConfig, dict[str, house_config.RoomSensorConfig]]:
    """Load `betonbatterij_local.json` from `config_dir` if present, else defaults."""
    local_json = config_dir / house_config.LOCAL_CONFIG_FILENAME
    if local_json.is_file():
        _LOGGER.info("Using entity overrides from %s", local_json)
        return house_config.load_house_config(local_json)
    return house_config.load_house_config(None)


def apply_scenario(
    client: ha_client.LocalHomeAssistantClient,
    scenario: str,
    *,
    config_dir: pathlib.Path | None = None,
) -> int:
    """Push all virtual house states for `scenario` into Home Assistant.

    Args:
        client: Authenticated `LocalHomeAssistantClient`.
        scenario: Scenario name from `SCENARIO_NAMES`.
        config_dir: Optional HA `/config` directory to inspect for
            `betonbatterij_local.json`.

    Returns:
        Number of entity states updated.
    """
    eff_dir = config_dir or (_LOCAL_HA_DIR / "config")
    entities, room_sensors = load_effective_house_config(eff_dir)
    states = build_scenario_states(scenario, entities, room_sensors)
    for entity_id, (state_val, attrs) in states.items():
        client.set_state(entity_id, state_val, attrs)
    _LOGGER.info(
        "Applied scenario '%s' (%d entities updated).", scenario, len(states)
    )
    return len(states)


def run_live_simulation(
    client: ha_client.LocalHomeAssistantClient,
    *,
    duration_s: float,
    interval_s: float = 10.0,
    config_dir: pathlib.Path | None = None,
) -> None:
    """Continuously wiggle indoor/buffer temperatures during a live CV run.

    Useful for watching BetonBatterij 5-minute ticks and smart-throttled
    sensors update live in the Home Assistant UI.
    """
    eff_dir = config_dir or (_LOCAL_HA_DIR / "config")
    entities, room_sensors = load_effective_house_config(eff_dir)
    apply_scenario(client, "cv_heating", config_dir=eff_dir)

    start = time.monotonic()
    step = 0
    wk_temp_id = room_sensors["wk"].temp_entity_id
    kt_temp_id = room_sensors["kantoor"].temp_entity_id

    while (time.monotonic() - start) < duration_s:
        elapsed = time.monotonic() - start
        phase = elapsed / 600.0
        t_wk = round(20.4 + 0.35 * math.sin(phase), 2)
        t_kt = round(18.9 + 0.20 * math.cos(phase), 2)
        t_buf_top = round(28.0 + 0.6 * math.sin(phase * 1.5), 2)
        t_wp_out = round(28.5 + 0.5 * math.sin(phase * 1.5), 2)

        client.set_state(wk_temp_id, t_wk, _temp_attrs("Temperatuur wk"))
        client.set_state(kt_temp_id, t_kt, _temp_attrs("Temperatuur kantoor"))
        client.set_state(
            entities.buf_top_temp, t_buf_top, _temp_attrs("Buffervat Boven")
        )
        client.set_state(
            entities.wp_outlet_temp, t_wp_out, _temp_attrs("WP Aanvoer")
        )
        step += 1
        _LOGGER.info(
            "Sim step %d (t=%.0fs): wk=%.2f C, kt=%.2f C, buf_top=%.2f C",
            step,
            elapsed,
            t_wk,
            t_kt,
            t_buf_top,
        )
        time.sleep(interval_s)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed virtual house entities into local Docker HA."
    )
    parser.add_argument(
        "--url",
        default=ha_client.DEFAULT_BASE_URL,
        help="Local Home Assistant base URL (default: http://127.0.0.1:8123).",
    )
    parser.add_argument(
        "--scenario",
        choices=SCENARIO_NAMES,
        default="cv_heating",
        help="Named house scenario to inject.",
    )
    parser.add_argument(
        "--simulate-seconds",
        type=float,
        default=0.0,
        help="If > 0, continuously animate live states for N seconds.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Update interval [s] during --simulate-seconds.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = _parse_args()
    client = ha_client.LocalHomeAssistantClient(args.url)
    client.wait_until_ready()
    client.ensure_authenticated(auto_onboard=True)

    if args.simulate_seconds > 0.0:
        run_live_simulation(
            client,
            duration_s=args.simulate_seconds,
            interval_s=args.interval,
        )
    else:
        apply_scenario(client, args.scenario)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
