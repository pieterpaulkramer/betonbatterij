# Lokale Docker Home Assistant Testomgeving (BetonBatterij)

Deze map bevat een volledig geïsoleerde, lokale Home Assistant Core
testopstelling in Docker. De generieke configuratie en scripts staan in
versiecontrole, terwijl alle gegenereerde runtime-bestanden (`.storage/`,
SQLite-databases, tokens, logs, exports en `betonbatterij_local.json`) via een
strikte allowlist in `.gitignore` buiten Git blijven.

> **Pytest & IDE-integratie:** De E2E-tests staan als `@pytest.mark.e2e` en
> `@pytest.mark.e2e_full` in
> [`tests/e2e/test_ha_docker.py`](../tests/e2e/test_ha_docker.py) en worden ook
> automatisch gedraaid in GitHub Actions
> ([`.github/workflows/ci.yml`](../.github/workflows/ci.yml)).
> `python -m pytest` draait standaard alleen de snelle offline unit tests;
> gebruik `python -m pytest -m e2e -v` voor de Docker smoke-suite of
> `python -m pytest -m "e2e or e2e_full" -v` voor de volledige suite.

## 1. Architectuur & Bestanden

```text
local_ha_test/
├── docker-compose.yml              # HA Core container + optionele rpi4- en laag2-profielen
├── .env.example                    # Optioneel kopiëren naar .env om HA-versie/poort te pinnen
├── run_test.ps1                    # 1-commando PowerShell wrapper (start/test/seed/logs/reset)
├── config/
│   ├── configuration.yaml          # Minimale HA-config met debug-logging op betonbatterij
│   └── packages/
│       └── kantoor_schedule.yaml   # Voorbeeld-helper (`schedule.kantoor_werktijden`)
├── share/                          # Gemount als `/share` (CSV/JSON-export doelmap)
├── mosquitto/
│   └── mosquitto.conf              # Lokale MQTT-broker voor toekomstige Laag 2 actuator-tests
└── scripts/
    ├── ha_client.py                # Zero-dependency HA REST client + automatische onboarding
    ├── mock_house.py               # Virtueel huis (62+ entiteiten & 9 fysieke scenario's)
    ├── e2e_smoke_test.py           # Volledige E2E smoke- & integratietest tegen echte HA Core
    └── import_prod_snapshot.py     # Veilige import van een gekopieerde productie Recorder-DB
```

Belangrijke ontwerpkeuzes:

- **Directe live-mount (read-only):** `../custom_components/betonbatterij` wordt
  als `/config/custom_components/betonbatterij:ro` gemount met
  `PYTHONDONTWRITEBYTECODE=1`. Codewijzigingen in de workspace zijn direct
  beschikbaar na een integratie-reload of container-restart, zonder dat de
  Linux-container `.pyc`-bestanden in Windows achterlaat.
- **Geen tweede MCP-server nodig:** Alle geautomatiseerde tests praten lokaal
  tegen `http://127.0.0.1:8123` via de HA REST API en lezen direct de gemounte
  `/config` en `/share` mappen uit. Daardoor is er nul risico op verwarring met
  de productie-HA MCP-server.
- **Automatische eerste onboarding:** Op een verse container maakt
  `ha_client.py` automatisch een lokaal dev-account aan (`dev` /
  `betonbatterij-dev-123`) via `/api/onboarding/*` en slaat de OAuth
  refresh-token op in `.ha_auth.json`. Je hoeft dus niet handmatig door de
  HA-wizard te klikken (al mag dat wel: zet in dat geval een Long-Lived Access
  Token in `local_ha_test/.ha_token`).

______________________________________________________________________

## 2. Quickstart (als je terugkomt)

Zorg dat **Docker Desktop** draait en voer vanuit de projectroot één commando
uit:

```powershell
.\local_ha_test\run_test.ps1 test
```

Dit doet volledig automatisch:

1. Start de container `betonbatterij-ha-test` op `http://127.0.0.1:8123` (eerste
   keer wordt de HA Core image gedownload).
1. Wacht tot Home Assistant reageert en voert automatische onboarding uit.
1. Injecteert alle virtuele huis-entiteiten (`cv_heating`-scenario).
1. Doorloopt de 1-click **ConfigFlow** voor `betonbatterij` en controleert dat
   een tweede ConfigFlow netjes afbreekt (`single_config_entry`).
1. Controleert dat alle **18 entiteiten** (13 sensoren, 1 switch, 4 knoppen)
   geregistreerd zijn met kloppende eenheden en attributen.
1. Test de **OptionsFlow** (zowel ongeldige waarde `99.0 L/min` als geldige
   aanpassing `5.0 -> 6.5 -> 5.0 L/min` + verificatie in diagnostics).
1. Schakelt `switch.betonbatterij_auto_kalibratie` (`off -> on`) en drukt op
   `exporteer_csv`, `rollback_parameters` en `reset_parameters`.
1. Controleert dat `/share/betonbatterij_export/telemetry_5m.csv` en
   `fit_report.json` valide zijn aangemaakt in `local_ha_test/share/`.
1. Haalt de geredacteerde **Diagnostics** JSON op en controleert de inhoud.
1. Draait alle 9 scenario's door de echte HA event-bus (`cv_heating`,
   `dhw_tank`, `defrost`, `buffer_harvest`, `ac_kantoor_heat`, `solar_day`,
   `summer_cooling`, `sensor_stale`, `cv_heating`).
1. Herlaadt de integratie en verifieert dat `.storage/betonbatterij.state`
   netjes op schijf staat.
1. Scant de Home Assistant logs en faalt bij elke `ERROR`, `Traceback`,
   `Detected blocking call` of deprecation-waarschuwing rond `betonbatterij`.

### UI in je browser openen

Ga naar `http://127.0.0.1:8123` en log in met:

- **Gebruikersnaam:** `dev`
- **Wachtwoord:** `betonbatterij-dev-123`

______________________________________________________________________

## 3. Handige Dagelijkse Commando's

| Commando                                               | Doel                                                                                                                               |
| ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| `.\local_ha_test\run_test.ps1 start`                   | Start de Docker HA container                                                                                                       |
| `.\local_ha_test\run_test.ps1 test`                    | Start HA (indien nodig) en draai de snelle E2E smoke-suite (~15s)                                                                  |
| `.\local_ha_test\run_test.ps1 test -Full`              | Uitgebreide E2E: ook `betonbatterij_local.json`, 1 echte 5m-tick + gevulde CSV-check & container-restart persistentie (max ~5 min) |
| `.\local_ha_test\run_test.ps1 seed -Scenario dhw_tank` | Zet het virtuele huis in een specifiek scenario                                                                                    |
| `.\local_ha_test\run_test.ps1 simulate -Seconds 600`   | Laat kamers en buffer live fluctueren (mooi in de UI)                                                                              |
| `.\local_ha_test\run_test.ps1 logs -Follow`            | Volg `docker logs` live (incl. debug-logging)                                                                                      |
| `.\local_ha_test\run_test.ps1 restart`                 | Herstart HA na Python-wijzigingen                                                                                                  |
| `.\local_ha_test\run_test.ps1 stop`                    | Stop en verwijder de container (state blijft bewaard)                                                                              |
| `.\local_ha_test\run_test.ps1 reset`                   | Schoon `.storage/`, DB, exports en auth-cache op voor een schone start                                                             |

### Beschikbare virtuele huis-scenario's (`mock_house.py`)

- `idle` — Winter ruststand (WP uit, kleppen dicht, `threeway_valve == Room`).
- `cv_heating` — Stabiele CV-run (`24 Hz`, `15 L/min`, `28.5 / 24.5 °C`,
  `800 W`, BG-klep + secundaire pomp aan).
- `dhw_tank` — SWW-boilerlading (`threeway_valve == Tank`, `58 °C` aanvoer,
  `z1_temp` blijft expres `heat` om de Panasonic-quirk na te bootsen).
- `defrost` — Ontdooicyclus (`wp_defrosting == on`, `1.0 °C` buiten).
- `buffer_harvest` — Laag 2 buffer-harvesting (`z1_temp == off`, secundaire pomp
  - BG-klep aan).
- `ac_kantoor_heat` — Hybride zolderkantoor verwarming via airco (`480 W`,
  occupancy aan, groeilampen aan).
- `solar_day` — Zonnige dag (`3600 W` PV, P1 teruglevering in `kW`, windsnelheid
  in `km/h` voor eenhedenconversie).
- `summer_cooling` — Zomerstand + vloer- en aircokoeling + dichte screens.
- `sensor_stale` — Woonkamer-thermometer, weer en EPEX gaan `unavailable`.

______________________________________________________________________

## 4. Testen met Echte Productie-Historie (Cold-Start Backfill & Kalibratie)

Wil je `backfill.py`, 14-daagse hindcast en MAP-kalibratie lokaal testen op je
echte huisdata?

1. Stop eerst de testcontainer:

   ```powershell
   .\local_ha_test\run_test.ps1 stop
   ```

1. Kopieer een backup van `home-assistant_v2.db` van je RPi 4 naar een
   tijdelijke plek op je PC en importeer hem (met `--fresh-backfill` zodat
   BetonBatterij opnieuw vanaf 0 de 90d LTS + 10d raw historie inleest):

   ```powershell
   python local_ha_test\scripts\import_prod_snapshot.py C:\pad\naar\home-assistant_v2.db --fresh-backfill
   ```

   *(Dit kopieert automatisch ook `betonbatterij_local.json` naar
   `local_ha_test/config/` zodat alle echte `entity_id`s matchen.)*

1. Start de container weer en volg de backfill in de logs:

   ```powershell
   .\local_ha_test\run_test.ps1 start
   .\local_ha_test\run_test.ps1 logs -Follow
   ```

______________________________________________________________________

## 5. Optionele Profielen: RPi 4-Begrenzing & Laag 2 MQTT

- **HA-versie pinnen op productie:** Kopieer `local_ha_test/.env.example` naar
  `local_ha_test/.env` en zet bijv.
  `HA_IMAGE=ghcr.io/home-assistant/home-assistant:2026.11.0`.

- **RPi 4 CPU/RAM-simulatie (2 CPU, 2 GB RAM op poort 8124):**

  ```powershell
  .\local_ha_test\run_test.ps1 start -Rpi4 -Url http://127.0.0.1:8124
  ```

- **Laag 2 virtuele actuatoren (Mosquitto MQTT):** Zodra Laag 2 echt kleppen,
  pomp, warmtepomp en airco gaat schakelen, kun je de lokale MQTT-broker
  meestarten
  (`docker compose -f local_ha_test/docker-compose.yml --profile laag2 up -d`)
  of een extra YAML-package in `local_ha_test/config/packages/` plaatsen met
  bestuurbare `template` switches en virtuele `climate`-entiteiten.
