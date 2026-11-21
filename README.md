# BetonBatterij — Slimme Vloerverwarming & Warmtepomp Regeling (Home Assistant)

Pragmatische verwarmings- en tapwaterregeling voor een multizone vloerverwarming
in Home Assistant op een **Raspberry Pi 4 (4 GB RAM)** — 100% pure Python
standaardbibliotheek zonder externe `numpy`/`scipy`-afhankelijkheden — specifiek
afgestemd op:

- **Warmtepomp:** Panasonic Aquarea 9 kW T-CAP (`WH-SXC09H3E5` / `WH-UX09HE5`)
  via **HeishaMon** (op de ingebouwde stooklijn + dynamische offsets)
- **Hydrauliek & Vaten:** 200L parallel CV-buffervat (`Tiki Buffer 200`),
  drukgeregelde EC-huispomp en 280–300L SWW-boiler (`OSO Delta Geocoil 300`) met
  4 extra Dallas 1-Wire sensoren
- **Afgiftesysteem:** 3 woonlagen met trage betonvloeren (beneden als 1 zone op
  HeishaMon-relais, 1e en 2e verdieping per kamer geschakeld via **ESPHome**) +
  **Mitsubishi Heavy Industries split-airco** op het zolderkantoor
- **Energie & Stuur-metric:** 14 zuid-gerichte zonnepanelen (5,74 kWp, 30°
  helling / 175° azimut), P1-slimme meter, weers-/zonvoorspelling en sturing op
  de **kale EPEX-uurprijs** ten behoeve van de energietransitie en netbalans

______________________________________________________________________

## Kernprobleem & Oplossingsrichting

Losse kamerthermostaten (`generic_thermostat` op alleen luchttemperatuur) en
losse HA-automatiseringen botsen op vier manieren met de praktijk in huis:

1. **Nachtelijk pendelen (het "04:00 uur probleem"):** Als alleen een kleine
   bovenkamer (zoals het kantoor op zolder) warmte vraagt terwijl de woonkamer
   dicht zit, kan de 9 kW T-CAP zijn minimumvermogen (~3,5 kW op 19–20 Hz) niet
   kwijt en pendelt hij elk uur in runs van ~13 minuten op lage COP, terwijl hij
   mét de woonkamer open 5 tot 9,5 uur ononderbroken en fluisterstil doorloopt.
1. **Faseverschuiving & kook-maskering in de woonkamer:** De dikke betonvloer
   beneden reageert met uren vertraging, terwijl koken of visite direct de
   kamerlucht opwarmt. Een gewone thermostaat ziet het verschil niet tussen een
   warme vloer en een warme pan op het fornuis.
1. **Onnodige ochtend-starts op tapwater & onbenutte EPEX-ruimte:** Het
   tapwatervat springt 's ochtends tijdens het douchen soms onnodig aan op de
   middensensor (terwijl er bovenin nog volop heet water zit), en
   ruimteverwarming benut zonnige/goedkope uren nog niet om koude, dure uren te
   overbruggen.
1. **Gebrek aan samenhang & uitlegbaarheid:** Verschillende kamers (zoals het
   zolderkantoor met split-airco of een thuiswerkkamer) vragen om
   werkdag-/weekschema's, tijdige opwarming en één helder overzicht in Home
   Assistant van *waarom* het systeem wacht of draait.

**Aanpak in twee lagen (simpel waar het kan, slim waar het telt):**

- **Laag 1 — Gericht Gebouw- & COP-Model + Fit-Evaluatie (Waarneming, Woonkamer,
  Zolderkantoor & WP):** Schat voor zowel de woonkamer als het zolderkantoor
  continu de vloertemperatuur, PVC-oppervlaktetemperatuur en warmtestromen via
  een exacte 2R2C ZOH Kalman Digital Twin met strak begrensde bouwkundige
  priors, fit daarnaast de **werkelijke COP-curve van de warmtepomp** (als
  functie van buitentemperatuur, aanvoertemperatuur en defrosts), met een
  veilige MAP batch-kalibratie op een eigen 90-dagen columnaire ringbuffer
  (`Herbereken`-knop met 1-klik Rollback + optionele gedempte wekelijkse update
  in de winter met out-of-sample validatiepoort) en een terugkijkende **6-uurs
  open-loop hindcast fit-evaluatie**. Slaap- en badkamers hebben géén zwaar
  fysisch model nodig.
- **Laag 2 — Pragmatische Planner & Nette Uitvoeringslaag (Aansturing):**
  Vervangt `generic_thermostat` door eigen Home Assistant `climate.*` entiteiten
  met koppelbare werkdag-/agenda-schema's en uitlegbare status, stuurt de
  Panasonic op zijn eigen stooklijn + ruime offsets, laat bovenkamers actief
  meeliften met beneden tegen pendelen, plant woonkamer en tapwater op
  **verwachte warmteprijs (`EPEX / verwachte COP`)** (in v1 met een
  begrijpelijke heuristiek, in v2 optioneel via een 24-uurs solver zoals
  EMHASS/DAO), en regelt het zolderkantoor hybride (airco + vloer).

______________________________________________________________________

## Installatie (HACS + `/config/betonbatterij_local.json`)

1. **Toevoegen via HACS / Marketplace:** Voeg deze GitHub-repository
   (`https://github.com/pieterpaulkramer/betonbatterij`) toe als *Custom
   Repository* (categorie **Integration**) en installeer **BetonBatterij**.
1. **Lokale sensor- en klepmapping aanmaken:** Kopieer
   [`betonbatterij_local.example.json`](betonbatterij_local.example.json) naar
   `/config/betonbatterij_local.json` op je Home Assistant-installatie en pas de
   `entity_id`'s onder `"entities"` en `"room_sensors"` aan op jouw eigen
   HeishaMon-, 1-Wire-, ZHA- en ESPHome-entiteiten. Zonder dit bestand valt de
   integratie terug op generieke placeholder-entiteiten (met een melding in het
   Home Assistant-logboek).
1. **Activeren & debieten instellen:** Herstart Home Assistant, ga naar
   **Instellingen → Apparaten & diensten → Integratie toevoegen** en kies
   **BetonBatterij**. Vul daarna via **Configureren** de afgelezen
   rotameter-debieten (`L/min`) per vloerverwarmingsgroep in.

______________________________________________________________________

## Opbouw van deze Repository

| Map / Bestand                                                          | Doel & Inhoud                                                                                                                                                                                                                                             |
| :--------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`custom_components/betonbatterij/`](custom_components/betonbatterij/) | Home Assistant Custom Integration: pure-Python `domain/` Digital Twin (`zoh_kalman`, `hydraulics`, `solar_weather`, `cop_model`, `evaluation`, `calibration`, `engine`) en async `ha_bridge/` (`accumulator`, `storage_ring`, `backfill`, `coordinator`). |
| [`betonbatterij_local.example.json`](betonbatterij_local.example.json) | Voorbeeldsjabloon voor `/config/betonbatterij_local.json` met generieke placeholder `entity_id`'s voor warmtepomp, kleppen en kamersensoren.                                                                                                              |
| [`tests/`](tests/)                                                     | Volledige offline `pytest`-suite voor ZOH-discretisatie, Kalman-filters, hydrauliek, COP-model, MAP-kalibratie, 90-dagen ringbuffer en lokale configuratie.                                                                                               |

______________________________________________________________________

## Globale Fasering

1. **Fase 0 — Praktische basis & metingen:** Kleine fixes in de bestaande
   opstelling (zoals kamersensoren en airco-temperatuurbron) en eenmalige
   niet-invasieve metingen van de rotameter-debieten op de verdelers.
1. **Fase 1 — Passief Gebouwmodel (*Shadow Mode*):** Eerst veilig meelezen naast
   `generic_thermostat`, historie opbouwen, vloertemperatuur van de woonkamer en
   het kantoormodel valideren en de `Herbereken parameters`-knop testen zonder
   nog één klep aan te sturen.
1. **Fase 2 — Actieve v1-Regeling (Thermostaten, Anti-Pendel, EPEX-Heuristiek,
   SWW & Hybride Zolder):** Overname van klep-, pomp- en warmtepompsturing (op
   Panasonic stooklijn + offsets) met native HA fail-safe watchdog, eigen
   `climate.*` entiteiten met schema's en uitlegbare status, slimme
   SWW/legionella-planning en hybride zoldersturing.
1. **Fase 3 (v2) — Uitgebreide 24-uurs Energie-Solver:** Optionele koppeling met
   (of uitbouw naar) een 24-uurs solver zoals EMHASS/DAO voor verdere
   co-optimalisatie en het een avond vooraf laden van geplande
   vloerverwarmingskamers.
