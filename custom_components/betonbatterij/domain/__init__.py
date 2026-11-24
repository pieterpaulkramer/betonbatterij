"""Pure-Python Domain Layer for BetonBatterij (Layer-1 Digital Twin).

Exports the curated public Layer-1 / Layer-2 domain API (`DigitalTwinEngine`,
`ThermalTwinProtocol`, and immutable domain contracts). Contains zero
`homeassistant.*`, `numpy`, `scipy`, or `house_config` dependencies.
"""

from __future__ import annotations

from custom_components.betonbatterij.domain import contracts
from custom_components.betonbatterij.domain import engine

ActuatorStepInput = contracts.ActuatorStepInput
BufferSensorOffsets = contracts.BufferSensorOffsets
CoastEstimate = contracts.CoastEstimate
CopParameters = contracts.CopParameters
DigitalTwinEngine = engine.DigitalTwinEngine
ExogenousStepInput = contracts.ExogenousStepInput
LinearizedDiscreteModel = contracts.LinearizedDiscreteModel
PreheatEstimate = contracts.PreheatEstimate
SimulationStepResult = contracts.SimulationStepResult
ThermalState = contracts.ThermalState
ThermalTwinProtocol = contracts.ThermalTwinProtocol
ZoneEvaluationReport = contracts.ZoneEvaluationReport
ZoneId = contracts.ZoneId
ZoneParameters = contracts.ZoneParameters
ZoneSnapshot = contracts.ZoneSnapshot

__all__ = [
    "ActuatorStepInput",
    "BufferSensorOffsets",
    "CoastEstimate",
    "CopParameters",
    "DigitalTwinEngine",
    "ExogenousStepInput",
    "LinearizedDiscreteModel",
    "PreheatEstimate",
    "SimulationStepResult",
    "ThermalState",
    "ThermalTwinProtocol",
    "ZoneEvaluationReport",
    "ZoneId",
    "ZoneParameters",
    "ZoneSnapshot",
]
