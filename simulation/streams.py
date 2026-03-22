"""
simulation/streams.py - Enhanced for Graph-based Sequential Modular

Features:
- stream_id attribute for automatic numbering (S1, S2, etc.)
- Enhanced todict() with full properties
- from_dict() classmethod for JSON serialization
- Better integration with NetworkX graphs

Date: 2026-01-05
Version: 8.2.1
"""

from __future__ import annotations
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class Stream:
    """
    Process stream with automatic ID assignment and graph integration.
    NEW: stream_id - Automatic numbering for graph edges.
    """
    name: str
    flowrate_kmol_h: float
    temperature_C: float
    pressure_bar: float
    composition: Dict[str, float]
    thermo: Any

    # NEW: Auto S1, S2, etc. (flowsheet will overwrite with topology IDs)
    stream_id: Optional[str] = None

    # Calculated properties - NOW ALLOW init=True so pump.py can pass phase
    phase: Optional[str] = None
    vapor_fraction: Optional[float] = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Initialize stream with thermo calculations."""
        # Auto-generate temporary stream_id if not provided
        if self.stream_id is None:
            self.stream_id = f"stream_{id(self)}"  # flowsheet assigns final ID

        # Normalize composition
        total = sum(self.composition.values())
        if total > 0.0:
            self.composition = {k: v / total for k, v in self.composition.items()}

        # Initial flash (no silent fallback) - only if phase not already set
        if self.phase is None:
            self._update_phase_properties()

    def _update_phase_properties(self) -> None:
        """
        Update phase and vapor fraction using thermo.
        IMPORTANT: flash_TP returns dict, not tuple.
        """
        flash_result = self.thermo.flash_TP(
            self.temperature_C,
            self.pressure_bar,
            self.composition,
        )
        self.vapor_fraction = flash_result['vapor_fraction']
        vf = self.vapor_fraction

        if vf >= 0.999:
            self.phase = "vapor"
        elif vf <= 0.001:
            self.phase = "liquid"
        else:
            self.phase = "two-phase"

    def enthalpy_kJ_kmol(self) -> float:
        """
        Calculate molar enthalpy.
        enthalpy_TP also returns float directly, not tuple.
        """
        if self.phase == "vapor":
            return self.thermo.enthalpy_TP(
                self.temperature_C,
                self.pressure_bar,
                self.composition,
                "vapor",
            )
        elif self.phase == "liquid":
            return self.thermo.enthalpy_TP(
                self.temperature_C,
                self.pressure_bar,
                self.composition,
                "liquid",
            )
        else:
            # Two-phase weighted average
            vf = self.vapor_fraction or 0.0
            H_v = self.thermo.enthalpy_TP(
                self.temperature_C,
                self.pressure_bar,
                self.composition,
                "vapor",
            )
            H_l = self.thermo.enthalpy_TP(
                self.temperature_C,
                self.pressure_bar,
                self.composition,
                "liquid",
            )
            return vf * H_v + (1.0 - vf) * H_l

    def with_TP(
        self,
        temperature_C: Optional[float] = None,
        pressure_bar: Optional[float] = None,
        name: Optional[str] = None,
    ) -> "Stream":
        """Create new stream with different T/P."""
        new_T = temperature_C if temperature_C is not None else self.temperature_C
        new_P = pressure_bar if pressure_bar is not None else self.pressure_bar
        new_name = name if name is not None else f"{self.name}_TP"

        return Stream(
            name=new_name,
            stream_id=None,  # Will be assigned by flowsheet
            flowrate_kmol_h=self.flowrate_kmol_h,
            temperature_C=new_T,
            pressure_bar=new_P,
            composition=dict(self.composition),
            thermo=self.thermo,
        )

    def todict(self) -> Dict[str, Any]:
        """
        Convert stream to dictionary with full properties.
        Enhanced for graph SM with stream_id and all properties.
        """
        return {
            "stream_id": self.stream_id,
            "name": self.name,
            "flowrate_kmol_h": round(self.flowrate_kmol_h, 4),
            "temperature_C": round(self.temperature_C, 2),
            "pressure_bar": round(self.pressure_bar, 3),
            "phase": self.phase,
            "vapor_fraction": (
                round(self.vapor_fraction, 4)
                if self.vapor_fraction is not None
                else None
            ),
            "enthalpy_kJ_kmol": round(self.enthalpy_kJ_kmol(), 2),
            "composition": {k: round(v, 6) for k, v in self.composition.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], thermo: Any) -> "Stream":
        """
        Create stream from dictionary.
        For JSON deserialization in graph SM.
        """
        return cls(
            name=data["name"],
            stream_id=data.get("stream_id"),
            flowrate_kmol_h=data["flowrate_kmol_h"],
            temperature_C=data["temperature_C"],
            pressure_bar=data["pressure_bar"],
            composition=data["composition"],
            thermo=thermo,
        )

    def copy(self, new_name: Optional[str] = None) -> "Stream":
        """Create a copy of this stream."""
        return Stream(
            name=new_name or f"{self.name}_copy",
            stream_id=None,
            flowrate_kmol_h=self.flowrate_kmol_h,
            temperature_C=self.temperature_C,
            pressure_bar=self.pressure_bar,
            composition=dict(self.composition),
            thermo=self.thermo,
        )

    def __repr__(self) -> str:
        return (
            f"Stream(id={self.stream_id}, name={self.name}, "
            f"F={self.flowrate_kmol_h:.1f} kmol/h, "
            f"T={self.temperature_C:.1f}°C, P={self.pressure_bar:.2f} bar, "
            f"phase={self.phase})"
        )
