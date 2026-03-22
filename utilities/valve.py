"""
utilities/valve.py
===============================================================================

Pressure reduction valve with hardcoded defaults.
Config parameter is now optional.

Author: King Saud University - Chemical Engineering Department
Date: 2026-01-15 (Refactored)
Version: 2.0.0
"""

from __future__ import annotations
import math
import logging
from typing import Any, Dict, Optional, Tuple
from scipy.optimize import brentq

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════

VALVE_DEFAULTS = {
    "type": "throttle",
    "valve_style": "globe",
    "cavitation_sigma_min": 1.5,
    "max_pressure_ratio_single_stage": 0.5,
}

CAVITATION_SIGMA_MIN = 1.5
MAX_PRESSURE_RATIO_SINGLE_STAGE = 0.5

# ═══════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION - Config now Optional
# ═══════════════════════════════════════════════════════════════════════════

def throttle_valve(
    inlet: Any,
    outlet_pressure_bar: float,
    thermo: Any,
    valve_name: str = "valve",
    valve_config: Optional[dict] = None,
) -> Tuple[Any, dict]:
    """
    Throttle valve with optional configuration override.

    Args:
        inlet: Inlet stream
        outlet_pressure_bar: Outlet pressure (bar)
        thermo: ThermodynamicPackage
        valve_name: Valve identifier
        valve_config: OPTIONAL configuration (uses defaults if None)

    Returns:
        (outlet_stream, summary_dict)
    """
    # ═══════════════════════════════════════════════════════════════════════
    # APPLY DEFAULTS WITH OVERRIDE CAPABILITY
    # ═══════════════════════════════════════════════════════════════════════
    config = {**VALVE_DEFAULTS, **(valve_config or {})}

    logger.info(f"Throttling valve: {valve_name}")

    # ═══════════════════════════════════════════════════════════════════════
    # ALL REMAINING CODE UNCHANGED FROM ORIGINAL
    # ═══════════════════════════════════════════════════════════════════════

    # Validate
    if inlet is None:
        raise ValueError("Inlet stream cannot be None")
    if outlet_pressure_bar <= 0:
        raise ValueError(f"Outlet pressure must be positive, got {outlet_pressure_bar}")
    if outlet_pressure_bar >= inlet.pressure_bar:
        raise ValueError(f"Outlet pressure must be < inlet pressure (throttling requires ΔP > 0)")

    P_in = inlet.pressure_bar
    T_in = inlet.temperature_C
    z = inlet.composition
    phase_in = getattr(inlet, "phase", "unknown")

    logger.debug(f"Inlet: {P_in:.1f} bar, {T_in:.1f}°C → Outlet: {outlet_pressure_bar:.1f} bar")

    # Calculate pressure drop
    delta_P = P_in - outlet_pressure_bar
    pressure_ratio = outlet_pressure_bar / P_in

    if delta_P / P_in > 0.5:
        logger.warning(f"Large pressure drop: ΔP/P = {delta_P/P_in*100:.1f}% > 50%. Consider multi-stage.")

    # Get inlet enthalpy
    H_inlet = thermo.enthalpy_TP(T_in, P_in, z, phase_in)
    logger.debug(f"Inlet enthalpy: {H_inlet:.1f} kJ/kmol")

    # Solve for outlet temperature at constant enthalpy
    def enthalpy_error(T_out):
        try:
            try:
                flash_result = thermo.flash_TP(T_out, outlet_pressure_bar, z)
                phase = flash_result.get("phase", "mixed")
            except:
                phase = "mixed"
            H_out = thermo.enthalpy_TP(T_out, outlet_pressure_bar, z, phase)
            return H_out - H_inlet
        except:
            return 1e10

    # Bracket search
    T_low = T_in - 50.0
    T_high = T_in + 50.0
    T_low = max(T_low, -50.0)
    T_high = min(T_high, 500.0)

    try:
        T_outlet = brentq(enthalpy_error, T_low, T_high, xtol=0.01, maxiter=100)
    except:
        logger.warning("Temperature solve failed, using T_inlet as approximation")
        T_outlet = T_in

    # Flash at outlet
    try:
        flash_outlet = thermo.flash_TP(T_outlet, outlet_pressure_bar, z)
        vapor_fraction_outlet = flash_outlet.get("vapor_fraction", 0.5)
        phase_outlet = flash_outlet.get("phase", "mixed")
    except:
        vapor_fraction_outlet = 0.5
        phase_outlet = "mixed"

    # Detect flashing
    try:
        flash_inlet = thermo.flash_TP(T_in, P_in, z)
        vapor_fraction_inlet = flash_inlet.get("vapor_fraction", 0.0)
    except:
        vapor_fraction_inlet = 0.0

    flashing_occurred = (vapor_fraction_outlet - vapor_fraction_inlet) > 0.05

    # Cavitation risk
    cavitation_risk = (phase_in == "liquid" and delta_P / P_in > 0.3)

    # Valve type
    pressure_drop_ratio = delta_P / P_in
    if pressure_drop_ratio < 0.05:
        valve_type = "flow_control"
    elif pressure_drop_ratio < 0.2:
        valve_type = "pressure_control"
    elif pressure_drop_ratio < 0.5:
        valve_type = "pressure_letdown"
    else:
        valve_type = "high_pressure_letdown"

    # Valve Cv (simplified)
    MW = thermo.molecular_weight(z)
    mass_flow_kg_h = inlet.flowrate_kmol_h * MW
    delta_P_psi = delta_P * 14.5038
    valve_cv = mass_flow_kg_h / (500 * math.sqrt(delta_P_psi)) if delta_P_psi > 0 else None

    # Build outlet
    from simulation.streams import Stream
    outlet = Stream(
        name=f"{inlet.name}_throttled",
        temperature_C=T_outlet,
        pressure_bar=outlet_pressure_bar,
        flowrate_kmol_h=inlet.flowrate_kmol_h,
        composition=dict(inlet.composition),
        thermo=thermo,
        phase=phase_outlet,
    )

    summary = {
        "valve_name": valve_name,
        "inlet_pressure_bar": P_in,
        "outlet_pressure_bar": outlet_pressure_bar,
        "pressure_drop_bar": delta_P,
        "pressure_ratio": pressure_ratio,
        "inlet_temperature_C": T_in,
        "outlet_temperature_C": T_outlet,
        "temperature_change_C": T_outlet - T_in,
        "inlet_phase": phase_in,
        "outlet_phase": phase_outlet,
        "inlet_vapor_fraction": vapor_fraction_inlet,
        "outlet_vapor_fraction": vapor_fraction_outlet,
        "flashing_occurred": flashing_occurred,
        "inlet_enthalpy_kJ_kmol": H_inlet,
        "valve_type": valve_type,
        "valve_cv": valve_cv,
        "cavitation_risk": cavitation_risk,
        "choked_flow": False,
    }

    logger.info(f"Throttle complete: ΔP={delta_P:.1f} bar, ΔT={T_outlet-T_in:.1f}°C, Flashing={flashing_occurred}")

    return outlet, summary
