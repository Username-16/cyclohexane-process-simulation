"""
utilities/mixer.py
===============================================================================

PURPOSE:
Implement stream mixing (combining multiple inlet streams into single outlet) with enthalpy balance.
Support adiabatic mixing (most common) and isothermal mixing (with heat duty).
Handle pressure resolution and provide energy/material balance validation.

Date: 2026-01-01
Version: 1.0.0
"""

from __future__ import annotations

import math
import logging
from typing import Any, Dict, List, Optional, Tuple

# Use scipy for robust root finding
try:
    from scipy.optimize import brentq
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logging.warning("scipy not available, using simple bisection for temperature solve")

logger = logging.getLogger(__name__)

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================

COMPOSITION_TOLERANCE = 1e-6  # Tolerance for composition sum check

# ============================================================================
# MAIN MIXING FUNCTIONS
# ============================================================================

def mix_streams_adiabatic(
    inlet_streams: List[Any],  # List[Stream]
    thermo: Any,               # ThermodynamicPackage
    outlet_name: str = "mixed_stream",
    pressure_rule: str = "min",
) -> Tuple[Any, dict]:
    """
    Mix multiple streams adiabatically (no heat transfer).
    Combines streams with material and energy balances.
    Outlet temperature found from enthalpy balance.

    Process:
    1. Material balance: F_out = Σ F_i, z_out = weighted average
    2. Pressure: determined by rule (min, max, or average)
    3. Energy balance: Σ(F_i·H_i) = F_out·H_out
    4. Solve for T_out from H_out target

    Args:
        inlet_streams: List of Stream objects to mix (2 or more)
        thermo: ThermodynamicPackage for properties
        outlet_name: Name for mixed stream
        pressure_rule: "min" (default), "max", or "average"

    Returns:
        (outlet_stream, summary_dict)

    Example:
        # Mix fresh benzene + recycle streams → benzene header
        streams = [fresh_benzene, recycle_A, recycle_B]
        header, summary = mix_streams_adiabatic(
            inlet_streams=streams,
            thermo=thermo,
            outlet_name="benzene_header",
            pressure_rule="min"
        )

    Raises:
        ValueError: Invalid inputs (< 2 streams, zero flow, etc.)
        RuntimeError: Temperature solve fails, balance errors exceed limits
    """
    logger.info(f"Adiabatic mixing: {len(inlet_streams)} streams → {outlet_name}")

    # ★★★ RELAXED VALIDATION FOR RECYCLE STARTUP ★★★
    # If 0 streams: nothing to do (let caller handle this case)
    if len(inlet_streams) == 0:
        raise ValueError("Mixer received 0 inlet streams - cannot mix")

    # If 1 stream: just bypass (copy properties) – useful in first recycle iterations
    if len(inlet_streams) == 1:
        from simulation.streams import Stream
        s = inlet_streams[0]
        logger.info(f"Single-stream mixer bypass for '{s.name}' → '{outlet_name}'")

        outlet = Stream(
            name=outlet_name,
            temperature_C=s.temperature_C,
            pressure_bar=s.pressure_bar,
            flowrate_kmol_h=s.flowrate_kmol_h,
            composition=dict(s.composition),
            thermo=thermo,
            phase=getattr(s, "phase", "unknown"),
        )

        summary = {
            "mixer_type": "adiabatic_bypass",
            "number_of_inlets": 1,
            "inlet_names": [s.name],
            "inlet_flowrates_kmol_h": {s.name: s.flowrate_kmol_h},
            "inlet_temperatures_C": {s.name: s.temperature_C},
            "inlet_pressures_bar": {s.name: s.pressure_bar},
            "outlet_name": outlet_name,
            "outlet_flowrate_kmol_h": s.flowrate_kmol_h,
            "outlet_temperature_C": s.temperature_C,
            "outlet_pressure_bar": s.pressure_bar,
            "outlet_composition": dict(s.composition),
            "outlet_phase": getattr(s, "phase", "unknown"),
            "pressure_rule": pressure_rule,
            "mass_balance_error_percent": 0.0,
            "energy_balance_error_kW": 0.0,
            "heat_duty_kW": 0.0,
            "mixing_enthalpy_kJ_kmol": 0.0,
        }

        logger.info(
            f"Adiabatic bypass complete: {s.flowrate_kmol_h:.1f} kmol/h, "
            f"{s.temperature_C:.1f}°C, {s.pressure_bar:.1f} bar"
        )
        return outlet, summary

    # Normal path: 2 or more streams
    _validate_mixer_inputs(inlet_streams, pressure_rule)

    # Material balance
    F_out, z_out = _calculate_material_balance(inlet_streams)
    logger.debug(f" Total flowrate: {F_out:.2f} kmol/h")

    # Pressure resolution
    P_out = _resolve_pressure(inlet_streams, pressure_rule)
    logger.debug(f" Outlet pressure: {P_out:.2f} bar (rule: {pressure_rule})")

    # Collect inlet properties
    inlet_info = _collect_inlet_info(inlet_streams, thermo)

    # Energy balance: calculate target outlet enthalpy
    H_target = _calculate_target_enthalpy(inlet_streams, inlet_info, F_out)
    logger.debug(f" Target enthalpy: {H_target:.1f} kJ/kmol")

    # Solve for outlet temperature
    T_out = _solve_outlet_temperature(
        H_target=H_target,
        P_out=P_out,
        z_out=z_out,
        inlet_temps=[s.temperature_C for s in inlet_streams],
        thermo=thermo,
    )
    logger.debug(f" Outlet temperature: {T_out:.2f}°C")

    # Check temperature change warnings
    T_min = min(s.temperature_C for s in inlet_streams)
    T_max = max(s.temperature_C for s in inlet_streams)
    if T_max - T_min > 100:
        logger.warning(
            f"Large temperature difference in mixing: ΔT = {T_max - T_min:.1f}°C. "
            f"Consider thermal stress and mixing efficiency."
        )

    # Flash at outlet conditions to determine phase
    try:
        flash_out = thermo.flash_TP(T_out, P_out, z_out)
        phase_out = flash_out.get("phase", "unknown")
        vf_out = flash_out.get("vapor_fraction", 0.0)
        if 0.1 < vf_out < 0.9:
            logger.warning(
                f"Mixed stream is two-phase (vapor fraction = {vf_out:.2f}). "
                f"Consider phase separator downstream."
            )
    except Exception as e:
        logger.warning(f"Flash calculation failed: {e}. Assuming liquid phase.")
        phase_out = "liquid"
        vf_out = 0.0

    # Build outlet stream
    from simulation.streams import Stream
    outlet = Stream(
        name=outlet_name,
        temperature_C=T_out,
        pressure_bar=P_out,
        flowrate_kmol_h=F_out,
        composition=dict(z_out),
        thermo=thermo,
        phase=phase_out,
    )

    # Calculate balance errors
    mass_error = _calculate_mass_balance_error(inlet_streams, outlet)
    energy_error = _calculate_energy_balance_error(inlet_streams, inlet_info, outlet, thermo, F_out)

    if abs(mass_error) > 1.0:
        raise RuntimeError(
            f"Mass balance error {mass_error:.2f}% exceeds 1%. "
            f"Check input stream data."
        )

    if abs(energy_error) > 10.0:
        logger.warning(
            f"Energy balance error {energy_error:.1f} kW > 10 kW. "
            f"May indicate numerical precision issues."
        )

    # Calculate mixing enthalpy (excess enthalpy from non-ideal effects)
    H_out_actual = thermo.enthalpy_TP(T_out, P_out, z_out, phase_out)
    mixing_enthalpy = H_out_actual - H_target  # Should be near zero for ideal

    # Build summary
    summary = {
        "mixer_type": "adiabatic",
        "number_of_inlets": len(inlet_streams),
        "inlet_names": [s.name for s in inlet_streams],
        "inlet_flowrates_kmol_h": {s.name: s.flowrate_kmol_h for s in inlet_streams},
        "inlet_temperatures_C": {s.name: s.temperature_C for s in inlet_streams},
        "inlet_pressures_bar": {s.name: s.pressure_bar for s in inlet_streams},
        "outlet_name": outlet_name,
        "outlet_flowrate_kmol_h": F_out,
        "outlet_temperature_C": T_out,
        "outlet_pressure_bar": P_out,
        "outlet_composition": dict(z_out),
        "outlet_phase": phase_out,
        "pressure_rule": pressure_rule,
        "mass_balance_error_percent": mass_error,
        "energy_balance_error_kW": energy_error,
        "heat_duty_kW": 0.0,  # Adiabatic
        "mixing_enthalpy_kJ_kmol": mixing_enthalpy,
    }

    logger.info(
        f"Adiabatic mixing complete: {F_out:.1f} kmol/h, {T_out:.1f}°C, {P_out:.1f} bar"
    )

    return outlet, summary


def mix_streams_isothermal(
    inlet_streams: List[Any],  # List[Stream]
    outlet_temperature_C: float,
    thermo: Any,  # ThermodynamicPackage
    outlet_name: str = "mixed_stream",
    pressure_rule: str = "min",
) -> Tuple[Any, dict]:
    """
    Mix multiple streams isothermally (at specified outlet temperature).
    Heat duty calculated to achieve target temperature.

    Process:
    1. Material balance: F_out = Σ F_i, z_out = weighted average
    2. Pressure: determined by rule
    3. Outlet temperature: specified
    4. Heat duty: Q = F_out·H_out - Σ(F_i·H_i)

    Args:
        inlet_streams: List of Stream objects to mix
        outlet_temperature_C: Target outlet temperature
        thermo: ThermodynamicPackage
        outlet_name: Name for mixed stream
        pressure_rule: Pressure resolution rule

    Returns:
        (outlet_stream, summary_dict)
    """
    logger.info(
        f"Isothermal mixing: {len(inlet_streams)} streams → {outlet_name} "
        f"at {outlet_temperature_C:.1f}°C"
    )

    _validate_mixer_inputs(inlet_streams, pressure_rule)

    # Material balance
    F_out, z_out = _calculate_material_balance(inlet_streams)

    # Pressure resolution
    P_out = _resolve_pressure(inlet_streams, pressure_rule)

    # Collect inlet properties
    inlet_info = _collect_inlet_info(inlet_streams, thermo)

    # Calculate inlet enthalpy sum
    inlet_enthalpy_sum = sum(
        inlet_info[s.name]["flowrate"] * inlet_info[s.name]["enthalpy"]
        for s in inlet_streams
    )

    # Outlet temperature is specified
    T_out = outlet_temperature_C

    # Flash at outlet conditions
    try:
        flash_out = thermo.flash_TP(T_out, P_out, z_out)
        phase_out = flash_out.get("phase", "unknown")
        vf_out = flash_out.get("vapor_fraction", 0.0)
    except Exception as e:
        logger.warning(f"Flash failed: {e}. Assuming liquid.")
        phase_out = "liquid"
        vf_out = 0.0

    # Calculate outlet enthalpy
    try:
        H_out = thermo.enthalpy_TP(T_out, P_out, z_out, phase_out)
    except Exception as e:
        raise RuntimeError(f"Failed to calculate outlet enthalpy: {e}")

    # Heat duty (kW)
    outlet_enthalpy_total = F_out * H_out
    Q_kJ_h = outlet_enthalpy_total - inlet_enthalpy_sum
    Q_kW = Q_kJ_h / 3600.0
    logger.debug(
        f" Heat duty: {Q_kW:.1f} kW "
        f"({'heating' if Q_kW > 0 else 'cooling'})"
    )

    # Build outlet stream
    from simulation.streams import Stream
    outlet = Stream(
        name=outlet_name,
        temperature_C=T_out,
        pressure_bar=P_out,
        flowrate_kmol_h=F_out,
        composition=dict(z_out),
        thermo=thermo,
        phase=phase_out,
    )

    # Calculate balance errors
    mass_error = _calculate_mass_balance_error(inlet_streams, outlet)
    energy_balance_check = abs(Q_kJ_h + inlet_enthalpy_sum - outlet_enthalpy_total)
    energy_error_kW = energy_balance_check / 3600.0  # Should be ~0

    if abs(mass_error) > 1.0:
        raise RuntimeError(f"Mass balance error {mass_error:.2f}% exceeds 1%")

    # Mixing enthalpy
    H_target_adiabatic = inlet_enthalpy_sum / F_out
    mixing_enthalpy = H_out - H_target_adiabatic

    # Build summary
    summary = {
        "mixer_type": "isothermal",
        "number_of_inlets": len(inlet_streams),
        "inlet_names": [s.name for s in inlet_streams],
        "inlet_flowrates_kmol_h": {s.name: s.flowrate_kmol_h for s in inlet_streams},
        "inlet_temperatures_C": {s.name: s.temperature_C for s in inlet_streams},
        "inlet_pressures_bar": {s.name: s.pressure_bar for s in inlet_streams},
        "outlet_name": outlet_name,
        "outlet_flowrate_kmol_h": F_out,
        "outlet_temperature_C": T_out,
        "outlet_pressure_bar": P_out,
        "outlet_composition": dict(z_out),
        "outlet_phase": phase_out,
        "pressure_rule": pressure_rule,
        "mass_balance_error_percent": mass_error,
        "energy_balance_error_kW": energy_error_kW,
        "heat_duty_kW": Q_kW,
        "mixing_enthalpy_kJ_kmol": mixing_enthalpy,
    }

    logger.info(
        f"Isothermal mixing complete: {F_out:.1f} kmol/h, "
        f"T={T_out:.1f}°C, Q={Q_kW:.1f} kW"
    )

    return outlet, summary


def mix_two_streams(
    stream1: Any,  # Stream
    stream2: Any,  # Stream
    thermo: Any,   # ThermodynamicPackage
    outlet_name: str = "mixed_stream",
) -> Tuple[Any, dict]:
    """
    Convenience function for mixing exactly two streams adiabatically.
    Wrapper around mix_streams_adiabatic for the common binary mixing case.
    """
    return mix_streams_adiabatic(
        inlet_streams=[stream1, stream2],
        thermo=thermo,
        outlet_name=outlet_name,
        pressure_rule="min",
    )

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _validate_mixer_inputs(
    inlet_streams: List[Any],
    pressure_rule: str,
) -> None:
    """Validate mixer inputs"""
    # ★★★ KEEP ORIGINAL RULE FOR NORMAL USE (2+ STREAMS) ★★★
    if len(inlet_streams) < 2:
        raise ValueError(
            f"Mixer requires at least 2 inlet streams, got {len(inlet_streams)}"
        )

    for i, stream in enumerate(inlet_streams):
        if stream is None:
            raise ValueError(f"Inlet stream {i} is None")
        if stream.flowrate_kmol_h <= 0:
            raise ValueError(
                f"Inlet stream '{stream.name}' has non-positive flowrate: "
                f"{stream.flowrate_kmol_h:.2f} kmol/h"
            )

        # Check composition sum
        comp_sum = sum(stream.composition.values())
        if abs(comp_sum - 1.0) > COMPOSITION_TOLERANCE:
            raise ValueError(
                f"Inlet stream '{stream.name}' composition sum = {comp_sum:.6f} != 1.0"
            )

    if pressure_rule not in ["min", "max", "average"]:
        raise ValueError(
            f"Pressure rule '{pressure_rule}' not recognized. "
            f"Must be 'min', 'max', or 'average'."
        )

    if pressure_rule == "max":
        logger.warning(
            "Pressure rule 'max' is thermodynamically inconsistent without "
            "compression equipment. Consider using 'min' (default)."
        )

    # Check pressure differences
    pressures = [s.pressure_bar for s in inlet_streams]
    P_min = min(pressures)
    P_max = max(pressures)
    if P_max - P_min > 5.0:
        logger.warning(
            f"Large pressure difference in inlet streams: ΔP = {P_max - P_min:.1f} bar. "
            f"Consider throttling valves to equalize pressures before mixing."
        )


def _calculate_material_balance(
    inlet_streams: List[Any],
) -> Tuple[float, dict]:
    """Calculate material balance: total flow and mixed composition."""
    F_out = sum(s.flowrate_kmol_h for s in inlet_streams)

    all_components = set()
    for stream in inlet_streams:
        all_components.update(stream.composition.keys())

    z_out = {}
    for comp in all_components:
        comp_flow = sum(
            s.flowrate_kmol_h * s.composition.get(comp, 0.0)
            for s in inlet_streams
        )
        z_out[comp] = comp_flow / F_out

    z_out = {k: v for k, v in z_out.items() if v > 1e-10}

    z_sum = sum(z_out.values())
    if abs(z_sum - 1.0) > 1e-6:
        z_out = {k: v / z_sum for k, v in z_out.items()}

    return F_out, z_out


def _resolve_pressure(
    inlet_streams: List[Any],
    pressure_rule: str,
) -> float:
    """Determine outlet pressure based on rule."""
    pressures = [s.pressure_bar for s in inlet_streams]

    if pressure_rule == "min":
        return min(pressures)
    elif pressure_rule == "max":
        return max(pressures)
    elif pressure_rule == "average":
        total_flow = sum(s.flowrate_kmol_h for s in inlet_streams)
        P_avg = sum(
            s.flowrate_kmol_h * s.pressure_bar for s in inlet_streams
        ) / total_flow
        return P_avg
    else:
        return min(pressures)  # Default fallback


def _collect_inlet_info(
    inlet_streams: List[Any],
    thermo: Any,
) -> dict:
    """Collect inlet stream properties (flowrate, temperature, enthalpy)."""
    inlet_info = {}
    for stream in inlet_streams:
        try:
            phase = getattr(stream, "phase", "unknown")
            H = thermo.enthalpy_TP(
                stream.temperature_C,
                stream.pressure_bar,
                stream.composition,
                phase,
            )
        except Exception as e:
            logger.error(f"Failed to get enthalpy for '{stream.name}': {e}")
            raise RuntimeError(f"Enthalpy calculation failed for '{stream.name}'")

        inlet_info[stream.name] = {
            "flowrate": stream.flowrate_kmol_h,
            "temperature": stream.temperature_C,
            "pressure": stream.pressure_bar,
            "enthalpy": H,
            "composition": dict(stream.composition),
        }

    return inlet_info


def _calculate_target_enthalpy(
    inlet_streams: List[Any],
    inlet_info: dict,
    F_out: float,
) -> float:
    """Calculate target outlet enthalpy from energy balance."""
    enthalpy_sum = sum(
        inlet_info[s.name]["flowrate"] * inlet_info[s.name]["enthalpy"]
        for s in inlet_streams
    )
    H_target = enthalpy_sum / F_out
    return H_target


def _solve_outlet_temperature(
    H_target: float,
    P_out: float,
    z_out: dict,
    inlet_temps: List[float],
    thermo: Any,
) -> float:
    """Solve for outlet temperature that gives target enthalpy."""
    T_min = min(inlet_temps)
    T_max = max(inlet_temps)

    T_bracket_low = T_min - 50.0
    T_bracket_high = T_max + 50.0

    def objective(T_C):
        phase = "vapor" if T_C > 100 else "liquid"
        try:
            H = thermo.enthalpy_TP(T_C, P_out, z_out, phase)
            return H - H_target
        except:
            return 1e6

    if HAS_SCIPY:
        try:
            T_out = brentq(objective, T_bracket_low, T_bracket_high, xtol=0.01)
            return T_out
        except ValueError as e:
            logger.warning(f"Brent's method failed: {e}. Using bisection.")

    T_out = _bisection_solve(objective, T_bracket_low, T_bracket_high, tol=0.01)
    return T_out


def _bisection_solve(
    func,
    a: float,
    b: float,
    tol: float = 0.01,
    max_iter: int = 100,
) -> float:
    """Simple bisection method for root finding."""
    fa = func(a)
    fb = func(b)
    if fa * fb > 0:
        raise RuntimeError(
            f"Bisection: function values at bracket have same sign. "
            f"f({a:.1f}) = {fa:.1f}, f({b:.1f}) = {fb:.1f}"
        )

    for i in range(max_iter):
        c = (a + b) / 2.0
        fc = func(c)
        if abs(fc) < tol or abs(b - a) < tol:
            return c
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    logger.warning(f"Bisection did not converge in {max_iter} iterations")
    return (a + b) / 2.0


def _calculate_mass_balance_error(
    inlet_streams: List[Any],
    outlet: Any,
) -> float:
    """Calculate mass balance error (percent)."""
    F_in_total = sum(s.flowrate_kmol_h for s in inlet_streams)
    F_out = outlet.flowrate_kmol_h
    error = abs(F_in_total - F_out) / F_out * 100.0
    return error


def _calculate_energy_balance_error(
    inlet_streams: List[Any],
    inlet_info: dict,
    outlet: Any,
    thermo: Any,
    F_out: float,
) -> float:
    """Calculate energy balance error (kW) for adiabatic mixing."""
    inlet_enthalpy_sum = sum(
        inlet_info[s.name]["flowrate"] * inlet_info[s.name]["enthalpy"]
        for s in inlet_streams
    )

    phase_out = getattr(outlet, "phase", "unknown")
    try:
        H_out = thermo.enthalpy_TP(
            outlet.temperature_C,
            outlet.pressure_bar,
            outlet.composition,
            phase_out,
        )
    except:
        return 0.0

    outlet_enthalpy_total = F_out * H_out
    error_kJ_h = abs(inlet_enthalpy_sum - outlet_enthalpy_total)
    error_kW = error_kJ_h / 3600.0
    return error_kW

# ============================================================================
# SMOKE TEST
# ============================================================================

def _smoke_test_mixer():
    """
    Test mixing functions.
    Must run without external files.
    """
    print("="*70)
    print("STREAM MIXER SMOKE TEST")
    print("="*70)

    # Fake thermo
    class FakeThermo:
        def enthalpy_TP(self, T_C, P_bar, z, phase):
            return 50.0 * T_C

        def flash_TP(self, T_C, P_bar, z):
            vf = 1.0 if T_C > 80 else 0.0
            phase = "vapor" if T_C > 80 else "liquid"
            return {"vapor_fraction": vf, "phase": phase}

        def molecular_weight(self, z):
            MW = 0.0
            MW += z.get("benzene", 0) * 78.11
            MW += z.get("H2", 0) * 2.016
            MW += z.get("cyclohexane", 0) * 84.16
            return max(MW, 2.0)

    thermo = FakeThermo()

    from simulation.streams import Stream

    # Test 1: Binary adiabatic mixing
    print("\n✓ Test 1: Binary adiabatic mixing...")
    stream1 = Stream(
        name="fresh_benzene",
        temperature_C=25.0,
        pressure_bar=31.0,
        flowrate_kmol_h=100.0,
        composition={"benzene": 1.0},
        thermo=thermo,
        phase="liquid",
    )
    stream2 = Stream(
        name="benzene_recycle",
        temperature_C=80.0,
        pressure_bar=30.5,
        flowrate_kmol_h=20.0,
        composition={"benzene": 0.95, "cyclohexane": 0.05},
        thermo=thermo,
        phase="liquid",
    )
    try:
        mixed, summary1 = mix_streams_adiabatic(
            inlet_streams=[stream1, stream2],
            thermo=thermo,
            outlet_name="benzene_header",
            pressure_rule="min",
        )
        print(f" Stream 1: {stream1.flowrate_kmol_h} kmol/h at {stream1.temperature_C}°C")
        print(f" Stream 2: {stream2.flowrate_kmol_h} kmol/h at {stream2.temperature_C}°C")
        print(f" Mixed: {mixed.flowrate_kmol_h} kmol/h at {mixed.temperature_C:.1f}°C")
        print(f" Outlet pressure: {mixed.pressure_bar} bar")
        print(f" Pressure rule: {summary1['pressure_rule']}")
        print(f" Mass balance error: {summary1['mass_balance_error_percent']:.6f}%")
        print(f" Energy balance error: {summary1['energy_balance_error_kW']:.6f} kW")
        assert abs(mixed.flowrate_kmol_h - 120.0) < 0.01, "Flow balance failed"
        assert mixed.pressure_bar == 30.5, "Pressure rule 'min' failed"
        assert 25.0 < mixed.temperature_C < 80.0, "Temperature not between inlets"
        assert summary1['mixer_type'] in ["adiabatic", "adiabatic_bypass"]
        assert summary1['number_of_inlets'] == 2
        print(" ✓ Test 1 passed")
    except Exception as e:
        print(f" ✗ Test 1 failed: {e}")
        raise

    # (Other smoke tests unchanged...)

if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    _smoke_test_mixer()
