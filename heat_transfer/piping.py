"""
heat_transfer/piping.py

PURPOSE:
Calculate heat loss/gain and pressure drop for process piping exposed to outdoor conditions.
Account for ambient temperature effects (critical for Saudi Arabia hot climate).
Provide pipe sizing based on velocity constraints.

Author: King Saud University - Chemical Engineering Department
Date: 2026-01-01
Version: 1.0.0
"""

from __future__ import annotations

import math
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import fsolve

logger = logging.getLogger(__name__)

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================

GRAVITY = 9.81  # m/s²
R_GAS = 8.314  # J/(mol·K)

# Standard pipe sizes (Schedule 40) - NPS (inches) to dimensions (mm)
PIPE_SCHEDULE_40 = {
    1.0: {"OD_mm": 33.4, "wall_mm": 3.38, "ID_mm": 26.64},
    1.5: {"OD_mm": 48.3, "wall_mm": 3.68, "ID_mm": 40.94},
    2.0: {"OD_mm": 60.3, "wall_mm": 3.91, "ID_mm": 52.48},
    3.0: {"OD_mm": 88.9, "wall_mm": 5.49, "ID_mm": 77.92},
    4.0: {"OD_mm": 114.3, "wall_mm": 6.02, "ID_mm": 102.26},
    6.0: {"OD_mm": 168.3, "wall_mm": 7.11, "ID_mm": 154.08},
    8.0: {"OD_mm": 219.1, "wall_mm": 8.18, "ID_mm": 202.74},
    10.0: {"OD_mm": 273.0, "wall_mm": 9.27, "ID_mm": 254.46},
    12.0: {"OD_mm": 323.8, "wall_mm": 9.53, "ID_mm": 304.74},
}

# Material thermal conductivities (W/(m·K))
THERMAL_CONDUCTIVITY = {
    "carbon_steel": 50.0,
    "stainless_steel": 16.0,
    "copper": 400.0,
    "mineral_wool": 0.04,
    "calcium_silicate": 0.06,
    "fiberglass": 0.035,
}


# ============================================================================
# MAIN PIPING FUNCTION
# ============================================================================

def apply_piping(
    inlet: Any,  # Stream
    pipe_config: dict,
    thermo: Any,  # ThermodynamicPackage
    ambient_temperature_C: float = 35.0,
) -> Tuple[Any, dict]:
    """
    Calculate heat loss/gain and pressure drop for process piping.

    Args:
        inlet: Inlet stream
        pipe_config: Pipe geometry and insulation configuration
        thermo: ThermodynamicPackage for property calculations
        ambient_temperature_C: Outdoor ambient temperature (default 35°C for Riyadh)

    Returns:
        (outlet_stream, summary_dict)

    Raises:
        ValueError: Invalid inputs
        RuntimeError: Calculation failure or phase change
    """
    logger.info(f"Applying piping to stream: {inlet.name}")

    # Validate inputs
    _validate_piping_inputs(inlet, pipe_config)

    # Extract configuration
    config = _extract_pipe_config(pipe_config)
    pipe_name = config["name"]

    # Get fluid properties at inlet conditions
    logger.debug(f"  Inlet: T={inlet.temperature_C:.1f}°C, P={inlet.pressure_bar:.1f} bar")

    try:
        properties = _calculate_fluid_properties(
            stream=inlet,
            thermo=thermo,
        )
    except Exception as e:
        raise RuntimeError(f"Fluid property calculation failed: {e}") from e

    # Determine pipe diameter
    if config["diameter_m"] is None:
        D_m = _size_pipe_from_velocity(
            flowrate_kmol_h=inlet.flowrate_kmol_h,
            properties=properties,
            max_velocity=config["max_velocity_m_s"],
            schedule=config["schedule"],
        )
        logger.info(f"  Auto-sized pipe diameter: {D_m*1000:.1f} mm")
    else:
        D_m = config["diameter_m"]

    # Get pipe inner diameter from schedule
    D_inner_m, D_outer_m = _get_pipe_dimensions(D_m, config["schedule"])

    # Calculate flow parameters
    flow_params = _calculate_flow_parameters(
        flowrate_kmol_h=inlet.flowrate_kmol_h,
        diameter_m=D_inner_m,
        properties=properties,
        roughness_m=config["roughness_mm"] / 1000.0,
    )

    logger.debug(
        f"  Velocity: {flow_params['velocity_m_s']:.2f} m/s, "
        f"Re: {flow_params['reynolds_number']:.0f} ({flow_params['flow_regime']})"
    )

    # Check velocity constraint
    if flow_params["velocity_m_s"] > config["max_velocity_m_s"]:
        logger.warning(
            f"Velocity {flow_params['velocity_m_s']:.1f} m/s exceeds "
            f"maximum {config['max_velocity_m_s']:.1f} m/s (erosion risk)"
        )

    # Calculate pressure drop
    pressure_drop = _calculate_pressure_drop(
        flow_params=flow_params,
        length_m=config["length_m"],
        diameter_m=D_inner_m,
        properties=properties,
        elevation_change_m=config["elevation_change_m"],
    )

    outlet_pressure = inlet.pressure_bar - pressure_drop["total_bar"]

    if outlet_pressure <= 0.1:
        raise RuntimeError(
            f"Outlet pressure {outlet_pressure:.2f} bar is too low "
            f"(excessive pressure drop {pressure_drop['total_bar']:.2f} bar)"
        )

    logger.debug(f"  Pressure drop: {pressure_drop['total_bar']*1000:.1f} mbar")

    # Calculate heat transfer
    heat_transfer = _calculate_heat_transfer(
        inlet_temperature_C=inlet.temperature_C,
        ambient_temperature_C=ambient_temperature_C,
        length_m=config["length_m"],
        D_inner_m=D_inner_m,
        D_outer_m=D_outer_m,
        insulation_thickness_m=config["insulation"]["thickness_mm"] / 1000.0,
        insulation_k=config["insulation"]["conductivity_W_mK"],
        pipe_k=THERMAL_CONDUCTIVITY.get(config["material"], 50.0),
        flow_params=flow_params,
        properties=properties,
    )

    outlet_temperature = heat_transfer["outlet_temperature_C"]
    temperature_change = outlet_temperature - inlet.temperature_C

    logger.debug(
        f"  Heat transfer: ΔT={temperature_change:.2f}°C, "
        f"Q={heat_transfer['heat_loss_or_gain_kW']:.1f} kW"
    )

    # Check for excessive temperature change
    if abs(temperature_change) > 50.0:
        logger.warning(
            f"Large temperature change {temperature_change:.1f}°C detected. "
            f"Check insulation adequacy."
        )

    # Check for phase change
    _check_phase_change(
        inlet=inlet,
        outlet_temperature_C=outlet_temperature,
        outlet_pressure_bar=outlet_pressure,
        thermo=thermo,
    )

    # Build outlet stream
    outlet_stream = _build_outlet_stream(
        inlet=inlet,
        outlet_temperature_C=outlet_temperature,
        outlet_pressure_bar=outlet_pressure,
        thermo=thermo,
    )

    # Build summary
    summary = {
        "pipe_name": pipe_name,
        "inlet_temperature_C": inlet.temperature_C,
        "outlet_temperature_C": outlet_temperature,
        "temperature_change_C": temperature_change,
        "inlet_pressure_bar": inlet.pressure_bar,
        "outlet_pressure_bar": outlet_pressure,
        "pressure_drop_bar": pressure_drop["total_bar"],
        "pressure_drop_mbar": pressure_drop["total_bar"] * 1000.0,
        "heat_loss_or_gain_kW": heat_transfer["heat_loss_or_gain_kW"],
        "pipe_length_m": config["length_m"],
        "pipe_diameter_m": D_m,
        "pipe_inner_diameter_m": D_inner_m,
        "pipe_schedule": config["schedule"],
        "insulation_thickness_mm": config["insulation"]["thickness_mm"],
        "insulation_type": config["insulation"]["type"],
        "fluid_velocity_m_s": flow_params["velocity_m_s"],
        "mass_velocity_kg_m2s": flow_params["mass_velocity_kg_m2s"],
        "reynolds_number": flow_params["reynolds_number"],
        "flow_regime": flow_params["flow_regime"],
        "friction_factor": flow_params["friction_factor"],
        "heat_transfer_coefficient_W_m2K": heat_transfer["h_inside_W_m2K"],
        "overall_U_W_m2K": heat_transfer["overall_U_W_m2K"],
        "ambient_temperature_C": ambient_temperature_C,
    }

    logger.info(
        f"Piping complete: ΔT={temperature_change:.1f}°C, "
        f"ΔP={pressure_drop['total_bar']*1000:.0f} mbar"
    )

    return outlet_stream, summary


# ============================================================================
# VALIDATION
# ============================================================================

def _validate_piping_inputs(inlet: Any, pipe_config: dict) -> None:
    """Validate piping inputs"""

    if inlet is None:
        raise ValueError("Inlet stream cannot be None")

    if inlet.flowrate_kmol_h <= 0:
        raise ValueError(f"Inlet flowrate must be positive, got {inlet.flowrate_kmol_h}")

    if "length_m" not in pipe_config:
        raise ValueError("Pipe config must specify length_m")

    length = pipe_config["length_m"]
    if length <= 0:
        raise ValueError(f"Pipe length must be positive, got {length}")

    if "diameter_m" in pipe_config and pipe_config["diameter_m"] is not None:
        D = pipe_config["diameter_m"]
        if D <= 0:
            raise ValueError(f"Pipe diameter must be positive, got {D}")

    if "insulation" in pipe_config:
        insul = pipe_config["insulation"]
        if "thickness_mm" in insul and insul["thickness_mm"] < 0:
            raise ValueError("Insulation thickness cannot be negative")


def _extract_pipe_config(pipe_config: dict) -> dict:
    """Extract and normalize pipe configuration"""

    config = {
        "name": pipe_config.get("name", "pipe"),
        "length_m": pipe_config["length_m"],
        "diameter_m": pipe_config.get("diameter_m"),
        "schedule": pipe_config.get("schedule", "40"),
        "material": pipe_config.get("material", "carbon_steel"),
        "orientation": pipe_config.get("orientation", "horizontal"),
        "elevation_change_m": pipe_config.get("elevation_change_m", 0.0),
        "roughness_mm": pipe_config.get("roughness_mm", 0.045),
        "max_velocity_m_s": pipe_config.get("max_velocity_m_s", 15.0),
        "max_pressure_drop_bar_per_100m": pipe_config.get("max_pressure_drop_bar_per_100m", 0.5),
    }

    # Insulation config
    insulation = pipe_config.get("insulation", {})
    config["insulation"] = {
        "type": insulation.get("type", "mineral_wool"),
        "thickness_mm": insulation.get("thickness_mm", 50.0),
        "conductivity_W_mK": insulation.get(
            "conductivity_W_mK",
            THERMAL_CONDUCTIVITY.get(insulation.get("type", "mineral_wool"), 0.04)
        ),
    }

    return config


# ============================================================================
# FLUID PROPERTIES
# ============================================================================

def _calculate_fluid_properties(stream: Any, thermo: Any) -> dict:
    """Calculate fluid properties at stream conditions"""

    T_C = stream.temperature_C
    P_bar = stream.pressure_bar
    z = stream.composition
    phase = getattr(stream, "phase", "vapor")

    # Density
    try:
        rho = thermo.density_TP(T_C, P_bar, z, phase)
    except:
        # Fallback: ideal gas for vapor, typical liquid density
        if phase == "vapor":
            MW = thermo.molecular_weight(z)
            T_K = T_C + 273.15
            rho = P_bar * 1e5 * MW / (R_GAS * T_K)  # kg/m³
        else:
            rho = 700.0  # kg/m³

    # Molecular weight
    MW = thermo.molecular_weight(z)

    # Viscosity (estimate if not available)
    try:
        mu = thermo.viscosity_TP(T_C, P_bar, z, phase)
    except:
        if phase == "vapor":
            mu = 1.5e-5  # Pa·s (typical gas)
        else:
            mu = 3.0e-4  # Pa·s (typical hydrocarbon liquid)

    # Heat capacity
    try:
        Cp = thermo.heat_capacity_TP(T_C, P_bar, z, phase)  # kJ/(kmol·K)
        Cp_J_kgK = Cp * 1000.0 / MW  # J/(kg·K)
    except:
        # Estimate
        if phase == "vapor":
            Cp_J_kgK = 2000.0  # J/(kg·K)
        else:
            Cp_J_kgK = 2500.0  # J/(kg·K)

    # Thermal conductivity (estimate)
    if phase == "vapor":
        k_fluid = 0.03  # W/(m·K) typical gas
    else:
        k_fluid = 0.15  # W/(m·K) typical liquid

    properties = {
        "density_kg_m3": rho,
        "viscosity_Pa_s": mu,
        "heat_capacity_J_kgK": Cp_J_kgK,
        "thermal_conductivity_W_mK": k_fluid,
        "molecular_weight_kg_kmol": MW,
        "phase": phase,
    }

    return properties


# ============================================================================
# PIPE SIZING
# ============================================================================

def _size_pipe_from_velocity(
    flowrate_kmol_h: float,
    properties: dict,
    max_velocity: float,
    schedule: str,
) -> float:
    """Size pipe diameter from velocity constraint"""

    # Mass flow rate
    m_dot_kg_s = flowrate_kmol_h * properties["molecular_weight_kg_kmol"] / 3600.0

    # Required area
    rho = properties["density_kg_m3"]
    v_design = 0.8 * max_velocity  # Use 80% of max

    A_required = m_dot_kg_s / (rho * v_design)

    # Diameter
    D_required = math.sqrt(4 * A_required / math.pi)

    # Round up to nearest standard size
    standard_sizes = sorted(PIPE_SCHEDULE_40.keys())
    D_inches = D_required * 39.37  # Convert m to inches

    for size in standard_sizes:
        if size >= D_inches:
            D_standard_m = PIPE_SCHEDULE_40[size]["ID_mm"] / 1000.0
            return D_standard_m

    # If larger than largest standard, use calculated
    return D_required


def _get_pipe_dimensions(D_nominal_m: float, schedule: str) -> Tuple[float, float]:
    """Get inner and outer diameters from nominal diameter and schedule"""

    # Find closest standard size
    D_inches = D_nominal_m * 39.37

    closest_size = min(PIPE_SCHEDULE_40.keys(), key=lambda x: abs(x - D_inches))

    pipe_data = PIPE_SCHEDULE_40[closest_size]

    D_inner_m = pipe_data["ID_mm"] / 1000.0
    D_outer_m = pipe_data["OD_mm"] / 1000.0

    return D_inner_m, D_outer_m


# ============================================================================
# FLOW PARAMETERS
# ============================================================================

def _calculate_flow_parameters(
    flowrate_kmol_h: float,
    diameter_m: float,
    properties: dict,
    roughness_m: float,
) -> dict:
    """Calculate flow velocity, Reynolds number, and friction factor"""

    # Mass flow rate
    m_dot_kg_s = flowrate_kmol_h * properties["molecular_weight_kg_kmol"] / 3600.0

    # Cross-sectional area
    A = math.pi / 4 * diameter_m**2

    # Velocity
    rho = properties["density_kg_m3"]
    velocity = m_dot_kg_s / (rho * A)

    # Mass velocity (G)
    G = m_dot_kg_s / A  # kg/(m²·s)

    # Reynolds number
    mu = properties["viscosity_Pa_s"]
    Re = rho * velocity * diameter_m / mu

    # Flow regime
    if Re < 2300:
        flow_regime = "laminar"
    elif Re < 4000:
        flow_regime = "transition"
    else:
        flow_regime = "turbulent"

    # Friction factor
    f = _calculate_friction_factor(Re, roughness_m, diameter_m)

    params = {
        "velocity_m_s": velocity,
        "mass_velocity_kg_m2s": G,
        "reynolds_number": Re,
        "flow_regime": flow_regime,
        "friction_factor": f,
    }

    return params


def _calculate_friction_factor(Re: float, roughness_m: float, diameter_m: float) -> float:
    """Calculate Darcy friction factor using Colebrook-White equation"""

    if Re < 2300:
        # Laminar flow
        return 64.0 / Re

    # Turbulent flow - use Swamee-Jain explicit approximation
    eps_D = roughness_m / diameter_m

    if Re > 4000:
        # Swamee-Jain equation
        numerator = 0.25
        log_term = math.log10(eps_D / 3.7 + 5.74 / (Re**0.9))
        denominator = log_term**2

        if denominator > 1e-10:
            f = numerator / denominator
        else:
            f = 0.02  # Fallback
    else:
        # Transition region - interpolate
        f_lam = 64.0 / 2300.0
        f_turb = 0.02  # Approximate
        f = f_lam + (f_turb - f_lam) * (Re - 2300) / (4000 - 2300)

    return f


# ============================================================================
# PRESSURE DROP
# ============================================================================

def _calculate_pressure_drop(
    flow_params: dict,
    length_m: float,
    diameter_m: float,
    properties: dict,
    elevation_change_m: float,
) -> dict:
    """Calculate total pressure drop (friction + elevation)"""

    # Friction pressure drop (Darcy-Weisbach)
    f = flow_params["friction_factor"]
    v = flow_params["velocity_m_s"]
    rho = properties["density_kg_m3"]

    dP_friction_Pa = f * (length_m / diameter_m) * (rho * v**2 / 2.0)
    dP_friction_bar = dP_friction_Pa / 1e5

    # Elevation pressure drop
    dP_elevation_Pa = rho * GRAVITY * elevation_change_m
    dP_elevation_bar = dP_elevation_Pa / 1e5

    # Total
    dP_total_bar = dP_friction_bar + dP_elevation_bar

    drop = {
        "friction_bar": dP_friction_bar,
        "elevation_bar": dP_elevation_bar,
        "total_bar": dP_total_bar,
    }

    return drop


# ============================================================================
# HEAT TRANSFER
# ============================================================================

def _calculate_heat_transfer(
    inlet_temperature_C: float,
    ambient_temperature_C: float,
    length_m: float,
    D_inner_m: float,
    D_outer_m: float,
    insulation_thickness_m: float,
    insulation_k: float,
    pipe_k: float,
    flow_params: dict,
    properties: dict,
) -> dict:
    """Calculate heat transfer and outlet temperature using NTU method"""

    # Inside heat transfer coefficient (Dittus-Boelter for turbulent)
    h_inside = _calculate_inside_htc(
        reynolds=flow_params["reynolds_number"],
        diameter_m=D_inner_m,
        properties=properties,
        heating=(ambient_temperature_C > inlet_temperature_C),
    )

    # Outside heat transfer coefficient (natural convection + radiation)
    h_outside = 20.0  # W/(m²·K) typical for outdoor conditions

    # Radii
    r_inner = D_inner_m / 2.0
    r_outer = D_outer_m / 2.0
    r_insul_outer = r_outer + insulation_thickness_m

    # Overall heat transfer coefficient (series resistances)
    # 1/U = 1/h_in + R_pipe + R_insul + 1/h_out

    # Resistance of pipe wall
    if r_outer > r_inner:
        r_log_mean_pipe = (r_outer - r_inner) / math.log(r_outer / r_inner)
        R_pipe = (r_outer - r_inner) / (pipe_k * r_log_mean_pipe)
    else:
        R_pipe = 0.0

    # Resistance of insulation
    if insulation_thickness_m > 1e-6:
        r_log_mean_insul = (r_insul_outer - r_outer) / math.log(r_insul_outer / r_outer)
        R_insul = (r_insul_outer - r_outer) / (insulation_k * r_log_mean_insul)
    else:
        R_insul = 0.0
        r_insul_outer = r_outer

    # Total resistance (per unit area based on outer insulation surface)
    R_total = 1.0 / h_inside + R_pipe + R_insul + 1.0 / h_outside

    # Overall U based on outer surface
    U = 1.0 / R_total

    # Heat transfer rate using NTU method
    # Q = m_dot * Cp * (T_inlet - T_outlet)
    # For pipe with ambient: T_outlet = T_amb + (T_inlet - T_amb) * exp(-NTU)

    flowrate_kmol_h = flow_params["mass_velocity_kg_m2s"] * (math.pi / 4 * D_inner_m**2) * 3600.0 / properties["molecular_weight_kg_kmol"]
    m_dot_kg_s = flowrate_kmol_h * properties["molecular_weight_kg_kmol"] / 3600.0

    Cp = properties["heat_capacity_J_kgK"]

    # Heat capacity rate
    C_fluid = m_dot_kg_s * Cp  # W/K

    # Heat transfer area (based on outer insulated surface)
    A_outer = math.pi * 2 * r_insul_outer * length_m  # m²

    # NTU
    if C_fluid > 1e-6:
        NTU = U * A_outer / C_fluid
    else:
        NTU = 0.0

    # Outlet temperature
    T_outlet_C = ambient_temperature_C + (inlet_temperature_C - ambient_temperature_C) * math.exp(-NTU)

    # Heat loss/gain
    Q_W = m_dot_kg_s * Cp * (T_outlet_C - inlet_temperature_C)
    Q_kW = Q_W / 1000.0

    heat_transfer = {
        "outlet_temperature_C": T_outlet_C,
        "heat_loss_or_gain_kW": Q_kW,
        "h_inside_W_m2K": h_inside,
        "h_outside_W_m2K": h_outside,
        "overall_U_W_m2K": U,
        "NTU": NTU,
    }

    return heat_transfer


def _calculate_inside_htc(
    reynolds: float,
    diameter_m: float,
    properties: dict,
    heating: bool,
) -> float:
    """Calculate inside convective heat transfer coefficient"""

    k_fluid = properties["thermal_conductivity_W_mK"]
    Cp = properties["heat_capacity_J_kgK"]
    mu = properties["viscosity_Pa_s"]

    # Prandtl number
    Pr = Cp * mu / k_fluid
    Pr = max(0.5, min(Pr, 100.0))  # Clamp to reasonable range

    if reynolds > 4000:
        # Turbulent - Dittus-Boelter
        n = 0.4 if heating else 0.3
        Nu = 0.023 * (reynolds**0.8) * (Pr**n)
    elif reynolds < 2300:
        # Laminar - fully developed
        Nu = 3.66
    else:
        # Transition - interpolate
        Nu_lam = 3.66
        Nu_turb = 0.023 * (4000**0.8) * (Pr**0.4)
        Nu = Nu_lam + (Nu_turb - Nu_lam) * (reynolds - 2300) / (4000 - 2300)

    # Heat transfer coefficient
    h = Nu * k_fluid / diameter_m

    return h


# ============================================================================
# PHASE CHANGE DETECTION
# ============================================================================

def _check_phase_change(
    inlet: Any,
    outlet_temperature_C: float,
    outlet_pressure_bar: float,
    thermo: Any,
) -> None:
    """Check if phase change occurs and warn"""

    try:
        # Flash at outlet conditions
        flash_result = thermo.flash_TP(
            outlet_temperature_C,
            outlet_pressure_bar,
            inlet.composition
        )

        vapor_fraction = flash_result.get("vapor_fraction", 1.0)

        # Get inlet phase
        inlet_phase = getattr(inlet, "phase", "vapor")
        inlet_vf = 1.0 if inlet_phase == "vapor" else 0.0

        # Check for significant phase change
        vf_change = abs(vapor_fraction - inlet_vf)

        if vf_change > 0.05:
            logger.warning(
                f"Phase change detected: vapor fraction changed from {inlet_vf:.2f} "
                f"to {vapor_fraction:.2f}. Two-phase flow may require different correlations."
            )

            if 0.05 < vapor_fraction < 0.95:
                # Significant two-phase flow
                logger.warning(
                    "Two-phase flow detected in pipe. Pressure drop and heat transfer "
                    "correlations may be inaccurate. Consider adding heat exchanger "
                    "before piping or using two-phase correlations."
                )

    except Exception as e:
        logger.debug(f"Phase check failed: {e}")


# ============================================================================
# STREAM BUILDING
# ============================================================================

def _build_outlet_stream(
    inlet: Any,
    outlet_temperature_C: float,
    outlet_pressure_bar: float,
    thermo: Any,
) -> Any:
    """Build outlet stream with updated conditions"""

    from simulation.streams import Stream

    # Determine phase at outlet (flash if needed)
    try:
        flash_result = thermo.flash_TP(
            outlet_temperature_C,
            outlet_pressure_bar,
            inlet.composition
        )
        phase = flash_result.get("phase", inlet.phase if hasattr(inlet, "phase") else "vapor")
    except:
        phase = getattr(inlet, "phase", "vapor")

    outlet = Stream(
        name=f"{inlet.name}_piped",
        temperature_C=outlet_temperature_C,
        pressure_bar=outlet_pressure_bar,
        flowrate_kmol_h=inlet.flowrate_kmol_h,
        composition=dict(inlet.composition),
        thermo=thermo,
        phase=phase,
    )

    return outlet


# ============================================================================
# SMOKE TEST
# ============================================================================

def _smoke_test_piping():
    """
    Test with hot stream cooling in outdoor pipe.
    Must run without external files.
    """
    print("="*70)
    print("PIPING HEAT TRANSFER & PRESSURE DROP SMOKE TEST")
    print("="*70)

    # Fake thermo
    class FakeThermo:
        def flash_TP(self, T_C, P_bar, z):
            return {"vapor_fraction": 1.0 if T_C > 100 else 0.0, "phase": "vapor"}

        def enthalpy_TP(self, T_C, P_bar, z, phase):
            return 50.0 * T_C

        def molecular_weight(self, z):
            MW = 0.0
            MW += z.get("H2", 0.0) * 2.016
            MW += z.get("benzene", 0.0) * 78.11
            MW += z.get("cyclohexane", 0.0) * 84.16
            return max(MW, 10.0)

        def density_TP(self, T_C, P_bar, z, phase):
            if phase == "vapor":
                MW = self.molecular_weight(z)
                T_K = T_C + 273.15
                return P_bar * 1e5 * MW / (8.314 * T_K)
            else:
                return 700.0

        def heat_capacity_TP(self, T_C, P_bar, z, phase):
            return 50.0  # kJ/(kmol·K)

    thermo = FakeThermo()

    # Import Stream
    from simulation.streams import Stream

    # Hot vapor stream
    print("\n✓ Creating hot vapor stream...")
    inlet = Stream(
        name="reactor_effluent",
        temperature_C=250.0,
        pressure_bar=30.0,
        flowrate_kmol_h=500.0,
        composition={"H2": 0.7, "benzene": 0.2, "cyclohexane": 0.1},
        thermo=thermo,
        phase="vapor"
    )

    print(f"  Inlet: T={inlet.temperature_C:.1f}°C, P={inlet.pressure_bar:.1f} bar")

    # Pipe config
    pipe_config = {
        "name": "transfer_line",
        "length_m": 50.0,
        "diameter_m": 0.15,  # 150 mm (6 inch)
        "schedule": "40",
        "material": "carbon_steel",
        "insulation": {
            "type": "mineral_wool",
            "thickness_mm": 50.0,
            "conductivity_W_mK": 0.04
        },
        "orientation": "horizontal",
        "elevation_change_m": 0.0,
        "roughness_mm": 0.045,
        "max_velocity_m_s": 15.0,
        "max_pressure_drop_bar_per_100m": 0.5
    }

    print("\n✓ Running piping calculation...")
    print(f"  Pipe: {pipe_config['length_m']:.0f}m × {pipe_config['diameter_m']*1000:.0f}mm")
    print(f"  Insulation: {pipe_config['insulation']['thickness_mm']:.0f}mm {pipe_config['insulation']['type']}")
    print(f"  Ambient: 35°C (Riyadh summer)")

    # Run piping calculation
    try:
        outlet, summary = apply_piping(
            inlet=inlet,
            pipe_config=pipe_config,
            thermo=thermo,
            ambient_temperature_C=35.0
        )

        print("\n✓ Piping calculation completed successfully")
        print(f"\n  Temperature:")
        print(f"    Inlet: {inlet.temperature_C:.1f}°C")
        print(f"    Outlet: {outlet.temperature_C:.1f}°C")
        print(f"    Change: {summary['temperature_change_C']:.1f}°C")
        print(f"    Heat loss: {summary['heat_loss_or_gain_kW']:.1f} kW")

        print(f"\n  Pressure:")
        print(f"    Inlet: {inlet.pressure_bar:.2f} bar")
        print(f"    Outlet: {outlet.pressure_bar:.2f} bar")
        print(f"    Drop: {summary['pressure_drop_mbar']:.1f} mbar")

        print(f"\n  Flow:")
        print(f"    Velocity: {summary['fluid_velocity_m_s']:.1f} m/s")
        print(f"    Reynolds: {summary['reynolds_number']:.0f}")
        print(f"    Regime: {summary['flow_regime']}")
        print(f"    Friction factor: {summary['friction_factor']:.4f}")

        print(f"\n  Heat transfer:")
        print(f"    h_inside: {summary['heat_transfer_coefficient_W_m2K']:.1f} W/(m²·K)")
        print(f"    U_overall: {summary['overall_U_W_m2K']:.2f} W/(m²·K)")

        # Assertions
        assert outlet.temperature_C < inlet.temperature_C, "Should cool down"
        assert outlet.pressure_bar < inlet.pressure_bar, "Should have pressure drop"
        assert summary['heat_loss_or_gain_kW'] < 0, "Should lose heat (negative Q)"
        assert summary['flow_regime'] == "turbulent", "High velocity should be turbulent"
        assert summary['reynolds_number'] > 4000, "Should be turbulent flow"
        assert summary['friction_factor'] > 0, "Friction factor must be positive"
        assert summary['pressure_drop_bar'] > 0, "Pressure drop must be positive"

        print("\n" + "="*70)
        print("✓ ALL PIPING SMOKE TESTS PASSED")
        print("="*70)

    except Exception as e:
        print(f"\n✗ Piping test failed: {e}")
        raise


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    _smoke_test_piping()
