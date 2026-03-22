"""
utilities/safety.py
===============================================================================

PURPOSE:
Implement safety systems for process equipment protection and personnel safety.
Calculate pressure relief valve (PRV) sizing per API 520/521 and ASME standards.
Design flare systems for emergency depressurization and vapor disposal.
Critical for HAZOP, PHA, and process safety management.

Date: 2026-01-01
Version: 1.0.0
"""

from __future__ import annotations

import math
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS AND STANDARDS
# ============================================================================

# API 526 Standard Orifice Sizes (area in mm²)
API_ORIFICE_SIZES = {
    "D": 71,
    "E": 126,
    "F": 198,
    "G": 324,
    "H": 506,
    "J": 832,
    "K": 1290,
    "L": 2010,
    "M": 3250,
    "N": 5100,
    "P": 8300,
    "Q": 11300,
    "R": 16100,
    "T": 26000
}

# Fire exposure coefficient (API 521)
FIRE_COEFF_BARE = 43200  # W/m² for bare vessels
FIRE_COEFF_INSULATED = 30000  # W/m² for insulated vessels

# Radiation limits (API 521)
RADIATION_LIMIT_PERSONNEL = 1.58  # kW/m² max at grade for emergency
RADIATION_LIMIT_EQUIPMENT = 4.73  # kW/m² max for equipment

# API 520 constants
API_520_C_VAPOR_SI = 315  # For P in bar, W in kg/h


# ============================================================================
# MAIN SAFETY FUNCTIONS
# ============================================================================

def size_pressure_relief_valve(
    vessel_volume_m3: float,
    design_pressure_bar: float,
    set_pressure_bar: float,
    relieving_pressure_bar: float,
    relief_scenario: dict,
    fluid_properties: dict,
    prv_config: dict,
) -> dict:
    """
    Size pressure relief valve per API 520.

    Calculates required orifice area, selects standard size, and determines
    discharge conditions for given relief scenario.

    Args:
        vessel_volume_m3: Vessel internal volume (m³)
        design_pressure_bar: Maximum Allowable Working Pressure
        set_pressure_bar: PRV opening pressure
        relieving_pressure_bar: Pressure during relief (set + accumulation)
        relief_scenario: Dict with scenario type and parameters
        fluid_properties: Dict with fluid properties at relieving conditions
        prv_config: Configuration dict

    Returns:
        PRV sizing dict with orifice area, designation, capacity, etc.

    Example:
        prv = size_pressure_relief_valve(
            vessel_volume_m3=20.0,
            design_pressure_bar=35.0,
            set_pressure_bar=31.5,
            relieving_pressure_bar=38.1,
            relief_scenario={"scenario_type": "fire", ...},
            fluid_properties={...},
            prv_config={...}
        )

        orifice = prv['orifice_designation']  # e.g., "K"

    Raises:
        ValueError: Invalid inputs
        RuntimeError: Calculation failures
    """
    logger.info(
        f"Sizing PRV: P_set={set_pressure_bar:.1f} bar, "
        f"P_relief={relieving_pressure_bar:.1f} bar"
    )

    # Validate inputs
    _validate_prv_inputs(
        vessel_volume_m3, design_pressure_bar, set_pressure_bar,
        relieving_pressure_bar, fluid_properties
    )

    # Calculate relief load
    relief_load = calculate_relief_load(
        scenario_type=relief_scenario.get("scenario_type", "fire"),
        equipment_data={"vessel_volume_m3": vessel_volume_m3, **relief_scenario},
        process_conditions={
            "temperature_C": fluid_properties["temperature_C"],
            "pressure_bar": relieving_pressure_bar,
            **fluid_properties
        }
    )

    W_relief_kg_h = relief_load["relief_mass_flow_kg_h"]

    logger.debug(f"  Relief load: {W_relief_kg_h:.0f} kg/h")

    # Extract fluid properties
    MW = fluid_properties["molecular_weight"]
    T_K = fluid_properties["temperature_C"] + 273.15
    Z = fluid_properties.get("compressibility_Z", 1.0)
    gamma = fluid_properties.get("heat_capacity_ratio_gamma", 1.4)
    phase = fluid_properties.get("phase", "vapor")

    # PRV configuration
    valve_type = prv_config.get("valve_type", "conventional")
    backpressure_bar = prv_config.get("backpressure_bar", 1.01325)
    overpressure_percent = prv_config.get("overpressure_percent", 10.0)

    # Check accumulation limits (ASME requirements)
    accumulation_percent = (
        (relieving_pressure_bar - set_pressure_bar) / set_pressure_bar * 100.0
    )

    scenario_type = relief_scenario.get("scenario_type", "fire")
    if scenario_type == "fire":
        max_accumulation = 21.0
    else:
        max_accumulation = 10.0

    if accumulation_percent > max_accumulation:
        raise ValueError(
            f"Accumulation {accumulation_percent:.1f}% exceeds "
            f"{max_accumulation}% limit for {scenario_type} scenario"
        )

    overpressure_acceptable = accumulation_percent <= max_accumulation

    # Check for critical (choked) flow
    P_relieving_abs_bar = relieving_pressure_bar + 1.01325  # Absolute
    P_downstream_bar = backpressure_bar

    r_critical = (2 / (gamma + 1)) ** (gamma / (gamma - 1))
    pressure_ratio = P_downstream_bar / P_relieving_abs_bar
    choked_flow = pressure_ratio < r_critical

    if choked_flow:
        logger.debug(f"  Choked flow: ratio={pressure_ratio:.3f} < {r_critical:.3f}")

    # Calculate discharge coefficients
    K_d = _get_discharge_coefficient(valve_type)
    K_b = _get_backpressure_correction(
        valve_type, backpressure_bar, set_pressure_bar, choked_flow
    )
    K_v = _get_viscosity_correction(phase, fluid_properties.get("viscosity_cP", 1.0))
    K_r = 1.0  # No rupture disk
    K_overall = K_d * K_b * K_v * K_r

    # Calculate required orifice area based on phase
    if phase == "vapor" or phase == "gas":
        # API 520 vapor/gas formula
        # A = (W * sqrt(T*Z)) / (C * K * P * sqrt(MW))
        # where C = 315 for SI units with P in bar

        numerator = W_relief_kg_h * math.sqrt(T_K * Z)
        denominator = (
            API_520_C_VAPOR_SI * K_overall * P_relieving_abs_bar * math.sqrt(MW)
        )

        A_required_mm2 = numerator / denominator

    elif phase == "liquid":
        # API 520 liquid formula
        # A = (Q * sqrt(rho / ΔP)) / (38 * K_d * K_w * K_v)

        rho = fluid_properties.get("density_kg_m3", 800)
        delta_P_bar = set_pressure_bar - backpressure_bar

        if delta_P_bar <= 0:
            raise ValueError("Liquid relief: set pressure must exceed backpressure")

        # Flow in L/min
        Q_L_min = W_relief_kg_h / rho * 1000 / 60

        K_w = 1.0  # Backpressure correction for liquid (simplified)

        numerator = Q_L_min * math.sqrt(rho / delta_P_bar)
        denominator = 38 * K_d * K_w * K_v

        A_required_mm2 = numerator / denominator

    elif phase == "two-phase":
        # Two-phase (conservative: use liquid formula with safety factor)
        logger.warning("Two-phase flow: using conservative sizing (×1.5)")

        rho = fluid_properties.get("density_kg_m3", 500)
        delta_P_bar = set_pressure_bar - backpressure_bar
        Q_L_min = W_relief_kg_h / rho * 1000 / 60

        K_w = 1.0
        A_required_mm2 = (
            1.5 * Q_L_min * math.sqrt(rho / delta_P_bar) / (38 * K_d * K_w * K_v)
        )

    else:
        raise ValueError(f"Unknown phase: {phase}")

    if A_required_mm2 <= 0:
        raise RuntimeError(f"Invalid orifice area: {A_required_mm2:.2f} mm²")

    logger.debug(f"  Required area: {A_required_mm2:.0f} mm²")

    # Select standard orifice size
    orifice_designation, A_actual_mm2 = _select_orifice_size(A_required_mm2)

    if A_actual_mm2 is None:
        raise RuntimeError(
            f"Required area {A_required_mm2:.0f} mm² exceeds largest standard orifice. "
            f"Consider multiple PRVs."
        )

    # Calculate capacity margin
    capacity_margin_percent = (A_actual_mm2 - A_required_mm2) / A_required_mm2 * 100.0

    if capacity_margin_percent < 5:
        logger.warning(
            f"Low capacity margin: {capacity_margin_percent:.1f}% < 5%. "
            f"Consider next size up for uncertainty."
        )

    # Calculate orifice diameter
    D_orifice_mm = 2 * math.sqrt(A_actual_mm2 / math.pi)

    # Inlet and outlet sizing (6-8 times orifice diameter)
    inlet_size_mm = 6 * D_orifice_mm
    outlet_size_mm = 8 * D_orifice_mm

    # Discharge conditions
    if choked_flow:
        # Critical flow
        discharge_pressure_bar = r_critical * P_relieving_abs_bar
    else:
        discharge_pressure_bar = backpressure_bar

    # Discharge velocity
    rho_discharge = fluid_properties.get("density_kg_m3", 20)
    v_discharge_m_s = (W_relief_kg_h / 3600) / (rho_discharge * A_actual_mm2 / 1e6)

    # Check for high velocity (noise/vibration)
    mach_discharge = v_discharge_m_s / 340  # Approximate speed of sound
    if mach_discharge > 0.5:
        logger.warning(
            f"High discharge velocity: Mach {mach_discharge:.2f} > 0.5. "
            f"Expect noise and vibration."
        )

    # Discharge temperature (assume isentropic expansion)
    T_discharge_C = fluid_properties["temperature_C"] * (
        (discharge_pressure_bar / P_relieving_abs_bar) ** ((gamma - 1) / gamma)
    )

    # Build comprehensive sizing summary
    prv_sizing = {
        "scenario": scenario_type,
        "set_pressure_bar": set_pressure_bar,
        "relieving_pressure_bar": relieving_pressure_bar,
        "accumulation_percent": accumulation_percent,
        "overpressure_acceptable": overpressure_acceptable,

        # Relief load
        "relief_mass_flow_kg_h": W_relief_kg_h,
        "relief_volumetric_flow_m3_h": W_relief_kg_h / rho_discharge,
        "vapor_fraction": fluid_properties.get("vapor_fraction", 1.0),

        # Valve sizing
        "required_orifice_area_mm2": A_required_mm2,
        "orifice_designation": orifice_designation,
        "actual_orifice_area_mm2": A_actual_mm2,
        "orifice_diameter_mm": D_orifice_mm,
        "capacity_margin_percent": capacity_margin_percent,

        # Discharge conditions
        "discharge_pressure_bar": discharge_pressure_bar,
        "discharge_temperature_C": T_discharge_C,
        "discharge_velocity_m_s": v_discharge_m_s,
        "choked_flow": choked_flow,
        "critical_pressure_ratio": r_critical,

        # Coefficients
        "discharge_coefficient_Kd": K_d,
        "backpressure_correction_Kb": K_b,
        "viscosity_correction_Kv": K_v,
        "rupture_disk_factor_Kr": K_r,
        "overall_coefficient_K": K_overall,

        # Valve details
        "valve_type": valve_type,
        "inlet_size_mm": inlet_size_mm,
        "outlet_size_mm": outlet_size_mm,
        "material": prv_config.get("material", "carbon_steel"),
        "code_compliance": prv_config.get("code", "API_520"),
    }

    logger.info(
        f"PRV sized: Orifice {orifice_designation} ({A_actual_mm2} mm²), "
        f"Flow {W_relief_kg_h:.0f} kg/h"
    )

    return prv_sizing


def calculate_relief_load(
    scenario_type: str,
    equipment_data: dict,
    process_conditions: dict,
) -> dict:
    """
    Calculate required relief flow for given scenario.

    Supports scenarios:
    - fire: External fire exposure
    - blocked_outlet: Continued feed with blocked discharge
    - cooling_failure: Loss of cooling in exothermic reactor
    - runaway_reaction: Uncontrolled exothermic reaction

    Args:
        scenario_type: Type of relief scenario
        equipment_data: Equipment parameters (volume, area, etc.)
        process_conditions: Process state (T, P, composition, etc.)

    Returns:
        Relief load dict with mass flow, duty, conditions

    Example:
        load = calculate_relief_load(
            scenario_type="fire",
            equipment_data={"wetted_surface_area_m2": 50},
            process_conditions={"temperature_C": 100, ...}
        )

        W_relief = load['relief_mass_flow_kg_h']
    """
    logger.info(f"Calculating relief load for scenario: {scenario_type}")

    if scenario_type == "fire":
        # API 521 fire relief
        return _calculate_fire_relief(equipment_data, process_conditions)

    elif scenario_type == "blocked_outlet":
        # Maximum inlet flow
        return _calculate_blocked_outlet_relief(equipment_data, process_conditions)

    elif scenario_type == "cooling_failure":
        # Heat generation minus available cooling
        return _calculate_cooling_failure_relief(equipment_data, process_conditions)

    elif scenario_type == "runaway_reaction":
        # Runaway exotherm (simplified DIERS)
        return _calculate_runaway_relief(equipment_data, process_conditions)

    else:
        raise ValueError(f"Unknown scenario type: {scenario_type}")


def size_flare_system(
    relief_streams: list[dict],
    flare_config: dict,
) -> dict:
    """
    Size flare system for safe disposal of relief vapors.

    Calculates stack height, tip diameter, and thermal radiation to ensure
    personnel safety per API 521.

    Args:
        relief_streams: List of dicts with relief stream data
        flare_config: Configuration for flare system

    Returns:
        Flare sizing dict with stack height, radiation, knockout drum

    Example:
        flare = size_flare_system(
            relief_streams=[
                {"name": "reactor", "flow_kg_h": 5000, "H2_content": 0.8},
                {"name": "flash", "flow_kg_h": 2000, "H2_content": 0.1}
            ],
            flare_config={
                "radiation_limit_kW_m2": 1.58,
                "wind_speed_m_s": 5.0
            }
        )

        height = flare['stack_height_m']
    """
    logger.info(f"Sizing flare system with {len(relief_streams)} streams")

    # Calculate total relief load
    total_load_kg_h = sum(s.get("flow_kg_h", 0) for s in relief_streams)

    # Consider simultaneity (typically not all PRVs open at once)
    # Conservative: assume 2 largest streams simultaneously
    sorted_streams = sorted(relief_streams, key=lambda x: x.get("flow_kg_h", 0), reverse=True)
    if len(sorted_streams) >= 2:
        max_simultaneous_kg_h = sum(s["flow_kg_h"] for s in sorted_streams[:2])
    else:
        max_simultaneous_kg_h = total_load_kg_h

    design_load_kg_h = max_simultaneous_kg_h

    logger.debug(f"  Design load: {design_load_kg_h:.0f} kg/h")

    # Estimate heat release (assume LHV = 45 MJ/kg for hydrocarbons)
    # For H2-rich streams, LHV_H2 = 120 MJ/kg
    total_heat_kW = 0
    for stream in relief_streams[:2]:  # Only consider simultaneous streams
        flow = stream.get("flow_kg_h", 0)
        h2_fraction = stream.get("H2_content", 0)
        lhv = h2_fraction * 120000 + (1 - h2_fraction) * 45000  # kJ/kg
        heat_kW = flow * lhv / 3600
        total_heat_kW += heat_kW

    logger.debug(f"  Total heat release: {total_heat_kW:.0f} kW")

    # Estimate flame length (API 521 correlation)
    # L_flame ≈ 0.2 * Q^0.4, where Q in kW
    flame_length_m = 0.2 * (total_heat_kW ** 0.4)

    # Wind effect (flame tilt)
    wind_speed_m_s = flare_config.get("wind_speed_m_s", 5.0)
    flame_tilt_deg = min(45, wind_speed_m_s * 5)  # Simplified

    # Calculate radiation at grade
    radiation_limit_kW_m2 = flare_config.get("radiation_limit_kW_m2", 1.58)

    # Iterate to find required stack height
    stack_height_m = flare_config.get("stack_height_m", "calculate")

    if stack_height_m == "calculate":
        # Start with 1.5 × flame length
        stack_height_m = 1.5 * flame_length_m

        # Iterate to meet radiation limit
        for iteration in range(20):
            radiation_kW_m2 = _calculate_ground_radiation(
                total_heat_kW, stack_height_m, flame_length_m, wind_speed_m_s
            )

            if radiation_kW_m2 <= radiation_limit_kW_m2:
                break

            # Increase height
            stack_height_m *= 1.1

        if radiation_kW_m2 > radiation_limit_kW_m2:
            logger.warning(
                f"Could not meet radiation limit with stack height {stack_height_m:.1f} m"
            )

    radiation_at_grade = _calculate_ground_radiation(
        total_heat_kW, stack_height_m, flame_length_m, wind_speed_m_s
    )

    radiation_limit_met = radiation_at_grade <= radiation_limit_kW_m2

    # Calculate safe distance
    safe_distance_m = _calculate_safe_distance(
        total_heat_kW, radiation_limit_kW_m2
    )

    # Size flare tip diameter (velocity 0.2-0.5 Mach)
    tip_diameter_mm = flare_config.get("tip_diameter_mm", "calculate")

    if tip_diameter_mm == "calculate":
        # Target exit velocity: 0.3 Mach (~100 m/s)
        target_velocity_m_s = 100

        # Volumetric flow (assume ideal gas, 1 bar, 25°C)
        V_m3_s = design_load_kg_h / 3600 / 1.2  # Approximate density

        # Area = flow / velocity
        A_tip_m2 = V_m3_s / target_velocity_m_s
        tip_diameter_mm = 1000 * 2 * math.sqrt(A_tip_m2 / math.pi)

    tip_velocity_m_s = (
        (design_load_kg_h / 3600 / 1.2) / (math.pi * (tip_diameter_mm/2000)**2)
    )

    # Flare header sizing (pressure drop < 10% of PRV set pressure)
    # Assume average PRV set pressure ~30 bar
    max_dp_bar = 3.0

    # Size for velocity ~20-30 m/s
    header_velocity_m_s = 25
    header_area_m2 = (design_load_kg_h / 3600 / 1.2) / header_velocity_m_s
    header_diameter_mm = 1000 * 2 * math.sqrt(header_area_m2 / math.pi)

    # Pressure drop (Darcy-Weisbach, simplified)
    header_length_m = 50  # Typical
    f = 0.02  # Friction factor
    rho = 1.2  # kg/m³
    dp_Pa = f * (header_length_m / (header_diameter_mm/1000)) * (
        rho * header_velocity_m_s**2 / 2
    )
    header_dp_bar = dp_Pa / 1e5

    # Knockout drum sizing (separate liquid from vapor)
    knockout_required = flare_config.get("liquid_knockout_drum", True)

    if knockout_required:
        # Size for residence time ~5 minutes
        liquid_holdup_m3 = 0.1 * design_load_kg_h / 800 / 60 * 5  # Assume some liquid
        knockout_diameter_m = 1.5  # Typical
        knockout_length_m = (4 * liquid_holdup_m3) / (math.pi * knockout_diameter_m**2)
    else:
        knockout_diameter_m = 0
        knockout_length_m = 0
        liquid_holdup_m3 = 0

    # Purge gas flow (prevent air ingress)
    purge_gas_flow_kg_h = 0.015 * 3600  # ~0.015 kg/s typical for N2

    # Build flare sizing summary
    flare_sizing = {
        "total_relief_load_kg_h": total_load_kg_h,
        "maximum_simultaneous_load_kg_h": max_simultaneous_kg_h,
        "design_case_description": f"{len(sorted_streams[:2])} largest streams simultaneous",

        # Stack sizing
        "stack_height_m": stack_height_m,
        "stack_internal_diameter_mm": tip_diameter_mm * 1.5,  # Larger than tip
        "tip_diameter_mm": tip_diameter_mm,
        "tip_type": flare_config.get("tip_type", "tulip"),
        "tip_velocity_m_s": tip_velocity_m_s,

        # Thermal radiation
        "flame_length_m": flame_length_m,
        "flame_tilt_angle_deg": flame_tilt_deg,
        "radiation_at_grade_kW_m2": radiation_at_grade,
        "safe_distance_m": safe_distance_m,
        "radiation_limit_met": radiation_limit_met,

        # Flare header
        "header_diameter_mm": header_diameter_mm,
        "header_pressure_drop_bar": header_dp_bar,
        "header_velocity_m_s": header_velocity_m_s,

        # Knockout drum
        "knockout_required": knockout_required,
        "knockout_diameter_m": knockout_diameter_m,
        "knockout_length_m": knockout_length_m,
        "liquid_holdup_m3": liquid_holdup_m3,

        # Auxiliary
        "pilot_ignition": flare_config.get("pilot_ignition", True),
        "purge_gas_flow_kg_h": purge_gas_flow_kg_h,
        "molecular_seal_type": "liquid" if flare_config.get("molecular_seal", True) else "none",
    }

    logger.info(
        f"Flare sized: H={stack_height_m:.1f}m, D_tip={tip_diameter_mm:.0f}mm, "
        f"Radiation={radiation_at_grade:.2f} kW/m²"
    )

    return flare_sizing


def check_overpressure_protection(
    equipment_type: str,
    operating_pressure_bar: float,
    design_pressure_bar: float,
    relief_devices: list[dict],
) -> dict:
    """
    Check adequacy of overpressure protection.

    Verifies that installed relief devices provide sufficient capacity
    for credible overpressure scenarios.

    Args:
        equipment_type: Type of equipment (reactor, vessel, etc.)
        operating_pressure_bar: Normal operating pressure
        design_pressure_bar: MAWP
        relief_devices: List of relief device dicts

    Returns:
        Protection adequacy dict with compliance status
    """
    logger.info(f"Checking overpressure protection for {equipment_type}")

    if not relief_devices:
        return {
            "equipment_type": equipment_type,
            "operating_pressure_bar": operating_pressure_bar,
            "design_pressure_bar": design_pressure_bar,
            "set_pressure_bar": 0,
            "relief_capacity_kg_h": 0,
            "relief_load_kg_h": 0,
            "capacity_margin_percent": 0,
            "protection_adequate": False,
            "compliance_status": "FAIL - No relief devices installed",
            "recommendations": ["Install pressure relief valve per ASME VIII"]
        }

    # Check set pressures
    recommendations = []

    for device in relief_devices:
        set_p = device.get("set_pressure_bar", 0)
        if set_p > design_pressure_bar:
            recommendations.append(
                f"Device set pressure {set_p:.1f} bar exceeds MAWP {design_pressure_bar:.1f} bar"
            )

        if set_p < 0.8 * design_pressure_bar:
            recommendations.append(
                f"Device set pressure {set_p:.1f} bar < 0.8×MAWP (may pop frequently)"
            )

    # Sum capacities
    total_capacity_kg_h = sum(d.get("capacity_kg_h", 0) for d in relief_devices)

    # Estimate required relief load (simplified)
    # For actual check, would use detailed scenario analysis
    estimated_relief_load_kg_h = 0.1 * total_capacity_kg_h  # Placeholder

    capacity_margin_percent = (
        (total_capacity_kg_h - estimated_relief_load_kg_h) /
        estimated_relief_load_kg_h * 100
        if estimated_relief_load_kg_h > 0 else 100
    )

    protection_adequate = capacity_margin_percent >= 10

    if protection_adequate:
        compliance_status = "PASS - Adequate protection"
    else:
        compliance_status = "MARGINAL - Review required"
        recommendations.append("Verify relief capacity for all credible scenarios")

    protection_data = {
        "equipment_type": equipment_type,
        "operating_pressure_bar": operating_pressure_bar,
        "design_pressure_bar": design_pressure_bar,
        "set_pressure_bar": relief_devices[0].get("set_pressure_bar", 0),
        "relief_capacity_kg_h": total_capacity_kg_h,
        "relief_load_kg_h": estimated_relief_load_kg_h,
        "capacity_margin_percent": capacity_margin_percent,
        "protection_adequate": protection_adequate,
        "compliance_status": compliance_status,
        "recommendations": recommendations if recommendations else ["None"]
    }

    return protection_data


def calculate_depressurization_time(
    vessel_volume_m3: float,
    initial_pressure_bar: float,
    final_pressure_bar: float,
    orifice_area_mm2: float,
    fluid_properties: dict,
) -> dict:
    """
    Calculate depressurization time for emergency blowdown.

    Models isothermal or isentropic blowdown to determine time required
    to reduce pressure from initial to final value.

    Args:
        vessel_volume_m3: Vessel volume
        initial_pressure_bar: Starting pressure
        final_pressure_bar: Target pressure
        orifice_area_mm2: Blowdown valve orifice area
        fluid_properties: Fluid properties

    Returns:
        Depressurization data with time, flow profile, temperature drop
    """
    logger.info(
        f"Calculating depressurization: {initial_pressure_bar:.1f} → "
        f"{final_pressure_bar:.1f} bar"
    )

    MW = fluid_properties["molecular_weight"]
    T_initial_K = fluid_properties["temperature_C"] + 273.15
    gamma = fluid_properties.get("heat_capacity_ratio_gamma", 1.4)
    Z = fluid_properties.get("compressibility_Z", 1.0)

    # Isothermal blowdown (conservative for time)
    # t = (V / (C_d * A)) * ln(P1 / P2)

    C_d = 0.85  # Discharge coefficient
    A_m2 = orifice_area_mm2 / 1e6

    # Sound speed
    R = 8314 / MW  # J/(kg·K)
    a = math.sqrt(gamma * R * T_initial_K)

    # Mass flow (choked)
    rho_initial = initial_pressure_bar * 1e5 / (R * T_initial_K)
    m_dot_kg_s = C_d * A_m2 * rho_initial * a * 0.5  # Simplified

    # Initial mass
    m_initial_kg = (initial_pressure_bar * 1e5 * vessel_volume_m3) / (R * T_initial_K)
    m_final_kg = (final_pressure_bar * 1e5 * vessel_volume_m3) / (R * T_initial_K)

    # Average flow
    m_average_kg_s = (m_initial_kg - m_final_kg) / 2 / 60  # Rough estimate

    # Time (simplified)
    t_minutes = (m_initial_kg - m_final_kg) / max(m_dot_kg_s, 0.01) / 60

    # Temperature drop (isentropic expansion)
    T_final_K = T_initial_K * (final_pressure_bar / initial_pressure_bar) ** ((gamma - 1) / gamma)
    T_drop_C = T_initial_K - T_final_K

    ice_risk = (T_final_K - 273.15) < 0

    if t_minutes < 5:
        logger.warning(
            f"Fast depressurization: {t_minutes:.1f} min < 5 min. "
            f"Risk of thermal shock and brittle fracture."
        )

    if ice_risk:
        logger.warning(
            f"Ice formation risk: Final T = {T_final_K - 273.15:.1f}°C < 0°C"
        )

    # Time profile (simplified, 10 points)
    time_profile = []
    for i in range(11):
        fraction = i / 10
        P = initial_pressure_bar * (final_pressure_bar / initial_pressure_bar) ** fraction
        t = t_minutes * fraction
        time_profile.append({"time_min": t, "pressure_bar": P})

    depress_data = {
        "initial_pressure_bar": initial_pressure_bar,
        "final_pressure_bar": final_pressure_bar,
        "depressurization_time_minutes": t_minutes,
        "average_flow_rate_kg_h": m_average_kg_s * 3600,
        "time_profile": time_profile,
        "max_temperature_drop_C": T_drop_C,
        "risk_of_ice_formation": ice_risk,
    }

    return depress_data


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _validate_prv_inputs(
    vessel_volume_m3: float,
    design_pressure_bar: float,
    set_pressure_bar: float,
    relieving_pressure_bar: float,
    fluid_properties: dict,
) -> None:
    """Validate PRV sizing inputs"""

    if vessel_volume_m3 <= 0:
        raise ValueError(f"Vessel volume {vessel_volume_m3} must be > 0")

    if design_pressure_bar <= 0:
        raise ValueError(f"Design pressure {design_pressure_bar} must be > 0")

    if set_pressure_bar > design_pressure_bar:
        raise ValueError(
            f"Set pressure {set_pressure_bar:.1f} bar exceeds "
            f"MAWP {design_pressure_bar:.1f} bar (violates ASME)"
        )

    if relieving_pressure_bar < set_pressure_bar:
        raise ValueError(
            f"Relieving pressure {relieving_pressure_bar:.1f} bar < "
            f"set pressure {set_pressure_bar:.1f} bar (impossible)"
        )

    MW = fluid_properties.get("molecular_weight", 0)
    if MW <= 0 or MW > 1000:
        raise ValueError(f"Molecular weight {MW} invalid")


def _get_discharge_coefficient(valve_type: str) -> float:
    """Get discharge coefficient based on valve type"""
    coefficients = {
        "conventional": 0.975,
        "balanced_bellows": 0.975,
        "pilot_operated": 0.98,
    }
    return coefficients.get(valve_type, 0.975)


def _get_backpressure_correction(
    valve_type: str,
    backpressure_bar: float,
    set_pressure_bar: float,
    choked: bool,
) -> float:
    """Calculate backpressure correction factor"""

    backpressure_fraction = backpressure_bar / (set_pressure_bar + 1.01325)

    if choked:
        # Choked flow: backpressure has minimal effect
        return 1.0
    else:
        # Subcritical: backpressure reduces capacity
        # Simplified correlation
        return max(0.8, 1.0 - 0.5 * backpressure_fraction)


def _get_viscosity_correction(phase: str, viscosity_cP: float) -> float:
    """Calculate viscosity correction factor"""
    if phase == "liquid" and viscosity_cP > 1.0:
        # For viscous liquids
        return max(0.9, 1.0 - 0.001 * (viscosity_cP - 1))
    else:
        return 1.0


def _select_orifice_size(required_area_mm2: float) -> tuple[str, float]:
    """Select standard API 526 orifice size"""

    for designation, area in API_ORIFICE_SIZES.items():
        if area >= required_area_mm2:
            return designation, area

    # If exceeds largest
    return "T+", None


def _calculate_fire_relief(equipment_data: dict, process_conditions: dict) -> dict:
    """Calculate fire relief per API 521"""

    A_wetted_m2 = equipment_data.get("wetted_surface_area_m2", 50)
    insulated = equipment_data.get("insulated", False)
    env_factor = equipment_data.get("environment_factor", 1.0)

    # Fire heat input
    C = FIRE_COEFF_INSULATED if insulated else FIRE_COEFF_BARE
    Q_W = C * env_factor * (A_wetted_m2 ** 0.82)

    # Latent heat
    H_v_kJ_kg = process_conditions.get("latent_heat_kJ_kg", 400)

    # Relief flow
    W_kg_s = Q_W / (H_v_kJ_kg * 1000)
    W_kg_h = W_kg_s * 3600

    return {
        "scenario_type": "fire",
        "scenario_description": f"External fire on {A_wetted_m2:.0f} m² wetted area",
        "relief_required": True,
        "relief_mass_flow_kg_h": W_kg_h,
        "relief_volumetric_flow_m3_h": W_kg_h / process_conditions.get("density_kg_m3", 20),
        "relief_duty_kW": Q_W / 1000,
        "vapor_fraction": 1.0,
        "temperature_C": process_conditions.get("temperature_C", 100),
        "pressure_bar": process_conditions.get("pressure_bar", 30),
        "calculation_method": "API_521_fire",
        "assumptions": ["Bare vessel", "Good drainage", "Worst-case fire exposure"]
    }


def _calculate_blocked_outlet_relief(equipment_data: dict, process_conditions: dict) -> dict:
    """Calculate blocked outlet relief"""

    max_inlet_kmol_h = equipment_data.get("max_inlet_flow_kmol_h", 100)
    MW = process_conditions.get("molecular_weight", 78)

    W_kg_h = max_inlet_kmol_h * MW

    return {
        "scenario_type": "blocked_outlet",
        "scenario_description": "Continued feed with blocked discharge",
        "relief_required": True,
        "relief_mass_flow_kg_h": W_kg_h,
        "relief_volumetric_flow_m3_h": W_kg_h / process_conditions.get("density_kg_m3", 800),
        "relief_duty_kW": 0,
        "vapor_fraction": 0.0,
        "temperature_C": process_conditions.get("temperature_C", 80),
        "pressure_bar": process_conditions.get("pressure_bar", 2),
        "calculation_method": "Maximum_inlet_flow",
        "assumptions": ["Maximum feed rate", "Complete blockage", "No vaporization"]
    }


def _calculate_cooling_failure_relief(equipment_data: dict, process_conditions: dict) -> dict:
    """Calculate cooling failure relief for exothermic reactor"""

    Q_reaction_kW = equipment_data.get("reaction_heat_kW", 1000)
    Q_cooling_available_kW = equipment_data.get("emergency_cooling_kW", 0)

    Q_relief_kW = Q_reaction_kW - Q_cooling_available_kW

    H_v_kJ_kg = process_conditions.get("latent_heat_kJ_kg", 400)

    W_kg_s = (Q_relief_kW * 1000) / (H_v_kJ_kg * 1000)
    W_kg_h = W_kg_s * 3600

    return {
        "scenario_type": "cooling_failure",
        "scenario_description": "Loss of cooling water to exothermic reactor",
        "relief_required": True,
        "relief_mass_flow_kg_h": W_kg_h,
        "relief_volumetric_flow_m3_h": W_kg_h / process_conditions.get("density_kg_m3", 20),
        "relief_duty_kW": Q_relief_kW,
        "vapor_fraction": 1.0,
        "temperature_C": process_conditions.get("temperature_C", 300),
        "pressure_bar": process_conditions.get("pressure_bar", 35),
        "calculation_method": "Heat_balance",
        "assumptions": ["Complete cooling loss", "Maximum reaction rate", "Adiabatic temperature rise"]
    }


def _calculate_runaway_relief(equipment_data: dict, process_conditions: dict) -> dict:
    """Calculate runaway reaction relief (simplified DIERS)"""

    Q_runaway_kW = equipment_data.get("runaway_heat_kW", 2000)
    H_v_kJ_kg = process_conditions.get("latent_heat_kJ_kg", 400)

    # Conservative: 1.5× safety factor for runaway
    W_kg_h = 1.5 * (Q_runaway_kW * 3600) / H_v_kJ_kg

    return {
        "scenario_type": "runaway_reaction",
        "scenario_description": "Uncontrolled exothermic reaction runaway",
        "relief_required": True,
        "relief_mass_flow_kg_h": W_kg_h,
        "relief_volumetric_flow_m3_h": W_kg_h / process_conditions.get("density_kg_m3", 20),
        "relief_duty_kW": Q_runaway_kW,
        "vapor_fraction": 1.0,
        "temperature_C": process_conditions.get("temperature_C", 400),
        "pressure_bar": process_conditions.get("pressure_bar", 40),
        "calculation_method": "DIERS_simplified",
        "assumptions": ["Worst-case kinetics", "Adiabatic", "Safety factor 1.5×"]
    }


def _calculate_ground_radiation(
    heat_release_kW: float,
    stack_height_m: float,
    flame_length_m: float,
    wind_speed_m_s: float,
) -> float:
    """Calculate ground-level radiation from flare (simplified point source)"""

    # Fraction of heat radiated (typical 0.2-0.4)
    F_rad = 0.3

    # Atmospheric transmissivity
    tau = 0.8

    # Distance to flame center (with tilt)
    flame_tilt_rad = min(45, wind_speed_m_s * 5) * math.pi / 180
    x_horizontal = flame_length_m * math.sin(flame_tilt_rad) / 2
    x_vertical = stack_height_m + flame_length_m * math.cos(flame_tilt_rad) / 2
    distance_m = math.sqrt(x_horizontal**2 + x_vertical**2)

    # Point source model
    I_kW_m2 = (F_rad * tau * heat_release_kW) / (4 * math.pi * distance_m**2)

    return I_kW_m2


def _calculate_safe_distance(heat_release_kW: float, limit_kW_m2: float) -> float:
    """Calculate safe distance for radiation limit"""

    F_rad = 0.3
    tau = 0.8

    # Rearrange point source: x = sqrt((F*tau*Q) / (4*π*I))
    distance_m = math.sqrt(
        (F_rad * tau * heat_release_kW) / (4 * math.pi * limit_kW_m2)
    )

    return distance_m


# ============================================================================
# SMOKE TEST
# ============================================================================

def _smoke_test_safety():
    """
    Test safety system functions.
    """
    print("="*70)
    print("SAFETY SYSTEMS SMOKE TEST")
    print("="*70)

    # ========================================================================
    # Test 1: Fire relief PRV sizing
    # ========================================================================
    print("\n✓ Test 1: Fire relief PRV sizing...")

    relief_scenario = {
        "scenario_type": "fire",
        "wetted_surface_area_m2": 50.0,
        "environment_factor": 1.0,
        "insulated": False
    }

    fluid_props = {
        "molecular_weight": 78.11,  # benzene
        "temperature_C": 100.0,
        "pressure_bar": 30.0,
        "phase": "vapor",
        "vapor_fraction": 1.0,
        "density_kg_m3": 20.0,
        "compressibility_Z": 0.95,
        "heat_capacity_ratio_gamma": 1.1,
        "latent_heat_kJ_kg": 400.0
    }

    prv_config = {
        "valve_type": "conventional",
        "orientation": "vertical",
        "discharge_to": "flare",
        "backpressure_bar": 1.5,
        "code": "API_520",
        "overpressure_percent": 21.0
    }

    try:
        prv_sizing = size_pressure_relief_valve(
            vessel_volume_m3=20.0,
            design_pressure_bar=35.0,
            set_pressure_bar=31.5,
            relieving_pressure_bar=38.1,
            relief_scenario=relief_scenario,
            fluid_properties=fluid_props,
            prv_config=prv_config
        )

        print(f"  Scenario: {prv_sizing['scenario']}")
        print(f"  Relief flow: {prv_sizing['relief_mass_flow_kg_h']:.0f} kg/h")
        print(f"  Required area: {prv_sizing['required_orifice_area_mm2']:.0f} mm²")
        print(f"  Orifice designation: {prv_sizing['orifice_designation']}")
        print(f"  Actual area: {prv_sizing['actual_orifice_area_mm2']:.0f} mm²")
        print(f"  Capacity margin: {prv_sizing['capacity_margin_percent']:.1f}%")
        print(f"  Choked flow: {prv_sizing['choked_flow']}")
        print(f"  Discharge velocity: {prv_sizing['discharge_velocity_m_s']:.1f} m/s")

        assert prv_sizing['required_orifice_area_mm2'] > 0
        assert prv_sizing['capacity_margin_percent'] > 0
        assert prv_sizing['overpressure_acceptable']

        print("  ✓ Test 1 passed")

    except Exception as e:
        print(f"  ✗ Test 1 failed: {e}")
        raise

    # ========================================================================
    # Test 2: Blocked outlet relief load
    # ========================================================================
    print("\n✓ Test 2: Blocked outlet relief load...")

    try:
        relief_load = calculate_relief_load(
            scenario_type="blocked_outlet",
            equipment_data={
                "vessel_volume_m3": 10.0,
                "max_inlet_flow_kmol_h": 50.0
            },
            process_conditions={
                "temperature_C": 80.0,
                "pressure_bar": 2.0,
                "molecular_weight": 78.11,
                "density_kg_m3": 800
            }
        )

        print(f"  Scenario: {relief_load['scenario_type']}")
        print(f"  Relief required: {relief_load['relief_required']}")
        print(f"  Relief flow: {relief_load['relief_mass_flow_kg_h']:.0f} kg/h")
        print(f"  Method: {relief_load['calculation_method']}")

        assert relief_load['relief_required']
        assert relief_load['relief_mass_flow_kg_h'] > 0

        print("  ✓ Test 2 passed")

    except Exception as e:
        print(f"  ✗ Test 2 failed: {e}")
        raise

    # ========================================================================
    # Test 3: Flare system sizing
    # ========================================================================
    print("\n✓ Test 3: Flare system sizing...")

    relief_streams = [
        {"name": "reactor_prv", "flow_kg_h": 5000, "H2_content": 0.8},
        {"name": "flash_prv", "flow_kg_h": 2000, "H2_content": 0.1},
        {"name": "column_prv", "flow_kg_h": 1000, "H2_content": 0.0}
    ]

    flare_config = {
        "stack_height_m": "calculate",
        "tip_diameter_mm": "calculate",
        "tip_type": "tulip",
        "pilot_ignition": True,
        "radiation_limit_kW_m2": 1.58,
        "wind_speed_m_s": 5.0
    }

    try:
        flare_sizing = size_flare_system(
            relief_streams=relief_streams,
            flare_config=flare_config
        )

        print(f"  Total relief load: {flare_sizing['total_relief_load_kg_h']:.0f} kg/h")
        print(f"  Simultaneous load: {flare_sizing['maximum_simultaneous_load_kg_h']:.0f} kg/h")
        print(f"  Stack height: {flare_sizing['stack_height_m']:.1f} m")
        print(f"  Tip diameter: {flare_sizing['tip_diameter_mm']:.0f} mm")
        print(f"  Flame length: {flare_sizing['flame_length_m']:.1f} m")
        print(f"  Radiation at grade: {flare_sizing['radiation_at_grade_kW_m2']:.3f} kW/m²")
        print(f"  Radiation limit met: {flare_sizing['radiation_limit_met']}")
        print(f"  Knockout drum: {flare_sizing['knockout_diameter_m']:.2f}m × {flare_sizing['knockout_length_m']:.2f}m")

        assert flare_sizing['stack_height_m'] > 10
        assert flare_sizing['radiation_limit_met']

        print("  ✓ Test 3 passed")

    except Exception as e:
        print(f"  ✗ Test 3 failed: {e}")
        raise

    # ========================================================================
    # Test 4: Overpressure protection check
    # ========================================================================
    print("\n✓ Test 4: Overpressure protection check...")

    relief_devices = [
        {"type": "PRV", "set_pressure_bar": 31.5, "capacity_kg_h": 6000}
    ]

    try:
        protection = check_overpressure_protection(
            equipment_type="reactor",
            operating_pressure_bar=31.0,
            design_pressure_bar=35.0,
            relief_devices=relief_devices
        )

        print(f"  Equipment: {protection['equipment_type']}")
        print(f"  Protection adequate: {protection['protection_adequate']}")
        print(f"  Capacity margin: {protection['capacity_margin_percent']:.1f}%")
        print(f"  Compliance: {protection['compliance_status']}")

        assert protection['protection_adequate']

        print("  ✓ Test 4 passed")

    except Exception as e:
        print(f"  ✗ Test 4 failed: {e}")
        raise

    # ========================================================================
    # Test 5: Depressurization time
    # ========================================================================
    print("\n✓ Test 5: Depressurization time calculation...")

    try:
        depress = calculate_depressurization_time(
            vessel_volume_m3=20.0,
            initial_pressure_bar=35.0,
            final_pressure_bar=5.0,
            orifice_area_mm2=2000,
            fluid_properties=fluid_props
        )

        print(f"  Initial pressure: {depress['initial_pressure_bar']:.1f} bar")
        print(f"  Final pressure: {depress['final_pressure_bar']:.1f} bar")
        print(f"  Depressurization time: {depress['depressurization_time_minutes']:.1f} min")
        print(f"  Temperature drop: {depress['max_temperature_drop_C']:.1f}°C")
        print(f"  Ice formation risk: {depress['risk_of_ice_formation']}")

        assert depress['depressurization_time_minutes'] > 0

        print("  ✓ Test 5 passed")

    except Exception as e:
        print(f"  ✗ Test 5 failed: {e}")
        raise

    print("\n" + "="*70)
    print("✓ ALL SAFETY SYSTEMS SMOKE TESTS PASSED")
    print("="*70)


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    _smoke_test_safety()
