"""
utilities/pump.py
===============================================================================

PURPOSE:
Rigorous centrifugal and positive displacement pump models with mechanical design.
Includes NPSH calculations, power requirements, and efficiency curves.
Multi-stage operation with interstage cooling for high pressure applications.

Date: 2026-02-13
Version: 3.0.0 - Multi-stage with interstage cooling
"""

from __future__ import annotations
import math
import logging
from typing import Any, Dict, Optional, Tuple
import numpy as np

# Try to import Stream, will be imported locally if not available
try:
    from simulation.streams import Stream
except ImportError:
    Stream = None

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

GRAVITY = 9.81  # m/s²
WATER_DENSITY_KG_M3 = 1000.0  # kg/m³ at 20°C

# ═══════════════════════════════════════════════════════════════════════════
# HARDCODED PUMP SPECIFICATIONS
# ═══════════════════════════════════════════════════════════════════════════

PUMP_DEFAULTS = {
    "type": "centrifugal",
    "efficiency": 0.75,
    "motor_efficiency": 0.95,
    "design_pressure_bar": 50.0,
    "design_temperature_C": 150.0,
    "npsh_required_m": 3.0,
    "material": "carbon_steel",
    "min_flow_fraction": 0.3,
    "max_flow_fraction": 1.2,
    "design_margin": 1.15,
}

# Safety limits
PUMP_SAFETY_LIMITS = {
    "max_discharge_pressure_bar": 50.0,
    "max_temperature_C": 250.0,
    "min_npsh_margin_m": 0.5,
    "max_power_kW": 500.0,
    "cavitation_alarm_threshold": 0.8,
}

# Material properties
MATERIAL_PROPERTIES = {
    "carbon_steel": {
        "max_temperature_C": 400.0,
        "allowable_stress_MPa": 138.0,
        "corrosion_allowance_mm": 3.0,
    },
    "stainless_steel_316": {
        "max_temperature_C": 538.0,
        "allowable_stress_MPa": 138.0,
        "corrosion_allowance_mm": 1.5,
    },
}

# Multi-stage pump configuration
PUMP_STAGE_CONFIG = {
    "max_temp_rise_per_stage_C": 15.0,      # Maximum ΔT per stage before cooling
    "target_temp_after_cooling_C": None,    # Target temp after cooling (None = inlet temp)
    "min_temp_C": 10.0,                     # Minimum allowable temperature
    "max_temp_C": 150.0,                    # Maximum allowable temperature
    "enable_interstage_cooling": True,      # Enable automatic staging and cooling
    "cooling_efficiency": 0.95,             # Cooling effectiveness
}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PUMP FUNCTION - MULTI-STAGE WITH COOLING
# ═══════════════════════════════════════════════════════════════════════════

def pump_liquid(
    inlet: Any,
    discharge_pressure_bar: float,
    pump_config: Optional[dict] = None,
    thermo: Any = None,
    pump_name: str = "P-101",
) -> Tuple[Any, dict]:
    """
    Rigorous multi-stage centrifugal pump with interstage cooling.

    Features:
    - Automatic staging when temperature rise exceeds limits
    - Interstage cooling to maintain liquid integrity
    - Tracks cooling duty per stage
    - Returns total cooling energy and number of stages

    Args:
        inlet: Inlet liquid stream
        discharge_pressure_bar: Target discharge pressure (bar)
        pump_config: Optional pump configuration (uses defaults if None)
        thermo: ThermodynamicPackage for property calculations
        pump_name: Pump identifier

    Returns:
        (outlet_stream, summary_dict)

        summary_dict includes:
            - num_stages: Number of pump stages used
            - total_cooling_duty_kW: Total cooling energy required
            - stage_cooling_duties_kW: List of cooling duty per stage
            - stage_temperatures_C: Temperature profile through stages
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"MULTI-STAGE PUMP: {pump_name}")
    logger.info("=" * 70)

    # Merge configs
    config = {**PUMP_DEFAULTS, **PUMP_STAGE_CONFIG, **(pump_config or {})}

    # Validate inputs
    _validate_pump_inputs(inlet, discharge_pressure_bar, config, pump_name)

    # Extract inlet conditions
    P_in_bar = inlet.pressure_bar
    T_in_C = inlet.temperature_C
    F_kmol_h = inlet.flowrate_kmol_h
    composition = inlet.composition

    # Total pressure rise
    total_delta_P = discharge_pressure_bar - P_in_bar
    if total_delta_P <= 0:
        raise ValueError(
            f"{pump_name}: Discharge pressure ({discharge_pressure_bar:.2f} bar) "
            f"must be > suction pressure ({P_in_bar:.2f} bar)"
        )

    logger.info(f"Inlet: {F_kmol_h:.1f} kmol/h @ {P_in_bar:.2f} bar, {T_in_C:.1f}°C")
    logger.info(f"Target discharge: {discharge_pressure_bar:.2f} bar")
    logger.info(f"Total ΔP: {total_delta_P:.2f} bar")
    logger.info("")

    # Get liquid properties at inlet
    properties = _calculate_liquid_properties(inlet, thermo, pump_name)

    # ========================================================================
    # DETERMINE NUMBER OF STAGES NEEDED
    # ========================================================================

    if not config["enable_interstage_cooling"]:
        # Single stage operation
        num_stages = 1
        delta_P_per_stage = total_delta_P
        logger.info("Interstage cooling: DISABLED")
        logger.info(f"Operating as single-stage pump")
    else:
        # Calculate temperature rise for single-stage operation
        single_stage_delta_T = _estimate_temperature_rise_for_delta_P(
            delta_P_bar=total_delta_P,
            T_in_C=T_in_C,
            properties=properties,
            efficiency=config["efficiency"]
        )

        max_temp_rise = config["max_temp_rise_per_stage_C"]

        if single_stage_delta_T <= max_temp_rise:
            # Single stage sufficient
            num_stages = 1
            delta_P_per_stage = total_delta_P
            logger.info("Interstage cooling: ENABLED (not needed)")
            logger.info(f"Estimated ΔT: {single_stage_delta_T:.1f}°C ≤ {max_temp_rise:.1f}°C")
            logger.info(f"Operating as single-stage pump")
        else:
            # Multiple stages needed
            num_stages = int(np.ceil(single_stage_delta_T / max_temp_rise))
            num_stages = max(2, min(num_stages, 10))  # Limit 2-10 stages
            delta_P_per_stage = total_delta_P / num_stages
            logger.info("Interstage cooling: ENABLED")
            logger.info(f"Estimated single-stage ΔT: {single_stage_delta_T:.1f}°C > {max_temp_rise:.1f}°C")
            logger.info(f"Using {num_stages} stages with ΔP = {delta_P_per_stage:.2f} bar per stage")

    logger.info("")

    # ========================================================================
    # STAGE-BY-STAGE CALCULATION
    # ========================================================================

    # Track stage outputs
    stage_summaries = []
    stage_cooling_duties = []
    stage_temperatures = [T_in_C]

    # Initialize for first stage
    current_stream = inlet
    total_hydraulic_power = 0.0
    total_brake_power = 0.0
    total_motor_power = 0.0
    total_cooling_duty = 0.0

    for stage_num in range(1, num_stages + 1):
        logger.info(f"--- Stage {stage_num}/{num_stages} ---")

        # Stage inlet conditions
        P_stage_in = current_stream.pressure_bar
        T_stage_in = current_stream.temperature_C

        # Stage discharge pressure
        if stage_num == num_stages:
            P_stage_out = discharge_pressure_bar  # Final stage hits target exactly
        else:
            P_stage_out = P_stage_in + delta_P_per_stage

        logger.info(f"  Inlet: {P_stage_in:.2f} bar, {T_stage_in:.1f}°C")
        logger.info(f"  Outlet pressure target: {P_stage_out:.2f} bar")

        # Calculate this stage
        stage_outlet, stage_summary = _calculate_single_pump_stage(
            inlet=current_stream,
            discharge_pressure_bar=P_stage_out,
            config=config,
            thermo=thermo,
            pump_name=f"{pump_name}-Stage{stage_num}"
        )

        # Accumulate power
        total_hydraulic_power += stage_summary["hydraulic_power_kW"]
        total_brake_power += stage_summary["brake_power_kW"]
        total_motor_power += stage_summary["motor_power_kW"]

        T_stage_out = stage_outlet.temperature_C
        logger.info(f"  Outlet: {P_stage_out:.2f} bar, {T_stage_out:.1f}°C")
        logger.info(f"  Power: {stage_summary['motor_power_kW']:.2f} kW")

        # Apply interstage cooling if not last stage
        cooling_duty_kW = 0.0
        if stage_num < num_stages and config["enable_interstage_cooling"]:
            # Determine cooling target
            if config["target_temp_after_cooling_C"] is not None:
                T_target = config["target_temp_after_cooling_C"]
            else:
                T_target = T_in_C  # Cool back to inlet temp

            # Clamp target
            T_target = max(config["min_temp_C"], min(T_target, T_stage_out))

            if T_stage_out > T_target:
                # Calculate cooling duty
                cooling_duty_kW = _calculate_cooling_duty(
                    stream=stage_outlet,
                    T_final_C=T_target,
                    thermo=thermo,
                    efficiency=config["cooling_efficiency"]
                )

                logger.info(f"  Interstage cooling: {T_stage_out:.1f}°C → {T_target:.1f}°C ({cooling_duty_kW:.2f} kW)")

                # Import Stream if not already done
                global Stream
                if Stream is None:
                    from simulation.streams import Stream

                # Create cooled stream for next stage
                current_stream = Stream(
                    name=f"{pump_name}_stage{stage_num}_cooled",
                    flowrate_kmol_h=stage_outlet.flowrate_kmol_h,
                    temperature_C=T_target,
                    pressure_bar=stage_outlet.pressure_bar,
                    composition=dict(stage_outlet.composition),
                    thermo=thermo,
                    phase="liquid"
                )

                stage_temperatures.append(T_target)
                total_cooling_duty += cooling_duty_kW
            else:
                current_stream = stage_outlet
                stage_temperatures.append(T_stage_out)
        else:
            current_stream = stage_outlet
            stage_temperatures.append(T_stage_out)

        stage_cooling_duties.append(cooling_duty_kW)
        stage_summaries.append(stage_summary)
        logger.info("")

    # Final outlet
    outlet = current_stream

    # ========================================================================
    # OVERALL SUMMARY
    # ========================================================================

    logger.info("=" * 70)
    logger.info(f"PUMP COMPLETE: {num_stages} stage(s)")
    logger.info(f"Total power: {total_motor_power:.2f} kW")
    logger.info(f"Total cooling duty: {total_cooling_duty:.2f} kW")
    logger.info(f"Final outlet: {outlet.pressure_bar:.2f} bar, {outlet.temperature_C:.1f}°C")
    logger.info("=" * 70)
    logger.info("")

    # Mechanical design (for overall pump)
    mechanical_design = _design_pump_mechanical(
        Q_m3_h=stage_summaries[0]["volumetric_flow_m3_h"],
        H_m=sum(s["head_m"] for s in stage_summaries),  # Total head
        P_brake_kW=total_brake_power,
        P_discharge_bar=discharge_pressure_bar,
        T_C=outlet.temperature_C,
        config=config
    )

    # Build comprehensive summary
    summary = {
        "pump_name": pump_name,
        "pump_type": config["type"],

        # Overall performance
        "num_stages": num_stages,
        "suction_pressure_bar": P_in_bar,
        "discharge_pressure_bar": discharge_pressure_bar,
        "pressure_rise_bar": total_delta_P,
        "suction_temperature_C": T_in_C,
        "discharge_temperature_C": outlet.temperature_C,
        "temperature_rise_C": outlet.temperature_C - T_in_C,

        # Power
        "total_hydraulic_power_kW": total_hydraulic_power,
        "total_brake_power_kW": total_brake_power,
        "total_motor_power_kW": total_motor_power,
        "motor_power_kW": total_motor_power,  # For compatibility

        # Cooling
        "interstage_cooling_enabled": config["enable_interstage_cooling"],
        "total_cooling_duty_kW": total_cooling_duty,
        "stage_cooling_duties_kW": stage_cooling_duties,

        # Stage details
        "stage_temperatures_C": stage_temperatures,
        "stage_summaries": stage_summaries,

        # Flow
        "volumetric_flow_m3_h": stage_summaries[0]["volumetric_flow_m3_h"],
        "mass_flow_kg_s": stage_summaries[0]["mass_flow_kg_s"],

        # Efficiency
        "average_pump_efficiency": np.mean([s["pump_efficiency"] for s in stage_summaries]),
        "motor_efficiency": config["motor_efficiency"],

        # NPSH (from first stage - most critical)
        "npsh_available_m": stage_summaries[0]["npsh_available_m"],
        "npsh_required_m": stage_summaries[0]["npsh_required_m"],
        "npsh_margin_m": stage_summaries[0]["npsh_margin_m"],

        # Properties
        "liquid_density_kg_m3": properties["density_kg_m3"],
        "liquid_viscosity_Pa_s": properties["viscosity_Pa_s"],

        # Mechanical design
        **mechanical_design,
    }

    # Safety check
    _check_pump_safety(
        P_discharge_bar=discharge_pressure_bar,
        T_C=outlet.temperature_C,
        P_motor_kW=total_motor_power,
        npsh_margin=stage_summaries[0]["npsh_margin_m"],
        pump_name=pump_name
    )

    return outlet, summary


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS FOR MULTI-STAGE OPERATION
# ═══════════════════════════════════════════════════════════════════════════

def _estimate_temperature_rise_for_delta_P(
    delta_P_bar: float,
    T_in_C: float,
    properties: dict,
    efficiency: float
) -> float:
    """Estimate temperature rise for a given pressure rise."""
    delta_T_isentropic = _calculate_isentropic_temperature_rise(
        P_in_bar=0,  # Dummy, only delta matters
        P_out_bar=delta_P_bar,
        T_in_C=T_in_C,
        properties=properties
    )
    delta_T_actual = delta_T_isentropic / efficiency
    return delta_T_actual


def _calculate_single_pump_stage(
    inlet: Any,
    discharge_pressure_bar: float,
    config: dict,
    thermo: Any,
    pump_name: str
) -> Tuple[Any, dict]:
    """
    Calculate a single pump stage.
    Returns (outlet_stream, stage_summary).
    """
    # Extract conditions
    P_in = inlet.pressure_bar
    T_in = inlet.temperature_C
    F_kmol_h = inlet.flowrate_kmol_h
    composition = inlet.composition

    delta_P_bar = discharge_pressure_bar - P_in

    # Get properties
    properties = _calculate_liquid_properties(inlet, thermo, pump_name)
    rho_kg_m3 = properties["density_kg_m3"]
    mu_Pa_s = properties["viscosity_Pa_s"]
    MW_kg_kmol = properties["molecular_weight_kg_kmol"]

    # Flow calculations
    m_dot_kg_s = F_kmol_h * MW_kg_kmol / 3600.0
    Q_m3_s = m_dot_kg_s / rho_kg_m3
    Q_m3_h = Q_m3_s * 3600.0

    # Head
    delta_P_Pa = delta_P_bar * 1e5
    H_m = delta_P_Pa / (rho_kg_m3 * GRAVITY)

    # Efficiency
    eta_pump = _calculate_pump_efficiency(Q_m3_h, H_m, config["efficiency"], mu_Pa_s)

    # Power
    P_hydraulic_kW = (m_dot_kg_s * delta_P_Pa / rho_kg_m3) / 1000.0
    P_brake_kW = P_hydraulic_kW / eta_pump
    P_motor_kW = P_brake_kW / config["motor_efficiency"]

    # Temperature rise
    delta_T_isentropic = _calculate_isentropic_temperature_rise(
        P_in, discharge_pressure_bar, T_in, properties
    )
    delta_T_actual = delta_T_isentropic / eta_pump

    # Clamp to physical limit
    max_dt = config.get("max_temp_rise_per_stage_C", 15.0)
    if delta_T_actual > max_dt:
        delta_T_actual = max_dt

    T_out = T_in + delta_T_actual

    # NPSH
    npsh_avail = _calculate_npsh_available(inlet, properties, thermo)
    npsh_req = config["npsh_required_m"]

    # Import Stream if not already done
    global Stream
    if Stream is None:
        from simulation.streams import Stream

    # Create outlet
    outlet = Stream(
        name=f"{inlet.name}_stage_out",
        flowrate_kmol_h=F_kmol_h,
        temperature_C=T_out,
        pressure_bar=discharge_pressure_bar,
        composition=dict(composition),
        thermo=thermo,
        phase="liquid"
    )

    summary = {
        "pressure_rise_bar": delta_P_bar,
        "temperature_rise_C": delta_T_actual,
        "volumetric_flow_m3_h": Q_m3_h,
        "mass_flow_kg_s": m_dot_kg_s,
        "head_m": H_m,
        "hydraulic_power_kW": P_hydraulic_kW,
        "brake_power_kW": P_brake_kW,
        "motor_power_kW": P_motor_kW,
        "pump_efficiency": eta_pump,
        "npsh_available_m": npsh_avail,
        "npsh_required_m": npsh_req,
        "npsh_margin_m": npsh_avail - npsh_req,
    }

    return outlet, summary


def _calculate_cooling_duty(
    stream: Any,
    T_final_C: float,
    thermo: Any,
    efficiency: float = 0.95
) -> float:
    """
    Calculate cooling duty to cool stream from current T to T_final.
    Returns duty in kW (positive value).
    """
    T_initial = stream.temperature_C
    delta_T = T_initial - T_final_C

    if delta_T <= 0:
        return 0.0

    # Get heat capacity
    if thermo and hasattr(thermo, "heat_capacity_TP"):
        try:
            Cp_kJ_kmolK = thermo.heat_capacity_TP(
                T_initial, stream.pressure_bar, stream.composition, phase="liquid"
            )
        except:
            Cp_kJ_kmolK = 135.0  # Default for hydrocarbons
    else:
        Cp_kJ_kmolK = 135.0

    # Cooling duty
    Q_cooling_kW = stream.flowrate_kmol_h * Cp_kJ_kmolK * delta_T / 3600.0

    # Account for cooling efficiency
    Q_actual_kW = Q_cooling_kW / efficiency

    return abs(Q_actual_kW)


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _validate_pump_inputs(
    inlet: Any,
    discharge_pressure_bar: float,
    config: dict,
    pump_name: str,
) -> None:
    """Validate pump inputs."""
    if inlet is None:
        raise ValueError(f"{pump_name}: Inlet stream cannot be None")

    if inlet.flowrate_kmol_h <= 0:
        raise ValueError(
            f"{pump_name}: Flowrate must be positive, got {inlet.flowrate_kmol_h}"
        )

    if discharge_pressure_bar <= 0:
        raise ValueError(
            f"{pump_name}: Discharge pressure must be positive, "
            f"got {discharge_pressure_bar}"
        )

    if discharge_pressure_bar > PUMP_SAFETY_LIMITS["max_discharge_pressure_bar"]:
        raise ValueError(
            f"{pump_name}: Discharge pressure {discharge_pressure_bar:.1f} bar "
            f"exceeds maximum {PUMP_SAFETY_LIMITS['max_discharge_pressure_bar']} bar"
        )

    # Check if liquid phase
    phase = getattr(inlet, "phase", "liquid")
    if phase != "liquid":
        logger.warning(
            f"{pump_name}: Stream phase is '{phase}', expected 'liquid'. "
            f"Pump designed for liquids only."
        )

    # Check efficiency range
    if not 0.1 <= config["efficiency"] <= 1.0:
        raise ValueError(
            f"{pump_name}: Pump efficiency {config['efficiency']} "
            f"must be between 0.1 and 1.0"
        )


# ═══════════════════════════════════════════════════════════════════════════
# PROPERTY CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════

def _calculate_liquid_properties(
    inlet: Any,
    thermo: Any,
    pump_name: str,
) -> dict:
    """Calculate liquid properties for pump design."""
    T_C = inlet.temperature_C
    P_bar = inlet.pressure_bar
    composition = inlet.composition

    properties: Dict[str, float] = {}

    # Molecular weight
    if thermo:
        try:
            MW = thermo.molecular_weight(composition)
        except Exception:
            MW = 80.0  # Default for hydrocarbons
    else:
        MW = 80.0
    properties["molecular_weight_kg_kmol"] = MW

    # Density
    if thermo and hasattr(thermo, "density_TP"):
        try:
            rho = thermo.density_TP(T_C, P_bar, composition, phase="liquid")
        except Exception:
            logger.warning(f"{pump_name}: Density calculation failed, using default")
            rho = 750.0  # kg/m³ for light hydrocarbons
    else:
        rho = 750.0
    properties["density_kg_m3"] = rho

    # Viscosity
    if thermo and hasattr(thermo, "viscosity_TP"):
        try:
            mu = thermo.viscosity_TP(T_C, P_bar, composition, phase="liquid")
        except Exception:
            logger.warning(f"{pump_name}: Viscosity calculation failed, using default")
            mu = 3.0e-4  # Pa·s for light hydrocarbons
    else:
        mu = 3.0e-4
    properties["viscosity_Pa_s"] = mu

    # Heat capacity
    if thermo and hasattr(thermo, "heat_capacity_TP"):
        try:
            Cp_kJ_kmolK = thermo.heat_capacity_TP(
                T_C, P_bar, composition, phase="liquid"
            )
            Cp_J_kgK = Cp_kJ_kmolK * 1000.0 / MW
        except Exception:
            Cp_J_kgK = 2000.0  # J/(kg·K)
    else:
        Cp_J_kgK = 2000.0
    properties["heat_capacity_J_kgK"] = Cp_J_kgK

    # Vapor pressure (for NPSH)
    if thermo and hasattr(thermo, "vapor_pressure"):
        try:
            Pv_bar = thermo.vapor_pressure(T_C, composition)
        except Exception:
            Pv_bar = 0.1  # bar (estimate)
    else:
        Pv_bar = 0.1
    properties["vapor_pressure_bar"] = Pv_bar

    return properties


# ═══════════════════════════════════════════════════════════════════════════
# EFFICIENCY CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════

def _calculate_pump_efficiency(
    Q_m3_h: float,
    H_m: float,
    base_efficiency: float,
    viscosity_Pa_s: float,
) -> float:
    """
    Calculate pump efficiency corrected for operating point and viscosity.
    Uses empirical correlations.
    """
    eta = base_efficiency

    # Viscosity correction (for water = 0.001 Pa·s as reference)
    mu_water = 0.001
    if viscosity_Pa_s > 0.005:  # Significant viscosity effect
        # Empirical correction factor
        eta_correction = (mu_water / viscosity_Pa_s) ** 0.1
        eta *= eta_correction
        logger.debug(f"  Viscosity correction: η × {eta_correction:.3f}")

    # Ensure reasonable bounds
    eta = max(0.3, min(eta, 0.92))

    return eta


def _calculate_isentropic_temperature_rise(
    P_in_bar: float,
    P_out_bar: float,
    T_in_C: float,
    properties: dict,
) -> float:
    """Calculate isentropic temperature rise for liquid compression."""
    # For liquids (nearly incompressible), temperature rise is small
    # ΔT ≈ (T·v·ΔP) / Cp where v = 1/ρ

    T_K = T_in_C + 273.15
    delta_P_Pa = (P_out_bar - P_in_bar) * 1e5
    rho = properties["density_kg_m3"]
    Cp = properties["heat_capacity_J_kgK"]

    v_m3_kg = 1.0 / rho
    delta_T_K = (T_K * v_m3_kg * delta_P_Pa) / Cp

    return delta_T_K  # °C (same as K for differences)


# ═══════════════════════════════════════════════════════════════════════════
# NPSH CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════

def _calculate_npsh_available(
    inlet: Any,
    properties: dict,
    thermo: Any,
) -> float:
    """
    Calculate Net Positive Suction Head Available (NPSHA).

    NPSHA = (P_suction - P_vapor) / (ρ·g) + elevation

    For process simulation, elevation assumed zero.
    """
    P_suction_Pa = inlet.pressure_bar * 1e5
    P_vapor_Pa = properties["vapor_pressure_bar"] * 1e5
    rho = properties["density_kg_m3"]

    npsha_m = (P_suction_Pa - P_vapor_Pa) / (rho * GRAVITY)

    # Add static head if available (assumed zero here)
    elevation_m = 0.0
    npsha_total = npsha_m + elevation_m

    return max(npsha_total, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# MECHANICAL DESIGN
# ═══════════════════════════════════════════════════════════════════════════

def _design_pump_mechanical(
    Q_m3_h: float,
    H_m: float,
    P_brake_kW: float,
    P_discharge_bar: float,
    T_C: float,
    config: dict,
) -> dict:
    """
    Mechanical design calculations for centrifugal pump.
    Returns casing thickness, shaft diameter, impeller size, etc.
    """
    design: Dict[str, float] = {}

    # Specific speed (dimensionless, for impeller selection)
    # N_s = N·√Q / H^0.75 (assuming 3600 rpm)
    N_rpm = 3600.0  # Standard motor speed
    Ns = N_rpm * math.sqrt(Q_m3_h) / (H_m ** 0.75) if H_m > 0 else 0.0
    design["specific_speed_dimensionless"] = Ns
    design["rotational_speed_rpm"] = N_rpm

    # Classify pump type based on specific speed
    if Ns < 500.0:
        pump_class = "radial_flow"
    elif Ns < 1500.0:
        pump_class = "mixed_flow"
    else:
        pump_class = "axial_flow"
    design["pump_class"] = pump_class

    # Impeller diameter estimate (simplified)
    # D ∝ √(H/N²)
    D_impeller_m = 0.2 * math.sqrt(H_m / ((N_rpm / 1000.0) ** 2)) if H_m > 0 else 0.1
    D_impeller_m = max(0.1, min(D_impeller_m, 1.5))  # Practical limits
    design["impeller_diameter_m"] = D_impeller_m

    # Casing thickness (ASME pressure vessel code)
    material = config.get("material", "carbon_steel")
    mat_props = MATERIAL_PROPERTIES.get(material, MATERIAL_PROPERTIES["carbon_steel"])

    P_design_bar = P_discharge_bar * 1.1  # Design pressure = 110% operating
    P_design_MPa = P_design_bar / 10.0
    S_allowable_MPa = mat_props["allowable_stress_MPa"]
    corrosion_mm = mat_props["corrosion_allowance_mm"]

    D_casing_m = D_impeller_m * 1.5  # Casing ~1.5× impeller

    # Simplified thickness: t = P·D / (2·S - P) + corrosion
    t_mm = (P_design_MPa * D_casing_m * 1000.0) / (
        2.0 * S_allowable_MPa - P_design_MPa
    ) + corrosion_mm
    t_mm = max(t_mm, 6.0)  # Minimum thickness 6 mm

    design["casing_diameter_m"] = D_casing_m
    design["casing_thickness_mm"] = t_mm
    design["casing_material"] = material

    # Shaft diameter (torsional strength)
    # τ = 16·T / (π·d³) ≤ S_s/n where T = P/(2πN)
    T_Nm = (P_brake_kW * 1000.0) / (2.0 * math.pi * N_rpm / 60.0) if N_rpm > 0 else 0.0
    S_shaft_MPa = 80.0  # Shear strength for steel
    n_safety = 3.0  # Safety factor

    if T_Nm > 0:
        d_shaft_m = (
            (16.0 * T_Nm * n_safety) / (math.pi * S_shaft_MPa * 1e6)
        ) ** (1.0 / 3.0)
    else:
        d_shaft_m = 0.025

    d_shaft_mm = max(d_shaft_m * 1000.0, 25.0)  # Minimum 25 mm
    design["shaft_diameter_mm"] = d_shaft_mm

    # Estimate weight
    V_casing_m3 = math.pi * (D_casing_m / 2.0) ** 2 * (t_mm / 1000.0) * 3.0  # Approx
    rho_steel = 7850.0  # kg/m³
    weight_kg = V_casing_m3 * rho_steel + 50.0  # + impeller weight
    design["estimated_weight_kg"] = weight_kg

    return design


# ═══════════════════════════════════════════════════════════════════════════
# SAFETY CHECKS
# ═══════════════════════════════════════════════════════════════════════════

def _check_pump_safety(
    P_discharge_bar: float,
    T_C: float,
    P_motor_kW: float,
    npsh_margin: float,
    pump_name: str,
) -> None:
    """Check pump operating conditions against safety limits."""
    limits = PUMP_SAFETY_LIMITS

    # Pressure check
    if P_discharge_bar > limits["max_discharge_pressure_bar"]:
        raise RuntimeError(
            f"{pump_name}: Discharge pressure {P_discharge_bar:.1f} bar "
            f"exceeds maximum {limits['max_discharge_pressure_bar']} bar"
        )

    # Temperature check
    if T_C > limits["max_temperature_C"]:
        raise RuntimeError(
            f"{pump_name}: Temperature {T_C:.1f}°C "
            f"exceeds maximum {limits['max_temperature_C']}°C"
        )

    # Power check
    if P_motor_kW > limits["max_power_kW"]:
        logger.warning(
            f"{pump_name}: Motor power {P_motor_kW:.1f} kW "
            f"exceeds typical maximum {limits['max_power_kW']} kW"
        )

    # # NPSH check
    # if npsh_margin < limits["min_npsh_margin_m"]:
    #     logger.error(
    #         f"{pump_name}: NPSH margin {npsh_margin:.2f} m < "
    #         f"minimum {limits['min_npsh_margin_m']} m. CAVITATION RISK!"
    #     )
    #

# ═══════════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("PUMP MODULE SMOKE TEST - MULTI-STAGE")
    print("=" * 70)

    # Create minimal thermo for testing
    class FakeThermo:
        def molecular_weight(self, comp):
            return 78.11 if "benzene" in comp else 2.0

        def density_TP(self, T, P, comp, phase):
            return 870.0  # kg/m³ for benzene

        def viscosity_TP(self, T, P, comp, phase):
            return 6.5e-4  # Pa·s for benzene

        def heat_capacity_TP(self, T, P, comp, phase):
            return 135.0  # kJ/(kmol·K)

        def vapor_pressure(self, T, comp):
            return 0.12  # bar at 25°C

    thermo = FakeThermo()

    # Import Stream
    from simulation.streams import Stream

    # TEST 1: Single-stage (low pressure)
    print("\n" + "=" * 70)
    print("TEST 1: Single-stage pump (low ΔP)")
    print("=" * 70)

    inlet1 = Stream(
        name="benzene_feed",
        flowrate_kmol_h=200.0,
        temperature_C=25.0,
        pressure_bar=1.013,
        composition={"benzene": 1.0},
        thermo=thermo,
        phase="liquid",
    )

    outlet1, summary1 = pump_liquid(
        inlet=inlet1,
        discharge_pressure_bar=10.0,
        thermo=thermo,
        pump_name="P-101-Test",
    )

    print("\n✓ Single-stage test completed")
    print(f"  Stages: {summary1['num_stages']}")
    print(f"  Power: {summary1['motor_power_kW']:.2f} kW")
    print(f"  Cooling: {summary1['total_cooling_duty_kW']:.2f} kW")
    print(f"  Outlet: {outlet1.pressure_bar:.2f} bar @ {outlet1.temperature_C:.1f}°C")

    # TEST 2: Multi-stage (high pressure)
    print("\n" + "=" * 70)
    print("TEST 2: Multi-stage pump (high ΔP)")
    print("=" * 70)

    inlet2 = Stream(
        name="benzene_feed_2",
        flowrate_kmol_h=200.0,
        temperature_C=25.0,
        pressure_bar=1.013,
        composition={"benzene": 1.0},
        thermo=thermo,
        phase="liquid",
    )

    outlet2, summary2 = pump_liquid(
        inlet=inlet2,
        discharge_pressure_bar=35.0,
        pump_config={"max_temp_rise_per_stage_C": 10.0},
        thermo=thermo,
        pump_name="P-102-Test",
    )

    print("\n✓ Multi-stage test completed")
    print(f"  Stages: {summary2['num_stages']}")
    print(f"  Power: {summary2['motor_power_kW']:.2f} kW")
    print(f"  Cooling: {summary2['total_cooling_duty_kW']:.2f} kW")
    print(f"  Outlet: {outlet2.pressure_bar:.2f} bar @ {outlet2.temperature_C:.1f}°C")
    print(f"  Temperature profile: {summary2['stage_temperatures_C']}")

    # Assertions
    assert summary1["num_stages"] == 1, "Test 1 should be single-stage"
    assert summary2["num_stages"] >= 2, "Test 2 should be multi-stage"
    assert summary2["total_cooling_duty_kW"] > 0, "Test 2 should have cooling"
    assert outlet2.pressure_bar == 35.0, "Final pressure mismatch"

    print("\n" + "=" * 70)
    print("✓ ALL PUMP SMOKE TESTS PASSED")
    print("=" * 70)
