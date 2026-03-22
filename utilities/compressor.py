"""
utilities/compressor.py
===============================================================================

All thermodynamic properties (MW, Cp, density, gamma) are now calculated
using the ThermodynamicPackage. No more hardcoded fallbacks causing errors.

Date: 2026-01-18
Version: 3.0.0
"""

from __future__ import annotations
import math
import logging
from typing import Any, Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================
R_GAS_J_MOLK = 8.314  # J/(mol·K)
R_GAS_BAR_M3_KMOLK = 0.08314  # bar·m³/(kmol·K)

# ============================================================================
# DEFAULT COMPRESSOR CONFIGURATION
# ============================================================================
COMPRESSOR_DEFAULTS = {
    "type": "centrifugal",
    "number_of_stages": 1,
    "polytropic_efficiency": 0.80,
    "mechanical_efficiency": 0.98,
    "driver_efficiency": 0.95,
    "intercooling": False,
    "intercooler_outlet_temperature_C": 40.0,
    "intercooler_pressure_drop_bar": 0.2,
    "design_pressure_bar": 50.0,
    "design_temperature_C": 200.0,
    "surge_margin_percent": 10.0,
    "pressure_ratio_per_stage_max": 3.5,
    "tip_speed_max_ms": 300.0,
}

COMPRESSOR_SAFETY_LIMITS = {
    "max_discharge_pressure_bar": 50.0,
    "max_discharge_temperature_C": 200.0,
    "max_compression_ratio": 12.0,
    "max_power_kW": 5000.0,
    "min_surge_margin_percent": 10.0,
    "max_polytropic_head_per_stage_kJ_kg": 80.0,
}


# ============================================================================
# MAIN COMPRESSOR FUNCTION
# ============================================================================
def compress_gas(
    inlet: Any,
    discharge_pressure_bar: float,
    compressor_config: Optional[dict] = None,
    thermo: Any = None,
    compressor_name: str = "C-101",
) -> Tuple[Any, dict]:
    """
    Rigorous multi-stage centrifugal compressor with intercooling.

    All thermodynamic properties calculated using thermo package.

    Args:
        inlet: Inlet gas stream
        discharge_pressure_bar: Target discharge pressure (bar)
        compressor_config: Optional compressor configuration
        thermo: ThermodynamicPackage (REQUIRED for property calculations)
        compressor_name: Compressor identifier

    Returns:
        (outlet_stream, summary_dict)

    Raises:
        ValueError: Invalid inputs or missing thermo package
        RuntimeError: Calculation failure
    """

    if thermo is None:
        raise ValueError(f"{compressor_name}: ThermodynamicPackage is required")

    logger.info(f"Running compressor calculation: {compressor_name}")

    # Merge config
    config = {**COMPRESSOR_DEFAULTS, **(compressor_config or {})}

    # Validate inputs
    _validate_compressor_inputs(inlet, discharge_pressure_bar, config, compressor_name)

    # Extract inlet conditions
    P_in_bar = inlet.pressure_bar
    T_in_C = inlet.temperature_C
    T_in_K = T_in_C + 273.15
    F_kmol_h = inlet.flowrate_kmol_h
    composition = inlet.composition

    # Compression ratio
    r_overall = discharge_pressure_bar / P_in_bar

    if r_overall <= 1.0:
        raise ValueError(
            f"{compressor_name}: Discharge pressure ({discharge_pressure_bar:.2f} bar) "
            f"must be > suction pressure ({P_in_bar:.2f} bar)"
        )

    if r_overall > COMPRESSOR_SAFETY_LIMITS["max_compression_ratio"]:
        raise ValueError(
            f"{compressor_name}: Compression ratio {r_overall:.2f} exceeds "
            f"maximum {COMPRESSOR_SAFETY_LIMITS['max_compression_ratio']}"
        )

    logger.debug(f"  Compression ratio: {r_overall:.2f}, Tin: {T_in_C:.1f}°C")

    # Get inlet gas properties from thermo package
    properties = _calculate_gas_properties(inlet, thermo, compressor_name)
    MW_kg_kmol = properties["molecular_weight_kg_kmol"]
    gamma = properties["heat_capacity_ratio"]
    Cp_J_kgK = properties["heat_capacity_J_kgK"]

    # Mass flowrate
    m_dot_kg_s = F_kmol_h * MW_kg_kmol / 3600.0
    logger.debug(f"  Mass flow: {m_dot_kg_s:.2f} kg/s, gamma={gamma:.3f}")

    # Determine number of stages
    intercooling = config["intercooling"]
    r_max_per_stage = config["pressure_ratio_per_stage_max"]

    if intercooling:
        n_stages = _calculate_optimal_stages(r_overall, r_max_per_stage)
    else:
        n_stages = config["number_of_stages"]

    if n_stages == 1 and r_overall > r_max_per_stage:
        logger.warning(
            f"{compressor_name}: Single-stage compression ratio {r_overall:.2f} "
            f"exceeds recommended maximum {r_max_per_stage}. "
            f"Consider multi-stage with intercooling."
        )

    logger.info(f"  Using {n_stages} stages, intercooling={intercooling}")

    # Equal pressure ratio per stage
    r_per_stage = r_overall ** (1.0 / n_stages)

    # Stage-by-stage calculation
    stage_results = []
    P_stage = P_in_bar
    T_stage_C = T_in_C
    total_power_kW = 0.0

    for stage_num in range(1, n_stages + 1):
        logger.debug(f"  Stage {stage_num}")

        # Stage inlet conditions
        P_stage_in = P_stage
        T_stage_in_C = T_stage_C
        T_stage_in_K = T_stage_in_C + 273.15

        # Stage outlet pressure
        P_stage_out = P_stage_in * r_per_stage

        # Polytropic efficiency
        eta_p = config["polytropic_efficiency"]

        # Polytropic compression calculation
        # Using: T2/T1 = (P2/P1)^[(gamma-1)/(gamma*eta_p)]
        n_polytropic = (gamma / (gamma - 1.0)) * eta_p

        T_stage_out_K = T_stage_in_K * (r_per_stage ** ((n_polytropic - 1.0) / n_polytropic))
        T_stage_out_C = T_stage_out_K - 273.15

        # Polytropic work per unit mass
        h_polytropic_kJ_kg = Cp_J_kgK * (T_stage_out_K - T_stage_in_K) / 1000.0

        # Stage power
        P_stage_kW = m_dot_kg_s * h_polytropic_kJ_kg

        # Apply mechanical efficiency
        eta_mech = config["mechanical_efficiency"]
        P_stage_shaft_kW = P_stage_kW / eta_mech
        total_power_kW += P_stage_shaft_kW

        logger.debug(f"    P: {P_stage_in:.2f} → {P_stage_out:.2f} bar")
        logger.debug(f"    T: {T_stage_in_C:.1f} → {T_stage_out_C:.1f}°C")
        logger.debug(f"    Power: {P_stage_shaft_kW:.1f} kW")

        # Store stage results
        stage_results.append({
            "stage_number": stage_num,
            "inlet_pressure_bar": P_stage_in,
            "outlet_pressure_bar": P_stage_out,
            "inlet_temperature_C": T_stage_in_C,
            "outlet_temperature_C": T_stage_out_C,
            "pressure_ratio": r_per_stage,
            "polytropic_head_kJ_kg": h_polytropic_kJ_kg,
            "shaft_power_kW": P_stage_shaft_kW,
        })

        # Update conditions for next stage
        P_stage = P_stage_out
        T_stage_C = T_stage_out_C

        # Apply intercooling (if not last stage)
        if intercooling and stage_num < n_stages:
            T_cooled_C = config["intercooler_outlet_temperature_C"]
            P_stage -= config["intercooler_pressure_drop_bar"]  # Pressure drop in cooler
            T_stage_C = T_cooled_C
            logger.debug(f"    Intercooling: {T_stage_out_C:.1f}°C → {T_cooled_C:.1f}°C")

    # Final outlet conditions
    P_out_bar = P_stage
    T_out_C = T_stage_C

    # Driver power
    eta_driver = config["driver_efficiency"]
    P_driver_kW = total_power_kW / eta_driver

    logger.info(
        f"  Total: Tout={T_out_C:.1f}°C, "
        f"Pshaft={total_power_kW:.1f} kW, "
        f"Pdriver={P_driver_kW:.1f} kW"
    )

    # Safety checks
    _check_compressor_safety(
        P_out_bar=P_out_bar,
        T_out_C=T_out_C,
        r_overall=r_overall,
        P_driver_kW=P_driver_kW,
        compressor_name=compressor_name,
    )

    # Build outlet stream
    from simulation.streams import Stream
    outlet = Stream(
        name=f"{inlet.name}_compressed",
        flowrate_kmol_h=F_kmol_h,
        temperature_C=T_out_C,
        pressure_bar=P_out_bar,
        composition=dict(composition),
        thermo=thermo,
        phase="vapor",
    )

    # ═══════════════════════════════════════════════════════════════
    # CALCULATE HEAT GENERATED (for cooling load)
    # ═══════════════════════════════════════════════════════════════
    # Compression work → Gas heating
    # If T_rise > threshold → This heat must be removed

    T_rise_C = T_out_C - T_in_C
    HEAT_THRESHOLD_C = 10.0  # Significant heating threshold

    if T_rise_C > HEAT_THRESHOLD_C:
        # Shaft work converted to gas enthalpy increase
        # This heat must be removed if gas needs cooling
        heat_to_remove_kW = total_power_kW  # Shaft power = heat in gas
        logger.info(
            f"{compressor_name}: T_rise={T_rise_C:.1f}°C > {HEAT_THRESHOLD_C:.1f}°C → "
            f"Heat to remove: {heat_to_remove_kW:.1f} kW"
        )
    else:
        heat_to_remove_kW = 0.0
        logger.debug(
            f"{compressor_name}: T_rise={T_rise_C:.1f}°C < {HEAT_THRESHOLD_C:.1f}°C → "
            f"No significant heating"
        )

    # Build summary
    summary = {
        "compressor_name": compressor_name,
        "compressor_type": config["type"],
        "number_of_stages": n_stages,
        "intercooling": intercooling,
        "suction_pressure_bar": P_in_bar,
        "discharge_pressure_bar": P_out_bar,
        "overall_pressure_ratio": r_overall,
        "pressure_ratio_per_stage": r_per_stage,
        "suction_temperature_C": T_in_C,
        "discharge_temperature_C": T_out_C,
        "temperature_rise_C": T_rise_C,
        "mass_flow_kg_s": m_dot_kg_s,
        "volumetric_flow_m3_s": m_dot_kg_s / properties["density_kg_m3"],
        "polytropic_efficiency": eta_p,
        "mechanical_efficiency": eta_mech,
        "driver_efficiency": eta_driver,
        "overall_efficiency": eta_p * eta_mech * eta_driver,
        "total_polytropic_work_kW": total_power_kW * eta_mech,
        "shaft_power_kW": total_power_kW,
        "driver_power_kW": P_driver_kW,
        "heat_capacity_ratio_gamma": gamma,
        "stage_results": stage_results,
        "heat_to_remove_kW": heat_to_remove_kW,  # ← ADD THIS LINE
    }

    logger.info(f"{compressor_name}: r={r_overall:.2f}, P={P_driver_kW:.0f} kW, Tout={T_out_C:.1f}°C")

    return outlet, summary


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _validate_compressor_inputs(
    inlet: Any,
    discharge_pressure_bar: float,
    config: dict,
    compressor_name: str,
) -> None:
    """Validate compressor inputs."""
    if inlet is None:
        raise ValueError(f"{compressor_name}: Inlet stream cannot be None")

    if inlet.flowrate_kmol_h <= 0:
        raise ValueError(
            f"{compressor_name}: Flowrate must be positive, got {inlet.flowrate_kmol_h}"
        )

    if discharge_pressure_bar <= 0:
        raise ValueError(f"{compressor_name}: Discharge pressure must be positive")

    # Check phase
    phase = getattr(inlet, "phase", "vapor")
    if phase != "vapor":
        logger.warning(
            f"{compressor_name}: Stream phase is '{phase}', expected 'vapor'. "
            f"Compressor designed for gases only."
        )

    # Check efficiency
    if not (0.5 <= config["polytropic_efficiency"] <= 0.95):
        raise ValueError(
            f"{compressor_name}: Polytropic efficiency {config['polytropic_efficiency']} "
            f"must be between 0.5 and 0.95"
        )


def _calculate_gas_properties(inlet: Any, thermo: Any, compressor_name: str) -> dict:
    """
    Calculate gas properties using ONLY the thermodynamic package.

    No more hardcoded fallbacks - everything comes from thermo.
    """
    T_C = inlet.temperature_C
    P_bar = inlet.pressure_bar
    composition = inlet.composition

    properties = {}

    # 1. MOLECULAR WEIGHT - from thermo package
    try:
        MW = thermo.molecular_weight(composition)
        if MW <= 0:
            raise ValueError(f"Molecular weight is {MW}, expected > 0")
        properties["molecular_weight_kg_kmol"] = MW
        logger.debug(f"  MW from thermo: {MW:.3f} kg/kmol")
    except Exception as e:
        raise RuntimeError(
            f"{compressor_name}: Failed to calculate molecular weight: {e}"
        )

    # 2. DENSITY - from thermo package
    try:
        rho = thermo.density_TP(T_C, P_bar, composition, phase="vapor")
        if rho <= 0:
            raise ValueError(f"Density is {rho}, expected > 0")
        properties["density_kg_m3"] = rho
        logger.debug(f"  Density from thermo: {rho:.3f} kg/m³")
    except Exception as e:
        raise RuntimeError(
            f"{compressor_name}: Failed to calculate density: {e}"
        )

    # 3. HEAT CAPACITY - from thermo package
    try:
        Cp_J_molK = thermo.ideal_gas_cp(T_C, composition)
        if Cp_J_molK <= 0:
            raise ValueError(f"Heat capacity is {Cp_J_molK}, expected > 0")
        Cp_J_kgK = Cp_J_molK / MW * 1000.0
        properties["heat_capacity_J_kgK"] = Cp_J_kgK
        logger.debug(f"  Cp from thermo: {Cp_J_kgK:.1f} J/(kg·K)")
    except Exception as e:
        raise RuntimeError(
            f"{compressor_name}: Failed to calculate heat capacity: {e}"
        )

    # 4. HEAT CAPACITY RATIO (gamma = Cp/Cv)
    # For ideal gas: Cv = Cp - R
    R_specific = R_GAS_J_MOLK / MW * 1000.0  # J/(kg·K)
    Cv_J_kgK = Cp_J_kgK - R_specific

    if Cv_J_kgK > 0:
        gamma = Cp_J_kgK / Cv_J_kgK
    else:
        gamma = 1.4  # Default for diatomic gases

    # Physical bounds for gamma
    gamma = max(1.1, min(gamma, 1.67))
    properties["heat_capacity_ratio"] = gamma
    logger.debug(f"  Gamma calculated: {gamma:.3f}")

    return properties


def _calculate_optimal_stages(r_overall: float, r_max_per_stage: float) -> int:
    """Calculate optimal number of stages for intercooled compression."""
    # Minimum stages needed to stay below maximum pressure ratio per stage
    n_min = math.ceil(math.log(r_overall) / math.log(r_max_per_stage))

    # Practical: use 2-4 stages for r_overall up to 12
    if r_overall <= 3.5:
        n_opt = 1
    elif r_overall <= 8.0:
        n_opt = 2
    elif r_overall <= 12.0:
        n_opt = 3
    else:
        n_opt = 4

    return max(n_min, n_opt)


def _check_compressor_safety(
    P_out_bar: float,
    T_out_C: float,
    r_overall: float,
    P_driver_kW: float,
    compressor_name: str,
) -> None:
    """Check compressor operating conditions against safety limits."""
    limits = COMPRESSOR_SAFETY_LIMITS

    # Pressure check
    if P_out_bar > limits["max_discharge_pressure_bar"]:
        raise RuntimeError(
            f"{compressor_name}: Discharge pressure {P_out_bar:.1f} bar "
            f"exceeds maximum {limits['max_discharge_pressure_bar']} bar"
        )

    # Temperature check
    if T_out_C > limits["max_discharge_temperature_C"]:
        raise RuntimeError(
            f"{compressor_name}: Discharge temperature {T_out_C:.1f}°C "
            f"exceeds maximum {limits['max_discharge_temperature_C']}°C"
        )

    # Compression ratio check
    if r_overall > limits["max_compression_ratio"]:
        raise RuntimeError(
            f"{compressor_name}: Compression ratio {r_overall:.2f} "
            f"exceeds maximum {limits['max_compression_ratio']}"
        )

    # Power check
    if P_driver_kW > limits["max_power_kW"]:
        logger.warning(
            f"{compressor_name}: Driver power {P_driver_kW:.1f} kW "
            f"exceeds typical maximum {limits['max_power_kW']} kW"
        )
