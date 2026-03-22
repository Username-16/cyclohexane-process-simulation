"""
heat_transfer/heat_exchanger.py
===============================================================================

PURPOSE:
Combined shell-and-tube heat exchanger models for:
- Feed-effluent heat exchange (FEHE)
- General cooling/heating services

Date: 2026-01-16
Version: 3.2.0
"""
from __future__ import annotations
import math
import logging
from typing import Any, Dict, Optional, Tuple
import numpy as np
logger = logging.getLogger(__name__)
# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================
GRAVITY = 9.81  # m/s²
# Standard tube specifications
STANDARD_TUBES = {
    19.05: {  # 3/4 inch OD (mm)
        14: {"wall_mm": 2.11, "ID_mm": 14.83},  # BWG 14
        16: {"wall_mm": 1.65, "ID_mm": 15.75},  # BWG 16
    },
    25.4: {  # 1 inch OD (mm)
        14: {"wall_mm": 2.11, "ID_mm": 21.18},
        16: {"wall_mm": 1.65, "ID_mm": 22.10},
    },
}

# Thermal conductivities (W/(m·K))
THERMAL_CONDUCTIVITY = {
    "carbon_steel": 50.0,
    "stainless_steel": 16.0,
    "copper": 400.0,
}
# ─────────────────────────────────────────────────────────────────────────────
# DEFAULTS (from refactored JSON-free HX)
# ─────────────────────────────────────────────────────────────────────────────
HX_DEFAULTS = {
    "name": "heat_exchanger_HX-101",
    "type": "shell_and_tube",
    "service": "process_cooling",  # for FEHE use 'FEHE'
    "tube_outer_diameter_mm": 19.05,
    "tube_BWG": 14,
    "tube_length_m": 6.0,
    "number_of_tube_passes": 2,
    "number_of_shell_passes": 1,
    "tube_layout": "triangular",
    "tube_pitch_mm": 25.0,
    "overall_U_W_m2K": 350.0,
    "pressure_drop_hot_bar": 0.5,
    "pressure_drop_cold_bar": 0.5,
    "tube_material": "carbon_steel",
    "shell_material": "carbon_steel",
    "fouling_resistance_hot_m2K_W": 0.0002,
    "fouling_resistance_cold_m2K_W": 0.0002,
    "baffle_cut_percent": 25.0,
    "baffle_spacing_m": 0.4,
}

# For detailed FEHE config we will normalize user config on top of HX_DEFAULTS.
# ============================================================================
# PUBLIC API
# ============================================================================
def run_heat_exchanger(
    hot_inlet: Any,  # Stream
    cold_inlet: Any,  # Stream
    exchanger_config: Optional[dict] = None,
    thermo: Any = None,  # ThermodynamicPackage
    mode: str = "design",  # "design" or "rating"
) -> Tuple[Any, Any, dict]:
    if exchanger_config is None:
        exchanger_config = {}
    logger.info(f"Running heat exchanger: {exchanger_config.get('name', 'HX')}")
    # Validate inputs
    _validate_hx_inputs(hot_inlet, cold_inlet, exchanger_config, mode)
    # Extract configuration on top of JSON-free defaults
    config = _extract_hx_config(exchanger_config)
    logger.debug(
        f"  Hot inlet: {hot_inlet.temperature_C:.1f}°C, "
        f"Cold inlet: {cold_inlet.temperature_C:.1f}°C"
    )
    # Assign fluids to shell vs tube
    hot_side, cold_side = _assign_fluids_to_sides(hot_inlet, cold_inlet, config)
    logger.debug(f"  Hot side: {hot_side}, Cold side: {cold_side}")
    if thermo is None:
        raise ValueError("ThermodynamicPackage (thermo) is required")
    # Calculate stream properties (for accurate duty and U)
    hot_props = _calculate_stream_properties(hot_inlet, thermo)
    cold_props = _calculate_stream_properties(cold_inlet, thermo)
    if mode == "design":
        result = _design_mode_calculation(
            hot_inlet=hot_inlet,
            cold_inlet=cold_inlet,
            hot_props=hot_props,
            cold_props=cold_props,
            config=config,
            thermo=thermo,
        )
    elif mode == "rating":
        result = _rating_mode_calculation(
            hot_inlet=hot_inlet,
            cold_inlet=cold_inlet,
            hot_props=hot_props,
            cold_props=cold_props,
            config=config,
            thermo=thermo,
        )
    else:
        raise ValueError(f"Invalid mode '{mode}', must be 'design' or 'rating'")
    # Build outlet streams
    hot_outlet = _build_outlet_stream(
        inlet=hot_inlet,
        outlet_temperature_C=result["hot_outlet_temperature_C"],
        pressure_drop_bar=config["pressure_drop_hot_bar"],
        thermo=thermo,
        name_suffix="_hot_out",
    )
    cold_outlet = _build_outlet_stream(
        inlet=cold_inlet,
        outlet_temperature_C=result["cold_outlet_temperature_C"],
        pressure_drop_bar=config["pressure_drop_cold_bar"],
        thermo=thermo,
        name_suffix="_cold_out",
    )
    # Build summary
    summary = {
        "exchanger_name": config["name"],
        "exchanger_type": config["type"],
        "service": config["service"],
        "mode": mode,
        "hot_inlet_temperature_C": hot_inlet.temperature_C,
        "hot_outlet_temperature_C": result["hot_outlet_temperature_C"],
        "cold_inlet_temperature_C": cold_inlet.temperature_C,
        "cold_outlet_temperature_C": result["cold_outlet_temperature_C"],
        "duty_kW": result["duty_kW"],
        "LMTD_C": result["LMTD_C"],
        "correction_factor_Ft": result["correction_factor_Ft"],
        "effective_temperature_difference_C": result["effective_temperature_difference_C"],
        "overall_heat_transfer_coefficient_W_m2K": result["overall_U_W_m2K"],
        "required_area_m2": result["required_area_m2"],
        "actual_area_m2": result.get("actual_area_m2", result["required_area_m2"]),
        "area_margin_percent": result.get("area_margin_percent", 0.0),
        "number_of_tubes": result["number_of_tubes"],
        "tube_length_m": config["tube_length_m"],
        "tube_outer_diameter_mm": config["tube_outer_diameter_mm"],
        "tube_inner_diameter_mm": result["tube_inner_diameter_mm"],
        "tube_passes": config["number_of_tube_passes"],
        "shell_diameter_m": result["shell_diameter_m"],
        "shell_passes": config["number_of_shell_passes"],
        "baffle_spacing_m": result["baffle_spacing_m"],
        "pressure_drop_hot_bar": config["pressure_drop_hot_bar"],
        "pressure_drop_cold_bar": config["pressure_drop_cold_bar"],
        "hot_side": hot_side,
        "cold_side": cold_side,
        "hot_side_velocity_m_s": result.get("hot_side_velocity_m_s", 0.0),
        "cold_side_velocity_m_s": result.get("cold_side_velocity_m_s", 0.0),
        "effectiveness": result.get("effectiveness", 0.0),
        "NTU": result.get("NTU", 0.0),
    }
    logger.info(
        f"Heat exchanger complete: Q={result['duty_kW']:.1f} kW, "
        f"A={result['required_area_m2']:.1f} m²"
    )
    return hot_outlet, cold_outlet, summary

def run_fehe_with_startup_control(
    hot_inlet: Any,
    cold_inlet: Any,
    thermo: Any,
    exchanger_config: Optional[dict] = None,
    startup_target_C: float = 100.0,
    startup_threshold_C: float = 80.0,
    mode: str = "design",
) -> Tuple[Any, Any, dict]:
    if exchanger_config is None:
        exchanger_config = {}
    # Ensure service is FEHE
    base_config = {
        "service": "FEHE",
    }
    base_config.update(exchanger_config)
    T_hot_in = hot_inlet.temperature_C
    T_cold_in = cold_inlet.temperature_C
    both_cold = (T_hot_in < startup_threshold_C) and (T_cold_in < startup_threshold_C)
    # ─────────────────────────────────────────────────────────
    # STARTUP MODE: both circuits cold → force to target
    # ─────────────────────────────────────────────────────────
    if both_cold:
        hx_config = dict(base_config)
        hx_config["hot_outlet_temperature_C"] = startup_target_C
        hx_config["cold_outlet_temperature_C"] = startup_target_C
        logger.info(
            "FEHE startup mode ACTIVE: "
            f"T_hot_in={T_hot_in:.1f}°C, T_cold_in={T_cold_in:.1f}°C < "
            f"{startup_threshold_C:.1f}°C → "
            f"T_hot_out=T_cold_out={startup_target_C:.1f}°C"
        )
        # Use design mode internally to compute duty/area for this target
        hot_out, cold_out, summary = run_heat_exchanger(
            hot_inlet=hot_inlet,
            cold_inlet=cold_inlet,
            exchanger_config=hx_config,
            thermo=thermo,
            mode="design",
        )
        summary["startup_mode_active"] = True
        summary["startup_target_temperature_C"] = startup_target_C
        summary["startup_threshold_temperature_C"] = startup_threshold_C
        return hot_out, cold_out, summary
    # ─────────────────────────────────────────────────────────
    # NORMAL OPERATION: use regular FEHE calculation
    # ─────────────────────────────────────────────────────────
    hx_config = dict(base_config)
    logger.info(
        "FEHE normal mode: "
        f"T_hot_in={T_hot_in:.1f}°C, T_cold_in={T_cold_in:.1f}°C, "
        f"threshold={startup_threshold_C:.1f}°C"
    )
    hot_out, cold_out, summary = run_heat_exchanger(
        hot_inlet=hot_inlet,
        cold_inlet=cold_inlet,
        exchanger_config=hx_config,
        thermo=thermo,
        mode=mode,
    )
    summary["startup_mode_active"] = False
    summary["startup_target_temperature_C"] = startup_target_C
    summary["startup_threshold_temperature_C"] = startup_threshold_C
    return hot_out, cold_out, summary

# ============================================================================
# VALIDATION
# ============================================================================
def _validate_hx_inputs(
    hot_inlet: Any,
    cold_inlet: Any,
    exchanger_config: dict,
    mode: str,
) -> None:
    """Validate heat exchanger inputs"""
    if hot_inlet is None or cold_inlet is None:
        raise ValueError("Inlet streams cannot be None")
    if hot_inlet.flowrate_kmol_h <= 0 or cold_inlet.flowrate_kmol_h <= 0:
        raise ValueError("Flowrates must be positive")
    if mode not in ["design", "rating"]:
        raise ValueError(f"Mode must be 'design' or 'rating', got '{mode}'")
    # In FEHE startup or electric heating cases we may temporarily have
    # hot_inlet.T <= cold_inlet.T, so do not enforce for all services.
    service = exchanger_config.get("service", "process_cooling")
    if service not in ["FEHE", "process_heating"]:
        # Standard constraint for coolers: hot inlet > cold inlet
        if hot_inlet.temperature_C <= cold_inlet.temperature_C:
            raise ValueError(
                f"Hot inlet T ({hot_inlet.temperature_C:.1f}°C) must be > "
                f"cold inlet T ({cold_inlet.temperature_C:.1f}°C)"
            )
    if mode == "design":
        if (
            "hot_outlet_temperature_C" not in exchanger_config
            and "cold_outlet_temperature_C" not in exchanger_config
        ):
            logger.warning(
                "Design mode called without explicit outlet T; "
                "defaults or FEHE logic will determine targets."
            )
    elif mode == "rating":
        if "area_m2" not in exchanger_config:
            raise ValueError("Rating mode requires area_m2 specification")

def _extract_hx_config(exchanger_config: dict) -> dict:
    """Extract and normalize heat exchanger configuration."""
    # Start from JSON-free defaults, then overlay user config
    config = dict(HX_DEFAULTS)
    config.update(exchanger_config)
    # Add detailed FEHE / old-model fields with safe defaults
    config_normalized = {
        "name": config.get("name", "HX"),
        "type": config.get("type", "shell_and_tube"),
        "service": config.get("service", "FEHE"),
        "hot_outlet_temperature_C": config.get("hot_outlet_temperature_C"),
        "cold_outlet_temperature_C": config.get("cold_outlet_temperature_C"),
        "min_approach_temp_C": config.get("min_approach_temp_C", 10.0),
        "area_m2": config.get("area_m2"),
        "overall_U_W_m2K": config.get("overall_U_W_m2K", 300.0),
        "tube_outer_diameter_mm": config.get("tube_outer_diameter_mm", 19.05),
        "tube_thickness_mm": config.get("tube_thickness_mm", 2.11),
        "tube_length_m": config.get("tube_length_m", 6.0),
        "tube_material": config.get("tube_material", "carbon_steel"),
        "shell_material": config.get("shell_material", "carbon_steel"),
        "tube_layout": config.get("tube_layout", "triangular"),
        "tube_pitch_ratio": config.get("tube_pitch_ratio", 1.25),
        "number_of_tube_passes": config.get("number_of_tube_passes", 2),
        "number_of_shell_passes": config.get("number_of_shell_passes", 1),
        "baffle_cut_percent": config.get("baffle_cut_percent", 25.0),
        "baffle_spacing_ratio": config.get("baffle_spacing_ratio", 0.4),
        "pressure_drop_hot_bar": config.get("pressure_drop_hot_bar", 0.3),
        "pressure_drop_cold_bar": config.get("pressure_drop_cold_bar", 0.5),
        "fouling_factor_hot_m2K_W": config.get("fouling_factor_hot_m2K_W", 0.0002),
        "fouling_factor_cold_m2K_W": config.get("fouling_factor_cold_m2K_W", 0.0002),
        "hot_side_location": config.get("hot_side_location"),
    }
    # Sanity checks
    if (
        config_normalized["tube_length_m"] <= 0
        or config_normalized["tube_length_m"] > 20.0
    ):
        raise ValueError(
            f"Tube length {config_normalized['tube_length_m']}m outside "
            f"practical range [0, 20]m"
        )
    if config_normalized["number_of_tube_passes"] < 1:
        raise ValueError("Number of tube passes must be >= 1")
    if config_normalized["min_approach_temp_C"] < 5.0:
        logger.warning(
            f"Approach temperature {config_normalized['min_approach_temp_C']}°C < 5°C "
            f"(tight, expensive)"
        )
    return config_normalized

# ============================================================================
# FLUID ASSIGNMENT
# ============================================================================
def _assign_fluids_to_sides(
    hot_inlet: Any,
    cold_inlet: Any,
    config: dict,
) -> Tuple[str, str]:
    """
    Assign hot and cold fluids to shell vs tube side.
    Returns: (hot_side, cold_side) where each is "shell" or "tube"
    """
    if config["hot_side_location"] is not None:
        hot_side = config["hot_side_location"]
        cold_side = "tube" if hot_side == "shell" else "shell"
        return hot_side, cold_side
    # Heuristics for FEHE: hot effluent typically in shell
    if config["service"] == "FEHE":
        return "shell", "tube"
    # For coolers: process fluid in tubes, cooling water in shell
    if config["service"] == "cooler":
        return "tube", "shell"
    # Default: higher pressure in tubes
    if hot_inlet.pressure_bar > cold_inlet.pressure_bar:
        return "tube", "shell"
    else:
        return "shell", "tube"

# ============================================================================
# STREAM PROPERTIES
# ============================================================================
def _calculate_stream_properties(stream: Any, thermo: Any) -> dict:
    """Calculate stream properties for heat transfer calculations."""
    T_C = stream.temperature_C
    P_bar = stream.pressure_bar
    z = stream.composition
    phase = getattr(stream, "phase", "vapor")
    # Molecular weight
    MW = thermo.molecular_weight(z)
    # Mass flow rate
    m_dot_kg_s = stream.flowrate_kmol_h * MW / 3600.0
    # Density
    try:
        rho = thermo.density_TP(T_C, P_bar, z, phase)
    except Exception:
        rho = 10.0 if phase == "vapor" else 700.0
    # Viscosity
    try:
        mu = thermo.viscosity_TP(T_C, P_bar, z, phase)
    except Exception:
        mu = 1.5e-5 if phase == "vapor" else 3.0e-4
    # Heat capacity
    try:
        Cp_kJ_kmolK = thermo.heat_capacity_TP(T_C, P_bar, z, phase)
        Cp_J_kgK = Cp_kJ_kmolK * 1000.0 / MW
    except Exception:
        Cp_J_kgK = 2000.0 if phase == "vapor" else 2500.0
    # Thermal conductivity
    if phase == "vapor":
        k = 0.03
    else:
        k = 0.15
    # Prandtl number
    Pr = Cp_J_kgK * mu / k
    Pr = max(0.5, min(Pr, 100.0))
    props = {
        "mass_flow_kg_s": m_dot_kg_s,
        "density_kg_m3": rho,
        "viscosity_Pa_s": mu,
        "heat_capacity_J_kgK": Cp_J_kgK,
        "thermal_conductivity_W_mK": k,
        "prandtl_number": Pr,
        "molecular_weight_kg_kmol": MW,
        "phase": phase,
    }
    return props

# ============================================================================
# DESIGN MODE
# ============================================================================
def _design_mode_calculation(
    hot_inlet: Any,
    cold_inlet: Any,
    hot_props: dict,
    cold_props: dict,
    config: dict,
    thermo: Any,
) -> dict:
    """
    Design mode: given outlet temperature(s), calculate required area.
    """
    T_hot_in = hot_inlet.temperature_C
    T_cold_in = cold_inlet.temperature_C

    # ═══════════════════════════════════════════════════════════════════════
    # FEHE DEFAULT: Apply intelligent outlet temperature target if not specified
    # ═══════════════════════════════════════════════════════════════════════
    if config["service"] == "FEHE" and config["cold_outlet_temperature_C"] is None and config["hot_outlet_temperature_C"] is None:
        # Heat cold side to 80% temperature approach from hot inlet
        # Example: T_hot_in=150°C, T_cold_in=31°C → ΔT=119°C → Target=31+0.8*119=126°C
        delta_T = T_hot_in - T_cold_in
        T_target_cold = T_cold_in + 0.8 * delta_T
        config["cold_outlet_temperature_C"] = T_target_cold
        logger.info(
            f"FEHE default applied: cold_out={T_target_cold:.1f}°C "
            f"(80% approach from ΔT={delta_T:.1f}°C)"
        )
    # Determine outlet temperatures
    if config["cold_outlet_temperature_C"] is not None:
        T_cold_out = config["cold_outlet_temperature_C"]
        # Duty from cold side
        Q_kW = (
            cold_props["mass_flow_kg_s"]
            * cold_props["heat_capacity_J_kgK"]
            * (T_cold_out - T_cold_in)
            / 1000.0
        )
        # Hot outlet from energy balance
        T_hot_out = T_hot_in - Q_kW * 1000.0 / (
            hot_props["mass_flow_kg_s"] * hot_props["heat_capacity_J_kgK"]
        )
    elif config["hot_outlet_temperature_C"] is not None:
        T_hot_out = config["hot_outlet_temperature_C"]
        # Duty from hot side
        Q_kW = (
            hot_props["mass_flow_kg_s"]
            * hot_props["heat_capacity_J_kgK"]
            * (T_hot_in - T_hot_out)
            / 1000.0
        )
        # Cold outlet from energy balance
        T_cold_out = T_cold_in + Q_kW * 1000.0 / (
            cold_props["mass_flow_kg_s"] * cold_props["heat_capacity_J_kgK"]
        )
    else:
        raise ValueError(
            "Design mode requires at least one outlet temperature specification"
        )
    # Pinch / approach constraint (for FEHE accuracy)
    approach = T_hot_out - T_cold_in
    if approach < config["min_approach_temp_C"]:
        logger.warning(
            f"Approach temperature {approach:.1f}°C < minimum "
            f"{config['min_approach_temp_C']:.1f}°C. Adjusting hot outlet temperature."
        )
        T_hot_out = T_cold_in + config["min_approach_temp_C"]
        # Recompute duty and cold outlet
        Q_kW = (
            hot_props["mass_flow_kg_s"]
            * hot_props["heat_capacity_J_kgK"]
            * (T_hot_in - T_hot_out)
            / 1000.0
        )
        T_cold_out = T_cold_in + Q_kW * 1000.0 / (
            cold_props["mass_flow_kg_s"] * cold_props["heat_capacity_J_kgK"]
        )
    # LMTD
    dT1 = T_hot_in - T_cold_out
    dT2 = T_hot_out - T_cold_in
    if dT1 <= 0 or dT2 <= 0:
        raise RuntimeError("Temperature crossover in LMTD calculation")
    if abs(dT1 - dT2) > 0.1:
        LMTD = (dT1 - dT2) / math.log(dT1 / dT2)
    else:
        LMTD = 0.5 * (dT1 + dT2)
    # Correction factor (simplified)
    Ft = 0.95
    DTM = Ft * LMTD
    # Overall U (already includes fouling through config)
    U = config["overall_U_W_m2K"]
    # Required area
    A_required = Q_kW * 1000.0 / (U * DTM)
    # Number of tubes and shell diameter
    tube_OD = config["tube_outer_diameter_mm"]
    tube_length = config["tube_length_m"]
    tube_area_per_tube = math.pi * (tube_OD / 1000.0) * tube_length
    num_tubes = max(int(A_required / tube_area_per_tube) + 1, 1)
    # Shell diameter (simplified heuristic)
    shell_diameter_m = 0.03 * math.sqrt(num_tubes * tube_length) + 0.5
    # Tube ID from standards
    tube_BWG = 14  # default; config could include this later if needed
    if tube_OD in STANDARD_TUBES and tube_BWG in STANDARD_TUBES[tube_OD]:
        tube_ID_mm = STANDARD_TUBES[tube_OD][tube_BWG]["ID_mm"]
    else:
        tube_ID_mm = tube_OD * 0.8
    # Baffle spacing
    baffle_spacing_m = config["baffle_spacing_ratio"] * shell_diameter_m
    result = {
        "hot_outlet_temperature_C": T_hot_out,
        "cold_outlet_temperature_C": T_cold_out,
        "duty_kW": Q_kW,
        "LMTD_C": LMTD,
        "correction_factor_Ft": Ft,
        "effective_temperature_difference_C": DTM,
        "overall_U_W_m2K": U,
        "required_area_m2": A_required,
        "number_of_tubes": num_tubes,
        "tube_inner_diameter_mm": tube_ID_mm,
        "shell_diameter_m": shell_diameter_m,
        "baffle_spacing_m": baffle_spacing_m,
        # Optional performance metrics (left basic here)
        "hot_side_velocity_m_s": 0.0,
        "cold_side_velocity_m_s": 0.0,
        "effectiveness": 0.0,
        "NTU": 0.0,
    }
    return result

# ============================================================================
# RATING MODE (simplified)
# ============================================================================
def _rating_mode_calculation(
    hot_inlet: Any,
    cold_inlet: Any,
    hot_props: dict,
    cold_props: dict,
    config: dict,
    thermo: Any,
) -> dict:
    """
    Rating mode: given area, find outlet temperatures.
    Uses ε-NTU method in a simplified way.
    """
    T_hot_in = hot_inlet.temperature_C
    T_cold_in = cold_inlet.temperature_C
    A = config["area_m2"]
    U = config["overall_U_W_m2K"]
    C_hot = hot_props["mass_flow_kg_s"] * hot_props["heat_capacity_J_kgK"]
    C_cold = cold_props["mass_flow_kg_s"] * cold_props["heat_capacity_J_kgK"]
    C_min = min(C_hot, C_cold)
    C_max = max(C_hot, C_cold)
    C_ratio = C_min / C_max if C_max > 0 else 0.0
    NTU = U * A / C_min if C_min > 0 else 0.0
    # Effectiveness for counter-current (simplified)
    if C_ratio != 1.0:
        effectiveness = (1.0 - math.exp(-NTU * (1.0 - C_ratio))) / (
            1.0 - C_ratio * math.exp(-NTU * (1.0 - C_ratio))
        )
    else:
        effectiveness = NTU / (1.0 + NTU)
    Q_max = C_min * (T_hot_in - T_cold_in)
    Q_kW = effectiveness * Q_max / 1000.0
    T_hot_out = T_hot_in - Q_kW * 1000.0 / C_hot
    T_cold_out = T_cold_in + Q_kW * 1000.0 / C_cold
    # LMTD
    dT1 = T_hot_in - T_cold_out
    dT2 = T_hot_out - T_cold_in
    if dT1 <= 0 or dT2 <= 0:
        raise RuntimeError("Temperature crossover in rating mode")
    if abs(dT1 - dT2) > 0.1:
        LMTD = (dT1 - dT2) / math.log(dT1 / dT2)
    else:
        LMTD = 0.5 * (dT1 + dT2)
    Ft = 0.95
    DTM = Ft * LMTD
    result = {
        "hot_outlet_temperature_C": T_hot_out,
        "cold_outlet_temperature_C": T_cold_out,
        "duty_kW": Q_kW,
        "LMTD_C": LMTD,
        "correction_factor_Ft": Ft,
        "effective_temperature_difference_C": DTM,
        "overall_U_W_m2K": U,
        "required_area_m2": A,
        "number_of_tubes": max(
            1,
            int(
                A
                / (
                    math.pi
                    * (config["tube_outer_diameter_mm"] / 1000.0)
                    * config["tube_length_m"]
                )
            ),
        ),
        "tube_inner_diameter_mm": config["tube_outer_diameter_mm"] * 0.8,
        "shell_diameter_m": 0.03 * math.sqrt(max(1, int(A))) + 0.5,
        "baffle_spacing_m": config["baffle_spacing_ratio"]
        * (0.03 * math.sqrt(max(1, int(A))) + 0.5),
        "hot_side_velocity_m_s": 0.0,
        "cold_side_velocity_m_s": 0.0,
        "effectiveness": effectiveness,
        "NTU": NTU,
    }
    return result

# ============================================================================
# OUTLET STREAM BUILDER
# ============================================================================
def _build_outlet_stream(
    inlet: Any,
    outlet_temperature_C: float,
    pressure_drop_bar: float,
    thermo: Any,
    name_suffix: str,
):
    """Create an outlet Stream from an inlet with new T and P."""
    from simulation.streams import Stream
    outlet = Stream(
        name=f"{inlet.name}{name_suffix}",
        temperature_C=outlet_temperature_C,
        pressure_bar=inlet.pressure_bar - pressure_drop_bar,
        flowrate_kmol_h=inlet.flowrate_kmol_h,
        composition=dict(inlet.composition),
        thermo=thermo,
        phase=inlet.phase,
    )
    return outlet