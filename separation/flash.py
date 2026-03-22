"""
separation/flash.py
===============================================================================

Author: King Saud University - Chemical Engineering Department
Date: 2026-01-30
Version: 3.0.0
"""

from __future__ import annotations
import math
import logging
from typing import Any, Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# HARDCODED DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════
FLASH_DEFAULTS = {
    "type": "vertical_flash_drum",
    "operating_temperature_C": 60.0,
    "operating_pressure_bar": 25.0,
    "residence_time_min": 7.0,
    "length_to_diameter_ratio": 3.0,
    "vapor_velocity_fraction": 0.75,
    "demister_pad": True,
    "liquid_level_fraction": 0.5,
}

K_SB_WITH_DEMISTER = 0.107
K_SB_WITHOUT_DEMISTER = 0.05
MIN_DIAMETER_M = 0.6
MAX_DIAMETER_M = 5.0
MIN_LENGTH_M = 1.5
MAX_LENGTH_M = 20.0

# ═══════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ═══════════════════════════════════════════════════════════════════════════
def run_flash(
    feed: Any,
    flash_config: Optional[dict] = None,
    thermo: Any = None,
    equipment_config: Optional[dict] = None,
) -> Tuple[Any, Any, dict]:
    """
    Run isothermal flash separator.

    Args:
        feed: Feed stream
        flash_config: OPTIONAL configuration (uses defaults if None)
        thermo: ThermodynamicPackage
        equipment_config: OPTIONAL equipment params

    Returns:
        (vapor_stream, liquid_stream, summary_dict)
    """

    # Apply defaults
    config = {**FLASH_DEFAULTS, **(flash_config or {}), **(equipment_config or {})}
    logger.info(f"Running flash separator for feed: {feed.name}")

    # Validate
    if feed is None or feed.flowrate_kmol_h <= 0:
        raise ValueError("Invalid feed stream")
    if thermo is None:
        raise ValueError("ThermodynamicPackage required")

    # Operating conditions
    T_operating = config["operating_temperature_C"]
    P_operating = config["operating_pressure_bar"]

    # Check feed conditioning
    T_diff = abs(feed.temperature_C - T_operating)
    P_diff = abs(feed.pressure_bar - P_operating)

    if T_diff > 20.0:
        logger.warning(
            f"Feed temperature ({feed.temperature_C:.1f}°C) differs from "
            f"flash temperature ({T_operating:.1f}°C) by {T_diff:.1f}°C"
        )

    if P_diff > 2.0:
        logger.warning(
            f"Feed pressure ({feed.pressure_bar:.1f} bar) differs from "
            f"flash pressure ({P_operating:.1f} bar) by {P_diff:.1f} bar"
        )

    # ═══════════════════════════════════════════════════════════════
    # Flash calculation - USE CORRECT KEYS!
    # ═══════════════════════════════════════════════════════════════
    flash_result = thermo.flash_TP(T_operating, P_operating, feed.composition)

    vapor_fraction = flash_result["vapor_fraction"]

    # CRITICAL: Use "y" and "x" keys from thermodynamics.py
    vapor_composition = flash_result["y"]  # NOT "vapor_composition"!
    liquid_composition = flash_result["x"]  # NOT "liquid_composition"!

    # K-values (if available)
    K_values = flash_result.get("K_values", {})

    # VALIDATION: Ensure compositions are valid (NO FALLBACKS!)
    if not vapor_composition or sum(vapor_composition.values()) < 1e-9:
        raise ValueError(
            f"thermodynamics.py returned invalid vapor composition: {vapor_composition}. "
            f"Check flash_TP() method!"
        )

    if not liquid_composition or sum(liquid_composition.values()) < 1e-9:
        raise ValueError(
            f"thermodynamics.py returned invalid liquid composition: {liquid_composition}. "
            f"Check flash_TP() method!"
        )

    # Edge case warnings
    if vapor_fraction < 0.01:
        logger.warning(
            f"Flash produces mostly liquid (β={vapor_fraction:.4f}). "
            f"Consider adjusting T or P for better separation."
        )
    elif vapor_fraction > 0.99:
        logger.warning(
            f"Flash produces mostly vapor (β={vapor_fraction:.4f}). "
            f"Limited liquid product."
        )

    # Flowrates
    F_feed = feed.flowrate_kmol_h
    F_vapor = vapor_fraction * F_feed
    F_liquid = (1.0 - vapor_fraction) * F_feed

    logger.debug(f"Vapor fraction: {vapor_fraction:.3f}")
    logger.debug(f"Vapor: {F_vapor:.1f} kmol/h, Liquid: {F_liquid:.1f} kmol/h")

    # ═══════════════════════════════════════════════════════════════
    # Properties from thermo package
    # ═══════════════════════════════════════════════════════════════
    MW_vapor = thermo.molecular_weight(vapor_composition)
    MW_liquid = thermo.molecular_weight(liquid_composition)

    if MW_vapor < 1e-6:
        raise ValueError(f"MW_vapor = {MW_vapor} is zero! Check vapor composition: {vapor_composition}")
    if MW_liquid < 1e-6:
        raise ValueError(f"MW_liquid = {MW_liquid} is zero! Check liquid composition: {liquid_composition}")

    logger.debug(f"  MW vapor: {MW_vapor:.3f} kg/kmol")
    logger.debug(f"  MW liquid: {MW_liquid:.3f} kg/kmol")

    # Try to get densities from thermo package
    try:
        rho_vapor = thermo.density_TP(
            T_operating, P_operating, vapor_composition, phase="vapor"
        )
        logger.debug(f"  Using calculated vapor density: {rho_vapor:.1f} kg/m³")
    except Exception as e:
        # Fallback: estimate from ideal gas law
        T_K = T_operating + 273.15
        rho_vapor = (P_operating * MW_vapor) / (0.08314 * T_K)
        logger.debug(f"  Using ideal gas density: {rho_vapor:.1f} kg/m³ (fallback)")

    try:
        rho_liquid = thermo.density_TP(
            T_operating, P_operating, liquid_composition, phase="liquid"
        )
        logger.debug(f"  Using calculated liquid density: {rho_liquid:.1f} kg/m³")
    except Exception as e:
        # Fallback to typical hydrocarbon liquid density
        rho_liquid = 700.0
        logger.debug(f"  Using default liquid density: {rho_liquid:.1f} kg/m³ (fallback)")

    # ═══════════════════════════════════════════════════════════════
    # Drum sizing
    # ═══════════════════════════════════════════════════════════════
    sizing = _size_drum_iterative(
        F_vapor=F_vapor,
        F_liquid=F_liquid,
        MW_vapor=MW_vapor,
        MW_liquid=MW_liquid,
        rho_vapor=rho_vapor,
        rho_liquid=rho_liquid,
        config=config
    )

    logger.info(
        f"Flash sizing: D={sizing['diameter_m']:.2f}m, "
        f"L={sizing['length_m']:.2f}m, β={vapor_fraction:.2f}"
    )

    # ═══════════════════════════════════════════════════════════════
    # Build output streams
    # ═══════════════════════════════════════════════════════════════
    from simulation.streams import Stream

    vapor_stream = Stream(
        name=f"{feed.name}_vapor",
        temperature_C=T_operating,
        pressure_bar=P_operating,
        flowrate_kmol_h=F_vapor,
        composition=vapor_composition,
        thermo=thermo,
        phase="vapor",
    )

    liquid_stream = Stream(
        name=f"{feed.name}_liquid",
        temperature_C=T_operating,
        pressure_bar=P_operating,
        flowrate_kmol_h=F_liquid,
        composition=liquid_composition,
        thermo=thermo,
        phase="liquid",
    )

    # Build summary
    summary = {
        "flash_name": f"{feed.name}_flash",
        "feed_flowrate_kmol_h": F_feed,
        "vapor_flowrate_kmol_h": F_vapor,
        "liquid_flowrate_kmol_h": F_liquid,
        "vapor_fraction": vapor_fraction,
        "operating_temperature_C": T_operating,
        "operating_pressure_bar": P_operating,
        "vapor_composition": dict(vapor_composition),
        "liquid_composition": dict(liquid_composition),
        "K_values": dict(K_values),
        **sizing,
    }

    logger.info(f"✓ Flash completed successfully")
    return vapor_stream, liquid_stream, summary


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════
def _size_drum_iterative(
    F_vapor: float,
    F_liquid: float,
    MW_vapor: float,
    MW_liquid: float,
    rho_vapor: float,
    rho_liquid: float,
    config: dict,
) -> dict:
    """
    Size vertical drum with SAFETY CHECKS for edge cases.
    Handles: all-liquid, all-vapor, zero vapor density, normal two-phase.
    """

    # Extract config
    t_res_target = config.get("residence_time_min", 5.0)
    L_D_ratio = config.get("length_to_diameter_ratio", 3.0)
    f_flood = config.get("vapor_velocity_fraction", 0.75)
    has_demister = config.get("demister_pad", True)

    # Souders-Brown constant
    K_SB = K_SB_WITH_DEMISTER if has_demister else K_SB_WITHOUT_DEMISTER

    # Convert flows to mass (kg/s)
    m_vapor_kg_s = (F_vapor * MW_vapor) / 3600.0
    m_liquid_kg_s = (F_liquid * MW_liquid) / 3600.0

    # ═══════════════════════════════════════════════════════════════
    # SAFETY CHECK 1: All-liquid case (no vapor)
    # ═══════════════════════════════════════════════════════════════
    if F_vapor < 1e-6 or m_vapor_kg_s < 1e-9:
        logger.warning("Flash is all-liquid (vapor < 1e-6 kmol/h). Sizing as surge tank.")

        # Size as liquid surge vessel
        V_liquid_required = (m_liquid_kg_s * t_res_target * 60.0) / rho_liquid if rho_liquid > 0 else 1.0

        # Assume L/D = 3 for horizontal tank
        D = (4 * V_liquid_required / (math.pi * L_D_ratio)) ** (1 / 3)
        L = L_D_ratio * D

        # Apply minimum constraints
        D = max(D, MIN_DIAMETER_M)
        L = max(L, MIN_LENGTH_M)

        # Round dimensions
        D_rounded = math.ceil(D * 10) / 10
        L_rounded = math.ceil(L * 2) / 2

        return {
            "diameter_m": D_rounded,
            "length_m": L_rounded,
            "volume_m3": math.pi / 4 * D_rounded ** 2 * L_rounded,
            "liquid_level_m": 0.5 * L_rounded,
            "residence_time_actual_min": t_res_target,
            "vapor_velocity_m_s": 0.0,
            "flooding_velocity_m_s": 0.0,
            "design_basis": "all_liquid_surge_tank"
        }

    # ═══════════════════════════════════════════════════════════════
    # SAFETY CHECK 2: All-vapor case (no liquid)
    # ═══════════════════════════════════════════════════════════════
    if F_liquid < 1e-6 or m_liquid_kg_s < 1e-9:
        logger.warning("Flash is all-vapor (liquid < 1e-6 kmol/h). Sizing as knockout drum.")

        residence_time_sec = config.get("vapor_residence_time_sec", 10.0)
        Q_vapor_m3_s = m_vapor_kg_s / rho_vapor if rho_vapor > 1e-6 else 0.1
        V_total = Q_vapor_m3_s * residence_time_sec

        D = (4 * V_total / (math.pi * L_D_ratio)) ** (1 / 3)
        L = L_D_ratio * D

        # Apply minimum constraints
        D = max(D, MIN_DIAMETER_M)
        L = max(L, MIN_LENGTH_M)

        # Round dimensions
        D_rounded = math.ceil(D * 10) / 10
        L_rounded = math.ceil(L * 2) / 2

        return {
            "diameter_m": D_rounded,
            "length_m": L_rounded,
            "volume_m3": math.pi / 4 * D_rounded ** 2 * L_rounded,
            "liquid_level_m": 0.0,
            "residence_time_actual_min": residence_time_sec / 60.0,
            "vapor_velocity_m_s": 0.0,
            "flooding_velocity_m_s": 0.0,
            "design_basis": "all_vapor_knockout_drum"
        }

    # ═══════════════════════════════════════════════════════════════
    # SAFETY CHECK 3: Prevent division by zero
    # ═══════════════════════════════════════════════════════════════
    if rho_vapor < 1e-6:
        logger.warning(f"Vapor density very low ({rho_vapor:.6f} kg/m³). Using minimum safe value.")
        rho_vapor = 1e-6

    # ═══════════════════════════════════════════════════════════════
    # NORMAL TWO-PHASE FLASH: Calculate flooding velocity
    # ═══════════════════════════════════════════════════════════════
    if rho_liquid > rho_vapor:
        u_flood = K_SB * math.sqrt((rho_liquid - rho_vapor) / rho_vapor)
    else:
        logger.warning("Liquid density <= vapor density, using minimum flooding velocity")
        u_flood = 0.5  # m/s minimum

    # Design vapor velocity
    u_design = f_flood * u_flood
    logger.debug(f"  u_flood={u_flood:.3f} m/s, u_design={u_design:.3f} m/s")

    # ═══════════════════════════════════════════════════════════════
    # Initial diameter from vapor flow
    # ═══════════════════════════════════════════════════════════════
    Q_vapor_m3_s = m_vapor_kg_s / rho_vapor if rho_vapor > 0 else 0.0

    if u_design > 1e-6 and Q_vapor_m3_s > 0:
        A_cross = Q_vapor_m3_s / u_design
        D = math.sqrt(4 * A_cross / math.pi)
    else:
        D = MIN_DIAMETER_M

    D = max(D, MIN_DIAMETER_M)

    # ═══════════════════════════════════════════════════════════════
    # Calculate length from L/D ratio
    # ═══════════════════════════════════════════════════════════════
    demister_height = 0.4 if has_demister else 0.0
    L_drum = L_D_ratio * D
    L_total = L_drum + demister_height
    L_total = max(L_total, MIN_LENGTH_M)

    # ═══════════════════════════════════════════════════════════════
    # Check liquid level constraint (ITERATIVE)
    # ═══════════════════════════════════════════════════════════════
    V_liquid_required = (m_liquid_kg_s * t_res_target * 60.0) / rho_liquid if rho_liquid > 0 else 0.0
    h_liquid = V_liquid_required / (math.pi / 4 * D ** 2) if D > 1e-6 else 0.0
    max_liquid_level = 0.5 * L_total

    iteration = 0
    while h_liquid > max_liquid_level and iteration < 10:
        D = D * 1.1
        L_total = L_D_ratio * D + demister_height
        h_liquid = V_liquid_required / (math.pi / 4 * D ** 2)
        max_liquid_level = 0.5 * L_total
        iteration += 1
        logger.debug(f"  Iteration {iteration}: D={D:.2f}m, h_liquid={h_liquid:.2f}m")

    if iteration > 0:
        logger.debug(f"  Adjusted diameter to {D:.2f}m after {iteration} iterations")

    # ═══════════════════════════════════════════════════════════════
    # Apply maximum constraints and round
    # ═══════════════════════════════════════════════════════════════
    if D > MAX_DIAMETER_M:
        logger.warning(
            f"Calculated diameter {D:.2f}m exceeds maximum {MAX_DIAMETER_M}m. "
            f"Design may be uneconomical."
        )
        D = MAX_DIAMETER_M

    if L_total > MAX_LENGTH_M:
        logger.warning(f"Calculated length {L_total:.2f}m exceeds maximum {MAX_LENGTH_M}m")
        L_total = MAX_LENGTH_M

    D_rounded = math.ceil(D * 10) / 10
    L_rounded = math.ceil(L_total * 2) / 2

    # ═══════════════════════════════════════════════════════════════
    # Calculate actual performance
    # ═══════════════════════════════════════════════════════════════
    A_cross_actual = math.pi / 4 * D_rounded ** 2
    V_total = A_cross_actual * L_rounded

    h_liquid_actual = V_liquid_required / A_cross_actual if A_cross_actual > 0 else 0.0
    V_liquid_actual = A_cross_actual * h_liquid_actual

    t_res_actual = (V_liquid_actual * rho_liquid) / (
        m_liquid_kg_s * 60.0) if m_liquid_kg_s > 0 and rho_liquid > 0 else 0.0

    u_actual = Q_vapor_m3_s / A_cross_actual if A_cross_actual > 0 and Q_vapor_m3_s > 0 else 0.0

    # ═══════════════════════════════════════════════════════════════
    # Validation checks
    # ═══════════════════════════════════════════════════════════════
    if t_res_target > 0:
        res_time_error = abs(t_res_actual - t_res_target) / t_res_target
        if res_time_error > 0.2:
            logger.warning(
                f"Achieved residence time {t_res_actual:.1f} min differs from "
                f"target {t_res_target:.1f} min by {res_time_error * 100:.1f}%"
            )

    if u_actual > 0.9 * u_flood:
        logger.warning(
            f"Vapor velocity {u_actual:.3f} m/s is {u_actual / u_flood * 100:.1f}% "
            f"of flooding velocity. Risk of liquid entrainment."
        )

    if h_liquid_actual > 0.6 * L_rounded:
        logger.warning(
            f"Liquid level {h_liquid_actual:.2f}m is "
            f"{h_liquid_actual / L_rounded * 100:.1f}% of drum height"
        )

    # ═══════════════════════════════════════════════════════════════
    # Build sizing dictionary
    # ═══════════════════════════════════════════════════════════════
    sizing = {
        "diameter_m": D_rounded,
        "length_m": L_rounded,
        "volume_m3": V_total,
        "liquid_level_m": h_liquid_actual,
        "residence_time_actual_min": t_res_actual,
        "vapor_velocity_m_s": u_actual,
        "flooding_velocity_m_s": u_flood,
        "design_basis": "vapor_velocity",
    }

    logger.debug(
        f"  Final: D={D_rounded:.2f}m, L={L_rounded:.2f}m, "
        f"h_liq={h_liquid_actual:.2f}m, t_res={t_res_actual:.1f}min"
    )

    return sizing
