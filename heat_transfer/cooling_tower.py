"""
heat_transfer/cooling_tower.py - REFACTORED (NO JSON DEPENDENCIES)

Cooling water system with hardcoded defaults.

Author: King Saud University - Chemical Engineering Department
Date: 2026-01-15 (Refactored)
Version: 2.0.0 - JSON-Free
"""

from __future__ import annotations
import math
import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# HARDCODED DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════

COOLING_TOWER_DEFAULTS = {
    "type": "induced_draft",
    "cycles_of_concentration": 4.0,
    "L_G_ratio": 1.0,
    "tank_residence_time_min": 10.0,
}

AMBIENT_DEFAULTS = {
    "dry_bulb_temperature_C": 45.0,
    "wet_bulb_temperature_C": 28.0,
    "relative_humidity_percent": 20.0,
    "atmospheric_pressure_bar": 1.013,
}

CP_WATER = 4.18
LAMBDA_VAP = 2400.0
RHO_WATER = 1000.0
RHO_AIR = 1.2
GRAVITY = 9.81

# ═══════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def run_cooling_tower_system(
    total_process_duty_kW: float,
    cw_supply_temperature_C: float,
    cw_return_temperature_C: float,
    ambient_config: Optional[dict] = None,
    tower_config: Optional[dict] = None,
) -> dict:
    """
    Calculate cooling tower system performance with optional config override.

    Args:
        total_process_duty_kW: Total heat duty to remove (kW)
        cw_supply_temperature_C: Cold water supply temperature (°C)
        cw_return_temperature_C: Hot water return temperature (°C)
        ambient_config: OPTIONAL ambient conditions (uses defaults if None)
        tower_config: OPTIONAL tower design parameters (uses defaults if None)

    Returns:
        summary: Complete cooling system performance dict
    """
    # Apply defaults
    ambient = {**AMBIENT_DEFAULTS, **(ambient_config or {})}
    tower = {**COOLING_TOWER_DEFAULTS, **(tower_config or {})}

    logger.info("Calculating cooling tower system performance")

    # Validate
    if total_process_duty_kW <= 0:
        raise ValueError(f"Duty must be positive, got {total_process_duty_kW}")
    if cw_return_temperature_C <= cw_supply_temperature_C:
        raise ValueError("Return temperature must be > supply temperature")

    # Calculate cooling water flowrate
    cooling_range_C = cw_return_temperature_C - cw_supply_temperature_C
    m_cw_kg_s = total_process_duty_kW / (CP_WATER * cooling_range_C)
    m_cw_kg_h = m_cw_kg_s * 3600.0
    V_cw_m3_h = m_cw_kg_h / RHO_WATER

    logger.debug(f"Cooling water flow: {V_cw_m3_h:.1f} m³/h")

    # Wet bulb temperature
    T_wb = ambient["wet_bulb_temperature_C"]

    # Check feasibility
    if T_wb >= cw_supply_temperature_C:
        raise ValueError(f"Wet bulb {T_wb:.1f}°C >= supply {cw_supply_temperature_C:.1f}°C. Not feasible.")

    # Approach
    approach_C = cw_supply_temperature_C - T_wb
    if approach_C < 2.0:
        raise ValueError(f"Approach {approach_C:.1f}°C < 2°C (infeasible)")

    # Effectiveness
    effectiveness = cooling_range_C / (cw_return_temperature_C - T_wb)
    effectiveness_percent = effectiveness * 100.0

    logger.debug(f"Approach: {approach_C:.1f}°C, Effectiveness: {effectiveness_percent:.1f}%")

    # Evaporation
    Q_total_kJ_h = total_process_duty_kW * 3600.0
    evap_kg_h_theory = Q_total_kJ_h / LAMBDA_VAP
    evap_m3_h_empirical = 0.00085 * V_cw_m3_h * cooling_range_C
    evap_m3_h = (evap_kg_h_theory / RHO_WATER + evap_m3_h_empirical) / 2.0
    evap_kg_h = evap_m3_h * RHO_WATER

    # Drift
    drift_percent = 0.001
    drift_m3_h = V_cw_m3_h * drift_percent / 100.0
    drift_kg_h = drift_m3_h * RHO_WATER

    # Blowdown
    COC = tower["cycles_of_concentration"]
    if COC < 2.0 or COC > 10.0:
        raise ValueError(f"COC {COC:.1f} outside valid range [2, 10]")

    blowdown_m3_h = evap_m3_h / (COC - 1.0) - drift_m3_h
    blowdown_m3_h = max(blowdown_m3_h, 0.0)
    blowdown_kg_h = blowdown_m3_h * RHO_WATER

    # Makeup
    makeup_m3_h = evap_m3_h + blowdown_m3_h + drift_m3_h
    makeup_kg_h = makeup_m3_h * RHO_WATER
    makeup_percent = (makeup_m3_h / V_cw_m3_h) * 100.0

    logger.debug(f"Evap: {evap_m3_h:.2f} m³/h, Blowdown: {blowdown_m3_h:.2f} m³/h, Makeup: {makeup_m3_h:.2f} m³/h")

    # Size cooling tower (simplified)
    diameter_m = 0.1 * math.sqrt(V_cw_m3_h) + 1.0
    diameter_m = max(2.0, min(diameter_m, 10.0))
    height_m = 15.0

    # Air flowrate
    L_G_ratio = tower["L_G_ratio"]
    G_air_kg_s = m_cw_kg_s / L_G_ratio

    # Fan power (estimate)
    fan_power_kW = 0.01 * total_process_duty_kW

    # Pump power (estimate)
    pump_head_m = height_m + 10.0
    pump_power_kW = (RHO_WATER * GRAVITY * (V_cw_m3_h / 3600.0) * pump_head_m) / (0.75 * 1000.0)

    total_power_kW = fan_power_kW + pump_power_kW

    logger.debug(f"Power: Fan={fan_power_kW:.1f} kW, Pump={pump_power_kW:.1f} kW, Total={total_power_kW:.1f} kW")

    # Tanks
    res_time_min = tower["tank_residence_time_min"]
    tank_volume_m3 = V_cw_m3_h * res_time_min / 60.0
    tank_diameter_m = (4 * tank_volume_m3 / (math.pi * 2.0)) ** (1/3)
    tank_height_m = 2.0 * tank_diameter_m

    # Cost
    water_cost_usd_per_h = makeup_m3_h * 1.0

    summary = {
        "cooling_duty_kW": total_process_duty_kW,
        "cooling_water_flowrate_m3_h": V_cw_m3_h,
        "cooling_water_flowrate_kg_h": m_cw_kg_h,
        "supply_temperature_C": cw_supply_temperature_C,
        "return_temperature_C": cw_return_temperature_C,
        "cooling_range_C": cooling_range_C,
        "approach_C": approach_C,
        "effectiveness_percent": effectiveness_percent,
        "wet_bulb_temperature_C": T_wb,
        "evaporation_m3_h": evap_m3_h,
        "evaporation_kg_h": evap_kg_h,
        "blowdown_m3_h": blowdown_m3_h,
        "blowdown_kg_h": blowdown_kg_h,
        "drift_m3_h": drift_m3_h,
        "drift_kg_h": drift_kg_h,
        "makeup_water_m3_h": makeup_m3_h,
        "makeup_water_kg_h": makeup_kg_h,
        "makeup_percent_of_circulation": makeup_percent,
        "cycles_of_concentration": COC,
        "tower_diameter_m": diameter_m,
        "tower_height_m": height_m,
        "tower_type": tower["type"],
        "air_flowrate_kg_s": G_air_kg_s,
        "fan_power_kW": fan_power_kW,
        "pump_power_kW": pump_power_kW,
        "total_power_kW": total_power_kW,
        "cold_tank_volume_m3": tank_volume_m3,
        "cold_tank_diameter_m": tank_diameter_m,
        "cold_tank_height_m": tank_height_m,
        "hot_tank_volume_m3": tank_volume_m3,
        "hot_tank_diameter_m": tank_diameter_m,
        "hot_tank_height_m": tank_height_m,
        "water_cost_usd_per_h": water_cost_usd_per_h,
    }

    return summary
