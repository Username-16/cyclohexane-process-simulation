"""
optimization/simulation_adapter.py

Version 3.0
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Callable
from datetime import datetime
import copy

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))
logger = logging.getLogger(__name__)

# Import simulation modules
from simulation.thermodynamics import ThermodynamicPackage
from simulation.flowsheet import Flowsheet, create_flowsheet


# ════════════════════════════════════════════════════════════════════════════
# SAUDI ARABIA JUBAIL ECONOMICS CALCULATOR
# ════════════════════════════════════════════════════════════════════════════

def calculate_saudi_economics(equipment_summaries: dict, streams: dict, design_dict: dict = None) -> dict:
    """
    Calculate economics with ACCURATE Saudi Arabia Jubail costs (2026).

    Returns:
        dict: Complete economics breakdown in USD
    """

    capex_equipment = 0.0
    equipment_breakdown = {}

    # ─────────────────────────────────────────────────────────────
    # 1. REACTORS
    # ─────────────────────────────────────────────────────────────
    for name, summary in equipment_summaries.items():
        if name.startswith("R-"):
            if design_dict:
                volumes = [design_dict.get(f"stage_volume_{i+1}", 2.5) for i in range(6)]
                total_volume_m3 = sum(volumes) / 2
            else:
                total_volume_m3 = summary.get("volume_m3", 12.0)

            # Six-tenths rule: Base $1.2M for 10 m³
            reactor_cost = 1_200_000 * (total_volume_m3 / 10.0) ** 0.6
            reactor_cost *= 1.15  # High-pressure fittings
            reactor_cost += 5000 * 65  # Catalyst loading

            capex_equipment += reactor_cost
            equipment_breakdown[name] = reactor_cost

    # ─────────────────────────────────────────────────────────────
    # 2. HEAT EXCHANGERS
    # ─────────────────────────────────────────────────────────────
    for name, summary in equipment_summaries.items():
        if name.startswith("E-"):
            duty_kW = abs(summary.get("duty_kW", 100.0))
            area_m2 = max((duty_kW * 1000) / (850 * 40), 5.0)
            hx_cost = 15_000 + 500 * area_m2

            if duty_kW < 0 or "cool" in name.lower():
                hx_cost *= 1.3  # SS316 premium

            capex_equipment += hx_cost
            equipment_breakdown[name] = hx_cost

    # ─────────────────────────────────────────────────────────────
    # 3. COMPRESSORS
    # ─────────────────────────────────────────────────────────────
    for name, summary in equipment_summaries.items():
        if name.startswith("C-"):
            power_kW = summary.get("driver_power_kW", 100.0)
            comp_cost = 250_000 * (power_kW / 100.0) ** 0.85
            comp_cost *= 1.20  # Explosion-proof

            capex_equipment += comp_cost
            equipment_breakdown[name] = comp_cost

    # ─────────────────────────────────────────────────────────────
    # 4. PUMPS
    # ─────────────────────────────────────────────────────────────
    for name, summary in equipment_summaries.items():
        if name.startswith("P-"):
            power_kW = summary.get("motor_power_kW", 5.0)
            pump_cost = (30_000 + 3_000 * power_kW) * 1.10

            capex_equipment += pump_cost
            equipment_breakdown[name] = pump_cost

    # ─────────────────────────────────────────────────────────────
    # 5. SEPARATORS
    # ─────────────────────────────────────────────────────────────
    if "V-101" in equipment_summaries:
        capex_equipment += 180_000
        equipment_breakdown["V-101"] = 180_000

    if "T-101" in equipment_summaries:
        column_cost = 450_000 + 8_000 * 20 + 120_000
        capex_equipment += column_cost
        equipment_breakdown["T-101"] = column_cost

    if "M-101" in equipment_summaries:
        feed_stream = streams.get("S24_bottoms")
        feed_kmolh = feed_stream.flowrate_kmol_h if (feed_stream and hasattr(feed_stream, "flowrate_kmol_h")) else 100.0
        membrane_cost = 80_000 + 1_500 * (feed_kmolh * 1.5)
        capex_equipment += membrane_cost
        equipment_breakdown["M-101"] = membrane_cost

    # Misc equipment
    capex_equipment += 75_000
    equipment_breakdown["MISC"] = 75_000

    # ─────────────────────────────────────────────────────────────
    # INSTALLED CAPEX (Lang Factor = 4.5)
    # ─────────────────────────────────────────────────────────────
    total_installed_capex = capex_equipment * 4.5

    # ════════════════════════════════════════════════════════════════════════════
    # ANNUAL OPEX (Saudi Arabia Jubail rates, 2026)
    # ════════════════════════════════════════════════════════════════════════════

    hours_per_year = 8760 * 0.92

    # Electricity: SAR 0.20/kWh = $0.053/kWh
    electricity_rate = 0.053
    total_electricity_kW = 0
    for name, summary in equipment_summaries.items():
        if name.startswith("P-"):
            total_electricity_kW += summary.get("motor_power_kW", 0)
        elif name.startswith("C-"):
            total_electricity_kW += summary.get("driver_power_kW", 0)

    electricity_cost_annual = total_electricity_kW * hours_per_year * electricity_rate

    # Steam: $20/tonne
    total_heating_kW = 0
    for name, summary in equipment_summaries.items():
        if name.startswith("E-") and summary.get("duty_kW", 0) > 0:
            total_heating_kW += summary.get("duty_kW", 0)

    steam_tonnes_per_year = (total_heating_kW / 0.556 * hours_per_year) / 1000
    steam_cost_annual = steam_tonnes_per_year * 20.0

    # Cooling water: SAR 1.65/m³ = $0.44/m³
    total_cooling_kW = 0
    for name, summary in equipment_summaries.items():
        if name.startswith("E-") and summary.get("duty_kW", 0) < 0:
            total_cooling_kW += abs(summary.get("duty_kW", 0))

    cooling_m3_per_year = (total_cooling_kW / 0.0116 * hours_per_year / 1000) * 1.03
    cooling_cost_annual = cooling_m3_per_year * 0.44

    # Maintenance: 2.5% of CAPEX
    maintenance_cost_annual = total_installed_capex * 0.025

    # Labor: 20 operators @ $35k/year
    labor_cost_annual = 20 * 35_000

    # Catalyst: Annualized
    catalyst_cost_annual = (10_000 * 65) / 4

    opex_annual = (
        electricity_cost_annual +
        steam_cost_annual +
        cooling_cost_annual +
        maintenance_cost_annual +
        labor_cost_annual +
        catalyst_cost_annual
    )

    return {
        "capex_USD": total_installed_capex,
        "capex_equipment_USD": capex_equipment,
        "opex_annual_USD": opex_annual,
        "electricity_cost_USD_yr": electricity_cost_annual,
        "steam_cost_USD_yr": steam_cost_annual,
        "cooling_cost_USD_yr": cooling_cost_annual,
        "maintenance_cost_USD_yr": maintenance_cost_annual,
        "labor_cost_USD_yr": labor_cost_annual,
        "catalyst_cost_USD_yr": catalyst_cost_annual,
    }


# ════════════════════════════════════════════════════════════════════════════
# FLOWSHEET EVALUATOR FACTORY
# ════════════════════════════════════════════════════════════════════════════

def create_flowsheet_evaluator(
    baseline_config_path: str = None,
    reports_dir: str = None
) -> Callable[[Dict[str, float]], Dict[str, Any]]:
    """
    Create a flowsheet evaluator function for optimization.

    Returns:
        Evaluator function that takes design_dict and returns results
    """

    # Setup paths
    if baseline_config_path is None:
        config_dir = Path(__file__).parent.parent / "config"
        baseline_config_path = config_dir / "process_parameters.json"

    if reports_dir is None:
        reports_dir = Path(__file__).parent.parent / "simulation" / "reports"

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Initialize thermodynamics ONCE
    thermo = ThermodynamicPackage()

    # Load baseline config
    with open(baseline_config_path, 'r') as f:
        baseline_config = json.load(f)

    # Design variable mapping
    # Design variable mapping
    DESIGN_VAR_MAPPING = {
        'stage_volume_1': ['reactor_configuration', 'stage_volumes', 0],
        'stage_volume_2': ['reactor_configuration', 'stage_volumes', 1],
        'stage_volume_3': ['reactor_configuration', 'stage_volumes', 2],
        'stage_volume_4': ['reactor_configuration', 'stage_volumes', 3],
        'stage_volume_5': ['reactor_configuration', 'stage_volumes', 4],
        'stage_volume_6': ['reactor_configuration', 'stage_volumes', 5],
        'h2_recycle_fraction': ['design_parameters', 'h2_recycle_fraction'],
        'liquid_recycle_fraction': ['design_parameters', 'liquid_recycle_fraction'],
        'distillate_recycle_fraction': ['design_parameters', 'distillate_recycle_fraction'],
        'h2_benzene_feed_ratio': ['feed_specifications', 'hydrogen_feed', 'h2_benzene_ratio'],
        'distillate_LK_mole_frac': ['separation_configuration', 'distillation_column', 'distillate_LK_mole_frac'],
        'distillate_HK_mole_frac': ['separation_configuration', 'distillation_column', 'distillate_HK_mole_frac'],
        'bottoms_LK_mole_frac': ['separation_configuration', 'distillation_column', 'bottoms_LK_mole_frac'],
        'bottoms_HK_mole_frac': ['separation_configuration', 'distillation_column', 'bottoms_HK_mole_frac'],
        'reflux_ratio_factor': ['separation_configuration', 'distillation_column', 'reflux_ratio_factor'],
    }

    def evaluator(design_dict: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate a single design point."""
        global total_dist_duty_kW, theoretical_stages, actual_stages
        try:
            # Create fresh config
            custom_config = copy.deepcopy(baseline_config)

            # Fix stage_volumes
            if 'reactor_configuration' in custom_config:
                if 'stage_volumes' not in custom_config['reactor_configuration']:
                    custom_config['reactor_configuration']['stage_volumes'] = [2.0] * 6
                elif not isinstance(custom_config['reactor_configuration']['stage_volumes'], list):
                    val = custom_config['reactor_configuration']['stage_volumes']
                    custom_config['reactor_configuration']['stage_volumes'] = [val] * 6

            # Apply design variables
            for var_name, var_value in design_dict.items():
                if var_name in DESIGN_VAR_MAPPING:
                    path = DESIGN_VAR_MAPPING[var_name]
                    target = custom_config
                    for key in path[:-1]:
                        target = target[key]
                    target[path[-1]] = var_value

            # Run simulation
            flowsheet = create_flowsheet(thermo=thermo, process_parameters=custom_config)
            results = flowsheet.run_simulation()

            converged = results.get("converged", False)

            if converged:
                streams = results.get("streams", {})
                equipment = results.get("equipment_summaries", {})
                dist_summary = equipment.get("T-101", {})
                actual_stages = dist_summary.get("number_of_actual_trays", 0)
                theoretical_stages = dist_summary.get("theoretical_stages", 0)
                reboiler_duty_kW = dist_summary.get("reboiler_duty_kW", 0)
                condenser_duty_kW = abs(dist_summary.get("condenser_duty_kW", 0))
                total_dist_duty_kW = reboiler_duty_kW + condenser_duty_kW
                # Extract KPIs
                product_stream = streams.get("S25")
                if product_stream:
                    cyclohexane_production = product_stream.flowrate_kmol_h
                    cyclohexane_purity = product_stream.composition.get("cyclohexane", 0) * 100
                else:
                    cyclohexane_production = 0
                    cyclohexane_purity = 0

                benzene_feed = streams.get("S1")
                benzene_feed_flow = benzene_feed.flowrate_kmol_h if benzene_feed else 0

                if benzene_feed_flow > 0:
                    benzene_conversion = (benzene_feed_flow - cyclohexane_production * 0.001) / benzene_feed_flow
                else:
                    benzene_conversion = 0

                total_reactor_volume = sum(design_dict.get(f"stage_volume_{i+1}", 0) for i in range(6))

                # ═══════════════════════════════════════════════════════════════
                # CALCULATE SAUDI ECONOMICS
                # ═══════════════════════════════════════════════════════════════
                economics = calculate_saudi_economics(equipment, streams, design_dict)

                # Extract utilities from KPIs
                kpis = results.get("KPIs", {})
                electricity_kW = kpis.get("compressor_power_kW", 0) + kpis.get("pump_power_kW", 0)
                heating_kW = kpis.get("heating_duty_kW", 0)
                cooling_kW = kpis.get("cooling_duty_kW", 0)

            else:
                cyclohexane_production = 0
                cyclohexane_purity = 0
                benzene_conversion = 0
                total_reactor_volume = sum(design_dict.get(f"stage_volume_{i+1}", 0) for i in range(6))
                economics = {"capex_USD": 0, "opex_annual_USD": 0}
                electricity_kW = 0
                heating_kW = 0
                cooling_kW = 0

            # Return results with economics
            return {
                "converged": converged,
                "products": {
                    "cyclohexane_kmol_h": cyclohexane_production,
                    "purity_percent": cyclohexane_purity,
                    "benzene_conversion": benzene_conversion,
                },
                "utilities": {
                    "electricity_kW": electricity_kW,
                    "heating_kW": heating_kW,
                    "cooling_kW": cooling_kW,
                },
                "equipment": {
                    "total_reactor_volume_m3": total_reactor_volume,
                    "distillation_actual_stages": actual_stages,
                    "distillation_theoretical_stages": theoretical_stages,
                    "distillation_total_duty_kW": total_dist_duty_kW,
                },
                "economics": economics,  # ← ADDED!
                "raw_results": results,
                "KPIs": results.get("KPIs", {}),
            }

        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return {
                "converged": False,
                "products": {"cyclohexane_kmol_h": 0, "purity_percent": 0, "benzene_conversion": 0},
                "utilities": {"electricity_kW": 0, "heating_kW": 0, "cooling_kW": 0},
                "equipment": {"total_reactor_volume_m3": sum(design_dict.get(f"stage_volume_{i+1}", 0) for i in range(6))},
                "economics": {"capex_USD": 0, "opex_annual_USD": 0},  # ← ADDED!
                "error": str(e),
            }

    return evaluator
