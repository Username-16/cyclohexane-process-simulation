"""
optimization/objective_functions.py

PURPOSE:
Define objective functions for process optimization including economic (NPV, ROI, payback),
performance (yield, conversion, selectivity), and operational (energy, utilities) metrics.
Support single and multi-objective optimization with automatic gradient calculation.

Date: 2026-01-02
Version: 2.0.0
"""

from __future__ import annotations

import math
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION DATACLASSES
# ============================================================================

@dataclass
class EconomicAssumptions:
    """Economic parameters for NPV/ROI calculations."""
    discount_rate: float = 0.10  # 10% per year
    plant_lifetime_years: int = 20
    construction_time_years: int = 2
    capacity_factor: float = 0.90  # 90% uptime
    salvage_value_fraction: float = 0.10  # 10% of CAPEX
    tax_rate: float = 0.25  # 25% corporate tax
    depreciation_years: int = 10  # Straight-line depreciation


@dataclass
class PriceAssumptions:
    """Market prices for products, feeds, and utilities."""
    # Products
    cyclohexane_USD_per_kg: float = 1.50

    # Feeds
    benzene_USD_per_kg: float = 1.20
    hydrogen_USD_per_kg: float = 2.50

    # Utilities
    electricity_USD_per_kWh: float = 0.08
    cooling_water_USD_per_m3: float = 0.20
    steam_LP_USD_per_kg: float = 0.015  # Low pressure
    steam_MP_USD_per_kg: float = 0.025  # Medium pressure
    steam_HP_USD_per_kg: float = 0.035  # High pressure

    # Other
    catalyst_USD_per_kg: float = 50.0
    co2_penalty_USD_per_tonne: float = 30.0  # Carbon price


# ============================================================================
# ABSTRACT BASE CLASS
# ============================================================================

class ObjectiveFunction(ABC):
    """
    Abstract base class for all objective functions.

    All concrete objectives must implement:
    - evaluate(): calculate objective value
    - get_name(): return objective name
    - get_direction(): return 'minimize' or 'maximize'
    """

    def __init__(self, config: dict):
        """
        Initialize objective function.

        Args:
            config: Configuration dict with objective-specific parameters
        """
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        # Subclasses can override for specific validation
        pass

    @abstractmethod
    def evaluate(
        self, 
        design_vector: np.ndarray, 
        flowsheet_results: dict
    ) -> float:
        """
        Evaluate objective function at given design point.

        Args:
            design_vector: Design variables (e.g., temperatures, pressures)
            flowsheet_results: Results from flowsheet simulation

        Returns:
            Objective function value (scalar)

        Raises:
            RuntimeError: If flowsheet did not converge
            ValueError: If inputs are invalid
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return objective function name."""
        pass

    @abstractmethod
    def get_direction(self) -> str:
        """Return optimization direction: 'minimize' or 'maximize'."""
        pass

    def get_units(self) -> str:
        """Return objective units (can be overridden)."""
        return "dimensionless"

    def gradient(
        self,
        design_vector: np.ndarray,
        flowsheet_results: dict,
        epsilon: float = 1e-6,
        flowsheet_evaluator: Optional[callable] = None
    ) -> np.ndarray:
        """
        Calculate gradient via finite differences.

        Args:
            design_vector: Design variables
            flowsheet_results: Results at current design
            epsilon: Perturbation size for finite differences
            flowsheet_evaluator: Function to re-evaluate flowsheet (if needed)

        Returns:
            Gradient vector (partial derivatives)

        Note:
            If flowsheet_evaluator is None, uses one-sided differences assuming
            flowsheet_results contains pre-computed perturbed results.
            If provided, performs full finite difference calculation.
        """
        n_vars = len(design_vector)
        gradient = np.zeros(n_vars)

        # Evaluate at current point
        f_0 = self.evaluate(design_vector, flowsheet_results)

        if flowsheet_evaluator is None:
            # Use provided epsilon or adaptive scaling
            for i in range(n_vars):
                # Adaptive epsilon based on variable magnitude
                eps = epsilon * max(abs(design_vector[i]), 1.0)
                gradient[i] = eps  # Placeholder - needs flowsheet evaluation

            logger.warning(
                "Gradient calculation incomplete: flowsheet_evaluator not provided. "
                "Use optimizer with gradient-free methods or provide evaluator."
            )
        else:
            # Full finite difference calculation
            for i in range(n_vars):
                # Perturb variable i
                design_perturbed = design_vector.copy()
                eps = epsilon * max(abs(design_vector[i]), 1.0)
                design_perturbed[i] += eps

                # Re-evaluate flowsheet
                try:
                    flowsheet_perturbed = flowsheet_evaluator(design_perturbed)
                    f_i = self.evaluate(design_perturbed, flowsheet_perturbed)

                    # Central difference (if possible)
                    gradient[i] = (f_i - f_0) / eps

                except Exception as e:
                    logger.warning(f"Gradient calculation failed for variable {i}: {e}")
                    gradient[i] = 0.0

        return gradient

    def _check_convergence(self, flowsheet_results: dict) -> None:
        """Check if flowsheet converged, raise error if not."""
        if not flowsheet_results.get("converged", False):
            raise RuntimeError(
                f"{self.get_name()}: Flowsheet simulation did not converge. "
                f"Cannot evaluate objective."
            )


# ============================================================================
# ECONOMIC OBJECTIVES
# ============================================================================

class NPVObjective(ObjectiveFunction):
    """
    Net Present Value (NPV) maximization.

    NPV = -CAPEX + Σ(t=1 to N) [CF_t / (1+r)^t] + Salvage/(1+r)^N

    Where:
    - CAPEX: Capital expenditure
    - CF_t: Annual cash flow (Revenue - OPEX - Tax)
    - r: Discount rate
    - N: Plant lifetime
    """

    def __init__(self, config: dict):
        super().__init__(config)

        self.econ = EconomicAssumptions(
            **config.get("economic_assumptions", {})
        )
        self.prices = PriceAssumptions(
            **config.get("price_assumptions", {})
        )

    def evaluate(
        self,
        design_vector: np.ndarray,
        flowsheet_results: dict
    ) -> float:
        """Calculate NPV."""

        self._check_convergence(flowsheet_results)

        # Extract from flowsheet
        econ = flowsheet_results.get("economics", {})

        # If flowsheet already calculated NPV, use it
        if "npv_USD" in econ:
            npv = econ["npv_USD"]
            logger.debug(f"NPV from flowsheet: ${npv:,.0f}")
            return npv

        # Otherwise calculate here
        capex = econ.get("capex_USD", 0)
        revenue_annual = econ.get("revenue_annual_USD", 0)
        opex_annual = econ.get("opex_annual_USD", 0)

        if capex <= 0:
            logger.warning("CAPEX is zero or negative, NPV calculation may be invalid")

        # Annual cash flow (simplified, no tax)
        cf_annual = revenue_annual - opex_annual

        # Salvage value
        salvage = capex * self.econ.salvage_value_fraction

        # NPV calculation
        r = self.econ.discount_rate
        n = self.econ.plant_lifetime_years

        # Present value of annuity: CF × [(1 - (1+r)^-n) / r]
        if r > 0:
            pv_annuity = cf_annual * (1 - (1 + r) ** (-n)) / r
        else:
            pv_annuity = cf_annual * n

        # Present value of salvage
        pv_salvage = salvage / ((1 + r) ** n)

        # Total NPV
        npv = -capex + pv_annuity + pv_salvage

        logger.debug(
            f"NPV calculation: CAPEX=${capex:,.0f}, "
            f"CF_annual=${cf_annual:,.0f}, NPV=${npv:,.0f}"
        )

        return npv

    def get_name(self) -> str:
        return "NPV"

    def get_direction(self) -> str:
        return "maximize"

    def get_units(self) -> str:
        return "USD"


class ROIObjective(ObjectiveFunction):
    """
    Return on Investment (ROI) maximization.

    ROI = (Average Annual Profit / CAPEX) × 100%
    """

    def evaluate(
        self,
        design_vector: np.ndarray,
        flowsheet_results: dict
    ) -> float:
        """Calculate ROI."""

        self._check_convergence(flowsheet_results)

        econ = flowsheet_results.get("economics", {})

        if "roi_percent" in econ:
            return econ["roi_percent"]

        # Calculate
        capex = econ.get("capex_USD", 1)  # Avoid division by zero
        revenue = econ.get("revenue_annual_USD", 0)
        opex = econ.get("opex_annual_USD", 0)

        profit = revenue - opex
        roi_percent = (profit / capex) * 100.0

        logger.debug(f"ROI: {roi_percent:.1f}%")

        return roi_percent

    def get_name(self) -> str:
        return "ROI"

    def get_direction(self) -> str:
        return "maximize"

    def get_units(self) -> str:
        return "percent"


class CAPEXObjective(ObjectiveFunction):
    """Capital expenditure (CAPEX) minimization."""

    # In objective_functions.py — find CAPEXObjective.evaluate() and fix the key:
    def evaluate(self, design_vector, flowsheet_results):
        economics = flowsheet_results.get("economics", {})

        # FIX: use capex_USD (confirmed key from debug_outputs.py)
        capex = economics.get("capex_USD",
                              economics.get("capex_MM", 0) * 1e6)  # fallback if old key exists

        return capex

    def get_name(self) -> str:
        return "CAPEX"

    def get_direction(self) -> str:
        return "minimize"

    def get_units(self) -> str:
        return "USD"


class OPEXObjective(ObjectiveFunction):
    """Operating expenditure (OPEX) minimization."""

    def evaluate(
        self,
        design_vector: np.ndarray,
        flowsheet_results: dict
    ) -> float:
        """Extract OPEX from flowsheet results."""

        self._check_convergence(flowsheet_results)

        opex = flowsheet_results.get("economics", {}).get("opex_annual_USD", 0)

        return opex

    def get_name(self) -> str:
        return "OPEX"

    def get_direction(self) -> str:
        return "minimize"

    def get_units(self) -> str:
        return "USD/year"


class PaybackObjective(ObjectiveFunction):
    """Payback period minimization."""

    def evaluate(
        self,
        design_vector: np.ndarray,
        flowsheet_results: dict
    ) -> float:
        """Calculate payback period."""

        self._check_convergence(flowsheet_results)

        econ = flowsheet_results.get("economics", {})

        if "payback_years" in econ:
            return econ["payback_years"]

        capex = econ.get("capex_USD", 1)
        revenue = econ.get("revenue_annual_USD", 0)
        opex = econ.get("opex_annual_USD", 0)

        annual_cf = revenue - opex

        if annual_cf <= 0:
            logger.warning("Negative cash flow - payback is infinite")
            return 999.0  # Very long payback

        payback = capex / annual_cf

        return payback

    def get_name(self) -> str:
        return "Payback"

    def get_direction(self) -> str:
        return "minimize"

    def get_units(self) -> str:
        return "years"



# ============================================================================
# Distillation-specific objective functions for optimization.
# ============================================================================

class DistillationStagesObjective(ObjectiveFunction):
    """Minimize number of distillation stages."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.weight = config.get("weight", 1.0) if config else 1.0
        self.target_stages = config.get("target_stages", 80) if config else 80

    def evaluate(self, design_vector: np.ndarray, flowsheet_results: Dict[str, Any]) -> float:
        """Minimize deviation from target stages."""
        if not flowsheet_results.get("converged", False):
            return 1e6

        actual_stages = flowsheet_results.get("equipment", {}).get("distillation_actual_stages", 999)

        # Penalize deviation from target
        deviation = abs(actual_stages - self.target_stages)

        # Add penalty for exceeding target
        if actual_stages > self.target_stages:
            deviation *= 2.0  # Double penalty for too many stages

        return self.weight * deviation

    def get_name(self) -> str:
        return f"DistillationStages(target={self.target_stages})"

    def get_direction(self) -> str:
        return "minimize"


class DistillationEnergyObjective(ObjectiveFunction):
    """Minimize distillation column heat duty."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.weight = config.get("weight", 1.0) if config else 1.0

    def evaluate(self, design_vector: np.ndarray, flowsheet_results: Dict[str, Any]) -> float:
        """Minimize total distillation duty (reboiler + condenser)."""
        if not flowsheet_results.get("converged", False):
            return 1e8

        duty_kW = flowsheet_results.get("equipment", {}).get("distillation_total_duty_kW", 1e6)

        return self.weight * duty_kW

    def get_name(self) -> str:
        return "DistillationEnergy"

    def get_direction(self) -> str:
        return "minimize"


class CombinedDistillationObjective(ObjectiveFunction):
    """Combined objective: minimize stages + energy."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.stage_weight = config.get("stage_weight", 100.0) if config else 100.0
        self.energy_weight = config.get("energy_weight", 0.01) if config else 0.01
        self.target_stages = config.get("target_stages", 80) if config else 80

    def evaluate(self, design_vector, flowsheet_results):
        if not flowsheet_results.get("converged", False):
            return 1e8

        # Stage count is fixed at 16 — skip that penalty, focus on duty
        duty_kW = flowsheet_results.get("equipment", {}).get(
            "distillation_total_duty_kW",
            flowsheet_results.get("KPIs", {}).get("total_energy_kW", 1e6)
        )

        # Penalize high reflux ratio (design_vector[14] = reflux_ratio_factor)
        reflux_factor = float(design_vector[14]) if len(design_vector) > 14 else 3.0
        reflux_penalty = max(0, reflux_factor - 3.0) * 1000.0  # penalize > 3×min

        return self.energy_weight * duty_kW + reflux_penalty

    def get_name(self) -> str:
        return f"DistillationCombined(stages={self.target_stages})"

    def get_direction(self) -> str:
        return "minimize"


# ============================================================================
# PERFORMANCE OBJECTIVES
# ============================================================================

class YieldObjective(ObjectiveFunction):
    """
    Product yield maximization.

    Yield = (Product flow / Feed flow) × 100%
    """

    def evaluate(
        self,
        design_vector: np.ndarray,
        flowsheet_results: dict
    ) -> float:
        """Calculate yield."""

        self._check_convergence(flowsheet_results)

        products = flowsheet_results.get("products", {})

        # For benzene → cyclohexane
        cyclohexane_kmol_h = products.get("cyclohexane_kmol_h", 0)
        benzene_feed_kmol_h = products.get("benzene_feed_kmol_h", 1)

        yield_percent = (cyclohexane_kmol_h / benzene_feed_kmol_h) * 100.0

        logger.debug(f"Yield: {yield_percent:.2f}%")

        return yield_percent

    def get_name(self) -> str:
        return "Yield"

    def get_direction(self) -> str:
        return "maximize"

    def get_units(self) -> str:
        return "percent"


class SelectivityObjective(ObjectiveFunction):
    """
    Product selectivity maximization.

    Selectivity = (Desired product / All products) × 100%
    """

    def evaluate(
        self,
        design_vector: np.ndarray,
        flowsheet_results: dict
    ) -> float:
        """Calculate selectivity."""

        self._check_convergence(flowsheet_results)

        products = flowsheet_results.get("products", {})

        cyclohexane_purity = products.get("cyclohexane_purity", 99.0)

        logger.debug(f"Selectivity (purity): {cyclohexane_purity:.2f}%")

        return cyclohexane_purity

    def get_name(self) -> str:
        return "Selectivity"

    def get_direction(self) -> str:
        return "maximize"

    def get_units(self) -> str:
        return "percent"


class ConversionObjective(ObjectiveFunction):
    """
    Reactant conversion maximization.

    Conversion = (Feed_in - Feed_out) / Feed_in × 100%
    """

    def evaluate(
        self,
        design_vector: np.ndarray,
        flowsheet_results: dict
    ) -> float:
        """Calculate conversion."""

        self._check_convergence(flowsheet_results)

        products = flowsheet_results.get("products", {})

        benzene_conversion = products.get("benzene_conversion", 0)

        if isinstance(benzene_conversion, float) and benzene_conversion < 1.0:
            # Already in fraction form
            conversion_percent = benzene_conversion * 100.0
        else:
            # Already in percent form
            conversion_percent = benzene_conversion

        logger.debug(f"Conversion: {conversion_percent:.2f}%")

        return conversion_percent

    def get_name(self) -> str:
        return "Conversion"

    def get_direction(self) -> str:
        return "maximize"

    def get_units(self) -> str:
        return "percent"


class ProductionRateObjective(ObjectiveFunction):
    """Production rate (throughput) maximization."""

    def evaluate(
        self,
        design_vector: np.ndarray,
        flowsheet_results: dict
    ) -> float:
        """Extract production rate."""

        self._check_convergence(flowsheet_results)

        products = flowsheet_results.get("products", {})
        production = products.get("cyclohexane_kmol_h", 0)

        logger.debug(f"Production: {production:.1f} kmol/h")

        return production

    def get_name(self) -> str:
        return "Production"

    def get_direction(self) -> str:
        return "maximize"

    def get_units(self) -> str:
        return "kmol/h"


# ============================================================================
# OPERATIONAL OBJECTIVES
# ============================================================================

class EnergyObjective(ObjectiveFunction):
    """
    Energy consumption minimization.

    Specific energy = Total energy / Product flow
    """

    def evaluate(self, design_vector, flowsheet_results):
        self._check_convergence(flowsheet_results)
        utilities = flowsheet_results.get("utilities", {})
        kpis = flowsheet_results.get("KPIs", {})
        products = flowsheet_results.get("products", {})

        heating = utilities.get("heating_kW", kpis.get("heating_duty_kW", 0))
        cooling = utilities.get("cooling_kW", kpis.get("cooling_duty_kW", 0))
        electricity = utilities.get("electricity_kW", kpis.get("compressor_power_kW", 0))

        total_energy_kW = heating + cooling + electricity
        production_kmol_h = products.get("cyclohexane_kmol_h", 1)
        specific_energy = total_energy_kW / max(production_kmol_h, 1)
        return specific_energy

    def get_name(self) -> str:
        return "Energy"

    def get_direction(self) -> str:
        return "minimize"

    def get_units(self) -> str:
        return "kWh/kmol"


class UtilityCostObjective(ObjectiveFunction):
    """Utility cost minimization."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.prices = PriceAssumptions(**config.get("price_assumptions", {}))

    def evaluate(
        self,
        design_vector: np.ndarray,
        flowsheet_results: dict
    ) -> float:
        """Calculate utility costs."""

        self._check_convergence(flowsheet_results)

        utilities = flowsheet_results.get("utilities", {})

        # Annual utility costs (assuming 8760 h/year × 0.9 capacity factor)
        hours_per_year = 8760 * 0.9

        heating_kW = utilities.get("heating_duty_kW", 0)
        cooling_kW = utilities.get("cooling_duty_kW", 0)
        electricity_kW = utilities.get("electricity_kW", 0)

        # Heating cost (convert kW to kg/h steam, assume 2000 kJ/kg latent heat)
        steam_kg_h = heating_kW * 3600 / 2000
        heating_cost_annual = steam_kg_h * self.prices.steam_MP_USD_per_kg * hours_per_year

        # Cooling cost (convert kW to m3/h, assume 20 kJ/kg × 1000 kg/m3)
        cw_m3_h = cooling_kW * 3600 / (20 * 1000)
        cooling_cost_annual = cw_m3_h * self.prices.cooling_water_USD_per_m3 * hours_per_year

        # Electricity cost
        elec_cost_annual = electricity_kW * self.prices.electricity_USD_per_kWh * hours_per_year

        total_utility_cost = heating_cost_annual + cooling_cost_annual + elec_cost_annual

        logger.debug(f"Utility cost: ${total_utility_cost:,.0f}/year")

        return total_utility_cost

    def get_name(self) -> str:
        return "Utility_Cost"

    def get_direction(self) -> str:
        return "minimize"

    def get_units(self) -> str:
        return "USD/year"


# ============================================================================
# ENVIRONMENTAL OBJECTIVES
# ============================================================================

class EmissionsObjective(ObjectiveFunction):
    """
    CO2 emissions minimization.

    Emissions = Utilities emissions + Process emissions
    """

    def __init__(self, config: dict):
        super().__init__(config)
        # CO2 intensity factors
        self.co2_electricity_kg_per_kWh = config.get("co2_electricity", 0.5)
        self.co2_steam_kg_per_GJ = config.get("co2_steam", 56.0)

    def evaluate(
        self,
        design_vector: np.ndarray,
        flowsheet_results: dict
    ) -> float:
        """Calculate CO2 emissions."""

        self._check_convergence(flowsheet_results)

        env = flowsheet_results.get("environmental", {})

        if "co2_emissions_kg_h" in env:
            return env["co2_emissions_kg_h"]

        # Calculate from utilities
        utilities = flowsheet_results.get("utilities", {})

        heating_kW = utilities.get("heating_duty_kW", 0)
        electricity_kW = utilities.get("electricity_kW", 0)

        # Emissions from electricity
        co2_elec_kg_h = electricity_kW * self.co2_electricity_kg_per_kWh

        # Emissions from steam (kW → GJ/h → kg CO2/h)
        heating_GJ_h = heating_kW * 3.6 / 1000
        co2_steam_kg_h = heating_GJ_h * self.co2_steam_kg_per_GJ

        total_co2_kg_h = co2_elec_kg_h + co2_steam_kg_h

        logger.debug(f"CO2 emissions: {total_co2_kg_h:.1f} kg/h")

        return total_co2_kg_h

    def get_name(self) -> str:
        return "CO2_Emissions"

    def get_direction(self) -> str:
        return "minimize"

    def get_units(self) -> str:
        return "kg_CO2/h"


class WaterConsumptionObjective(ObjectiveFunction):
    """Water consumption minimization."""

    def evaluate(
        self,
        design_vector: np.ndarray,
        flowsheet_results: dict
    ) -> float:
        """Calculate water consumption."""

        self._checkconvergence(flowsheet_results)

        env = flowsheet_results.get("environmental", {})
        utilities = flowsheet_results.get("utilities", {})

        if "water_consumption_m3_h" in env:
            return env["water_consumption_m3_h"]

        # Estimate from cooling water duty
        cooling_kW = utilities.get("cooling_duty_kW", 0)

        # Assume cooling tower with 2% evaporation loss
        cw_m3_h = cooling_kW * 3600 / (20 * 1000)  # 20 kJ/kg temperature rise
        evaporation_m3_h = cw_m3_h * 0.02

        logger.debug(f"Water consumption: {evaporation_m3_h:.2f} m3/h")

        return evaporation_m3_h

    def get_name(self) -> str:
        return "Water"

    def get_direction(self) -> str:
        return "minimize"

    def get_units(self) -> str:
        return "m3/h"


# ============================================================================
# SAFETY OBJECTIVES
# ============================================================================

class SafetyIndexObjective(ObjectiveFunction):
    """
    Safety index minimization (lower is safer).

    Combines pressure, temperature, flammability, toxicity hazards.
    """

    def evaluate(
        self,
        design_vector: np.ndarray,
        flowsheet_results: dict
    ) -> float:
        """Calculate safety hazard index."""

        self._check_convergence(flowsheet_results)

        safety = flowsheet_results.get("safety", {})

        if "hazard_index" in safety:
            return safety["hazard_index"]

        # Simple hazard index based on operating conditions
        equipment = flowsheet_results.get("equipment", {})

        hazard_index = 0.0

        # Pressure hazard
        for equip_name, equip_data in equipment.items():
            P_bar = equip_data.get("pressure_bar", 1.0)
            if P_bar > 10:
                hazard_index += (P_bar / 10) ** 2

        # Temperature hazard
        for equip_name, equip_data in equipment.items():
            T_C = equip_data.get("temperature_C", 25.0)
            if T_C > 100:
                hazard_index += (T_C / 100) ** 1.5

        # Flare load hazard
        flare_load = safety.get("flare_load_kg_h", 0)
        hazard_index += flare_load / 1000  # Normalize

        logger.debug(f"Safety hazard index: {hazard_index:.2f}")

        return hazard_index

    def get_name(self) -> str:
        return "Safety_Index"

    def get_direction(self) -> str:
        return "minimize"

    def get_units(self) -> str:
        return "dimensionless"


# ============================================================================
# FACTORY AND UTILITIES
# ============================================================================

def create_objective_function(
    objective_type: str,
    config: dict
) -> ObjectiveFunction:
    """
    Factory function to create objective from string type.

    Args:
        objective_type: Type of objective (e.g., "npv", "capex", "yield")
        config: Configuration dict

    Returns:
        Objective function instance

    Example:
        obj = create_objective_function("npv", config)
        value = obj.evaluate(design_vector, flowsheet_results)
    """

    objectives_map = {
        "npv": NPVObjective,
        "roi": ROIObjective,
        "capex": CAPEXObjective,
        "opex": OPEXObjective,
        "payback": PaybackObjective,
        "yield": YieldObjective,
        "selectivity": SelectivityObjective,
        "conversion": ConversionObjective,
        "production": ProductionRateObjective,
        "energy": EnergyObjective,
        "utility_cost": UtilityCostObjective,
        "emissions": EmissionsObjective,
        "water": WaterConsumptionObjective,
        "safety": SafetyIndexObjective,
    }

    obj_class = objectives_map.get(objective_type.lower())

    if obj_class is None:
        raise ValueError(
            f"Unknown objective type: {objective_type}. "
            f"Available: {list(objectives_map.keys())}"
        )

    return obj_class(config)


def evaluate_multiple_objectives(
    design_vector: np.ndarray,
    objectives: list[ObjectiveFunction],
    flowsheet_results: dict,
    weights: Optional[dict[str, float]] = None,
    normalize: bool = True,
    bounds: Optional[dict[str, tuple[float, float]]] = None
) -> dict:
    """
    Evaluate multiple objectives at design point.

    Args:
        design_vector: Design variables
        objectives: List of objective function instances
        flowsheet_results: Flowsheet simulation results
        weights: Optional weights for scalarization (sum to 1.0)
        normalize: Whether to normalize objectives to [0, 1]
        bounds: Optional bounds for normalization {name: (min, max)}

    Returns:
        Dict with objective values, normalized values, weighted sum
    """

    objective_values = {}
    directions = {}
    units = {}

    # Evaluate all objectives
    for obj in objectives:
        name = obj.get_name()
        try:
            value = obj.evaluate(design_vector, flowsheet_results)
            objective_values[name] = value
            directions[name] = obj.get_direction()
            units[name] = obj.get_units()
        except Exception as e:
            logger.error(f"Error evaluating {name}: {e}")
            objective_values[name] = float('nan')
            directions[name] = obj.get_direction()
            units[name] = obj.get_units()

    # Normalize if requested
    normalized_values = {}

    if normalize and bounds:
        for name, value in objective_values.items():
            if name in bounds:
                min_val, max_val = bounds[name]

                if directions[name] == "minimize":
                    # For minimization: 0 is best, 1 is worst
                    norm = (value - min_val) / (max_val - min_val)
                else:
                    # For maximization: 1 is best, 0 is worst
                    norm = (value - min_val) / (max_val - min_val)

                normalized_values[name] = np.clip(norm, 0, 1)
            else:
                normalized_values[name] = value  # No normalization
    else:
        normalized_values = objective_values.copy()

    # Calculate weighted sum if weights provided
    weighted_sum = None
    if weights:
        weighted_sum = 0.0
        for name, weight in weights.items():
            if name in normalized_values:
                # Use normalized values for weighted sum
                val = normalized_values[name]

                # Flip sign if maximizing (want high values)
                if directions[name] == "maximize":
                    val = 1.0 - val

                weighted_sum += weight * val

    return {
        "objective_values": objective_values,
        "normalized_values": normalized_values,
        "directions": directions,
        "units": units,
        "weighted_sum": weighted_sum,
    }


def calculate_weighted_objective(
    objective_values: dict[str, float],
    weights: dict[str, float],
    directions: dict[str, str]
) -> float:
    """
    Calculate weighted sum of objectives.

    Args:
        objective_values: Dict of {objective_name: value}
        weights: Dict of {objective_name: weight} (should sum to 1.0)
        directions: Dict of {objective_name: 'minimize' or 'maximize'}

    Returns:
        Weighted objective value (for minimization)
    """

    weighted = 0.0

    for name, weight in weights.items():
        if name in objective_values:
            value = objective_values[name]

            # If maximizing, flip sign
            if directions[name] == "maximize":
                value = -value

            weighted += weight * value

    return weighted


# ============================================================================
# SMOKE TEST
# ============================================================================

def _smoke_test_objectives():
    """
    Test objective function calculations.
    """
    print("="*70)
    print("OBJECTIVE FUNCTIONS SMOKE TEST")
    print("="*70)

    # Mock flowsheet results
    flowsheet_results = {
        "converged": True,
        "streams": {},
        "equipment": {
            "reactor": {"pressure_bar": 31.0, "temperature_C": 350.0},
            "flash": {"pressure_bar": 30.0, "temperature_C": 100.0}
        },
        "utilities": {
            "heating_duty_kW": 5000,
            "cooling_duty_kW": 8000,
            "electricity_kW": 500
        },
        "products": {
            "cyclohexane_kmol_h": 100.0,
            "cyclohexane_purity": 99.5,
            "benzene_recovery": 98.0,
            "benzene_feed_kmol_h": 102.0,
            "benzene_conversion": 0.98
        },
        "economics": {
            "capex_USD": 10_000_000,
            "opex_annual_USD": 5_000_000,
            "revenue_annual_USD": 12_000_000,
            "npv_USD": 25_000_000,
            "roi_percent": 70.0,
            "payback_years": 1.43
        },
        "environmental": {
            "co2_emissions_kg_h": 200.0
        },
        "safety": {
            "flare_load_kg_h": 1000.0
        }
    }

    design_vector = np.array([350, 340, 330, 320, 5.0, 31.0])

    # ========================================================================
    # Test 1: NPV objective
    # ========================================================================
    print("\n✓ Test 1: NPV objective...")

    npv_config = {
        "economic_assumptions": {
            "discount_rate": 0.10,
            "plant_lifetime_years": 20
        }
    }

    try:
        npv_obj = NPVObjective(npv_config)
        npv_value = npv_obj.evaluate(design_vector, flowsheet_results)

        print(f"  NPV: ${npv_value:,.0f}")
        print(f"  Direction: {npv_obj.get_direction()}")
        print(f"  Units: {npv_obj.get_units()}")

        assert npv_value > 0, "NPV should be positive"
        assert npv_obj.get_direction() == "maximize"

        print("  ✓ Test 1 passed")

    except Exception as e:
        print(f"  ✗ Test 1 failed: {e}")
        raise

    # ========================================================================
    # Test 2: Economic objectives
    # ========================================================================
    print("\n✓ Test 2: Economic objectives...")

    try:
        capex_obj = CAPEXObjective({})
        roi_obj = ROIObjective({})
        payback_obj = PaybackObjective({})

        capex = capex_obj.evaluate(design_vector, flowsheet_results)
        roi = roi_obj.evaluate(design_vector, flowsheet_results)
        payback = payback_obj.evaluate(design_vector, flowsheet_results)

        print(f"  CAPEX: ${capex:,.0f}")
        print(f"  ROI: {roi:.1f}%")
        print(f"  Payback: {payback:.2f} years")

        assert capex == 10_000_000
        assert roi == 70.0
        assert 1.0 < payback < 2.0

        print("  ✓ Test 2 passed")

    except Exception as e:
        print(f"  ✗ Test 2 failed: {e}")
        raise

    # ========================================================================
    # Test 3: Performance objectives
    # ========================================================================
    print("\n✓ Test 3: Performance objectives...")

    try:
        yield_obj = YieldObjective({})
        sel_obj = SelectivityObjective({})
        conv_obj = ConversionObjective({})

        yield_val = yield_obj.evaluate(design_vector, flowsheet_results)
        selectivity = sel_obj.evaluate(design_vector, flowsheet_results)
        conversion = conv_obj.evaluate(design_vector, flowsheet_results)

        print(f"  Yield: {yield_val:.2f}%")
        print(f"  Selectivity: {selectivity:.2f}%")
        print(f"  Conversion: {conversion:.2f}%")

        assert 90 <= yield_val <= 100
        assert 99 <= selectivity <= 100
        assert 90 <= conversion <= 100

        print("  ✓ Test 3 passed")

    except Exception as e:
        print(f"  ✗ Test 3 failed: {e}")
        raise

    # ========================================================================
    # Test 4: Operational objectives
    # ========================================================================
    print("\n✓ Test 4: Operational objectives...")

    try:
        energy_obj = EnergyObjective({})
        emissions_obj = EmissionsObjective({})

        energy = energy_obj.evaluate(design_vector, flowsheet_results)
        emissions = emissions_obj.evaluate(design_vector, flowsheet_results)

        print(f"  Energy: {energy:.1f} kWh/kmol")
        print(f"  Emissions: {emissions:.1f} kg CO2/h")

        assert energy > 0
        assert emissions > 0

        print("  ✓ Test 4 passed")

    except Exception as e:
        print(f"  ✗ Test 4 failed: {e}")
        raise

    # ========================================================================
    # Test 5: Multiple objectives
    # ========================================================================
    print("\n✓ Test 5: Multiple objectives evaluation...")

    try:
        objectives = [npv_obj, capex_obj, yield_obj, emissions_obj]

        weights = {
            "NPV": 0.5,
            "CAPEX": 0.3,
            "Yield": 0.1,
            "CO2_Emissions": 0.1
        }

        bounds = {
            "NPV": (0, 50_000_000),
            "CAPEX": (5_000_000, 20_000_000),
            "Yield": (90, 100),
            "CO2_Emissions": (100, 500)
        }

        multi_results = evaluate_multiple_objectives(
            design_vector, objectives, flowsheet_results,
            weights=weights, normalize=True, bounds=bounds
        )

        print("  Objective values:")
        for name, value in multi_results["objective_values"].items():
            units = multi_results["units"][name]
            print(f"    {name}: {value:.2e} {units}")

        print("  Normalized values:")
        for name, value in multi_results["normalized_values"].items():
            print(f"    {name}: {value:.4f}")

        print(f"  Weighted sum: {multi_results['weighted_sum']:.4f}")

        assert len(multi_results["objective_values"]) == 4
        assert multi_results["weighted_sum"] is not None

        print("  ✓ Test 5 passed")

    except Exception as e:
        print(f"  ✗ Test 5 failed: {e}")
        raise

    # ========================================================================
    # Test 6: Factory function
    # ========================================================================
    print("\n✓ Test 6: Objective factory...")

    try:
        obj1 = create_objective_function("npv", npv_config)
        obj2 = create_objective_function("capex", {})
        obj3 = create_objective_function("yield", {})

        assert isinstance(obj1, NPVObjective)
        assert isinstance(obj2, CAPEXObjective)
        assert isinstance(obj3, YieldObjective)

        print("  Created NPVObjective, CAPEXObjective, YieldObjective")
        print("  ✓ Test 6 passed")

    except Exception as e:
        print(f"  ✗ Test 6 failed: {e}")
        raise

    print("\n" + "="*70)
    print("✓ ALL OBJECTIVE FUNCTION SMOKE TESTS PASSED")
    print("="*70)


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    _smoke_test_objectives()
