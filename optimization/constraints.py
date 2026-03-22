"""
optimization/constraints.py

PURPOSE:
Define and evaluate constraints for process optimization including process specifications,
safety limits, mechanical design limits, and operational bounds.
Support equality and inequality constraints with automatic violation detection.

Author: King Saud University - Chemical Engineering Department
Date: 2026-01-02
Version: 1.0.0
"""

from __future__ import annotations

import math
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# ABSTRACT BASE CLASS
# ============================================================================

class Constraint(ABC):
    """
    Abstract base class for all constraints.

    Convention:
    - Inequality: g(x) ≤ 0 (satisfied when g ≤ 0)
    - Equality: h(x) = 0 (satisfied when |h| < tolerance)
    """

    def __init__(self, config: dict):
        """
        Initialize constraint.

        Args:
            config: Configuration dict with constraint parameters
        """
        self.config = config
        self.name = config.get("name", self.__class__.__name__)
        self.critical = config.get("critical", False)
        self.tolerance = config.get("tolerance", 1e-6)
        self.penalty_weight = config.get("penalty_weight", 1000.0)

        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.tolerance < 0:
            raise ValueError(f"Tolerance must be non-negative, got {self.tolerance}")
        if self.penalty_weight < 0:
            raise ValueError(f"Penalty weight must be non-negative, got {self.penalty_weight}")

    @abstractmethod
    def evaluate(
        self,
        design_vector: np.ndarray,
        flowsheet_results: dict
    ) -> float:
        """
        Evaluate constraint at design point.

        Args:
            design_vector: Design variables
            flowsheet_results: Flowsheet simulation results

        Returns:
            Constraint value:
            - Inequality: g(x) ≤ 0 satisfied means return ≤ 0
            - Equality: h(x) = 0 satisfied means return ≈ 0
            - Positive value indicates violation

        Raises:
            RuntimeError: If flowsheet did not converge
            ValueError: If inputs are invalid
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return constraint name."""
        pass

    @abstractmethod
    def get_type(self) -> str:
        """Return 'inequality' or 'equality'."""
        pass

    @abstractmethod
    def get_bounds(self) -> Tuple[float, float]:
        """
        Return (lower_bound, upper_bound) for constraint.

        For inequality g(x) ≤ 0: bounds are (-inf, 0)
        For equality h(x) = 0: bounds are (-tol, tol)
        """
        pass

    def is_violated(
        self,
        design_vector: np.ndarray,
        flowsheet_results: dict,
        tolerance: Optional[float] = None
    ) -> bool:
        """
        Check if constraint is violated.

        Args:
            design_vector: Design variables
            flowsheet_results: Flowsheet results
            tolerance: Override default tolerance

        Returns:
            True if violated, False if satisfied
        """
        tol = tolerance if tolerance is not None else self.tolerance
        value = self.evaluate(design_vector, flowsheet_results)

        if self.get_type() == "inequality":
            return value > tol
        else:  # equality
            return abs(value) > tol

    def violation_magnitude(
        self,
        design_vector: np.ndarray,
        flowsheet_results: dict
    ) -> float:
        """
        Return magnitude of constraint violation.

        Returns:
            0.0 if satisfied, positive value if violated
        """
        value = self.evaluate(design_vector, flowsheet_results)

        if self.get_type() == "inequality":
            return max(0.0, value)
        else:  # equality
            return abs(value) if abs(value) > self.tolerance else 0.0

    def gradient(
        self,
        design_vector: np.ndarray,
        flowsheet_results: dict,
        epsilon: float = 1e-6,
        flowsheet_evaluator: Optional[callable] = None
    ) -> np.ndarray:
        """
        Calculate constraint gradient via finite differences.

        Args:
            design_vector: Design variables
            flowsheet_results: Results at current point
            epsilon: Perturbation size
            flowsheet_evaluator: Function to re-evaluate flowsheet

        Returns:
            Gradient vector (partial derivatives)
        """
        n_vars = len(design_vector)
        gradient = np.zeros(n_vars)

        # Evaluate at current point
        g_0 = self.evaluate(design_vector, flowsheet_results)

        if flowsheet_evaluator is None:
            logger.warning(
                f"Gradient for {self.get_name()}: flowsheet_evaluator not provided"
            )
            return gradient

        # Finite differences
        for i in range(n_vars):
            design_perturbed = design_vector.copy()
            eps = epsilon * max(abs(design_vector[i]), 1.0)
            design_perturbed[i] += eps

            try:
                flowsheet_perturbed = flowsheet_evaluator(design_perturbed)
                g_i = self.evaluate(design_perturbed, flowsheet_perturbed)
                gradient[i] = (g_i - g_0) / eps
            except Exception as e:
                logger.warning(f"Gradient calculation failed for variable {i}: {e}")
                gradient[i] = 0.0

        return gradient

    def _check_convergence(self, flowsheet_results: dict) -> None:
        """Check if flowsheet converged."""
        if flowsheet_results is None:
            raise RuntimeError(f"{self.get_name()}: Flowsheet results are None")
        if not flowsheet_results.get("converged", False):
            raise RuntimeError(
                f"{self.get_name()}: Flowsheet did not converge. "
                f"Cannot evaluate constraint."
            )


# ============================================================================
# PROCESS CONSTRAINTS
# ============================================================================

class ProductPurityConstraint(Constraint):
    """
    Minimum product purity requirement.

    g = purity_min - purity_actual
    Satisfied if g ≤ 0 (purity_actual ≥ purity_min)
    """

    def __init__(self, config: dict):
        super().__init__(config)
        spec = config.get("specification", {})
        self.product_name = spec.get("product_name", "cyclohexane")
        self.minimum_purity = spec.get("minimum_purity", 0.995)

    def evaluate(
        self,
        design_vector: np.ndarray,
        flowsheet_results: dict
    ) -> float:
        """Evaluate purity constraint."""

        self._check_convergence(flowsheet_results)

        products = flowsheet_results.get("products", {})
        purity_key = f"{self.product_name}_purity"

        purity_actual = products.get(purity_key, 0)

        # g = min - actual (want actual ≥ min, so g ≤ 0)
        constraint_value = self.minimum_purity - purity_actual

        logger.debug(
            f"Purity constraint: actual={purity_actual:.4f}, "
            f"min={self.minimum_purity:.4f}, g={constraint_value:.6f}"
        )

        return constraint_value

    def get_name(self) -> str:
        return f"{self.product_name}_purity"

    def get_type(self) -> str:
        return "inequality"

    def get_bounds(self) -> Tuple[float, float]:
        return (-float('inf'), 0.0)


class ConversionConstraint(Constraint):
    """
    Minimum reactant conversion requirement.

    g = conversion_min - conversion_actual
    """

    def __init__(self, config: dict):
        super().__init__(config)
        spec = config.get("specification", {})
        self.reactant = spec.get("reactant", "benzene")
        self.minimum_conversion = spec.get("minimum_conversion", 0.98)

    def evaluate(
        self,
        design_vector: np.ndarray,
        flowsheet_results: dict
    ) -> float:
        """Evaluate conversion constraint."""

        self._check_convergence(flowsheet_results)

        products = flowsheet_results.get("products", {})
        conversion_key = f"conversion_{self.reactant}"

        conversion_actual = products.get(conversion_key, 0)

        # Handle if conversion is given as percent
        if conversion_actual > 1.0:
            conversion_actual = conversion_actual / 100.0

        constraint_value = self.minimum_conversion - conversion_actual

        logger.debug(
            f"Conversion constraint: actual={conversion_actual:.4f}, "
            f"min={self.minimum_conversion:.4f}, g={constraint_value:.6f}"
        )

        return constraint_value

    def get_name(self) -> str:
        return f"{self.reactant}_conversion"

    def get_type(self) -> str:
        return "inequality"

    def get_bounds(self) -> Tuple[float, float]:
        return (-float('inf'), 0.0)


class RecoveryConstraint(Constraint):
    """
    Minimum component recovery in separation.

    g = recovery_min - recovery_actual
    """

    def __init__(self, config: dict):
        super().__init__(config)
        spec = config.get("specification", {})
        self.component = spec.get("component", "benzene")
        self.minimum_recovery = spec.get("minimum_recovery", 0.98)

    def evaluate(
        self,
        design_vector: np.ndarray,
        flowsheet_results: dict
    ) -> float:
        """Evaluate recovery constraint."""

        self._check_convergence(flowsheet_results)

        products = flowsheet_results.get("products", {})
        recovery_key = f"{self.component}_recovery"

        recovery_actual = products.get(recovery_key, 0)

        # Handle if recovery is given as percent
        if recovery_actual > 1.0:
            recovery_actual = recovery_actual / 100.0

        constraint_value = self.minimum_recovery - recovery_actual

        logger.debug(
            f"Recovery constraint: actual={recovery_actual:.4f}, "
            f"min={self.minimum_recovery:.4f}, g={constraint_value:.6f}"
        )

        return constraint_value

    def get_name(self) -> str:
        return f"{self.component}_recovery"

    def get_type(self) -> str:
        return "inequality"

    def get_bounds(self) -> Tuple[float, float]:
        return (-float('inf'), 0.0)


class FlowRateConstraint(Constraint):
    """
    Flow rate bounds (min/max).

    Creates two inequality constraints:
    - g1 = flow_min - flow_actual (for minimum)
    - g2 = flow_actual - flow_max (for maximum)
    """

    def __init__(self, config: dict):
        super().__init__(config)
        spec = config.get("specification", {})
        self.stream_name = spec.get("stream_name", "product")
        self.minimum_flow = spec.get("minimum_flow", 0)
        self.maximum_flow = spec.get("maximum_flow", float('inf'))
        self.bound_type = spec.get("bound_type", "both")  # "min", "max", "both"

    def evaluate(
        self,
        design_vector: np.ndarray,
        flowsheet_results: dict
    ) -> float:
        """Evaluate flow rate constraint."""

        self._check_convergence(flowsheet_results)

        # Try to get flow from products or streams
        products = flowsheet_results.get("products", {})
        flow_key = f"{self.stream_name}_kmol_h"

        flow_actual = products.get(flow_key, 0)

        if flow_actual == 0:
            # Try streams
            streams = flowsheet_results.get("streams", {})
            if self.stream_name in streams:
                flow_actual = streams[self.stream_name].get("molar_flow_kmol_h", 0)

        # Return worst violation
        if self.bound_type == "min":
            return self.minimum_flow - flow_actual
        elif self.bound_type == "max":
            return flow_actual - self.maximum_flow
        else:  # both
            violation_min = self.minimum_flow - flow_actual
            violation_max = flow_actual - self.maximum_flow
            return max(violation_min, violation_max)

    def get_name(self) -> str:
        return f"{self.stream_name}_flow_{self.bound_type}"

    def get_type(self) -> str:
        return "inequality"

    def get_bounds(self) -> Tuple[float, float]:
        return (-float('inf'), 0.0)


# ============================================================================
# SAFETY CONSTRAINTS
# ============================================================================

class TemperatureConstraint(Constraint):
    """
    Temperature within safety/material limits.

    g = T_actual - T_max (for maximum constraint)
    """

    def __init__(self, config: dict):
        super().__init__(config)
        spec = config.get("specification", {})
        self.equipment = spec.get("equipment", "reactor")
        self.minimum_C = spec.get("minimum_C", -float('inf'))
        self.maximum_C = spec.get("maximum_C", float('inf'))
        self.bound_type = spec.get("bound_type", "max")  # "min", "max", "both"

    def evaluate(
        self,
        design_vector: np.ndarray,
        flowsheet_results: dict
    ) -> float:
        """Evaluate temperature constraint."""

        self._check_convergence(flowsheet_results)

        safety = flowsheet_results.get("safety", {})
        temp_key = f"max_temperature_C"

        temp_actual = safety.get(temp_key, 0)

        if temp_actual == 0:
            # Try equipment dict
            equipment = flowsheet_results.get("equipment", {})
            if self.equipment in equipment:
                temp_actual = equipment[self.equipment].get("temperature_C", 0)

        # Return worst violation
        if self.bound_type == "min":
            return self.minimum_C - temp_actual
        elif self.bound_type == "max":
            return temp_actual - self.maximum_C
        else:  # both
            violation_min = self.minimum_C - temp_actual
            violation_max = temp_actual - self.maximum_C
            return max(violation_min, violation_max)

    def get_name(self) -> str:
        return f"{self.equipment}_temperature_{self.bound_type}"

    def get_type(self) -> str:
        return "inequality"

    def get_bounds(self) -> Tuple[float, float]:
        return (-float('inf'), 0.0)


class PressureConstraint(Constraint):
    """
    Pressure within design limits (MAWP).

    g = P_actual - MAWP
    """

    def __init__(self, config: dict):
        super().__init__(config)
        spec = config.get("specification", {})
        self.equipment = spec.get("equipment", "reactor")
        self.MAWP_bar = spec.get("MAWP_bar", 35.0)
        self.safety_margin = spec.get("safety_margin", 0.9)  # Use 90% of MAWP

    def evaluate(
        self,
        design_vector: np.ndarray,
        flowsheet_results: dict
    ) -> float:
        """Evaluate pressure constraint."""

        self._check_convergence(flowsheet_results)

        safety = flowsheet_results.get("safety", {})
        pressure_actual = safety.get("max_pressure_bar", 0)

        if pressure_actual == 0:
            equipment = flowsheet_results.get("equipment", {})
            if self.equipment in equipment:
                pressure_actual = equipment[self.equipment].get("pressure_bar", 0)

        # Apply safety margin
        pressure_limit = self.MAWP_bar * self.safety_margin

        constraint_value = pressure_actual - pressure_limit

        logger.debug(
            f"Pressure constraint: actual={pressure_actual:.1f} bar, "
            f"limit={pressure_limit:.1f} bar, g={constraint_value:.3f}"
        )

        return constraint_value

    def get_name(self) -> str:
        return f"{self.equipment}_pressure"

    def get_type(self) -> str:
        return "inequality"

    def get_bounds(self) -> Tuple[float, float]:
        return (-float('inf'), 0.0)


class ReliefCapacityConstraint(Constraint):
    """
    Adequate pressure relief capacity.

    g = relief_load - relief_capacity
    Satisfied if capacity ≥ load
    """

    def __init__(self, config: dict):
        super().__init__(config)
        spec = config.get("specification", {})
        self.equipment = spec.get("equipment", "reactor")
        self.safety_factor = spec.get("safety_factor", 1.1)  # 10% margin

    def evaluate(
        self,
        design_vector: np.ndarray,
        flowsheet_results: dict
    ) -> float:
        """Evaluate relief capacity constraint."""

        self._check_convergence(flowsheet_results)

        safety = flowsheet_results.get("safety", {})

        relief_load = safety.get("relief_load_kg_h", 0)
        relief_capacity = safety.get("relief_capacity_kg_h", 0)

        # Apply safety factor to required capacity
        required_capacity = relief_load * self.safety_factor

        constraint_value = required_capacity - relief_capacity

        logger.debug(
            f"Relief capacity: load={relief_load:.0f} kg/h, "
            f"capacity={relief_capacity:.0f} kg/h, g={constraint_value:.1f}"
        )

        return constraint_value

    def get_name(self) -> str:
        return f"{self.equipment}_relief_capacity"

    def get_type(self) -> str:
        return "inequality"

    def get_bounds(self) -> Tuple[float, float]:
        return (-float('inf'), 0.0)


class FlammabilityConstraint(Constraint):
    """
    Keep compositions outside flammability envelope.

    g = y_component - LFL or UFL - y_component
    """

    def __init__(self, config: dict):
        super().__init__(config)
        spec = config.get("specification", {})
        self.component = spec.get("component", "H2")
        self.LFL = spec.get("LFL", 0.04)  # Lower flammability limit
        self.UFL = spec.get("UFL", 0.75)  # Upper flammability limit
        self.limit_type = spec.get("limit_type", "lower")  # "lower" or "upper"

    def evaluate(
        self,
        design_vector: np.ndarray,
        flowsheet_results: dict
    ) -> float:
        """Evaluate flammability constraint."""

        self._check_convergence(flowsheet_results)

        safety = flowsheet_results.get("safety", {})
        composition = safety.get(f"{self.component}_mole_fraction", 0)

        if self.limit_type == "lower":
            # Keep below LFL
            return composition - self.LFL
        else:
            # Keep above UFL (for inert systems)
            return self.UFL - composition

    def get_name(self) -> str:
        return f"{self.component}_flammability_{self.limit_type}"

    def get_type(self) -> str:
        return "inequality"

    def get_bounds(self) -> Tuple[float, float]:
        return (-float('inf'), 0.0)


# ============================================================================
# MECHANICAL CONSTRAINTS
# ============================================================================

class VesselSlendernessConstraint(Constraint):
    """
    Height/diameter ratio within limits.

    g = H/D - (H/D)_max
    """

    def __init__(self, config: dict):
        super().__init__(config)
        spec = config.get("specification", {})
        self.equipment = spec.get("equipment", "reactor")
        self.maximum_ratio = spec.get("maximum_ratio", 10.0)

    def evaluate(
        self,
        design_vector: np.ndarray,
        flowsheet_results: dict
    ) -> float:
        """Evaluate slenderness constraint."""

        self._check_convergence(flowsheet_results)

        mechanical = flowsheet_results.get("mechanical", {})

        height = mechanical.get(f"{self.equipment}_height_m", 0)
        diameter = mechanical.get(f"{self.equipment}_diameter_m", 1)

        if diameter == 0:
            diameter = 1.0  # Avoid division by zero

        ratio_actual = height / diameter

        constraint_value = ratio_actual - self.maximum_ratio

        logger.debug(
            f"Slenderness: H/D={ratio_actual:.2f}, "
            f"max={self.maximum_ratio:.2f}, g={constraint_value:.3f}"
        )

        return constraint_value

    def get_name(self) -> str:
        return f"{self.equipment}_slenderness"

    def get_type(self) -> str:
        return "inequality"

    def get_bounds(self) -> Tuple[float, float]:
        return (-float('inf'), 0.0)


class WallThicknessConstraint(Constraint):
    """
    Wall thickness within fabrication limits.

    Two constraints:
    - g1 = t_min - t_actual (minimum for fabrication)
    - g2 = t_actual - t_max (maximum for practicality)
    """

    def __init__(self, config: dict):
        super().__init__(config)
        spec = config.get("specification", {})
        self.equipment = spec.get("equipment", "reactor")
        self.minimum_mm = spec.get("minimum_mm", 6.0)
        self.maximum_mm = spec.get("maximum_mm", 100.0)
        self.bound_type = spec.get("bound_type", "both")

    def evaluate(
        self,
        design_vector: np.ndarray,
        flowsheet_results: dict
    ) -> float:
        """Evaluate wall thickness constraint."""

        self._check_convergence(flowsheet_results)

        mechanical = flowsheet_results.get("mechanical", {})
        thickness_actual = mechanical.get("wall_thickness_mm", 0)

        if self.bound_type == "min":
            return self.minimum_mm - thickness_actual
        elif self.bound_type == "max":
            return thickness_actual - self.maximum_mm
        else:  # both
            violation_min = self.minimum_mm - thickness_actual
            violation_max = thickness_actual - self.maximum_mm
            return max(violation_min, violation_max)

    def get_name(self) -> str:
        return f"{self.equipment}_wall_thickness_{self.bound_type}"

    def get_type(self) -> str:
        return "inequality"

    def get_bounds(self) -> Tuple[float, float]:
        return (-float('inf'), 0.0)


class FoundationLoadConstraint(Constraint):
    """
    Weight within foundation capacity.

    g = weight_actual - weight_capacity
    """

    def __init__(self, config: dict):
        super().__init__(config)
        spec = config.get("specification", {})
        self.equipment = spec.get("equipment", "reactor")
        self.capacity_kg = spec.get("capacity_kg", 100000)
        self.safety_factor = spec.get("safety_factor", 0.8)

    def evaluate(
        self,
        design_vector: np.ndarray,
        flowsheet_results: dict
    ) -> float:
        """Evaluate foundation load constraint."""

        self._check_convergence(flowsheet_results)

        mechanical = flowsheet_results.get("mechanical", {})
        weight_actual = mechanical.get(f"{self.equipment}_weight_kg", 0)

        # Apply safety factor
        capacity_limit = self.capacity_kg * self.safety_factor

        constraint_value = weight_actual - capacity_limit

        logger.debug(
            f"Foundation load: weight={weight_actual:.0f} kg, "
            f"capacity={capacity_limit:.0f} kg, g={constraint_value:.1f}"
        )

        return constraint_value

    def get_name(self) -> str:
        return f"{self.equipment}_foundation_load"

    def get_type(self) -> str:
        return "inequality"

    def get_bounds(self) -> Tuple[float, float]:
        return (-float('inf'), 0.0)


# ============================================================================
# OPERATIONAL CONSTRAINTS
# ============================================================================

class TurndownConstraint(Constraint):
    """
    Operation within equipment turndown range.

    g = flow_min - flow_actual
    """

    def __init__(self, config: dict):
        super().__init__(config)
        spec = config.get("specification", {})
        self.equipment = spec.get("equipment", "compressor")
        self.minimum_flow_fraction = spec.get("minimum_flow_fraction", 0.5)
        self.design_flow = spec.get("design_flow_kmol_h", 100)

    def evaluate(
        self,
        design_vector: np.ndarray,
        flowsheet_results: dict
    ) -> float:
        """Evaluate turndown constraint."""

        self._check_convergence(flowsheet_results)

        operational = flowsheet_results.get("operational", {})
        flow_actual = operational.get(f"{self.equipment}_flow_kmol_h", self.design_flow)

        minimum_flow = self.design_flow * self.minimum_flow_fraction

        constraint_value = minimum_flow - flow_actual

        logger.debug(
            f"Turndown: flow={flow_actual:.1f}, "
            f"min={minimum_flow:.1f}, g={constraint_value:.2f}"
        )

        return constraint_value

    def get_name(self) -> str:
        return f"{self.equipment}_turndown"

    def get_type(self) -> str:
        return "inequality"

    def get_bounds(self) -> Tuple[float, float]:
        return (-float('inf'), 0.0)


class StabilityConstraint(Constraint):
    """
    Process stability (avoid oscillations/runaway).

    Simplified stability criterion for exothermic reactors.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        spec = config.get("specification", {})
        self.critical_value = spec.get("critical_value", 10.0)

    def evaluate(
        self,
        design_vector: np.ndarray,
        flowsheet_results: dict
    ) -> float:
        """Evaluate stability constraint."""

        self._check_convergence(flowsheet_results)

        operational = flowsheet_results.get("operational", {})
        stability_margin = operational.get("stability_margin", 100.0)

        # Constraint: margin should be > critical value
        # g = critical - margin (want margin > critical, so g < 0)
        constraint_value = self.critical_value - stability_margin

        logger.debug(
            f"Stability: margin={stability_margin:.2f}, "
            f"critical={self.critical_value:.2f}, g={constraint_value:.3f}"
        )

        return constraint_value

    def get_name(self) -> str:
        return "stability"

    def get_type(self) -> str:
        return "inequality"

    def get_bounds(self) -> Tuple[float, float]:
        return (-float('inf'), 0.0)


# ============================================================================
# BALANCE CONSTRAINTS (EQUALITY)
# ============================================================================

class MaterialBalanceConstraint(Constraint):
    """
    Material balance closure (equality constraint).

    h = mass_in - mass_out (should be 0)
    """

    def __init__(self, config: dict):
        super().__init__(config)
        # Equality constraints typically have tighter tolerance
        self.tolerance = config.get("tolerance", 1e-4)

    def evaluate(
        self,
        design_vector: np.ndarray,
        flowsheet_results: dict
    ) -> float:
        """Evaluate material balance."""

        self._check_convergence(flowsheet_results)

        balances = flowsheet_results.get("balances", {})
        error_percent = balances.get("mass_balance_error_percent", 0)

        # Return absolute error
        return abs(error_percent)

    def get_name(self) -> str:
        return "material_balance"

    def get_type(self) -> str:
        return "equality"

    def get_bounds(self) -> Tuple[float, float]:
        return (-self.tolerance, self.tolerance)


class EnergyBalanceConstraint(Constraint):
    """
    Energy balance closure (equality constraint).

    h = energy_in - energy_out + Q - W (should be 0)
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.tolerance = config.get("tolerance", 1.0)  # kW

    def evaluate(
        self,
        design_vector: np.ndarray,
        flowsheet_results: dict
    ) -> float:
        """Evaluate energy balance."""

        self._check_convergence(flowsheet_results)

        balances = flowsheet_results.get("balances", {})
        error_kW = balances.get("energy_balance_error_kW", 0)

        return abs(error_kW)

    def get_name(self) -> str:
        return "energy_balance"

    def get_type(self) -> str:
        return "equality"

    def get_bounds(self) -> Tuple[float, float]:
        return (-self.tolerance, self.tolerance)


# ============================================================================
# FACTORY AND UTILITIES
# ============================================================================

def create_constraint(
    constraint_type: str,
    config: dict
) -> Constraint:
    """
    Factory function to create constraint from string type.

    Args:
        constraint_type: Type of constraint
        config: Configuration dict

    Returns:
        Constraint instance

    Example:
        constraint = create_constraint("product_purity", config)
        value = constraint.evaluate(design_vector, flowsheet_results)
    """

    constraints_map = {
        # Process
        "product_purity": ProductPurityConstraint,
        "conversion": ConversionConstraint,
        "recovery": RecoveryConstraint,
        "flow_rate": FlowRateConstraint,

        # Safety
        "temperature": TemperatureConstraint,
        "pressure": PressureConstraint,
        "relief_capacity": ReliefCapacityConstraint,
        "flammability": FlammabilityConstraint,

        # Mechanical
        "vessel_slenderness": VesselSlendernessConstraint,
        "wall_thickness": WallThicknessConstraint,
        "foundation_load": FoundationLoadConstraint,

        # Operational
        "turndown": TurndownConstraint,
        "stability": StabilityConstraint,

        # Balances
        "material_balance": MaterialBalanceConstraint,
        "energy_balance": EnergyBalanceConstraint,
    }

    constraint_class = constraints_map.get(constraint_type.lower())

    if constraint_class is None:
        raise ValueError(
            f"Unknown constraint type: {constraint_type}. "
            f"Available: {list(constraints_map.keys())}"
        )

    return constraint_class(config)


def check_all_constraints(
    design_vector: np.ndarray,
    constraints: List[Constraint],
    flowsheet_results: dict
) -> dict:
    """
    Check all constraints and return comprehensive results.

    Args:
        design_vector: Design variables
        constraints: List of constraint instances
        flowsheet_results: Flowsheet results

    Returns:
        Dict with comprehensive constraint evaluation results
    """

    constraint_values = {}
    violation_magnitudes = {}
    violated_constraints = []
    critical_violations = []

    for constraint in constraints:
        name = constraint.get_name()

        try:
            value = constraint.evaluate(design_vector, flowsheet_results)
            violated = constraint.is_violated(design_vector, flowsheet_results)
            violation = constraint.violation_magnitude(design_vector, flowsheet_results)

            constraint_values[name] = value
            violation_magnitudes[name] = violation

            if violated:
                violated_constraints.append(name)

                if constraint.critical:
                    critical_violations.append(name)

        except Exception as e:
            logger.error(f"Error evaluating constraint {name}: {e}")
            constraint_values[name] = float('nan')
            violation_magnitudes[name] = float('inf')
            violated_constraints.append(name)

    num_violated = len(violated_constraints)
    all_satisfied = (num_violated == 0)
    total_violation = sum(violation_magnitudes.values())

    return {
        "all_satisfied": all_satisfied,
        "num_violated": num_violated,
        "violated_constraints": violated_constraints,
        "constraint_values": constraint_values,
        "violation_magnitudes": violation_magnitudes,
        "critical_violations": critical_violations,
        "total_violation": total_violation
    }


def calculate_constraint_violations(
    design_vector: np.ndarray,
    constraints: List[Constraint],
    flowsheet_results: dict
) -> dict:
    """
    Calculate detailed constraint violation information.

    Args:
        design_vector: Design variables
        constraints: List of constraints
        flowsheet_results: Flowsheet results

    Returns:
        Dict mapping constraint name to violation details
    """

    violations = {}

    for constraint in constraints:
        name = constraint.get_name()

        try:
            value = constraint.evaluate(design_vector, flowsheet_results)
            satisfied = not constraint.is_violated(design_vector, flowsheet_results)
            violation = constraint.violation_magnitude(design_vector, flowsheet_results)

            violations[name] = {
                "value": value,
                "satisfied": satisfied,
                "violation": violation,
                "type": constraint.get_type(),
                "critical": constraint.critical
            }

        except Exception as e:
            logger.error(f"Error calculating violation for {name}: {e}")
            violations[name] = {
                "value": float('nan'),
                "satisfied": False,
                "violation": float('inf'),
                "type": constraint.get_type(),
                "critical": constraint.critical
            }

    return violations


def apply_penalty_function(
    objective_value: float,
    constraint_violations: dict,
    penalty_config: dict
) -> float:
    """
    Apply penalty function for constraint violations.

    Args:
        objective_value: Original objective value
        constraint_violations: Dict from calculate_constraint_violations()
        penalty_config: Penalty method configuration

    Returns:
        Penalized objective value

    Note:
        For maximization objectives (like NPV), penalty is subtracted.
        For minimization objectives, penalty is added.
        This function assumes maximization (NPV). For minimization,
        change sign in caller.
    """

    method = penalty_config.get("method", "quadratic")
    coefficient = penalty_config.get("coefficient", 1000.0)
    power = penalty_config.get("power", 2)

    total_penalty = 0.0

    for name, info in constraint_violations.items():
        if not info["satisfied"]:
            violation = info["violation"]

            if method == "linear":
                penalty = coefficient * violation
            elif method == "quadratic":
                penalty = coefficient * (violation ** power)
            elif method == "exponential":
                penalty = coefficient * (np.exp(violation) - 1)
            else:
                penalty = coefficient * violation

            # Critical constraints get higher penalty
            if info["critical"]:
                penalty *= 10.0

            total_penalty += penalty

    # For maximization: subtract penalty
    # For minimization: add penalty
    # Here we assume maximization (NPV)
    penalized_objective = objective_value - total_penalty

    logger.debug(
        f"Penalty function: original={objective_value:.2e}, "
        f"penalty={total_penalty:.2e}, penalized={penalized_objective:.2e}"
    )

    return penalized_objective


def get_feasible_region_bounds(
    constraints: List[Constraint],
    design_space_bounds: np.ndarray
) -> np.ndarray:
    """
    Estimate feasible region bounds from constraints.

    Args:
        constraints: List of constraints
        design_space_bounds: Initial bounds [(min1, max1), (min2, max2), ...]

    Returns:
        Tightened bounds array

    Note:
        This is a simple implementation. For complex constraints,
        use more sophisticated methods.
    """

    # For now, just return original bounds
    # In a full implementation, would analyze constraints to tighten bounds

    logger.info("Feasible region estimation not fully implemented yet")

    return design_space_bounds


# ============================================================================
# SMOKE TEST
# ============================================================================

def _smoke_test_constraints():
    """
    Test constraint evaluation and violation detection.
    """
    print("="*70)
    print("CONSTRAINTS SMOKE TEST")
    print("="*70)

    # Mock flowsheet results
    flowsheet_results = {
        "converged": True,
        "products": {
            "cyclohexane_purity": 0.997,  # 99.7%
            "benzene_recovery": 0.985,  # 98.5%
            "conversion_benzene": 0.99  # 99%
        },
        "safety": {
            "max_temperature_C": 385.0,
            "max_pressure_bar": 31.5,
            "relief_capacity_kg_h": 6000,
            "relief_load_kg_h": 5000
        },
        "mechanical": {
            "reactor_height_m": 20.0,
            "reactor_diameter_m": 3.0,
            "wall_thickness_mm": 28.0
        },
        "balances": {
            "mass_balance_error_percent": 0.005,
            "energy_balance_error_kW": 0.08
        }
    }

    design_vector = np.array([380, 360, 340, 320, 5.0, 31.0])

    # ========================================================================
    # Test 1: Product purity constraint (satisfied)
    # ========================================================================
    print("\n✓ Test 1: Product purity constraint (satisfied)...")

    purity_config = {
        "name": "cyclohexane_purity",
        "type": "inequality",
        "specification": {
            "product_name": "cyclohexane",
            "minimum_purity": 0.995
        }
    }

    try:
        purity_constraint = ProductPurityConstraint(purity_config)
        purity_value = purity_constraint.evaluate(design_vector, flowsheet_results)
        purity_violated = purity_constraint.is_violated(design_vector, flowsheet_results)

        print(f"  Actual purity: 99.7%")
        print(f"  Minimum purity: 99.5%")
        print(f"  Constraint value: {purity_value:.6f}")
        print(f"  Violated: {purity_violated}")

        assert purity_value <= 0, "Should be satisfied"
        assert not purity_violated
        print("  ✓ Test 1 passed")

    except Exception as e:
        print(f"  ✗ Test 1 failed: {e}")
        raise

    # ========================================================================
    # Test 2: Temperature constraint (satisfied)
    # ========================================================================
    print("\n✓ Test 2: Temperature constraint (satisfied)...")

    temp_config = {
        "name": "reactor_temperature",
        "type": "inequality",
        "specification": {
            "equipment": "reactor",
            "maximum_C": 400.0,
            "bound_type": "max"
        }
    }

    try:
        temp_constraint = TemperatureConstraint(temp_config)
        temp_value = temp_constraint.evaluate(design_vector, flowsheet_results)
        temp_violated = temp_constraint.is_violated(design_vector, flowsheet_results)

        print(f"  Actual max T: 385°C")
        print(f"  Limit: 400°C")
        print(f"  Constraint value: {temp_value:.1f}")
        print(f"  Violated: {temp_violated}")

        assert temp_value <= 0, "Should be satisfied"
        assert not temp_violated
        print("  ✓ Test 2 passed")

    except Exception as e:
        print(f"  ✗ Test 2 failed: {e}")
        raise

    # ========================================================================
    # Test 3: Vessel slenderness (satisfied)
    # ========================================================================
    print("\n✓ Test 3: Vessel slenderness constraint (satisfied)...")

    slenderness_config = {
        "name": "reactor_slenderness",
        "type": "inequality",
        "specification": {
            "equipment": "reactor",
            "maximum_ratio": 10.0
        }
    }

    try:
        slenderness_constraint = VesselSlendernessConstraint(slenderness_config)
        slenderness_value = slenderness_constraint.evaluate(design_vector, flowsheet_results)

        ratio = 20.0 / 3.0
        print(f"  H/D: {ratio:.2f}")
        print(f"  Max H/D: 10.0")
        print(f"  Constraint value: {slenderness_value:.2f}")

        assert slenderness_value <= 0, "Should be satisfied"
        print("  ✓ Test 3 passed")

    except Exception as e:
        print(f"  ✗ Test 3 failed: {e}")
        raise

    # ========================================================================
    # Test 4: Check all constraints
    # ========================================================================
    print("\n✓ Test 4: Check all constraints...")

    try:
        constraints = [purity_constraint, temp_constraint, slenderness_constraint]
        check_results = check_all_constraints(design_vector, constraints, flowsheet_results)

        print(f"  All satisfied: {check_results['all_satisfied']}")
        print(f"  Number violated: {check_results['num_violated']}")
        print(f"  Total violation: {check_results['total_violation']:.6f}")

        assert check_results['all_satisfied'], "All should be satisfied"
        assert check_results['num_violated'] == 0
        print("  ✓ Test 4 passed")

    except Exception as e:
        print(f"  ✗ Test 4 failed: {e}")
        raise

    # ========================================================================
    # Test 5: Violated constraint
    # ========================================================================
    print("\n✓ Test 5: Intentionally violated constraint...")

    try:
        # Set purity below spec
        flowsheet_results["products"]["cyclohexane_purity"] = 0.990

        purity_value_2 = purity_constraint.evaluate(design_vector, flowsheet_results)
        purity_violated_2 = purity_constraint.is_violated(design_vector, flowsheet_results)
        violation_mag = purity_constraint.violation_magnitude(design_vector, flowsheet_results)

        print(f"  Actual purity: 99.0% (below spec)")
        print(f"  Constraint value: {purity_value_2:.6f}")
        print(f"  Violated: {purity_violated_2}")
        print(f"  Violation magnitude: {violation_mag:.6f}")

        assert purity_value_2 > 0, "Should be violated"
        assert purity_violated_2
        assert violation_mag > 0
        print("  ✓ Test 5 passed")

    except Exception as e:
        print(f"  ✗ Test 5 failed: {e}")
        raise

    # ========================================================================
    # Test 6: Penalty function
    # ========================================================================
    print("\n✓ Test 6: Penalty function...")

    try:
        penalty_config = {
            "method": "quadratic",
            "coefficient": 1000000.0,  # High penalty for NPV
            "power": 2
        }

        objective_value = 25_000_000.0  # NPV
        violations = calculate_constraint_violations(
            design_vector, constraints, flowsheet_results
        )

        penalized_objective = apply_penalty_function(
            objective_value, violations, penalty_config
        )

        penalty_amount = objective_value - penalized_objective

        print(f"  Original objective: ${objective_value:,.0f}")
        print(f"  Penalized objective: ${penalized_objective:,.0f}")
        print(f"  Penalty: ${penalty_amount:,.0f}")

        assert penalized_objective < objective_value, "Penalty should reduce NPV"
        assert penalty_amount > 0
        print("  ✓ Test 6 passed")

    except Exception as e:
        print(f"  ✗ Test 6 failed: {e}")
        raise

    # ========================================================================
    # Test 7: Equality constraint
    # ========================================================================
    print("\n✓ Test 7: Equality constraint (material balance)...")

    try:
        balance_config = {
            "name": "material_balance",
            "type": "equality",
            "tolerance": 0.01
        }

        balance_constraint = MaterialBalanceConstraint(balance_config)

        # Restore flowsheet results
        flowsheet_results["products"]["cyclohexane_purity"] = 0.997

        balance_value = balance_constraint.evaluate(design_vector, flowsheet_results)
        balance_violated = balance_constraint.is_violated(design_vector, flowsheet_results)

        print(f"  Balance error: {balance_value:.6f}%")
        print(f"  Tolerance: 0.01%")
        print(f"  Violated: {balance_violated}")

        assert not balance_violated, "Balance should be satisfied"
        print("  ✓ Test 7 passed")

    except Exception as e:
        print(f"  ✗ Test 7 failed: {e}")
        raise

    # ========================================================================
    # Test 8: Factory function
    # ========================================================================
    print("\n✓ Test 8: Constraint factory...")

    try:
        c1 = create_constraint("product_purity", purity_config)
        c2 = create_constraint("temperature", temp_config)
        c3 = create_constraint("vessel_slenderness", slenderness_config)

        assert isinstance(c1, ProductPurityConstraint)
        assert isinstance(c2, TemperatureConstraint)
        assert isinstance(c3, VesselSlendernessConstraint)

        print("  Created ProductPurityConstraint, TemperatureConstraint, VesselSlendernessConstraint")
        print("  ✓ Test 8 passed")

    except Exception as e:
        print(f"  ✗ Test 8 failed: {e}")
        raise

    print("\n" + "="*70)
    print("✓ ALL CONSTRAINT SMOKE TESTS PASSED")
    print("="*70)


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    _smoke_test_constraints()
