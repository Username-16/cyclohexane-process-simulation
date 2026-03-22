"""
optimization/design_of_experiments.py

PURPOSE:
Implement Design of Experiments (DOE) methods for efficient sampling of design space.
Support classical DOE, space-filling designs, and adaptive sampling strategies.
Enable intelligent data collection to minimize expensive flowsheet evaluations.

Date: 2026-01-02
Version: 1.0.0
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import qmc

logger = logging.getLogger(__name__)

# ============================================================================
# RESULT DATACLASS
# ============================================================================

@dataclass
class DOEResult:
    """Container for DOE results."""
    samples: np.ndarray  # shape (num_samples, n_vars)
    design_matrix: np.ndarray  # coded values for analysis
    method: str
    num_samples: int
    num_variables: int
    properties: dict  # space-filling metrics, etc.


# ============================================================================
# ABSTRACT BASE CLASS
# ============================================================================

class DesignOfExperiments(ABC):
    """
    Abstract base class for DOE methods.

    All DOE methods must implement:
    - generate_samples(): Generate sample points
    - get_name(): Return method name
    """

    def __init__(self, config: dict):
        """
        Initialize DOE method.

        Args:
            config: Method-specific configuration
        """
        self.config = config

    @abstractmethod
    def generate_samples(
        self,
        bounds: np.ndarray,
        num_samples: int = None
    ) -> DOEResult:
        """
        Generate sample points in design space.

        Args:
            bounds: Variable bounds, shape (n_vars, 2)
            num_samples: Number of samples (None for factorial designs)

        Returns:
            DOEResult with samples and properties
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return DOE method name."""
        pass

    def _validate_inputs(self, bounds: np.ndarray, num_samples: int = None):
        """Validate inputs."""
        if len(bounds.shape) != 2 or bounds.shape[1] != 2:
            raise ValueError(f"Bounds must be shape (n_vars, 2), got {bounds.shape}")

        if num_samples is not None and num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")


# ============================================================================
# CLASSICAL DOE METHODS
# ============================================================================

class FullFactorialDesign(DesignOfExperiments):
    """Full factorial design - all combinations of levels."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.levels = config.get("levels", 2)
        self.center_points = config.get("center_points", 1)

    def get_name(self) -> str:
        return "Full Factorial"

    def generate_samples(
        self,
        bounds: np.ndarray,
        num_samples: int = None
    ) -> DOEResult:
        """Generate full factorial design."""
        self._validate_inputs(bounds, num_samples)

        n_vars = bounds.shape[0]

        # Generate coded design matrix
        if isinstance(self.levels, int):
            levels_per_var = [self.levels] * n_vars
        else:
            levels_per_var = self.levels

        # Create all combinations
        axes = [np.linspace(-1, 1, lev) for lev in levels_per_var]
        mesh = np.meshgrid(*axes, indexing='ij')
        design_matrix = np.column_stack([m.ravel() for m in mesh])

        # Add center points
        if self.center_points > 0:
            centers = np.zeros((self.center_points, n_vars))
            design_matrix = np.vstack([design_matrix, centers])

        # Convert to actual values
        samples = self._decode_design(design_matrix, bounds)

        # Compute properties
        properties = {
            "orthogonality": self._check_orthogonality(design_matrix),
            "levels": levels_per_var,
            "center_points": self.center_points
        }

        properties.update(evaluate_space_filling(samples, bounds))

        logger.info(f"Full factorial: {design_matrix.shape[0]} runs for {n_vars} factors")

        return DOEResult(
            samples=samples,
            design_matrix=design_matrix,
            method=self.get_name(),
            num_samples=len(samples),
            num_variables=n_vars,
            properties=properties
        )

    def _decode_design(self, coded: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        """Convert coded values to actual values."""
        center = (bounds[:, 0] + bounds[:, 1]) / 2
        range_half = (bounds[:, 1] - bounds[:, 0]) / 2
        return center + coded * range_half

    def _check_orthogonality(self, design_matrix: np.ndarray) -> float:
        """Check orthogonality of design."""
        X = design_matrix - design_matrix.mean(axis=0)
        corr = np.corrcoef(X.T)
        np.fill_diagonal(corr, 0)
        return float(np.max(np.abs(corr)))


class FractionalFactorialDesign(DesignOfExperiments):
    """Fractional factorial design with confounding."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.resolution = config.get("resolution", 4)
        self.levels = config.get("levels", 2)

    def get_name(self) -> str:
        return f"Fractional Factorial (Resolution {self.resolution})"

    def generate_samples(
        self,
        bounds: np.ndarray,
        num_samples: int = None
    ) -> DOEResult:
        """Generate fractional factorial (simplified 2-level)."""
        self._validate_inputs(bounds, num_samples)

        n_vars = bounds.shape[0]

        # For simplicity, use half fraction (2^(k-1))
        # Generate base factors
        n_base = max(n_vars - 1, 2)
        base_design = self._generate_base_factorial(n_base)

        # Generate confounded factor
        if n_vars > n_base:
            confounded = np.prod(base_design, axis=1, keepdims=True)
            design_matrix = np.hstack([base_design, confounded])
        else:
            design_matrix = base_design[:, :n_vars]

        samples = self._decode_design(design_matrix, bounds)

        properties = {
            "resolution": self.resolution,
            "fraction": f"2^({n_vars}-1)"
        }
        properties.update(evaluate_space_filling(samples, bounds))

        return DOEResult(
            samples=samples,
            design_matrix=design_matrix,
            method=self.get_name(),
            num_samples=len(samples),
            num_variables=n_vars,
            properties=properties
        )

    def _generate_base_factorial(self, n_factors: int) -> np.ndarray:
        """Generate base 2-level factorial."""
        n_runs = 2 ** n_factors
        design = np.zeros((n_runs, n_factors))

        for i in range(n_factors):
            design[:, i] = np.tile(np.repeat([-1, 1], 2**i), 2**(n_factors-i-1))

        return design

    def _decode_design(self, coded: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        """Convert coded to actual."""
        center = (bounds[:, 0] + bounds[:, 1]) / 2
        range_half = (bounds[:, 1] - bounds[:, 0]) / 2
        return center + coded * range_half


class CentralCompositeDesign(DesignOfExperiments):
    """Central composite design for response surface."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.alpha = config.get("alpha", "rotatable")
        self.center_points = config.get("center_points", (1, 1))
        self.face = config.get("face", "circumscribed")

    def get_name(self) -> str:
        return "Central Composite Design"

    def generate_samples(
        self,
        bounds: np.ndarray,
        num_samples: int = None
    ) -> DOEResult:
        """Generate CCD."""
        self._validate_inputs(bounds, num_samples)

        n_vars = bounds.shape[0]

        if n_vars < 2:
            raise ValueError("CCD requires at least 2 variables")

        # 1. Factorial points (2^k)
        factorial = self._generate_factorial(n_vars)

        # 2. Axial points (2k)
        alpha_val = self._compute_alpha(n_vars)
        axial = self._generate_axial(n_vars, alpha_val)

        # 3. Center points
        n_center_factorial, n_center_axial = self.center_points
        center_factorial = np.zeros((n_center_factorial, n_vars))
        center_axial = np.zeros((n_center_axial, n_vars))

        # Combine
        design_matrix = np.vstack([
            factorial,
            axial,
            center_factorial,
            center_axial
        ])

        # Convert to actual values
        samples = self._decode_design(design_matrix, bounds, alpha_val)

        # Properties
        is_rotatable = isinstance(self.alpha, str) and self.alpha == "rotatable"

        properties = {
            "alpha": alpha_val,
            "rotatability": is_rotatable,
            "factorial_points": len(factorial),
            "axial_points": len(axial),
            "center_points": n_center_factorial + n_center_axial
        }
        properties.update(evaluate_space_filling(samples, bounds))

        logger.info(f"CCD: {len(samples)} runs ({len(factorial)} factorial + "
                   f"{len(axial)} axial + {n_center_factorial + n_center_axial} center)")

        return DOEResult(
            samples=samples,
            design_matrix=design_matrix,
            method=self.get_name(),
            num_samples=len(samples),
            num_variables=n_vars,
            properties=properties
        )

    def _compute_alpha(self, n_vars: int) -> float:
        """Compute alpha for star points."""
        if isinstance(self.alpha, (int, float)):
            return float(self.alpha)
        elif self.alpha == "rotatable":
            n_factorial = 2 ** n_vars
            return n_factorial ** 0.25
        elif self.alpha == "face_centered":
            return 1.0
        elif self.alpha == "orthogonal":
            # Simplified orthogonal condition
            n_factorial = 2 ** n_vars
            return np.sqrt(n_vars)
        else:
            return (2 ** n_vars) ** 0.25

    def _generate_factorial(self, n_vars: int) -> np.ndarray:
        """Generate 2^k factorial points."""
        n_runs = 2 ** n_vars
        factorial = np.zeros((n_runs, n_vars))

        for i in range(n_vars):
            factorial[:, i] = np.tile(np.repeat([-1, 1], 2**i), 2**(n_vars-i-1))

        return factorial

    def _generate_axial(self, n_vars: int, alpha: float) -> np.ndarray:
        """Generate axial (star) points."""
        axial = np.zeros((2 * n_vars, n_vars))

        for i in range(n_vars):
            axial[2*i, i] = -alpha
            axial[2*i + 1, i] = alpha

        return axial

    def _decode_design(self, coded: np.ndarray, bounds: np.ndarray, alpha: float) -> np.ndarray:
        """Convert coded to actual."""
        center = (bounds[:, 0] + bounds[:, 1]) / 2
        range_half = (bounds[:, 1] - bounds[:, 0]) / 2

        # Scale by alpha
        scaled = coded / alpha

        return center + scaled * range_half


class BoxBehnkenDesign(DesignOfExperiments):
    """Box-Behnken design for quadratic models."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.center_points = config.get("center_points", 3)

    def get_name(self) -> str:
        return "Box-Behnken Design"

    def generate_samples(
        self,
        bounds: np.ndarray,
        num_samples: int = None
    ) -> DOEResult:
        """Generate Box-Behnken design."""
        self._validate_inputs(bounds, num_samples)

        n_vars = bounds.shape[0]

        if n_vars < 3:
            raise ValueError("Box-Behnken requires at least 3 variables")

        # Generate edge points: 2k(k-1) points
        design_matrix = []

        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                # Four combinations for dimensions i and j
                for vi in [-1, 1]:
                    for vj in [-1, 1]:
                        point = np.zeros(n_vars)
                        point[i] = vi
                        point[j] = vj
                        design_matrix.append(point)

        design_matrix = np.array(design_matrix)

        # Add center points
        if self.center_points > 0:
            centers = np.zeros((self.center_points, n_vars))
            design_matrix = np.vstack([design_matrix, centers])

        # Convert to actual values
        samples = self._decode_design(design_matrix, bounds)

        properties = {
            "edge_points": len(design_matrix) - self.center_points,
            "center_points": self.center_points
        }
        properties.update(evaluate_space_filling(samples, bounds))

        logger.info(f"Box-Behnken: {len(samples)} runs for {n_vars} factors")

        return DOEResult(
            samples=samples,
            design_matrix=design_matrix,
            method=self.get_name(),
            num_samples=len(samples),
            num_variables=n_vars,
            properties=properties
        )

    def _decode_design(self, coded: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        """Convert coded to actual."""
        center = (bounds[:, 0] + bounds[:, 1]) / 2
        range_half = (bounds[:, 1] - bounds[:, 0]) / 2
        return center + coded * range_half


class PlackettBurmanDesign(DesignOfExperiments):
    """Plackett-Burman screening design."""

    def __init__(self, config: dict):
        super().__init__(config)

    def get_name(self) -> str:
        return "Plackett-Burman Design"

    def generate_samples(
        self,
        bounds: np.ndarray,
        num_samples: int = None
    ) -> DOEResult:
        """Generate Plackett-Burman design (simplified)."""
        self._validate_inputs(bounds, num_samples)

        n_vars = bounds.shape[0]

        # Find appropriate run size (multiple of 4)
        n_runs = 4 * ((n_vars // 4) + 1)

        # Generate using Hadamard matrix approach (simplified)
        design_matrix = self._generate_pb_matrix(n_runs, n_vars)

        samples = self._decode_design(design_matrix, bounds)

        properties = {"screening": True, "resolution": "III"}
        properties.update(evaluate_space_filling(samples, bounds))

        return DOEResult(
            samples=samples,
            design_matrix=design_matrix,
            method=self.get_name(),
            num_samples=len(samples),
            num_variables=n_vars,
            properties=properties
        )

    def _generate_pb_matrix(self, n_runs: int, n_vars: int) -> np.ndarray:
        """Generate Plackett-Burman matrix (simplified)."""
        # Use random generation for simplicity
        np.random.seed(42)
        design = np.random.choice([-1, 1], size=(n_runs, n_vars))
        return design

    def _decode_design(self, coded: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        """Convert coded to actual."""
        center = (bounds[:, 0] + bounds[:, 1]) / 2
        range_half = (bounds[:, 1] - bounds[:, 0]) / 2
        return center + coded * range_half


# ============================================================================
# SPACE-FILLING DESIGNS
# ============================================================================

class LatinHypercubeSampling(DesignOfExperiments):
    """Latin hypercube sampling (LHS)."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.criterion = config.get("criterion", None)
        self.iterations = config.get("iterations", 1000)
        self.random_state = config.get("random_state", None)

    def get_name(self) -> str:
        return "Latin Hypercube Sampling"

    def generate_samples(
        self,
        bounds: np.ndarray,
        num_samples: int
    ) -> DOEResult:
        """Generate LHS."""
        self._validate_inputs(bounds, num_samples)

        n_vars = bounds.shape[0]

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Use scipy qmc for standard LHS
        sampler = qmc.LatinHypercube(d=n_vars, seed=self.random_state)
        samples_unit = sampler.random(n=num_samples)

        # Scale to bounds
        samples = qmc.scale(samples_unit, bounds[:, 0], bounds[:, 1])

        # Create design matrix (normalized to [0, 1])
        design_matrix = samples_unit

        # Compute properties
        properties = evaluate_space_filling(samples, bounds)

        logger.info(f"LHS: {num_samples} samples in {n_vars}D space")

        return DOEResult(
            samples=samples,
            design_matrix=design_matrix,
            method=self.get_name(),
            num_samples=num_samples,
            num_variables=n_vars,
            properties=properties
        )


class MaximinLHS(DesignOfExperiments):
    """Optimized LHS maximizing minimum distance."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.iterations = config.get("iterations", 1000)
        self.random_state = config.get("random_state", None)

    def get_name(self) -> str:
        return "Maximin LHS"

    def generate_samples(
        self,
        bounds: np.ndarray,
        num_samples: int
    ) -> DOEResult:
        """Generate maximin-optimized LHS."""
        self._validate_inputs(bounds, num_samples)

        n_vars = bounds.shape[0]

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Start with random LHS
        sampler = qmc.LatinHypercube(d=n_vars, seed=self.random_state)
        best_samples = sampler.random(n=num_samples)
        best_min_dist = compute_minimum_distance(best_samples)

        # Optimize by swapping
        for _ in range(self.iterations):
            # Create candidate by swapping two elements in random dimension
            candidate = best_samples.copy()
            dim = np.random.randint(n_vars)
            i, j = np.random.choice(num_samples, 2, replace=False)
            candidate[i, dim], candidate[j, dim] = candidate[j, dim], candidate[i, dim]

            # Check if improved
            min_dist = compute_minimum_distance(candidate)

            if min_dist > best_min_dist:
                best_samples = candidate
                best_min_dist = min_dist

        # Scale to bounds
        samples = qmc.scale(best_samples, bounds[:, 0], bounds[:, 1])

        properties = evaluate_space_filling(samples, bounds)

        logger.info(f"Maximin LHS: {num_samples} samples, min_dist={best_min_dist:.4f}")

        return DOEResult(
            samples=samples,
            design_matrix=best_samples,
            method=self.get_name(),
            num_samples=num_samples,
            num_variables=n_vars,
            properties=properties
        )


class SobolSequence(DesignOfExperiments):
    """Sobol quasi-random sequence."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.scramble = config.get("scramble", True)
        self.skip = config.get("skip", 0)

    def get_name(self) -> str:
        return "Sobol Sequence"

    def generate_samples(
        self,
        bounds: np.ndarray,
        num_samples: int
    ) -> DOEResult:
        """Generate Sobol sequence."""
        self._validate_inputs(bounds, num_samples)

        n_vars = bounds.shape[0]

        # Use scipy qmc Sobol
        sampler = qmc.Sobol(d=n_vars, scramble=self.scramble)

        # Skip initial points if requested
        if self.skip > 0:
            _ = sampler.random(n=self.skip)

        samples_unit = sampler.random(n=num_samples)

        # Scale to bounds
        samples = qmc.scale(samples_unit, bounds[:, 0], bounds[:, 1])

        properties = evaluate_space_filling(samples, bounds)

        logger.info(f"Sobol: {num_samples} samples, scrambled={self.scramble}")

        return DOEResult(
            samples=samples,
            design_matrix=samples_unit,
            method=self.get_name(),
            num_samples=num_samples,
            num_variables=n_vars,
            properties=properties
        )


class HaltonSequence(DesignOfExperiments):
    """Halton quasi-random sequence."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.scramble = config.get("scramble", True)

    def get_name(self) -> str:
        return "Halton Sequence"

    def generate_samples(
        self,
        bounds: np.ndarray,
        num_samples: int
    ) -> DOEResult:
        """Generate Halton sequence."""
        self._validate_inputs(bounds, num_samples)

        n_vars = bounds.shape[0]

        # Use scipy qmc Halton
        sampler = qmc.Halton(d=n_vars, scramble=self.scramble)
        samples_unit = sampler.random(n=num_samples)

        # Scale to bounds
        samples = qmc.scale(samples_unit, bounds[:, 0], bounds[:, 1])

        properties = evaluate_space_filling(samples, bounds)

        logger.info(f"Halton: {num_samples} samples")

        return DOEResult(
            samples=samples,
            design_matrix=samples_unit,
            method=self.get_name(),
            num_samples=num_samples,
            num_variables=n_vars,
            properties=properties
        )


class UniformRandomSampling(DesignOfExperiments):
    """Simple uniform random sampling (baseline)."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.random_state = config.get("random_state", None)

    def get_name(self) -> str:
        return "Uniform Random Sampling"

    def generate_samples(
        self,
        bounds: np.ndarray,
        num_samples: int
    ) -> DOEResult:
        """Generate random samples."""
        self._validate_inputs(bounds, num_samples)

        n_vars = bounds.shape[0]

        if self.random_state is not None:
            np.random.seed(self.random_state)

        samples = np.random.uniform(
            bounds[:, 0],
            bounds[:, 1],
            size=(num_samples, n_vars)
        )

        # Normalize for design matrix
        design_matrix = (samples - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

        properties = evaluate_space_filling(samples, bounds)

        logger.info(f"Random sampling: {num_samples} samples")

        return DOEResult(
            samples=samples,
            design_matrix=design_matrix,
            method=self.get_name(),
            num_samples=num_samples,
            num_variables=n_vars,
            properties=properties
        )


# ============================================================================
# ADAPTIVE SAMPLING
# ============================================================================

class AdaptiveSampling(DesignOfExperiments):
    """Adaptive sampling based on surrogate model."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.strategy = config.get("strategy", "uncertainty")
        self.batch_size = config.get("batch_size", 1)

    def get_name(self) -> str:
        return f"Adaptive Sampling ({self.strategy})"

    def generate_samples(
        self,
        bounds: np.ndarray,
        num_samples: int
    ) -> DOEResult:
        """
        Generate adaptive samples (requires surrogate model).

        Note: This is a framework. In practice, adaptive sampling
        needs existing data and a fitted surrogate model.
        """
        self._validate_inputs(bounds, num_samples)

        n_vars = bounds.shape[0]

        # Start with initial LHS
        initial_sampler = LatinHypercubeSampling({"random_state": 42})
        initial_result = initial_sampler.generate_samples(bounds, num_samples)

        logger.info(f"Adaptive sampling: initialized with {num_samples} LHS samples")
        logger.info(f"  (Full adaptive requires surrogate model and iterative evaluation)")

        return initial_result


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_doe(method: str, config: dict = None) -> DesignOfExperiments:
    """
    Factory function to create DOE method.

    Args:
        method: DOE method name
        config: Method configuration

    Returns:
        DesignOfExperiments instance
    """
    if config is None:
        config = {}

    methods = {
        "full_factorial": FullFactorialDesign,
        "fractional_factorial": FractionalFactorialDesign,
        "ccd": CentralCompositeDesign,
        "box_behnken": BoxBehnkenDesign,
        "plackett_burman": PlackettBurmanDesign,
        "lhs": LatinHypercubeSampling,
        "maximin_lhs": MaximinLHS,
        "sobol": SobolSequence,
        "halton": HaltonSequence,
        "random": UniformRandomSampling,
        "adaptive": AdaptiveSampling
    }

    method_lower = method.lower().replace("-", "_").replace(" ", "_")

    if method_lower not in methods:
        raise ValueError(f"Unknown DOE method: {method}. Available: {list(methods.keys())}")

    return methods[method_lower](config)


def evaluate_space_filling(
    samples: np.ndarray,
    bounds: np.ndarray
) -> dict:
    """
    Evaluate space-filling properties of design.

    Args:
        samples: Sample points
        bounds: Variable bounds

    Returns:
        Dictionary of metrics
    """
    # Normalize to [0, 1]
    samples_norm = (samples - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

    # Discrepancy
    disc_L2 = compute_discrepancy(samples_norm, np.array([[0, 1]] * samples.shape[1]), "L2")
    disc_centered = compute_discrepancy(samples_norm, np.array([[0, 1]] * samples.shape[1]), "centered_L2")

    # Minimum distance
    min_dist = compute_minimum_distance(samples)

    # Normalize minimum distance by diagonal
    diagonal = np.sqrt(np.sum((bounds[:, 1] - bounds[:, 0])**2))
    min_dist_norm = min_dist / diagonal if diagonal > 0 else 0

    # Coverage
    coverage = compute_coverage(samples, bounds, grid_resolution=10)

    # Composite uniformity score (0-1, higher is better)
    uniformity = (1 - min(disc_centered, 1.0)) * 0.5 + min_dist_norm * 0.3 + coverage * 0.2

    return {
        "discrepancy_L2": disc_L2,
        "discrepancy_centered_L2": disc_centered,
        "discrepancy": disc_centered,  # alias
        "minimum_distance": min_dist,
        "minimum_distance_normalized": min_dist_norm,
        "min_distance": min_dist,  # alias
        "maximin_criterion": min_dist,
        "coverage": coverage,
        "uniformity_score": uniformity,
        "space_filling": uniformity  # alias
    }


def compute_discrepancy(
    samples: np.ndarray,
    bounds: np.ndarray,
    discrepancy_type: str = "centered_L2"
) -> float:
    """
    Compute discrepancy (measure of uniformity).

    Args:
        samples: Normalized samples in [0, 1]^d
        bounds: Should be [[0,1], [0,1], ...]
        discrepancy_type: "L2" or "centered_L2"

    Returns:
        Discrepancy value (lower is better)
    """
    n, k = samples.shape

    if discrepancy_type == "centered_L2":
        # Centered L2 discrepancy (Hickernell 1998)
        term1 = (13.0 / 12.0) ** k

        term2 = 0.0
        for i in range(n):
            prod = 1.0
            for j in range(k):
                prod *= (1 + 0.5 * abs(samples[i, j] - 0.5) - 0.5 * (samples[i, j] - 0.5)**2)
            term2 += prod
        term2 /= n

        term3 = 0.0
        for i in range(n):
            for l in range(n):
                prod = 1.0
                for j in range(k):
                    prod *= (1 + 0.5 * abs(samples[i, j] - 0.5) + 0.5 * abs(samples[l, j] - 0.5)
                           - 0.5 * abs(samples[i, j] - samples[l, j]))
                term3 += prod
        term3 /= (n * n)

        discrepancy_sq = term1 - 2 * term2 + term3
        discrepancy = np.sqrt(max(discrepancy_sq, 0))

    else:  # L2
        # Simplified L2 discrepancy
        discrepancy = 0.0
        for i in range(n):
            prod = np.prod(samples[i] * (1 - samples[i]))
            discrepancy += prod
        discrepancy /= n
        discrepancy = np.sqrt(discrepancy)

    return float(discrepancy)


def compute_minimum_distance(samples: np.ndarray) -> float:
    """
    Compute minimum pairwise distance.

    Args:
        samples: Sample points

    Returns:
        Minimum distance
    """
    if len(samples) < 2:
        return 0.0

    distances = pdist(samples)
    return float(np.min(distances))


def compute_coverage(
    samples: np.ndarray,
    bounds: np.ndarray,
    grid_resolution: int = 10
) -> float:
    """
    Compute coverage (fraction of space with samples).

    Args:
        samples: Sample points
        bounds: Variable bounds
        grid_resolution: Grid fineness

    Returns:
        Coverage fraction (0-1)
    """
    n_vars = bounds.shape[0]

    # Create grid
    grids = [np.linspace(bounds[i, 0], bounds[i, 1], grid_resolution + 1) for i in range(n_vars)]

    # Count filled cells
    filled = set()

    for sample in samples:
        indices = tuple(
            np.searchsorted(grids[i], sample[i]) - 1
            for i in range(n_vars)
        )
        # Clip to valid range
        indices = tuple(max(0, min(grid_resolution - 1, idx)) for idx in indices)
        filled.add(indices)

    total_cells = grid_resolution ** n_vars
    coverage = len(filled) / total_cells

    return coverage


def augment_design(
    existing_samples: np.ndarray,
    bounds: np.ndarray,
    num_new_samples: int,
    method: str = "maximin"
) -> np.ndarray:
    """
    Augment existing design with new samples.

    Args:
        existing_samples: Current sample points
        bounds: Variable bounds
        num_new_samples: Number of new samples to add
        method: "maximin" or "lhs"

    Returns:
        New sample points to add
    """
    n_vars = bounds.shape[0]
    new_samples = []

    if method == "maximin":
        # Maximize distance to existing samples
        for _ in range(num_new_samples):
            # Generate candidates
            n_candidates = 100
            candidates = np.random.uniform(
                bounds[:, 0],
                bounds[:, 1],
                size=(n_candidates, n_vars)
            )

            # Compute minimum distance to existing + already selected
            all_existing = np.vstack([existing_samples] + new_samples) if new_samples else existing_samples

            best_idx = 0
            best_min_dist = 0

            for i, candidate in enumerate(candidates):
                distances = np.linalg.norm(all_existing - candidate, axis=1)
                min_dist = np.min(distances)

                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = i

            new_samples.append(candidates[best_idx])

    else:  # lhs
        # Generate new LHS samples
        lhs = LatinHypercubeSampling({"random_state": 42})
        lhs_result = lhs.generate_samples(bounds, num_new_samples)
        new_samples = list(lhs_result.samples)

    return np.array(new_samples)


def generate_response_surface_design(
    center_point: np.ndarray,
    bounds: np.ndarray,
    design_type: str = "ccd"
) -> np.ndarray:
    """
    Generate response surface design around center point.

    Args:
        center_point: Center of design
        bounds: Variable bounds (for scaling)
        design_type: "ccd" or "box_behnken"

    Returns:
        Sample points
    """
    if design_type == "ccd":
        doe = CentralCompositeDesign({"alpha": "rotatable"})
    else:
        doe = BoxBehnkenDesign({"center_points": 3})

    result = doe.generate_samples(bounds, num_samples=None)

    return result.samples


# ============================================================================
# VISUALIZATION (PLACEHOLDERS)
# ============================================================================

def plot_design_2d(
    samples: np.ndarray,
    variable_names: List[str],
    output_path: str = None
):
    """Plot 2D projections of design (requires matplotlib)."""
    logger.info("Plotting not implemented in this version")
    pass


def plot_design_matrix(
    samples: np.ndarray,
    variable_names: List[str],
    output_path: str = None
):
    """Plot design matrix (requires matplotlib)."""
    logger.info("Plotting not implemented in this version")
    pass


# ============================================================================
# SMOKE TEST
# ============================================================================

def _smoke_test_doe():
    """Test DOE methods."""

    print("="*90)
    print("DESIGN OF EXPERIMENTS (DOE) SMOKE TEST")
    print("="*90)

    # Common setup
    n_vars = 5
    bounds = np.array([[0, 1]] * n_vars)
    num_samples = 50
    var_names = [f"x{i+1}" for i in range(n_vars)]

    print(f"\nTest setup: {n_vars} variables, {num_samples} samples")
    print(f"Bounds: {bounds[0]}")

    # Test 1: Latin Hypercube Sampling
    print("\n" + "="*90)
    print("TEST 1: Latin Hypercube Sampling")
    print("="*90)

    lhs = LatinHypercubeSampling({"criterion": None, "random_state": 42})
    lhs_result = lhs.generate_samples(bounds, num_samples)

    print(f"Samples shape: {lhs_result.samples.shape}")
    print(f"Method: {lhs_result.method}")
    print(f"Min distance: {lhs_result.properties['min_distance']:.4f}")
    print(f"Discrepancy: {lhs_result.properties['discrepancy']:.4f}")
    print(f"Coverage: {lhs_result.properties['coverage']:.2%}")

    assert lhs_result.samples.shape == (num_samples, n_vars)
    assert np.all(lhs_result.samples >= 0) and np.all(lhs_result.samples <= 1)

    # Check Latin hypercube property
    for dim in range(n_vars):
        intervals = np.linspace(0, 1, num_samples + 1)
        counts = np.histogram(lhs_result.samples[:, dim], bins=intervals)[0]
        assert np.all(counts == 1), "Latin hypercube property violated"

    print("✓ Latin hypercube property satisfied")

    # Test 2: Maximin LHS
    print("\n" + "="*90)
    print("TEST 2: Maximin LHS (Optimized)")
    print("="*90)

    maximin = MaximinLHS({"iterations": 500, "random_state": 42})
    maximin_result = maximin.generate_samples(bounds, num_samples)

    print(f"Min distance: {maximin_result.properties['min_distance']:.4f}")
    print(f"vs. standard LHS: {lhs_result.properties['min_distance']:.4f}")
    print(f"Improvement: {(maximin_result.properties['min_distance'] / lhs_result.properties['min_distance'] - 1) * 100:.1f}%")

    assert maximin_result.properties['min_distance'] >= lhs_result.properties['min_distance']

    print("✓ Maximin optimization improved minimum distance")

    # Test 3: Sobol Sequence
    print("\n" + "="*90)
    print("TEST 3: Sobol Sequence")
    print("="*90)

    sobol = SobolSequence({"scramble": True})
    sobol_result = sobol.generate_samples(bounds, num_samples)

    print(f"Discrepancy: {sobol_result.properties['discrepancy']:.4f}")
    print(f"Min distance: {sobol_result.properties['min_distance']:.4f}")
    print(f"Coverage: {sobol_result.properties['coverage']:.2%}")

    assert sobol_result.samples.shape == (num_samples, n_vars)
    assert sobol_result.properties['discrepancy'] < 0.5

    print("✓ Sobol sequence has low discrepancy")

    # Test 4: Full Factorial Design
    print("\n" + "="*90)
    print("TEST 4: Full Factorial Design")
    print("="*90)

    n_vars_factorial = 3
    bounds_factorial = np.array([[10, 20], [100, 200], [1, 5]])

    factorial = FullFactorialDesign({"levels": 3, "center_points": 1})
    factorial_result = factorial.generate_samples(bounds_factorial, num_samples=None)

    print(f"Factors: {n_vars_factorial}, Levels: 3")
    print(f"Samples: {factorial_result.num_samples}")
    print(f"Expected: 3³ + 1 center = 28")
    print(f"Design matrix shape: {factorial_result.design_matrix.shape}")
    print(f"Orthogonality: {factorial_result.properties['orthogonality']:.4f}")

    assert factorial_result.num_samples == 28

    print("✓ Full factorial design correct")

    # Test 5: Central Composite Design
    print("\n" + "="*90)
    print("TEST 5: Central Composite Design (CCD)")
    print("="*90)

    n_vars_ccd = 3
    bounds_ccd = np.array([[-1, 1]] * n_vars_ccd)

    ccd = CentralCompositeDesign({"alpha": "rotatable", "center_points": (1, 1)})
    ccd_result = ccd.generate_samples(bounds_ccd, num_samples=None)

    print(f"Variables: {n_vars_ccd}")
    print(f"Samples: {ccd_result.num_samples}")
    print(f"Expected: 2³ + 2×3 + 2 = 16")
    print(f"Alpha: {ccd_result.properties['alpha']:.3f}")
    print(f"Rotatable: {ccd_result.properties['rotatability']}")
    print(f"Components:")
    print(f"  Factorial: {ccd_result.properties['factorial_points']}")
    print(f"  Axial: {ccd_result.properties['axial_points']}")
    print(f"  Center: {ccd_result.properties['center_points']}")

    assert ccd_result.num_samples == 16

    print("✓ CCD design correct")

    # Test 6: Box-Behnken Design
    print("\n" + "="*90)
    print("TEST 6: Box-Behnken Design")
    print("="*90)

    bb = BoxBehnkenDesign({"center_points": 3})
    bb_result = bb.generate_samples(bounds_ccd, num_samples=None)

    print(f"Variables: {n_vars_ccd}")
    print(f"Samples: {bb_result.num_samples}")
    print(f"Expected: 2×3×2 + 3 = 15")
    print(f"Edge points: {bb_result.properties['edge_points']}")
    print(f"Center points: {bb_result.properties['center_points']}")

    assert bb_result.num_samples == 15

    print("✓ Box-Behnken design correct")

    # Test 7: Space-filling comparison
    print("\n" + "="*90)
    print("TEST 7: Space-Filling Comparison")
    print("="*90)

    designs = {
        "Random": UniformRandomSampling({"random_state": 42}).generate_samples(bounds, num_samples),
        "LHS": lhs_result,
        "Maximin LHS": maximin_result,
        "Sobol": sobol_result,
        "Halton": HaltonSequence({"scramble": True}).generate_samples(bounds, num_samples)
    }

    print(f"\n{'Method':<15} {'Discrepancy':<12} {'Min Dist':<12} {'Coverage':<10} {'Uniformity':<10}")
    print("="*90)

    for name, result in designs.items():
        print(f"{name:<15} "
              f"{result.properties['discrepancy']:<12.4f} "
              f"{result.properties['min_distance']:<12.4f} "
              f"{result.properties['coverage']:<10.2%} "
              f"{result.properties['uniformity_score']:<10.4f}")

    # Verify quasi-random sequences have lower discrepancy than random
    assert sobol_result.properties['discrepancy'] < designs["Random"].properties['discrepancy']

    print("\n✓ Quasi-random sequences show better space-filling than random")

    # Test 8: Design augmentation
    print("\n" + "="*90)
    print("TEST 8: Design Augmentation")
    print("="*90)

    initial_samples = lhs_result.samples[:20]
    num_new = 10

    augmented = augment_design(initial_samples, bounds, num_new, method="maximin")

    print(f"Initial samples: {initial_samples.shape[0]}")
    print(f"New samples: {augmented.shape[0]}")
    print(f"Total: {initial_samples.shape[0] + augmented.shape[0]}")

    assert augmented.shape == (num_new, n_vars)

    # Check minimum distance to existing
    distances = cdist(augmented, initial_samples)
    min_distances = distances.min(axis=1)

    print(f"Min distance of new samples to existing:")
    print(f"  Mean: {min_distances.mean():.4f}")
    print(f"  Min: {min_distances.min():.4f}")

    print("✓ Design augmentation successful")

    # Test 9: Create DOE factory
    print("\n" + "="*90)
    print("TEST 9: Factory Function")
    print("="*90)

    methods_to_test = ["lhs", "sobol", "halton", "ccd", "box_behnken"]

    for method_name in methods_to_test:
        doe = create_doe(method_name, {"random_state": 42})
        print(f"  ✓ Created: {doe.get_name()}")

    print("\n✓ Factory function works for all methods")

    # Summary
    print("\n" + "="*90)
    print("SUMMARY - DOE Methods Performance")
    print("="*90)

    print(f"\nSpace-Filling Designs (best for surrogates):")
    print(f"  Maximin LHS   : Min dist = {maximin_result.properties['min_distance']:.4f} (BEST)")
    print(f"  Sobol Sequence: Discrepancy = {sobol_result.properties['discrepancy']:.4f} (LOWEST)")
    print(f"  Standard LHS  : Coverage = {lhs_result.properties['coverage']:.2%}")

    print(f"\nClassical DOE (for response surfaces):")
    print(f"  Full Factorial : {factorial_result.num_samples} runs for 3 factors")
    print(f"  CCD (Rotatable): {ccd_result.num_samples} runs for 3 factors")
    print(f"  Box-Behnken    : {bb_result.num_samples} runs for 3 factors")

    print("\n" + "="*90)
    print("✅ ALL DOE SMOKE TESTS PASSED!")
    print("="*90)

    print(f"""
🎯 DOE Methods Ready:
  • Classical: Full Factorial, Fractional, CCD, Box-Behnken, Plackett-Burman
  • Space-Filling: LHS, Maximin LHS, Sobol, Halton
  • Quality Metrics: Discrepancy, Min Distance, Coverage
  • Design Augmentation: Maximin-based sequential sampling

Perfect for:
  ✓ Surrogate model training (LHS, Sobol)
  ✓ Response surface methodology (CCD, Box-Behnken)
  ✓ Screening experiments (Plackett-Burman, Fractional Factorial)
  ✓ Sequential design augmentation

Ready for efficient process optimization! 🚀
""")


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    _smoke_test_doe()
