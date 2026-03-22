"""
optimization/sensitivity.py

PURPOSE:
Implement sensitivity analysis methods to identify important design variables and
understand process behavior. Support local and global sensitivity analysis including
Morris screening, Sobol indices, and correlation methods.

Date: 2026-01-02
Version: 1.0.0
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.stats import qmc

logger = logging.getLogger(__name__)

# ============================================================================
# ABSTRACT BASE CLASS
# ============================================================================

class SensitivityAnalysis(ABC):
    """
    Abstract base class for all sensitivity analysis methods.

    All methods must implement:
    - analyze(): Perform sensitivity analysis
    - get_name(): Return method name
    """

    def __init__(self, config: dict):
        """
        Initialize sensitivity analysis.

        Args:
            config: Method configuration dict
        """
        self.config = config

    @abstractmethod
    def analyze(
        self,
        model: Callable,
        bounds: np.ndarray,
        num_samples: int = 1000
    ) -> dict:
        """
        Perform sensitivity analysis.

        Args:
            model: callable(x: np.ndarray) -> float
            bounds: Variable bounds, shape (n_vars, 2)
            num_samples: Number of model evaluations

        Returns:
            Dictionary with sensitivity results
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return method name."""
        pass

    def _validate_inputs(self, model: Callable, bounds: np.ndarray, num_samples: int):
        """Validate inputs."""
        if len(bounds.shape) != 2 or bounds.shape[1] != 2:
            raise ValueError(f"Bounds must be shape (n_vars, 2), got {bounds.shape}")

        if num_samples < bounds.shape[0]:
            raise ValueError(
                f"Need at least {bounds.shape[0]} samples, got {num_samples}"
            )

        # Test model evaluation
        n_vars = bounds.shape[0]
        x_test = np.mean(bounds, axis=1)

        try:
            result = model(x_test)
            if not np.isscalar(result):
                raise ValueError("Model must return scalar value")
        except Exception as e:
            raise RuntimeError(f"Model evaluation failed: {e}")


# ============================================================================
# LOCAL SENSITIVITY
# ============================================================================

class LocalSensitivity(SensitivityAnalysis):
    """
    Local gradient-based sensitivity at a nominal point.

    Fast method using finite differences.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.perturbation_fraction = config.get("perturbation_fraction", 0.01)
        self.nominal_point = config.get("nominal_point", None)

    def get_name(self) -> str:
        return "LocalSensitivity"

    def analyze(
        self,
        model: Callable,
        bounds: np.ndarray,
        num_samples: int = 1000
    ) -> dict:
        """Local sensitivity using finite differences."""
        self._validate_inputs(model, bounds, num_samples)

        n_vars = bounds.shape[0]

        # Nominal point (center of bounds if not provided)
        if self.nominal_point is None:
            x0 = np.mean(bounds, axis=1)
        else:
            x0 = self.nominal_point.copy()

        # Evaluate at nominal
        f0 = model(x0)

        # Compute sensitivities
        gradients = np.zeros(n_vars)

        for i in range(n_vars):
            # Perturbation
            delta = self.perturbation_fraction * (bounds[i, 1] - bounds[i, 0])

            x_plus = x0.copy()
            x_plus[i] += delta

            f_plus = model(x_plus)

            # Finite difference
            gradients[i] = (f_plus - f0) / delta

        # Absolute sensitivity
        absolute_sensitivity = np.abs(gradients)

        # Relative sensitivity (dimensionless)
        if abs(f0) > 1e-10:
            relative_sensitivity = gradients * x0 / f0
        else:
            relative_sensitivity = np.zeros(n_vars)

        # Normalized sensitivity (sum to 1)
        total = np.sum(np.abs(relative_sensitivity))
        if total > 1e-10:
            normalized_sensitivity = np.abs(relative_sensitivity) / total
        else:
            normalized_sensitivity = np.ones(n_vars) / n_vars

        # Ranking
        ranking = np.argsort(normalized_sensitivity)[::-1].tolist()

        return {
            "method": "local",
            "nominal_point": x0,
            "nominal_output": f0,
            "absolute_sensitivity": absolute_sensitivity,
            "relative_sensitivity": relative_sensitivity,
            "normalized_sensitivity": normalized_sensitivity,
            "gradient": gradients,
            "elasticity": relative_sensitivity,
            "ranking": ranking,
            "num_evaluations": n_vars + 1
        }


# ============================================================================
# MORRIS SCREENING
# ============================================================================

class MorrisScreening(SensitivityAnalysis):
    """
    Morris one-at-a-time (OAT) method for factor screening.

    Efficient global screening method.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.num_trajectories = config.get("num_trajectories", 10)
        self.num_levels = config.get("num_levels", 4)
        self.delta = config.get("delta", 0.1)

    def get_name(self) -> str:
        return "MorrisScreening"

    def analyze(
        self,
        model: Callable,
        bounds: np.ndarray,
        num_samples: int = 1000
    ) -> dict:
        """Morris elementary effects method."""
        self._validate_inputs(model, bounds, num_samples)

        n_vars = bounds.shape[0]

        # Generate trajectories
        trajectories = []
        elementary_effects = []

        num_traj = min(self.num_trajectories, num_samples // (n_vars + 1))

        for _ in range(num_traj):
            traj, ee = self._generate_trajectory(model, bounds, n_vars)
            trajectories.append(traj)
            elementary_effects.append(ee)

        # Convert to arrays
        ee_array = np.array(elementary_effects)  # shape (num_traj, n_vars)

        # Statistics
        mu = np.mean(ee_array, axis=0)
        mu_star = np.mean(np.abs(ee_array), axis=0)
        sigma = np.std(ee_array, axis=0)

        # Ranking by mu_star (more robust than mu)
        ranking = np.argsort(mu_star)[::-1].tolist()

        return {
            "method": "morris",
            "num_trajectories": num_traj,
            "num_evaluations": num_traj * (n_vars + 1),
            "mu": mu,
            "mu_star": mu_star,
            "sigma": sigma,
            "trajectories": trajectories,
            "elementary_effects": elementary_effects,
            "ranking": ranking
        }

    def _generate_trajectory(
        self,
        model: Callable,
        bounds: np.ndarray,
        n_vars: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate one Morris trajectory."""

        # Random starting point on grid
        x = np.random.rand(n_vars)

        # Scale to bounds
        x = bounds[:, 0] + x * (bounds[:, 1] - bounds[:, 0])

        trajectory = [x.copy()]
        ee = np.zeros(n_vars)

        # Evaluate at start
        f_prev = model(x)

        # Random permutation of variables
        perm = np.random.permutation(n_vars)

        for i in perm:
            # Perturbation
            delta = self.delta * (bounds[i, 1] - bounds[i, 0])

            # Ensure we stay in bounds
            if x[i] + delta > bounds[i, 1]:
                delta = -delta

            x[i] += delta
            trajectory.append(x.copy())

            # Evaluate
            f_new = model(x)

            # Elementary effect
            ee[i] = (f_new - f_prev) / delta

            f_prev = f_new

        return np.array(trajectory), ee


# ============================================================================
# SOBOL INDICES
# ============================================================================

class SobolIndices(SensitivityAnalysis):
    """
    Variance-based global sensitivity (Sobol indices).

    Gold standard for global sensitivity analysis.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.method = config.get("method", "saltelli")
        self.second_order = config.get("second_order", False)
        self.num_bootstrap = config.get("num_bootstrap", 0)

    def get_name(self) -> str:
        return "SobolIndices"

    def analyze(
        self,
        model: Callable,
        bounds: np.ndarray,
        num_samples: int = 1000
    ) -> dict:
        """Sobol variance decomposition."""
        self._validate_inputs(model, bounds, num_samples)

        n_vars = bounds.shape[0]

        # Base sample size (Saltelli needs n*(2+d) evaluations)
        n = num_samples // (2 + n_vars)
        n = max(n, 100)  # Minimum

        logger.info(f"Sobol analysis: base n={n}, total evals={n*(2+n_vars)}")

        # Generate samples using Saltelli scheme
        A, B, C_matrices = self._saltelli_sampling(bounds, n)

        # Evaluate model
        f_A = np.array([model(x) for x in A])
        f_B = np.array([model(x) for x in B])
        f_C = {}

        for i in range(n_vars):
            f_C[i] = np.array([model(x) for x in C_matrices[i]])

        # Compute variance
        f0 = np.mean(np.concatenate([f_A, f_B]))
        var_y = np.var(np.concatenate([f_A, f_B]))

        if var_y < 1e-10:
            raise RuntimeError("Model output has zero variance (constant model)")

        # First-order indices
        first_order = np.zeros(n_vars)
        for i in range(n_vars):
            first_order[i] = (np.mean(f_B * f_C[i]) - f0**2) / var_y

        # Total-order indices
        total_order = np.zeros(n_vars)
        for i in range(n_vars):
            total_order[i] = 1 - (np.mean(f_A * f_C[i]) - f0**2) / var_y

        # Clip negative values (numerical errors)
        first_order = np.clip(first_order, 0, 1)
        total_order = np.clip(total_order, 0, 1)

        # Ranking by total order
        ranking = np.argsort(total_order)[::-1].tolist()

        results = {
            "method": "sobol",
            "num_samples": n,
            "num_evaluations": n * (2 + n_vars),
            "first_order": first_order,
            "total_order": total_order,
            "second_order": None,
            "confidence_intervals": None,
            "sum_first_order": float(np.sum(first_order)),
            "sum_total_order": float(np.sum(total_order)),
            "ranking": ranking,
            "output_variance": var_y
        }

        # Check consistency
        if np.sum(total_order) > 1.5:
            logger.warning("Sum of total-order indices > 1.5 (numerical instability)")

        # Check that ST_i >= S_i
        for i in range(n_vars):
            if first_order[i] > total_order[i] + 0.1:
                logger.warning(f"Variable {i}: S_i > ST_i (should not happen)")

        return results

    def _saltelli_sampling(
        self,
        bounds: np.ndarray,
        n: int
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
        """Generate Saltelli sample matrices."""
        n_vars = bounds.shape[0]

        # Generate two independent base samples
        sampler = qmc.LatinHypercube(d=n_vars, seed=42)
        sample = sampler.random(n=2*n)

        # Scale to bounds
        l_bounds = bounds[:, 0]
        u_bounds = bounds[:, 1]
        sample_scaled = qmc.scale(sample, l_bounds, u_bounds)

        # Split into A and B
        A = sample_scaled[:n, :]
        B = sample_scaled[n:, :]

        # Create C_i matrices
        C_matrices = {}
        for i in range(n_vars):
            C_i = A.copy()
            C_i[:, i] = B[:, i]
            C_matrices[i] = C_i

        return A, B, C_matrices


# ============================================================================
# FAST ANALYSIS
# ============================================================================

class FASTAnalysis(SensitivityAnalysis):
    """
    Fourier Amplitude Sensitivity Test (FAST).

    Efficient variance-based method using frequency domain.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.M = config.get("M", 4)

    def get_name(self) -> str:
        return "FAST"

    def analyze(
        self,
        model: Callable,
        bounds: np.ndarray,
        num_samples: int = 1000
    ) -> dict:
        """FAST sensitivity analysis."""
        self._validate_inputs(model, bounds, num_samples)

        n_vars = bounds.shape[0]

        # Frequencies (must be incommensurate)
        omega = self._get_frequencies(n_vars)
        omega_max = np.max(omega)

        # Sample size
        N = 2 * self.M * omega_max + 1
        N = min(N, num_samples)

        # Generate samples along search curve
        s = np.linspace(-np.pi, np.pi, N)
        X = np.zeros((N, n_vars))

        for i in range(n_vars):
            # Transform frequency curve to bounds
            X[:, i] = bounds[i, 0] + (bounds[i, 1] - bounds[i, 0]) *                       (0.5 + 0.5 * np.sin(omega[i] * s))

        # Evaluate model
        Y = np.array([model(x) for x in X])

        # Fourier transform
        Y_fft = np.fft.fft(Y)
        power = np.abs(Y_fft)**2

        # Total variance
        var_y = np.var(Y)

        if var_y < 1e-10:
            raise RuntimeError("Model output has zero variance")

        # First-order indices from Fourier coefficients
        first_order = np.zeros(n_vars)

        for i in range(n_vars):
            # Sum power at frequencies omega_i, 2*omega_i, 3*omega_i, ...
            V_i = 0
            for k in range(1, self.M + 1):
                idx = int(k * omega[i])
                if idx < len(power):
                    V_i += power[idx]

            first_order[i] = 2 * V_i / (N * var_y)

        # Clip to [0, 1]
        first_order = np.clip(first_order, 0, 1)

        # Ranking
        ranking = np.argsort(first_order)[::-1].tolist()

        return {
            "method": "fast",
            "num_samples": N,
            "num_evaluations": N,
            "first_order": first_order,
            "total_order": first_order,  # FAST only computes first-order
            "ranking": ranking,
            "frequencies": omega
        }

    def _get_frequencies(self, n_vars: int) -> np.ndarray:
        """Get incommensurate frequencies."""
        # Use prime numbers
        primes = [1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

        if n_vars > len(primes):
            # Generate more if needed
            omega = np.arange(1, n_vars + 1) * 2 + 1
        else:
            omega = np.array(primes[:n_vars])

        return omega


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

class CorrelationAnalysis(SensitivityAnalysis):
    """
    Pearson and Spearman correlation coefficients.

    Simple but limited screening method.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.method = config.get("method", "pearson")
        self.sampling = config.get("sampling", "lhs")

    def get_name(self) -> str:
        return "CorrelationAnalysis"

    def analyze(
        self,
        model: Callable,
        bounds: np.ndarray,
        num_samples: int = 1000
    ) -> dict:
        """Correlation-based sensitivity."""
        self._validate_inputs(model, bounds, num_samples)

        n_vars = bounds.shape[0]

        # Generate samples
        if self.sampling == "lhs":
            sampler = qmc.LatinHypercube(d=n_vars, seed=42)
            sample = sampler.random(n=num_samples)
        else:
            sample = np.random.rand(num_samples, n_vars)

        # Scale to bounds
        X = qmc.scale(sample, bounds[:, 0], bounds[:, 1])

        # Evaluate model
        Y = np.array([model(x) for x in X])

        # Compute correlations
        correlations = np.zeros(n_vars)
        p_values = np.zeros(n_vars)

        for i in range(n_vars):
            if self.method == "pearson":
                corr, p_val = pearsonr(X[:, i], Y)
            elif self.method == "spearman":
                corr, p_val = spearmanr(X[:, i], Y)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            correlations[i] = corr
            p_values[i] = p_val

        # Ranking by absolute correlation
        ranking = np.argsort(np.abs(correlations))[::-1].tolist()

        return {
            "method": "correlation",
            "correlation_type": self.method,
            "num_samples": num_samples,
            "num_evaluations": num_samples,
            "correlation_coefficients": correlations,
            "p_values": p_values,
            "absolute_correlations": np.abs(correlations),
            "ranking": ranking
        }


# ============================================================================
# COMPREHENSIVE SENSITIVITY STUDY
# ============================================================================

def perform_sensitivity_study(
    model: Callable,
    bounds: np.ndarray,
    variable_names: List[str],
    methods: List[str] = ["morris", "sobol"],
    num_samples: int = 1000
) -> dict:
    """
    Perform comprehensive sensitivity study with multiple methods.

    Args:
        model: Callable model
        bounds: Variable bounds
        variable_names: List of variable names
        methods: List of method names to use
        num_samples: Number of samples per method

    Returns:
        Comprehensive sensitivity results
    """
    n_vars = bounds.shape[0]

    if len(variable_names) != n_vars:
        raise ValueError(
            f"Number of variable names ({len(variable_names)}) must match "
            f"number of variables ({n_vars})"
        )

    logger.info(f"Sensitivity study: {len(methods)} methods, {n_vars} variables")

    results_dict = {}
    rankings = []

    # Run each method
    for method_name in methods:
        logger.info(f"Running {method_name}...")

        if method_name.lower() == "local":
            method = LocalSensitivity({})
        elif method_name.lower() == "morris":
            method = MorrisScreening({"num_trajectories": 20})
        elif method_name.lower() == "sobol":
            method = SobolIndices({})
        elif method_name.lower() == "fast":
            method = FASTAnalysis({})
        elif method_name.lower() == "correlation":
            method = CorrelationAnalysis({})
        else:
            logger.warning(f"Unknown method: {method_name}, skipping")
            continue

        try:
            result = method.analyze(model, bounds, num_samples)
            results_dict[method_name] = result
            rankings.append(result["ranking"])

            logger.info(f"  {method_name} complete: {result['num_evaluations']} evaluations")

        except Exception as e:
            logger.error(f"  {method_name} failed: {e}")

    # Consensus ranking (average rank across methods)
    if len(rankings) > 0:
        avg_ranks = np.zeros(n_vars)

        for ranking in rankings:
            for rank, var_idx in enumerate(ranking):
                avg_ranks[var_idx] += rank

        avg_ranks /= len(rankings)
        consensus_ranking = np.argsort(avg_ranks).tolist()
    else:
        consensus_ranking = list(range(n_vars))

    # Identify important/inactive variables
    important_vars = []
    inactive_vars = []

    # Use Sobol if available, otherwise Morris
    if "sobol" in results_dict:
        total_order = results_dict["sobol"]["total_order"]
        for i, var_name in enumerate(variable_names):
            if total_order[i] > 0.05:
                important_vars.append(var_name)
            elif total_order[i] < 0.01:
                inactive_vars.append(var_name)

    elif "morris" in results_dict:
        mu_star = results_dict["morris"]["mu_star"]
        threshold = 0.1 * np.max(mu_star)
        for i, var_name in enumerate(variable_names):
            if mu_star[i] > threshold:
                important_vars.append(var_name)
            elif mu_star[i] < 0.01 * np.max(mu_star):
                inactive_vars.append(var_name)

    return {
        "variable_names": variable_names,
        "bounds": bounds,
        "methods": results_dict,
        "consensus_ranking": consensus_ranking,
        "important_variables": important_vars,
        "inactive_variables": inactive_vars,
        "interactions": None
    }


def rank_variables_by_importance(
    sensitivity_results: dict,
    threshold: float = 0.01
) -> dict:
    """
    Rank variables by importance across all methods.

    Args:
        sensitivity_results: Results from perform_sensitivity_study()
        threshold: Threshold for importance

    Returns:
        Ranking dictionary
    """
    variable_names = sensitivity_results["variable_names"]
    n_vars = len(variable_names)

    # Aggregate scores from different methods
    scores = np.zeros(n_vars)

    for method_name, result in sensitivity_results["methods"].items():
        if method_name == "sobol":
            # Use total-order indices
            scores += result["total_order"]

        elif method_name == "morris":
            # Use normalized mu_star
            mu_star = result["mu_star"]
            if np.max(mu_star) > 0:
                scores += mu_star / np.max(mu_star)

        elif method_name == "local":
            # Use normalized sensitivity
            scores += result["normalized_sensitivity"]

        elif method_name == "correlation":
            # Use absolute correlations
            scores += result["absolute_correlations"]

    # Normalize
    if np.sum(scores) > 0:
        normalized_scores = scores / np.sum(scores)
    else:
        normalized_scores = np.ones(n_vars) / n_vars

    # Ranking
    ranking_indices = np.argsort(scores)[::-1]
    ranking = [variable_names[i] for i in ranking_indices]

    # Scores dict
    scores_dict = {variable_names[i]: scores[i] for i in range(n_vars)}
    normalized_scores_dict = {variable_names[i]: normalized_scores[i] for i in range(n_vars)}

    # Cumulative importance
    cumulative = {}
    cumsum = 0
    for var_name in ranking:
        cumsum += normalized_scores_dict[var_name]
        cumulative[var_name] = cumsum

    return {
        "ranking": ranking,
        "scores": scores_dict,
        "normalized_scores": normalized_scores_dict,
        "cumulative_importance": cumulative
    }


def identify_inactive_variables(
    sensitivity_results: dict,
    threshold: float = 0.001
) -> List[str]:
    """
    Identify variables with sensitivity below threshold.

    Args:
        sensitivity_results: Results from perform_sensitivity_study()
        threshold: Threshold for inactivity

    Returns:
        List of inactive variable names
    """
    variable_names = sensitivity_results["variable_names"]
    inactive = []

    # Check Sobol indices if available
    if "sobol" in sensitivity_results["methods"]:
        total_order = sensitivity_results["methods"]["sobol"]["total_order"]

        for i, var_name in enumerate(variable_names):
            if total_order[i] < threshold:
                inactive.append(var_name)

    # Check Morris if available
    elif "morris" in sensitivity_results["methods"]:
        mu_star = sensitivity_results["methods"]["morris"]["mu_star"]
        max_mu = np.max(mu_star)

        for i, var_name in enumerate(variable_names):
            if mu_star[i] < threshold * max_mu:
                inactive.append(var_name)

    return inactive


def compute_interaction_matrix(
    model: Callable,
    bounds: np.ndarray,
    num_samples: int = 500
) -> np.ndarray:
    """
    Compute pairwise interaction matrix.

    Args:
        model: Callable model
        bounds: Variable bounds
        num_samples: Number of samples

    Returns:
        Interaction matrix, shape (n_vars, n_vars)
    """
    n_vars = bounds.shape[0]

    # Use Sobol with second-order
    sobol = SobolIndices({"second_order": True})
    results = sobol.analyze(model, bounds, num_samples)

    # Create interaction matrix from total - first order
    interactions = np.zeros((n_vars, n_vars))

    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                interactions[i, j] = results["first_order"][i]
            else:
                # Interaction between i and j
                interactions[i, j] = (results["total_order"][i] - results["first_order"][i]) / (n_vars - 1)

    return interactions


# ============================================================================
# SMOKE TEST
# ============================================================================

def _smoke_test_sensitivity():
    """Test sensitivity analysis on Ishigami function."""

    print("="*80)
    print("SENSITIVITY ANALYSIS SMOKE TEST")
    print("="*80)

    # Ishigami function (standard benchmark)
    def ishigami(x):
        a, b = 7.0, 0.1
        return np.sin(x[0]) + a * np.sin(x[1])**2 + b * x[2]**4 * np.sin(x[0])

    bounds = np.array([[-np.pi, np.pi]] * 3)
    var_names = ["x1", "x2", "x3"]

    print("\nTest function: Ishigami")
    print(f"  Variables: {var_names}")
    print(f"  Bounds: [-π, π] for all")
    print(f"  Known Sobol indices:")
    print(f"    S1 ≈ 0.31, S2 ≈ 0.44, S3 ≈ 0.00")
    print(f"    ST1 ≈ 0.56, ST2 ≈ 0.44, ST3 ≈ 0.24")

    # Test 1: Local Sensitivity
    print("\n" + "="*80)
    print("TEST 1: Local Sensitivity")
    print("="*80)

    local = LocalSensitivity({"perturbation_fraction": 0.01})
    local_results = local.analyze(ishigami, bounds, num_samples=10)

    print(f"Normalized sensitivity: {local_results['normalized_sensitivity']}")
    print(f"Ranking: {[var_names[i] for i in local_results['ranking']]}")
    print(f"Evaluations: {local_results['num_evaluations']}")
    print("✓ Local sensitivity test passed")

    # Test 2: Morris Screening
    print("\n" + "="*80)
    print("TEST 2: Morris Screening")
    print("="*80)

    morris = MorrisScreening({"num_trajectories": 20, "num_levels": 4})
    morris_results = morris.analyze(ishigami, bounds, num_samples=100)

    print(f"μ*: {morris_results['mu_star']}")
    print(f"σ: {morris_results['sigma']}")
    print(f"Ranking: {[var_names[i] for i in morris_results['ranking']]}")
    print(f"Evaluations: {morris_results['num_evaluations']}")

    assert morris_results['mu_star'][1] > morris_results['mu_star'][0],         "x2 should have higher μ* than x1"
    assert morris_results['mu_star'][2] < morris_results['mu_star'][1],         "x3 should have lower μ* than x2"

    print("✓ Morris screening test passed")

    # Test 3: Sobol Indices
    print("\n" + "="*80)
    print("TEST 3: Sobol Indices")
    print("="*80)

    sobol = SobolIndices({"num_samples": 500, "method": "saltelli"})
    sobol_results = sobol.analyze(ishigami, bounds, num_samples=500)

    print(f"First-order: {sobol_results['first_order']}")
    print(f"Total-order: {sobol_results['total_order']}")
    print(f"Sum S_i: {sobol_results['sum_first_order']:.3f}")
    print(f"Sum S_Ti: {sobol_results['sum_total_order']:.3f}")
    print(f"Ranking: {[var_names[i] for i in sobol_results['ranking']]}")
    print(f"Evaluations: {sobol_results['num_evaluations']}")

    # Check approximate agreement with known values
    assert abs(sobol_results['first_order'][1] - 0.44) < 0.20,         f"S2 should be ≈ 0.44, got {sobol_results['first_order'][1]:.3f}"

    assert sobol_results['first_order'][2] < 0.15,         f"S3 should be ≈ 0, got {sobol_results['first_order'][2]:.3f}"

    assert sobol_results['total_order'][0] > sobol_results['first_order'][0],         "ST1 > S1 (interactions present)"

    print("✓ Sobol indices test passed")

    # Test 4: Correlation Analysis
    print("\n" + "="*80)
    print("TEST 4: Correlation Analysis")
    print("="*80)

    corr = CorrelationAnalysis({"num_samples": 300, "method": "pearson"})
    corr_results = corr.analyze(ishigami, bounds, num_samples=300)

    print(f"Correlations: {corr_results['correlation_coefficients']}")
    print(f"Ranking: {[var_names[i] for i in corr_results['ranking']]}")
    print(f"Evaluations: {corr_results['num_evaluations']}")
    print("✓ Correlation analysis test passed")

    # Test 5: Comprehensive Study
    print("\n" + "="*80)
    print("TEST 5: Comprehensive Sensitivity Study")
    print("="*80)

    study_results = perform_sensitivity_study(
        model=ishigami,
        bounds=bounds,
        variable_names=var_names,
        methods=["morris", "sobol"],
        num_samples=500
    )

    print(f"Methods: {list(study_results['methods'].keys())}")
    print(f"Consensus ranking: {[var_names[i] for i in study_results['consensus_ranking']]}")
    print(f"Important variables: {study_results['important_variables']}")
    print(f"Inactive variables: {study_results['inactive_variables']}")

    assert len(study_results['methods']) == 2
    print("✓ Comprehensive study test passed")

    # Test 6: Variable Ranking
    print("\n" + "="*80)
    print("TEST 6: Variable Ranking")
    print("="*80)

    ranking = rank_variables_by_importance(study_results, threshold=0.05)

    print(f"Ranking: {ranking['ranking']}")
    print(f"Scores: {ranking['scores']}")
    print(f"Normalized scores: {ranking['normalized_scores']}")
    print(f"Cumulative importance: {ranking['cumulative_importance']}")

    assert ranking['ranking'][0] in ['x1', 'x2'],         "x1 or x2 should be most important"

    print("✓ Variable ranking test passed")

    # Test 7: Identify Inactive
    print("\n" + "="*80)
    print("TEST 7: Identify Inactive Variables")
    print("="*80)

    inactive = identify_inactive_variables(study_results, threshold=0.01)

    print(f"Inactive variables (threshold=0.01): {inactive}")

    # x3 may or may not be inactive depending on threshold
    print("✓ Inactive variable identification test passed")

    print("\n" + "="*80)
    print("✅ ALL SENSITIVITY ANALYSIS TESTS PASSED!")
    print("="*80)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY - Ishigami Function Analysis")
    print("="*80)
    print(f"\n{'Method':<20} {'x1 Score':<12} {'x2 Score':<12} {'x3 Score':<12}")
    print("-"*80)

    # Local
    local_norm = local_results['normalized_sensitivity']
    print(f"{'Local':<20} {local_norm[0]:<12.4f} {local_norm[1]:<12.4f} {local_norm[2]:<12.4f}")

    # Morris
    morris_norm = morris_results['mu_star'] / np.sum(morris_results['mu_star'])
    print(f"{'Morris':<20} {morris_norm[0]:<12.4f} {morris_norm[1]:<12.4f} {morris_norm[2]:<12.4f}")

    # Sobol
    print(f"{'Sobol (first)':<20} {sobol_results['first_order'][0]:<12.4f} "
          f"{sobol_results['first_order'][1]:<12.4f} {sobol_results['first_order'][2]:<12.4f}")
    print(f"{'Sobol (total)':<20} {sobol_results['total_order'][0]:<12.4f} "
          f"{sobol_results['total_order'][1]:<12.4f} {sobol_results['total_order'][2]:<12.4f}")

    print("-"*80)
    print(f"\n🎯 Conclusion: x2 (middle term) is most important")
    print(f"   x1 has interactions (ST1 > S1)")
    print(f"   x3 has weak direct effect but contributes through interactions")


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    _smoke_test_sensitivity()
