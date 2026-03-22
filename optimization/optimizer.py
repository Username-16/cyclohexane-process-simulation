"""
optimization/optimizer.py - FINAL HIGH-PERFORMANCE VERSION v2.0.0

Date: 2026-02-07
Version: 2.0.0 - FINAL HIGH-PERFORMANCE RELEASE
"""

from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from functools import partial

import numpy as np

from optimization.objective_functions import ObjectiveFunction
from optimization.constraints import Constraint
from optimization.algorithms import (
    ParticleSwarmOptimization,
    GeneticAlgorithm,
    DifferentialEvolution,
    SequentialQuadraticProgramming
)
from optimization.surrogate_models import create_surrogate, train_surrogate, validate_surrogate, SurrogateModel
from optimization.sensitivity import MorrisScreening, SobolIndices, LocalSensitivity, FASTAnalysis
from optimization.multiobjective import NSGAII, MultiObjectiveResult, hypervolume_indicator, select_solution_from_pareto
from optimization.design_of_experiments import create_doe, DOEResult
from optimization.uncertainty import propagate_uncertainty, Distribution, UncertaintyResult, RobustOptimization

logger = logging.getLogger(__name__)


# ============================================================================
# MODULE-LEVEL WRAPPERS (REQUIRED FOR WINDOWS MULTIPROCESSING)
# ============================================================================

def _objective_wrapper_module_level(x, flowsheet_evaluator, objective_func, design_variables):
    """
    Module-level objective wrapper for Windows multiprocessing.
    This function MUST be at module level to be picklable on Windows.

    Args:
        x: Design vector (numpy array)
        flowsheet_evaluator: Function to evaluate flowsheet
        objective_func: Objective function instance
        design_variables: Dict mapping variable names to indices

    Returns:
        float: Objective value or penalty (1e10 if failed)
    """
    # Convert vector to dict
    design_dict = {name: x[i] for i, name in enumerate(design_variables.keys())}

    try:
        result = flowsheet_evaluator(design_dict)
        if result is None:
            logger.warning("Flowsheet evaluator returned None")
            return 1e10

        obj_value = objective_func.evaluate(x, result)
        return float(obj_value)

    except Exception as e:
        logger.warning(f"Evaluation failed: {e}")
        return 1e10


def _surrogate_objective_wrapper(x, surrogate_model):
    """
    Module-level surrogate objective wrapper for Windows multiprocessing.

    This function MUST be at module level to be picklable on Windows.

    Args:
        x: Design vector (numpy array)
        surrogate_model: Trained surrogate model

    Returns:
        float: Predicted objective value
    """
    try:
        # Reshape to (1, n_features) for prediction
        x_reshaped = x.reshape(1, -1)
        prediction = surrogate_model.predict(x_reshaped)

        # Return scalar value
        if hasattr(prediction, '__len__'):
            return float(prediction[0])
        else:
            return float(prediction)

    except Exception as e:
        logger.warning(f"Surrogate prediction failed: {e}")
        return 1e10


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class OptimizationConfig:
    """
    Comprehensive configuration for optimization workflow.
    """

    # Algorithm selection
    algorithm: str = "pso"  # pso, de, ga, sqp, nsga2, auto

    # Surrogate modeling
    use_surrogates: bool = False
    surrogate_type: str = "gpr"  # gpr, rf, gbr, svr
    initial_samples: int = 30
    adaptive_sampling: bool = True

    # Sensitivity analysis
    perform_sensitivity: bool = False
    sensitivity_methods: List[str] = field(default_factory=lambda: ["morris"])
    dimension_reduction: bool = False
    sensitivity_threshold: float = 0.01

    # Optimization parameters
    max_iterations: int = 30
    max_evaluations: int = 1000
    parallel: bool = True
    num_workers: int = 15

    # Convergence criteria
    tolerance_objective: float = 1e-6
    tolerance_constraint: float = 1e-6
    tolerance_design: float = 1e-6
    convergence_window: int = 10

    # Multi-objective optimization
    pareto_population_size: int = 50
    pareto_generations: int = 30

    # Uncertainty quantification
    uncertainty_analysis: bool = False
    uncertainty_method: str = "lhs_mc"  # lhs_mc, sobol, pce
    uncertainty_samples: int = 1000

    # Validation
    validate_optimum: bool = True
    validation_runs: int = 3

    # Output and logging
    save_history: bool = True
    save_surrogate: bool = True
    verbose: bool = True
    output_dir: str = "./optimization_results"

    # Algorithm-specific parameters
    pso_population_size: int = 24
    pso_inertia: float = 0.7
    pso_cognitive: float = 1.5
    pso_social: float = 1.5

    de_population_size: int = 30
    de_mutation_factor: float = 0.8
    de_crossover_rate: float = 0.9

    ga_population_size: int = 50
    ga_mutation_rate: float = 0.1
    ga_crossover_rate: float = 0.8


# ============================================================================
# OPTIMIZATION RESULTS
# ============================================================================

@dataclass
class OptimizationResult:
    """Comprehensive optimization results container."""

    # Status
    success: bool
    message: str

    # Optimal solution
    optimal_design: np.ndarray
    optimal_design_dict: dict
    optimal_objective: float | np.ndarray
    constraint_violations: dict

    # Solution quality
    optimality_gap: float = None
    constraint_satisfaction: bool = True
    validation_error: float = None

    # Convergence history
    num_iterations: int = 0
    num_evaluations: int = 0
    convergence_history: dict = field(default_factory=dict)

    # Analysis results
    sensitivity_results: dict = None
    uncertainty_results: UncertaintyResult = None
    pareto_front: np.ndarray = None

    # Surrogate model
    surrogate_model: SurrogateModel = None
    surrogate_accuracy: dict = None

    # Execution metrics
    execution_time_seconds: float = 0.0
    algorithm_used: str = ""

    # Metadata
    config: OptimizationConfig = None
    timestamp: str = ""
    hypervolume: float = None

    def __getitem__(self, key: str):
        """Allow dict-like access to result attributes."""
        return getattr(self, key)

    def __repr__(self):
        return (f"OptimizationResult(success={self.success}, "
                f"algorithm={self.algorithm_used}, "
                f"objective={self.optimal_objective:.4f}, "
                f"evaluations={self.num_evaluations})")


# ============================================================================
# PROCESS OPTIMIZER (MAIN CLASS)
# ============================================================================

class ProcessOptimizer:
    """
    Main optimization coordinator for process design.
    """

    def __init__(self, config: OptimizationConfig | dict):
        """
        Initialize process optimizer.

        Args:
            config: OptimizationConfig instance or dict of configuration parameters
        """
        if isinstance(config, dict):
            self.config = OptimizationConfig(**config)
        else:
            self.config = config

        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tracking
        self.evaluation_count = 0
        self.failure_count = 0
        self.callback = None
        self.pretrained_surrogate = None

        logger.info("=" * 80)
        logger.info("ProcessOptimizer initialized - HIGH PERFORMANCE MODE")
        logger.info(f"  Parallel processing: {self.config.parallel}")
        logger.info(f"  Workers: {self.config.num_workers}")
        logger.info(f"  Algorithm: {self.config.algorithm.upper()}")
        logger.info("=" * 80)

    def set_callback(self, callback: Callable):
        """
        Set callback function for optimization progress tracking.

        Args:
            callback: Function called after each iteration
                     Signature: callback(iteration, design, objective, constraints) -> bool
        """
        self.callback = callback
        logger.info("Progress callback registered")

    def optimize(
        self,
        flowsheet_evaluator: Callable,
        objectives: List[ObjectiveFunction] | ObjectiveFunction,
        constraints: List[Constraint],
        design_variables: dict,
        bounds: np.ndarray
    ) -> OptimizationResult:
        """
        Main optimization method with complete workflow.

        Args:
            flowsheet_evaluator: Function to evaluate flowsheet simulation
            objectives: Single or list of objective functions
            constraints: List of constraint functions
            design_variables: Dict of design variable definitions
            bounds: Numpy array of variable bounds (n_vars, 2)

        Returns:
            OptimizationResult: Comprehensive results object
        """
        start_time = time.time()

        logger.info("=" * 80)
        logger.info("STARTING OPTIMIZATION CAMPAIGN")
        logger.info("=" * 80)

        # Validate inputs
        self._validate_inputs(objectives, constraints, design_variables, bounds)

        # Convert single objective to list
        if isinstance(objectives, ObjectiveFunction):
            objectives = [objectives]
            is_multiobjective = False
        else:
            is_multiobjective = len(objectives) > 1

        # Initialize tracking
        convergence_history = {
            "iteration": [],
            "evaluation": [],
            "objective": [],
            "constraint_violation": [],
            "time": []
        }

        # Pre-optimization analysis
        sensitivity_results = None
        active_bounds = bounds.copy()
        active_variables = list(design_variables.keys())

        if self.config.perform_sensitivity and not is_multiobjective:
            logger.info("Performing sensitivity analysis...")
            sensitivity_results = self._run_sensitivity_analysis(
                flowsheet_evaluator, objectives[0], design_variables, bounds
            )

            if self.config.dimension_reduction:
                active_variables, active_bounds = self._reduce_dimensions(
                    sensitivity_results, design_variables, bounds
                )

        # Main optimization
        if is_multiobjective:
            logger.info("Multi-objective optimization detected")
            result = self._optimize_multiobjective(
                flowsheet_evaluator, objectives, constraints,
                design_variables, bounds, convergence_history
            )

        elif self.config.use_surrogates and self.config.max_iterations > 10:
            logger.info("Surrogate-assisted optimization")
            result = self._optimize_with_surrogate(
                flowsheet_evaluator, objectives[0], constraints,
                design_variables, active_bounds, convergence_history
            )

        else:
            if self.config.use_surrogates and self.config.max_iterations <= 10:
                logger.warning(
                    f"Surrogates disabled: not worth overhead for {self.config.max_iterations} iterations"
                )

            logger.info("Direct high-performance optimization")
            result = self._optimize_direct(
                flowsheet_evaluator, objectives[0], constraints,
                design_variables, active_bounds, convergence_history
            )

        # Post-optimization analysis
        uncertainty_results = None
        if self.config.uncertainty_analysis and result.success:
            logger.info("Performing uncertainty analysis at optimum...")
            uncertainty_results = self._analyze_uncertainty_at_optimum(
                flowsheet_evaluator, result.optimal_design_dict
            )

        # Finalize result
        result.sensitivity_results = sensitivity_results
        result.uncertainty_results = uncertainty_results
        result.execution_time_seconds = time.time() - start_time
        result.timestamp = datetime.now().isoformat()
        result.config = self.config

        logger.info("=" * 80)
        logger.info(f"OPTIMIZATION COMPLETE: {result.message}")
        logger.info(f"  Time: {result.execution_time_seconds:.2f}s")
        logger.info(f"  Evaluations: {result.num_evaluations}")
        logger.info(f"  Success: {result.success}")
        logger.info("=" * 80)

        return result

    def _optimize_direct(
        self,
        flowsheet_evaluator: Callable,
        objective: ObjectiveFunction,
        constraints: List[Constraint],
        design_variables: dict,
        bounds: np.ndarray,
        convergence_history: dict
    ) -> OptimizationResult:
        """
        Direct optimization without surrogates.
        """
        algorithm = self._select_algorithm()
        logger.info(f"Using algorithm: {algorithm.upper()}")
        logger.info(f"  Parallel: {self.config.parallel}")
        logger.info(f"  Workers: {self.config.num_workers}")

        # Create optimizer with optimized config
        if algorithm == "pso":
            algo_config = {
                "max_iterations": self.config.max_iterations,
                "pso": {
                    "population_size": self.config.pso_population_size,
                    "inertia_weight": self.config.pso_inertia,
                    "cognitive_coeff": self.config.pso_cognitive,
                    "social_coeff": self.config.pso_social,
                    "velocity_limit": 0.5
                },
                "verbose": self.config.verbose,
                "parallel": self.config.parallel,
                "num_workers": self.config.num_workers
            }
            optimizer = ParticleSwarmOptimization(algo_config)

        elif algorithm == "ga":
            algo_config = {
                "max_iterations": self.config.max_iterations,
                "ga": {
                    "population_size": self.config.ga_population_size,
                    "mutation_rate": self.config.ga_mutation_rate,
                    "crossover_rate": self.config.ga_crossover_rate
                },
                "verbose": self.config.verbose,
                "parallel": self.config.parallel,
                "num_workers": self.config.num_workers
            }
            optimizer = GeneticAlgorithm(algo_config)

        elif algorithm == "de" or algorithm == "differential_evolution":
            algo_config = {
                "max_iterations": self.config.max_iterations,
                "de": {
                    "population_size": self.config.de_population_size,
                    "mutation_factor": self.config.de_mutation_factor,
                    "crossover_rate": self.config.de_crossover_rate,
                    "strategy": "best1bin"
                },
                "verbose": self.config.verbose,
                "parallel": self.config.parallel,
                "num_workers": self.config.num_workers
            }
            optimizer = DifferentialEvolution(algo_config)

        else:  # sqp
            algo_config = {
                "max_iterations": self.config.max_iterations,
                "verbose": self.config.verbose
            }
            optimizer = SequentialQuadraticProgramming(algo_config)

        # Pass callback to algorithm if set
        if hasattr(self, 'callback') and self.callback is not None:
            optimizer.set_callback(self.callback)
            logger.info("Callback attached to algorithm")

        objective_wrapper = partial(
            _objective_wrapper_module_level,
            flowsheet_evaluator=flowsheet_evaluator,
            objective_func=objective,
            design_variables=design_variables
        )

        # Run optimization
        logger.info("Starting optimization...")
        opt_result = optimizer.optimize(objective_wrapper, constraints, bounds)

        if isinstance(opt_result, dict):
            self.evaluation_count = opt_result.get('num_function_evaluations', 0)
            optimal_design = opt_result.get("optimal_design", np.mean(bounds, axis=1))
            success = opt_result.get("converged", False)
            optimal_objective = opt_result.get("optimal_objective", 1e10)
            num_iterations = opt_result.get("num_iterations", 0)
        else:
            self.evaluation_count = getattr(opt_result, 'num_function_evaluations', 0)
            optimal_design = getattr(opt_result, 'optimal_design', np.mean(bounds, axis=1))
            success = getattr(opt_result, 'success', False)
            optimal_objective = getattr(opt_result, 'optimal_objective', 1e10)
            num_iterations = getattr(opt_result, 'num_iterations', 0)

        optimal_design_dict = self._vector_to_dict(optimal_design, design_variables)

        logger.info(f"Optimization complete:")
        logger.info(f"  Iterations: {num_iterations}")
        logger.info(f"  Evaluations: {self.evaluation_count}")
        logger.info(f"  Optimal objective: {optimal_objective:.6f}")

        return OptimizationResult(
            success=success,
            message=f"Optimization complete using {algorithm.upper()}",
            optimal_design=optimal_design,
            optimal_design_dict=optimal_design_dict,
            optimal_objective=optimal_objective,
            constraint_violations={},
            optimality_gap=None,
            constraint_satisfaction=True,
            validation_error=None,
            num_iterations=num_iterations,
            num_evaluations=self.evaluation_count,
            convergence_history=convergence_history,
            algorithm_used=algorithm.upper()
        )

    def _optimize_with_surrogate(
        self,
        flowsheet_evaluator: Callable,
        objective: ObjectiveFunction,
        constraints: List[Constraint],
        design_variables: dict,
        bounds: np.ndarray,
        convergence_history: dict
    ) -> OptimizationResult:
        """Optimize with surrogate model (with pretrained support)."""
        return self._optimize_with_surrogates(
            flowsheet_evaluator,
            objective,
            constraints,
            design_variables,
            bounds,
            initial_samples=self.config.initial_samples
        )

    def _optimize_with_surrogates(
        self,
        flowsheet_evaluator: Callable,
        objective: ObjectiveFunction,
        constraints: List[Constraint],
        design_variables: dict,
        bounds: np.ndarray,
        initial_samples: int = 30,
        adaptive_samples: int = 20,
        validation_samples: int = 5
    ) -> OptimizationResult:
        """
        Surrogate-assisted optimization with aggressive sampling.

        Supports pretrained surrogates for faster optimization.
        """
        start_time = time.time()
        logger.info("Surrogate-assisted optimization")

        # Check for pretrained surrogate
        if hasattr(self, 'pretrained_surrogate') and self.pretrained_surrogate is not None:
            logger.info("Using pre-trained surrogate model (skipping DOE)")
            surrogate = self.pretrained_surrogate

            # Optimize directly on pretrained surrogate
            surr_objective = partial(_surrogate_objective_wrapper, surrogate_model=surrogate)

            pso_config = {
                "max_iterations": 100,
                "pso": {
                    "population_size": 30,
                    "inertia_weight": 0.7,
                    "cognitive_coeff": 1.5,
                    "social_coeff": 1.5
                },
                "verbose": False,
                "parallel": False  # Surrogate is fast, no need for parallel
            }

            pso = ParticleSwarmOptimization(pso_config)
            opt_result = pso.optimize(surr_objective, [], bounds)

            if isinstance(opt_result, dict):
                optimal_design = opt_result.get("optimal_design", np.mean(bounds, axis=1))
                optimal_objective_val = opt_result.get("optimal_objective", 1e10)
            else:
                optimal_design = getattr(opt_result, 'optimal_design', np.mean(bounds, axis=1))
                optimal_objective_val = getattr(opt_result, 'optimal_objective', 1e10)

            optimal_design_dict = self._vector_to_dict(optimal_design, design_variables)

            # Validate with high-fidelity
            validation_results = []
            for _ in range(validation_samples):
                try:
                    result = flowsheet_evaluator(optimal_design_dict)
                    obj_value = objective.evaluate(optimal_design, result)
                    validation_results.append(obj_value)
                except:
                    pass

            if validation_results:
                validated_objective = np.mean(validation_results)
                surrogate_prediction = surrogate.predict(optimal_design.reshape(1, -1))[0]
                validation_error = abs(validated_objective - surrogate_prediction) / abs(validated_objective)
            else:
                validated_objective = optimal_objective_val
                validation_error = None

            return OptimizationResult(
                success=True,
                message="Surrogate-assisted optimization complete (pretrained)",
                optimal_design=optimal_design,
                optimal_design_dict=optimal_design_dict,
                optimal_objective=validated_objective,
                constraint_violations={},
                num_iterations=100,
                num_evaluations=len(validation_results),
                surrogate_model=surrogate,
                surrogate_accuracy={"pretrained": True},
                execution_time_seconds=time.time() - start_time,
                algorithm_used="PSO+PretrainedSurrogate"
            )

        # No pretrained surrogate - train new one
        logger.info("No pretrained surrogate - training new model")
        logger.info(f"  Initial DOE samples: {initial_samples}")

        # Generate initial DOE
        doe = create_doe("lhs", config={"random_state": 42})
        doe_result = doe.generate_samples(bounds, initial_samples)

        # Evaluate at DOE points
        X_train = []
        y_train = []

        logger.info("Evaluating DOE samples...")
        for i, sample in enumerate(doe_result.samples, 1):
            design_dict = self._vector_to_dict(sample, design_variables)
            try:
                result = flowsheet_evaluator(design_dict)
                obj_value = objective.evaluate(sample, result)
                X_train.append(sample)
                y_train.append(obj_value)

                if i % 10 == 0:
                    logger.info(f"  Progress: {i}/{initial_samples}")
            except Exception as e:
                logger.warning(f"Evaluation {i} failed: {e}")
                self.failure_count += 1

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        logger.info(f"Initial DOE: {len(y_train)}/{initial_samples} successful")

        # Train surrogate
        logger.info(f"Training {self.config.surrogate_type.upper()} surrogate...")
        surrogate = train_surrogate(
            X_train, y_train,
            model_type=self.config.surrogate_type,
            config={"n_estimators": 100, "n_restarts_optimizer": 5}
        )

        # Validate surrogate
        validation_metrics = validate_surrogate(
            surrogate, X_train, y_train,
            bounds=bounds,
            num_test_samples=20
        )

        logger.info(f"Surrogate trained: R² = {validation_metrics['r2_score']:.4f}")

        # Optimize on surrogate
        surr_objective = partial(_surrogate_objective_wrapper, surrogate_model=surrogate)

        pso_config = {
            "max_iterations": 100,
            "pso": {
                "population_size": 30,
                "inertia_weight": 0.7,
                "cognitive_coeff": 1.5,
                "social_coeff": 1.5
            },
            "verbose": False,
            "parallel": False  # Surrogate is fast
        }

        pso = ParticleSwarmOptimization(pso_config)
        opt_result = pso.optimize(surr_objective, [], bounds)

        if isinstance(opt_result, dict):
            optimal_design = opt_result.get("optimal_design", np.mean(bounds, axis=1))
            optimal_objective_val = opt_result.get("optimal_objective", 1e10)
        else:
            optimal_design = getattr(opt_result, 'optimal_design', np.mean(bounds, axis=1))
            optimal_objective_val = getattr(opt_result, 'optimal_objective', 1e10)

        optimal_design_dict = self._vector_to_dict(optimal_design, design_variables)

        # Validate with high-fidelity
        validation_results = []
        for _ in range(validation_samples):
            try:
                result = flowsheet_evaluator(optimal_design_dict)
                obj_value = objective.evaluate(optimal_design, result)
                validation_results.append(obj_value)
            except:
                pass

        if validation_results:
            validated_objective = np.mean(validation_results)
            surrogate_prediction = surrogate.predict(optimal_design.reshape(1, -1))[0]
            validation_error = abs(validated_objective - surrogate_prediction) / abs(validated_objective)
        else:
            validated_objective = optimal_objective_val
            validation_error = None

        logger.info(f"Validation: Surrogate={optimal_objective_val:.4f}, True={validated_objective:.4f}")

        return OptimizationResult(
            success=True,
            message="Surrogate-assisted optimization complete",
            optimal_design=optimal_design,
            optimal_design_dict=optimal_design_dict,
            optimal_objective=validated_objective,
            constraint_violations={},
            validation_error=validation_error,
            num_iterations=100,
            num_evaluations=len(y_train) + len(validation_results),
            surrogate_model=surrogate,
            surrogate_accuracy=validation_metrics,
            execution_time_seconds=time.time() - start_time,
            algorithm_used="PSO+Surrogate"
        )

    def _optimize_multiobjective(
        self,
        flowsheet_evaluator: Callable,
        objectives: List[ObjectiveFunction],
        constraints: List[Constraint],
        design_variables: dict,
        bounds: np.ndarray,
        convergence_history: dict
    ) -> OptimizationResult:
        """Multi-objective optimization using NSGA-II."""
        return self._multiobjective_optimize(
            flowsheet_evaluator,
            objectives,
            constraints,
            design_variables,
            bounds,
            population_size=self.config.pareto_population_size
        )

    def _multiobjective_optimize(
        self,
        flowsheet_evaluator: Callable,
        objectives: List[ObjectiveFunction],
        constraints: List[Constraint],
        design_variables: dict,
        bounds: np.ndarray,
        population_size: int = 50
    ) -> OptimizationResult:
        """
        Multi-objective Pareto optimization using NSGA-II.

        Finds Pareto-optimal trade-off solutions.
        """
        start_time = time.time()
        logger.info(f"Multi-objective optimization: {len(objectives)} objectives")

        try:
            # Wrapper class to bridge flowsheet evaluation
            class WrappedObjective:
                """Wrapper that includes flowsheet evaluation."""
                def __init__(self, inner_obj, evaluator, design_vars):
                    self.inner_obj = inner_obj
                    self.evaluator = evaluator
                    self.design_vars = design_vars

                def evaluate(self, x, flowsheet_results=None):
                    """Evaluate with flowsheet simulation."""
                    design_dict = {name: x[i] for i, name in enumerate(self.design_vars.keys())}
                    try:
                        results = self.evaluator(design_dict)
                        return self.inner_obj.evaluate(x, results)
                    except Exception as e:
                        logger.warning(f"Objective {self.inner_obj.get_name()} failed: {e}")
                        return 1e10

                def get_name(self):
                    return self.inner_obj.get_name()

                def get_direction(self):
                    return self.inner_obj.get_direction()

            # Wrap all objectives
            wrapped_objectives = [
                WrappedObjective(obj, flowsheet_evaluator, design_variables)
                for obj in objectives
            ]

            # Run NSGA-II
            nsga2_config = {
                "population_size": population_size,
                "num_generations": self.config.pareto_generations,
                "crossover_probability": 0.9,
                "crossover_eta": 20.0,
                "mutation_eta": 20.0,
                "save_history": True
            }

            nsga2 = NSGAII(nsga2_config)
            logger.info("Running NSGA-II...")

            moo_result = nsga2.optimize(
                objective_functions=wrapped_objectives,
                constraints=constraints,
                bounds=bounds,
                population_size=population_size
            )

            logger.info(f"Pareto front found: {len(moo_result.pareto_front)} solutions")

            # Select representative solution (knee point)
            optimal_design, optimal_objectives = select_solution_from_pareto(
                moo_result.pareto_front,
                moo_result.pareto_set,
                selection_method="knee"
            )

            optimal_design_dict = self._vector_to_dict(optimal_design, design_variables)

            # Compute hypervolume
            reference_point = np.max(moo_result.pareto_front, axis=0) * 1.1
            hv = hypervolume_indicator(moo_result.pareto_front, reference_point)

            execution_time = time.time() - start_time
            logger.info(f"Multi-objective optimization complete in {execution_time:.2f}s")

            return OptimizationResult(
                success=True,
                message=f"Pareto front found: {len(moo_result.pareto_front)} solutions",
                optimal_design=optimal_design,
                optimal_design_dict=optimal_design_dict,
                optimal_objective=optimal_objectives,
                constraint_violations={},
                num_iterations=moo_result.num_generations,
                num_evaluations=moo_result.num_evaluations,
                pareto_front=moo_result.pareto_front,
                execution_time_seconds=execution_time,
                algorithm_used="NSGA-II",
                hypervolume=hv
            )

        except Exception as e:
            logger.error(f"Multi-objective optimization failed: {e}", exc_info=True)

            return OptimizationResult(
                success=False,
                message=f"Multi-objective optimization failed: {str(e)}",
                optimal_design=np.mean(bounds, axis=1),
                optimal_design_dict=self._vector_to_dict(np.mean(bounds, axis=1), design_variables),
                optimal_objective=np.array([1e10] * len(objectives)),
                constraint_violations={},
                num_iterations=0,
                num_evaluations=0,
                execution_time_seconds=time.time() - start_time,
                algorithm_used="NSGA-II"
            )

    def _run_sensitivity_analysis(
        self,
        flowsheet_evaluator: Callable,
        objective: ObjectiveFunction,
        design_variables: dict,
        bounds: np.ndarray
    ) -> dict:
        """Run sensitivity analysis using Morris screening."""
        logger.info("Running Morris sensitivity analysis...")

        morris = MorrisScreening(config={"num_trajectories": 10})

        def model_wrapper(x):
            design_dict = self._vector_to_dict(x, design_variables)
            try:
                result = flowsheet_evaluator(design_dict)
                return objective.evaluate(x, result)
            except:
                return 1e10

        morris_result = morris.analyze(model_wrapper, bounds)

        logger.info("Sensitivity analysis complete")

        return {
            "method": "Morris",
            "mu_star": morris_result["mu_star"],
            "sigma": morris_result["sigma"],
            "variable_names": list(design_variables.keys())
        }

    def _reduce_dimensions(
        self,
        sensitivity_results: dict,
        design_variables: dict,
        bounds: np.ndarray
    ) -> Tuple[List[str], np.ndarray]:
        """Reduce dimensions based on sensitivity analysis."""
        mu_star = sensitivity_results["mu_star"]
        var_names = sensitivity_results["variable_names"]

        # Identify important variables
        threshold = self.config.sensitivity_threshold * np.max(mu_star)
        important_idx = np.where(mu_star >= threshold)[0]

        active_variables = [var_names[i] for i in important_idx]
        active_bounds = bounds[important_idx]

        logger.info(f"Dimension reduction: {len(active_variables)}/{len(var_names)} variables active")

        return active_variables, active_bounds

    def _analyze_uncertainty_at_optimum(
        self,
        flowsheet_evaluator: Callable,
        optimal_design_dict: dict
    ) -> UncertaintyResult:
        """Analyze uncertainty at optimal point."""
        logger.info("Analyzing uncertainty at optimum...")

        uncertain_params = {}
        for var, value in optimal_design_dict.items():
            uncertain_params[var] = Distribution(
                "normal",
                mean=value,
                std=abs(value) * 0.05  # 5% uncertainty
            )

        def model_wrapper(params):
            try:
                result = flowsheet_evaluator(params)
                return {"output": result.get("objective", 0)}
            except:
                return {"output": 0}

        unc_result = propagate_uncertainty(
            model_wrapper,
            uncertain_params,
            method=self.config.uncertainty_method,
            num_samples=self.config.uncertainty_samples
        )

        logger.info("Uncertainty analysis complete")

        return unc_result

    def _select_algorithm(self) -> str:
        """Automatic algorithm selection based on problem characteristics."""
        if self.config.algorithm != "auto":
            return self.config.algorithm.lower()

        logger.info("Auto-selecting PSO for general optimization")
        return "pso"

    def _vector_to_dict(self, x: np.ndarray, design_variables: dict) -> dict:
        """Convert design vector to named dict."""
        return {name: x[i] for i, name in enumerate(design_variables.keys())}

    def _validate_inputs(self, objectives, constraints, design_variables, bounds):
        """Validate optimization inputs."""
        if not objectives:
            raise ValueError("No objectives specified")

        n_vars = len(design_variables)
        if bounds.shape != (n_vars, 2):
            raise ValueError(f"Bounds shape {bounds.shape} doesn't match {n_vars} variables")

        logger.info(f"Validated: {n_vars} variables, {len(constraints)} constraints")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_optimizer(config: dict | OptimizationConfig) -> ProcessOptimizer:
    """
    Create optimizer with configuration.

    Args:
        config: Configuration dict or OptimizationConfig instance

    Returns:
        ProcessOptimizer: Configured optimizer instance
    """
    return ProcessOptimizer(config)


def save_optimization_results(result: OptimizationResult, filepath: str):
    """
    Save optimization results to file.

    Args:
        result: OptimizationResult instance
        filepath: Path to save file
    """
    with open(filepath, 'wb') as f:
        pickle.dump(result, f)
    logger.info(f"Results saved to {filepath}")


def load_optimization_results(filepath: str) -> OptimizationResult:
    """
    Load optimization results from file.

    Args:
        filepath: Path to saved results

    Returns:
        OptimizationResult: Loaded results
    """
    with open(filepath, 'rb') as f:
        result = pickle.load(f)
    logger.info(f"Results loaded from {filepath}")
    return result


# ============================================================================
# MODULE INFO
# ============================================================================

__version__ = "2.0.0"
__author__ = "KSU Chemical Engineering Department"
__status__ = "Production"

logger.info(f"optimizer.py v{__version__} loaded - HIGH PERFORMANCE MODE")
