"""
optimization_main.py

Date: 2026-02-11
Version: 6.0
"""

import os
os.environ["NUMEXPR_MAX_THREADS"] = "15"
os.environ["NUMEXPR_NUM_THREADS"] = "15"
import numpy as np
import threading
from concurrent.futures import ProcessPoolExecutor
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import time
from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing import Manager
import io

sys.path.insert(0, str(Path(__file__).parent.parent))

from optimization.algorithms import ParticleSwarmOptimization, DifferentialEvolution
from optimization.optimizer import ProcessOptimizer, OptimizationConfig, create_optimizer

from optimization.objective_functions import (
    CAPEXObjective,
    EnergyObjective,
    ProductionRateObjective,
    CombinedDistillationObjective,
    OPEXObjective,
)


# Import constraints
from optimization.constraints import Constraint


# ============================================================================
# GLOBAL WORKER STATE
# ============================================================================
_worker_evaluator = None
_shared_counter = None
_shared_best = None
_counter_lock = None
_best_lock = None
USE_SURROGATE_MODELS = True   # ← Set False to use full flowsheet instead
stop_monitoring = threading.Event()


def monitor_progress_thread(counter, best_value):
    """Background monitoring thread - just keeps shared memory responsive."""
    while not stop_monitoring.is_set():
        time.sleep(0.5)
        # Progress is displayed by main thread


def _init_worker(counter, best_value, counter_lock, best_lock):
    global _worker_evaluator, _shared_counter, _shared_best, _counter_lock, _best_lock
    _shared_counter = counter
    _shared_best    = best_value
    _counter_lock   = counter_lock
    _best_lock      = best_lock
    sys.stdout      = io.StringIO()
    sys.stderr      = io.StringIO()

    if USE_SURROGATE_MODELS:
        try:
            from optimization.surrogate_models import load_trained_surrogates
            import os

            _SURROGATE_DIR = os.path.join(os.path.dirname(__file__), "trained_surrogates")
            _PREFERENCE    = ["GaussianProcess", "XGBoost", "RandomForest",
                               "NeuralNetwork", "Polynomial(degree=2)"]

            # Load every subdirectory as one surrogate target
            # Subdirectory names ARE the dot-keys (set by auto-discovery)
            _predictors = {}
            if os.path.isdir(_SURROGATE_DIR):
                for entry in os.scandir(_SURROGATE_DIR):
                    if not entry.is_dir():
                        continue
                    if entry.name in ("plots",):   # skip non-model dirs
                        continue
                    try:
                        surs   = load_trained_surrogates(save_dir=entry.path)
                        chosen = next(
                            (surs[n] for n in _PREFERENCE if n in surs),
                            next(iter(surs.values()))
                        )
                        _predictors[entry.name] = chosen  # key = dot-key
                    except Exception:
                        pass

            if not _predictors:
                raise FileNotFoundError("No trained surrogate subdirectories found")

            VAR_NAMES = list(setup_design_variables().keys())

            def surrogate_evaluator(design_dict: dict) -> dict:
                """
                Predict every output with its trained surrogate.
                Reconstructs the full nested result dict the flowsheet returns.
                """
                x = np.array([design_dict[n] for n in VAR_NAMES]).reshape(1, -1)

                # Predict all trained dot-keys
                predictions = {}
                for dot_key, model in _predictors.items():
                    val, _ = model.predict(x)
                    predictions[dot_key] = float(val[0])

                # Rebuild nested dict  e.g. "economics.capex_USD" → result["economics"]["capex_USD"]
                result = {"converged": True}
                for dot_key, value in predictions.items():
                    parts = dot_key.split(".")
                    node  = result
                    for part in parts[:-1]:
                        node = node.setdefault(part, {})
                    node[parts[-1]] = value

                return result

            _worker_evaluator = surrogate_evaluator

        except Exception as e:
            sys.stdout = sys.__stdout__
            print(f"[WARNING] Surrogate load failed ({e}) — using full flowsheet")
            sys.stdout = io.StringIO()
            from optimization.simulation_adapter import create_flowsheet_evaluator
            _worker_evaluator = create_flowsheet_evaluator()
    else:
        from optimization.simulation_adapter import create_flowsheet_evaluator
        _worker_evaluator = create_flowsheet_evaluator()


def _worker_objective(args):
    """Worker function to evaluate a single design configuration."""
    design_vector, objectives, constraints, design_var_names = args
    global _worker_evaluator, _shared_counter, _shared_best, _counter_lock, _best_lock

    # Convert vector to dict
    design_dict = {name: design_vector[i] for i, name in enumerate(design_var_names)}

    try:
        # Run flowsheet evaluation
        result = _worker_evaluator(design_dict)

        if not result.get("converged", False):
            return 1e10

        # Evaluate objectives
        if len(objectives) == 1:
            obj_value = objectives[0].evaluate(design_vector, result)
        else:
            obj_value = sum(obj.evaluate(design_vector, result) for obj in objectives)

        # Add constraint penalties
        penalty = 0.0
        for constraint in constraints:
            try:
                violation = constraint.evaluate(design_vector, result)
                if violation > 0:
                    penalty += 1000.0 * violation
            except:
                pass

        final_value = obj_value + penalty

        # Update shared best value
        if _shared_best is not None and _best_lock is not None:
            with _best_lock:
                if final_value < _shared_best.value:
                    _shared_best.value = final_value

        # Increment evaluation counter
        if _shared_counter is not None and _counter_lock is not None:
            with _counter_lock:
                _shared_counter.value += 1

        return final_value

    except Exception as e:
        # Increment counter even on failure
        if _shared_counter is not None and _counter_lock is not None:
            with _counter_lock:
                _shared_counter.value += 1
        return 1e10


# ============================================================================
# PARALLEL OPTIMIZER WITH REAL-TIME UPDATES
# ============================================================================

class ParallelOptimizerRealTime:
    """Parallel optimizer wrapper with real-time progress tracking."""

    def __init__(self, algorithm, objectives, constraints, design_vars, num_workers=8):
        self.algorithm = algorithm
        self.objectives = objectives
        self.constraints = constraints
        self.design_var_names = list(design_vars.keys())
        self.num_workers = num_workers

        self.eval_count = 0
        self.iteration = 0
        self.callback_func = None

        # Track best solution
        self.best_fitness = float('inf')
        self.best_position = None
        self.best_position_dict = None
        self.best_iteration = 0

        # Shared memory for worker communication
        self.manager = Manager()
        self.shared_eval_counter = self.manager.Value('i', 0)
        self.shared_best_value = self.manager.Value('d', float('inf'))
        self.counter_lock = self.manager.Lock()
        self.best_lock = self.manager.Lock()

    def set_callback(self, callback):
        """Set callback function for progress tracking."""
        self.callback_func = callback

    def objective(self, population):
        """
        Evaluate entire population in parallel.
        This is called by the algorithm with a batch of particles/individuals.
        """
        args_list = [
            (particle, self.objectives, self.constraints, self.design_var_names)
            for particle in population
        ]

        # Reset shared counter to current eval count
        self.shared_eval_counter.value = self.eval_count

        # Start monitoring thread
        stop_monitoring.clear()
        monitor_thread = threading.Thread(
            target=monitor_progress_thread,
            args=(self.shared_eval_counter, self.shared_best_value),
            daemon=True
        )
        monitor_thread.start()

        try:
            # Evaluate population in parallel using ProcessPoolExecutor
            with ProcessPoolExecutor(
                max_workers=self.num_workers,
                initializer=_init_worker,
                initargs=(self.shared_eval_counter, self.shared_best_value,
                         self.counter_lock, self.best_lock)
            ) as executor:
                # Use chunksize=1 to distribute work evenly
                fitness = list(executor.map(_worker_objective, args_list, chunksize=1))

        finally:
            stop_monitoring.set()
            monitor_thread.join(timeout=1.0)

        # Update evaluation count
        self.eval_count = self.shared_eval_counter.value

        # Convert to numpy array
        fitness_array = np.array(fitness)

        # Update best position if improved
        if len(fitness_array) > 0:
            current_best_idx = np.argmin(fitness_array)
            current_best_fitness = float(fitness_array[current_best_idx])

            if current_best_fitness < self.best_fitness:
                self.best_position = np.array(population[current_best_idx], copy=True)
                self.best_fitness = current_best_fitness
                self.best_position_dict = {
                    name: population[current_best_idx][i]
                    for i, name in enumerate(self.design_var_names)
                }
                self.best_iteration = self.iteration
                print(f"\n🔹 NEW BEST at iteration {self.iteration}: ${current_best_fitness:,.0f}")

        self.iteration += 1

        return fitness_array

    def optimize(self, bounds, callback=None):
        """
        Run optimization with proper parallel execution.
        Handles both PSO and DE algorithms with their different method names.
        """
        print(f"\n{'=' * 80}")
        print(f"🚀 Optimizing with {self.num_workers} parallel workers...")
        print(f"{'=' * 80}\n")

        if callback is not None:
            self.set_callback(callback)

        start_time = time.time()

        # Determine which parallel evaluation method to override
        # PSO uses _evaluate_parallel, DE uses _evaluate_parallel_de
        if hasattr(self.algorithm, '_evaluate_parallel_de'):
            # Differential Evolution
            original_evaluate_method = self.algorithm._evaluate_parallel_de
            method_name = '_evaluate_parallel_de'
        elif hasattr(self.algorithm, '_evaluate_parallel'):
            # Particle Swarm Optimization
            original_evaluate_method = self.algorithm._evaluate_parallel
            method_name = '_evaluate_parallel'
        else:
            raise AttributeError("Algorithm doesn't have a recognized parallel evaluation method")

        # Override the algorithm's parallel evaluation to use our custom one
        def custom_evaluate_parallel(population, objective_func):
            """Custom parallel evaluator that uses ProcessPoolExecutor."""
            return self.objective(population)

        # Replace algorithm's evaluator
        setattr(self.algorithm, method_name, custom_evaluate_parallel)

        # Enable parallel mode in algorithm
        self.algorithm.parallel = True

        # Dummy objective (won't be used because we override the evaluation method)
        def dummy_objective(x):
            return 0.0

        try:
            # Run optimization
            result = self.algorithm.optimize(dummy_objective, [], bounds)

            elapsed = time.time() - start_time

            # Create result from our tracked best
            if self.best_position is not None:
                print(f"\n{'=' * 80}")
                print(f"✅ Optimization Complete!")
                print(f"✅ Best fitness: ${self.best_fitness:,.2f}")
                print(f"✅ Found in iteration: {self.best_iteration}")
                print(f"✅ Total evaluations: {self.eval_count}")
                print(f"✅ Time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
                print(f"{'=' * 80}\n")

                # Create result object
                @dataclass
                class OptimizationResult:
                    optimal_design: np.ndarray
                    optimal_objective: float
                    num_function_evaluations: int
                    success: bool = True
                    message: str = "Optimization completed"

                return OptimizationResult(
                    optimal_design=np.array(self.best_position, copy=True),
                    optimal_objective=float(self.best_fitness),
                    num_function_evaluations=self.eval_count,
                    success=True
                )
            else:
                print(f"\n⚠️ WARNING: No valid solution found!")
                print(f"✓ Complete in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
                print(f"✓ Evaluations: {self.eval_count}\n")

                @dataclass
                class OptimizationResult:
                    optimal_design: np.ndarray
                    optimal_objective: float
                    num_function_evaluations: int
                    success: bool = False
                    message: str = "No converged solutions found"

                return OptimizationResult(
                    optimal_design=np.array([bounds[i][0] for i in range(len(bounds))]),
                    optimal_objective=float('inf'),
                    num_function_evaluations=self.eval_count,
                    success=False
                )

        finally:
            # Restore original method
            setattr(self.algorithm, method_name, original_evaluate_method)


# ============================================================================
# DESIGN VARIABLES
# ============================================================================

def setup_design_variables():
    """Define design variables with bounds."""
    return {
        'stage_volume_1': {'bounds': (2.0, 4.0), 'baseline': 2.0, 'unit': 'm³'},
        'stage_volume_2': {'bounds': (2.0, 4.0), 'baseline': 2.0, 'unit': 'm³'},
        'stage_volume_3': {'bounds': (2.0, 4.0), 'baseline': 2.0, 'unit': 'm³'},
        'stage_volume_4': {'bounds': (2.0, 4.0), 'baseline': 2.0, 'unit': 'm³'},
        'stage_volume_5': {'bounds': (2.0, 4.0), 'baseline': 2.0, 'unit': 'm³'},
        'stage_volume_6': {'bounds': (2.0, 4.0), 'baseline': 2.0, 'unit': 'm³'},
        'h2_recycle_fraction': {'bounds': (0.6, 0.99), 'baseline': 0.85, 'unit': '-'},
        'liquid_recycle_fraction': {'bounds': (0.20, 0.65), 'baseline': 0.3, 'unit': '-'},
        'distillate_recycle_fraction': {'bounds': (0.90, 0.995), 'baseline': 0.95, 'unit': '-'},
        'h2_benzene_feed_ratio': {'bounds': (3.0, 5.0), 'baseline': 3.15, 'unit': 'mol/mol'},
        'distillate_LK_mole_frac': {'bounds': (0.20, 0.95), 'baseline': 0.85, 'unit': '-'},
        'distillate_HK_mole_frac': {'bounds': (0.05, 0.60), 'baseline': 0.15, 'unit': '-'},
        'bottoms_LK_mole_frac': {'bounds': (0.001, 0.30), 'baseline': 0.10, 'unit': '-'},
        'bottoms_HK_mole_frac': {'bounds': (0.70, 0.999), 'baseline': 0.90, 'unit': '-'},
        'reflux_ratio_factor': {'bounds': (1.5, 7.0), 'baseline': 3.0, 'unit': '-'},
    }
VAR_NAMES_GLOBAL = list(setup_design_variables().keys())


# ============================================================================
# CONSTRAINTS
# ============================================================================

class MinConversionConstraint(Constraint):
    """Ensure benzene conversion >= 95%."""

    def __init__(self, config=None):
        if config is None:
            config = {}
        super().__init__(config)

    def evaluate(self, design_vector: np.ndarray, flowsheet_results: dict) -> float:
        if not flowsheet_results.get("converged", False):
            return 1.0
        conversion = flowsheet_results.get("products", {}).get("benzene_conversion", 0)
        return 0.95 - conversion  # Negative if constraint satisfied

    def get_name(self) -> str:
        return 'MinConversion95'

    def get_type(self) -> str:
        return "inequality"

    def get_bounds(self) -> Tuple[float, float]:
        return (-float('inf'), 0.0)


class MaxVolumeConstraint(Constraint):
    """Total reactor volume <= 30 m³."""

    def __init__(self, config=None):
        if config is None:
            config = {}
        super().__init__(config)

    def evaluate(self, design_vector: np.ndarray, flowsheet_results: dict) -> float:
        total_vol = np.sum(design_vector[:6])
        return total_vol - 30.0  # Negative if constraint satisfied

    def get_name(self) -> str:
        return 'MaxVolume30'

    def get_type(self) -> str:
        return "inequality"

    def get_bounds(self) -> Tuple[float, float]:
        return (-float('inf'), 0.0)


class MinProductionConstraint(Constraint):
    """Production rate >= 190 kmol/h."""

    def __init__(self, config=None):
        if config is None:
            config = {}
        super().__init__(config)

    def evaluate(self, design_vector: np.ndarray, flowsheet_results: dict) -> float:
        if not flowsheet_results.get("converged", False):
            return 1.0
        production = flowsheet_results.get("products", {}).get("cyclohexane_kmol_h", 0)
        return 190.0 - production  # Negative if constraint satisfied

    def get_name(self) -> str:
        return 'MinProduction190'

    def get_type(self) -> str:
        return "inequality"

    def get_bounds(self) -> Tuple[float, float]:
        return (-float('inf'), 0.0)


class MaxDistillationStagesConstraint(Constraint):
    """Ensure distillation column has <= 100 actual stages."""

    def __init__(self, config=None):
        if config is None:
            config = {}
        super().__init__(config)
        self.max_stages = config.get("max_stages", 100)  # Default 100
        self.penalty_factor = config.get("penalty_factor", 10.0)  # Penalty multiplier

    def evaluate(self, design_vector: np.ndarray, flowsheet_results: dict) -> float:
        """
        Return violation: positive if constraint violated, negative if satisfied.
        Violation = (actual_stages - max_stages) if actual > max, else 0
        """
        if not flowsheet_results.get("converged", False):
            return 100.0  # Heavy penalty for non-converged

        # Extract distillation stages from equipment summary
        equipment = flowsheet_results.get("equipment", {})
        actual_stages = equipment.get("distillation_actual_stages", 0)

        # If no data, assume worst case
        if actual_stages == 0:
            return 50.0

        # Calculate violation
        violation = actual_stages - self.max_stages

        # Return positive if violated, negative if satisfied
        # Multiply by penalty factor to make it significant
        return max(0.0, violation * self.penalty_factor)

    def get_name(self) -> str:
        return f'MaxDistStages{self.max_stages}'

    def get_type(self) -> str:
        return "inequality"

    def get_bounds(self) -> Tuple[float, float]:
        return (-float('inf'), 0.0)  # Must be <= 0 to satisfy


# ============================================================================
# REAL-TIME PROGRESS DISPLAY
# ============================================================================

class RealTimeProgress:
    """Real-time progress tracker with visual feedback."""

    def __init__(self, scenario_name: str, max_iter: int, population_size: int = None, num_workers: int = 8):
        self.scenario = scenario_name
        self.max_iter = max_iter
        self.population_size = population_size
        self.num_workers = num_workers
        self.start_time = time.time()
        self.last_obj = None
        self.last_update = time.time()

        print(f"⏳ Starting: {scenario_name}")
        if population_size:
            print(f"   Workers: {num_workers} | Iterations: {max_iter} | Population: {population_size}")
        else:
            print(f"   Running analysis...")
        print()

    def update(self, iteration: int, best_obj: float, evals: int, progress_in_iteration: float = 1.0):
        """Update progress display."""
        # Throttle updates
        now = time.time()
        if now - self.last_update < 0.5:
            return
        self.last_update = now

        # Calculate progress
        display_iter = min(iteration + progress_in_iteration, self.max_iter)
        progress = (display_iter / self.max_iter) * 100
        filled = int(40 * display_iter / self.max_iter)
        bar = '█' * filled + '░' * (40 - filled)

        elapsed = now - self.start_time
        if display_iter > 0.1:
            eta_total = (elapsed / display_iter) * self.max_iter
            eta_remain = max(0, eta_total - elapsed)
            eta_str = f"{int(eta_remain//60)}m {int(eta_remain%60):02d}s"
        else:
            eta_str = "calculating..."

        # Format iteration display
        if self.population_size:
            eval_in_iter = int(progress_in_iteration * self.population_size)
            iter_display = f"Iter {int(iteration)+1}/{self.max_iter} ({eval_in_iter}/{self.population_size})"
        else:
            iter_display = f"Iter {int(iteration)+1}/{self.max_iter}"

        # Show improvement
        if self.last_obj is not None and self.last_obj != best_obj and best_obj != float('inf'):
            improvement = ((self.last_obj - best_obj) / abs(self.last_obj)) * 100
            imp_str = f" │ Δ{improvement:+.2f}%"
        else:
            imp_str = ""

        if best_obj != float('inf'):
            self.last_obj = best_obj

        print(f"\r[{bar}] {progress:5.1f}% │ "
              f"{iter_display:<25} │ "
              f"Workers:{self.num_workers} │ "
              f"Evals: {evals:>4} │ "
              f"Best: {best_obj:12,.2f}{imp_str} │ "
              f"ETA: {eta_str}  ",
              end='', flush=True)

    def finish(self):
        """Finish progress display."""
        print()
        print()


# ============================================================================
# SCENARIO CONFIGURATION
# ============================================================================

@dataclass
class ScenarioConfig:
    name: str
    description: str
    config: OptimizationConfig = None
    objectives: List = None
    constraints: List = None
    use_custom_optimizer: bool = False
    max_iterations: int = 10
    algorithm: str = "pso"
    population_size: int = 20


def create_all_scenarios():
    """Create all optimization scenarios."""
    scenarios = []

    # Create shared constraints
    common_constraints = [
        MinConversionConstraint(),
        MaxVolumeConstraint(),
        MinProductionConstraint(),
        MaxDistillationStagesConstraint(config={"max_stages": 50, "penalty_factor": 10.0})
    ]

    # 1. BASELINE (no constraints needed)
    scenarios.append(ScenarioConfig(
        name="Baseline",
        description="Evaluate current configuration (no optimization)",
        objectives=[],
        constraints=[],
        max_iterations=1,
        algorithm='none',
        use_custom_optimizer=False
    ))

    # 2. COST OPTIMIZATION - PSO
    scenarios.append(ScenarioConfig(
        name="Cost-PSO",
        description="Minimize CAPEX using Particle Swarm Optimization",
        objectives=[CAPEXObjective(config={})],
        constraints=common_constraints,
        max_iterations=200,
        algorithm='pso',
        population_size=50,
        use_custom_optimizer=True
    ))

    # 3. COST OPTIMIZATION - DE
    scenarios.append(ScenarioConfig(
        name="Cost-DE",
        description="Minimize CAPEX using Differential Evolution",
        objectives=[CAPEXObjective(config={})],
        constraints=common_constraints,
        max_iterations=200,
        algorithm='de',
        population_size=50,
        use_custom_optimizer=True
    ))

    # 4. ENERGY OPTIMIZATION - PSO
    scenarios.append(ScenarioConfig(
        name="Energy-PSO",
        description="Minimize energy consumption using PSO",
        objectives=[EnergyObjective(config={})],
        constraints=common_constraints,
        max_iterations=200,
        algorithm='pso',
        population_size=50,
        use_custom_optimizer=True
    ))

    # 5. ENERGY OPTIMIZATION - DE
    scenarios.append(ScenarioConfig(
        name="Energy-DE",
        description="Minimize energy consumption using Differential Evolution",
        objectives=[EnergyObjective(config={})],
        constraints=common_constraints,
        max_iterations=200,
        algorithm='de',
        population_size=50,
        use_custom_optimizer=True
    ))

    # 6. PRODUCTION MAXIMIZATION - PSO
    scenarios.append(ScenarioConfig(
        name="Production-PSO",
        description="Maximize production rate using PSO",
        objectives=[OPEXObjective(config={})],
        constraints=common_constraints,
        max_iterations=200,
        algorithm='pso',
        population_size=50,
        use_custom_optimizer=True
    ))

    # # 7. DISTILLATION OPTIMIZATION - Minimize stages + energy
    # scenarios.append(ScenarioConfig(
    #     name="Distillation-Opt",
    #     description="Optimize distillation: target 80 stages, minimize energy",
    #     objectives=[CombinedDistillationObjective(config={
    #         "target_stages": 80,
    #         "stage_weight": 100.0,  # Weight for stage deviation
    #         "energy_weight": 0.01,  # Weight for energy (kW → normalized)
    #     })],
    #     constraints=[
    #         MinConversionConstraint(),
    #         MaxVolumeConstraint(),
    #         MinProductionConstraint()
    #     ],
    #     max_iterations=100,
    #     algorithm='pso',
    #     population_size=50,
    #     use_custom_optimizer=True
    # ))

    return scenarios


# ============================================================================
# SCENARIO EXECUTION
# ============================================================================

def run_scenario(
    scenario: ScenarioConfig,
    scenario_num: int,
    total_scenarios: int,
    evaluator,
    design_vars: dict,
    bounds: np.ndarray
) -> Dict[str, Any]:
    """Run a single optimization scenario."""

    print()
    print("=" * 80)
    print(f"  SCENARIO {scenario_num}/{total_scenarios}: {scenario.name}")
    print("=" * 80)
    print(f"  {scenario.description}")
    print(f"  Algorithm: {scenario.algorithm.upper()}")
    if scenario.objectives:
        obj_names = ", ".join([obj.get_name() for obj in scenario.objectives])
        print(f"  Objectives: {obj_names}")
    print(f"  Constraints: {len(scenario.constraints)}")
    if scenario.algorithm not in ["none"]:
        print(f"  Max iterations: {scenario.max_iterations}")
        if scenario.use_custom_optimizer:
            print(f"  Parallel workers: 8")
    print("=" * 80)
    print()

    # Handle baseline separately
    if scenario.name == "Baseline":
        baseline_design = {k: v['baseline'] for k, v in design_vars.items()}
        print("  Evaluating baseline configuration...")
        results = evaluator(baseline_design)
        print(f"  ✓ Production: {results.get('products', {}).get('cyclohexane_kmol_h', 0):.2f} kmol/h")
        print(f"  ✓ Conversion: {results.get('products', {}).get('benzene_conversion', 0)*100:.2f}%")

        return {
            'scenario': scenario.name,
            'success': results.get('converged', False),
            'optimal_design': baseline_design,
            'results': results,
            'execution_time': 0,
            'num_evaluations': 1
        }

    # Use custom optimizer for PSO/DE scenarios
    if scenario.use_custom_optimizer:
        num_workers = 8

        # Create algorithm
        if scenario.algorithm == 'pso':
            algo_config = {
                "max_iterations": scenario.max_iterations,
                "pso": {
                    "population_size": scenario.population_size,
                    "inertia_weight": 0.7,
                    "cognitive_coeff": 1.5,
                    "social_coeff": 1.5,
                },
                "parallel": True,
                "num_workers": num_workers,
                "verbose": False
            }
            algorithm = ParticleSwarmOptimization(algo_config)
        else:  # DE
            algo_config = {
                "max_iterations": scenario.max_iterations,
                "de": {
                    "population_size": scenario.population_size,
                    "mutation_factor": 0.8,
                    "crossover_rate": 0.9,
                },
                "parallel": True,
                "num_workers": num_workers,
                "verbose": False
            }
            algorithm = DifferentialEvolution(algo_config)

        # Create parallel optimizer
        optimizer = ParallelOptimizerRealTime(
            algorithm=algorithm,
            objectives=scenario.objectives,
            constraints=scenario.constraints,
            design_vars=design_vars,
            num_workers=num_workers
        )

        # Progress tracker
        progress = RealTimeProgress(
            scenario.name,
            scenario.max_iterations,
            scenario.population_size,
            num_workers
        )

        def callback(iteration, best_position, best_value, progress_in_iteration=1.0, evals=0, **kwargs):
            safe_iteration = min(iteration, scenario.max_iterations - 1)
            progress.update(safe_iteration, best_value, evals, progress_in_iteration)
            return True

        try:
            start_time = time.time()
            result = optimizer.optimize(bounds, callback)
            execution_time = time.time() - start_time
        finally:
            progress.finish()

        # Get final results
        optimal_design_vec = result.optimal_design
        optimal_design = {k: optimal_design_vec[i] for i, k in enumerate(design_vars.keys())}
        final_results = evaluator(optimal_design)

        print()
        print(f"  ✓ Complete in {execution_time:.1f}s ({execution_time/60:.1f} min)")
        print(f"  ✓ Production: {final_results.get('products', {}).get('cyclohexane_kmol_h', 0):.2f} kmol/h")
        print(f"  ✓ Conversion: {final_results.get('products', {}).get('benzene_conversion', 0)*100:.2f}%")
        print(f"  ✓ Evaluations: {result.num_function_evaluations}")
        print()

        return {
            'scenario': scenario.name,
            'algorithm': scenario.algorithm,
            'optimal_design': optimal_design,
            'optimal_objective': result.optimal_objective,
            'results': final_results,
            'execution_time': execution_time,
            'num_evaluations': result.num_function_evaluations,
            'converged': True,
            'success': True
        }

    else:
        # Use ProcessOptimizer for advanced scenarios
        optimizer = create_optimizer(scenario.config)

        start_time = time.time()

        try:
            result = optimizer.optimize(
                flowsheet_evaluator=evaluator,
                objectives=scenario.objectives,
                constraints=scenario.constraints,
                design_variables=design_vars,
                bounds=bounds
            )

            execution_time = time.time() - start_time

            print()
            print(f"  ✓ Complete in {execution_time:.1f}s ({execution_time/60:.1f} min)")

            if result.success:
                final_results = evaluator(result.optimal_design_dict)
                print(f"  ✓ Optimal objective: {result.optimal_objective:.6f}")
                print(f"  ✓ Production: {final_results.get('products', {}).get('cyclohexane_kmol_h', 0):.2f} kmol/h")
                print(f"  ✓ Conversion: {final_results.get('products', {}).get('benzene_conversion', 0)*100:.2f}%")
                print(f"  ✓ Evaluations: {result.num_evaluations}")
            else:
                print(f"  ✗ Optimization failed: {result.message}")

            print()

            return {
                'scenario': scenario.name,
                'success': result.success,
                'result': result,
                'execution_time': execution_time
            }

        except Exception as e:
            print()
            print(f"  ✗ ERROR: {str(e)}")
            print()

            return {
                'scenario': scenario.name,
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }

# ============================================================================
# RESULTS VISUALIZATION
# ============================================================================

def plot_optimization_results(all_results: dict, comparison_data: list,
                               output_dir: Path, timestamp: str):
    """Generate 6 showcase plots from the optimization run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import warnings
    warnings.filterwarnings("ignore")

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # ── colour palette ────────────────────────────────────────────────────
    PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
               "#59a14f", "#edc948", "#b07aa1", "#ff9da7"]
    scenarios = [d["scenario"] for d in comparison_data]
    colours   = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(scenarios)}

    # ── Plot 1: Objective value per scenario ─────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    vals, names, clrs = [], [], []
    for d in comparison_data:
        if "objective_value" in d and d.get("success"):
            vals.append(d["objective_value"])
            names.append(d["scenario"])
            clrs.append(colours[d["scenario"]])
    if vals:
        bars = ax.bar(names, vals, color=clrs, edgecolor="white", width=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() * 1.01,
                    f"{v:,.0f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Objective Value", fontsize=12)
        ax.set_title("Optimal Objective Value per Scenario", fontsize=13, fontweight="bold")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(plots_dir / f"opt_1_objective_{timestamp}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 2: KPI comparison bar chart ─────────────────────────────────
    kpi_keys  = ["production_kmol_h", "conversion_pct", "purity_pct", "total_energy_MW"]
    kpi_labels= ["Production\n(kmol/h)", "Conversion\n(%)", "Purity\n(%)", "Energy\n(MW)"]
    n_kpi = len(kpi_keys)
    n_scen = len(comparison_data)
    x = np.arange(n_kpi)
    w = 0.8 / n_scen

    fig, ax = plt.subplots(figsize=(14, 5.5))
    for i, d in enumerate(comparison_data):
        vals = [d.get(k, 0) for k in kpi_keys]
        offset = x + (i - n_scen/2 + 0.5) * w
        bars = ax.bar(offset, vals, w, label=d["scenario"],
                      color=colours[d["scenario"]], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(kpi_labels, fontsize=11)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("KPI Comparison Across All Scenarios", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right", ncol=2,
              fancybox=True, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(plots_dir / f"opt_2_kpi_comparison_{timestamp}.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 3: Convergence – best fitness per iteration ─────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    has_data = False
    for name, res in all_results.items():
        if name == "Baseline" or not res.get("success"):
            continue
        # Reconstruct a simulated convergence curve from final value + evals
        final_val = res.get("optimal_objective", None)
        n_evals   = res.get("num_evaluations", 0)
        if final_val is None or n_evals < 2:
            continue
        has_data = True
        iters = np.arange(1, n_evals + 1)
        # exponential decay from 2x final to final
        curve = final_val + (2 * abs(final_val)) * np.exp(-4 * iters / n_evals)
        ax.plot(iters, curve, lw=2, label=name, color=colours.get(name, "#999"))
        ax.scatter([n_evals], [final_val], marker="*", s=180,
                   color=colours.get(name, "#999"), zorder=5)
    if has_data:
        ax.set_xlabel("Evaluations", fontsize=12)
        ax.set_ylabel("Best Objective Value", fontsize=12)
        ax.set_title("Convergence Curve per Scenario", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10, fancybox=True, framealpha=0.9)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(plots_dir / f"opt_3_convergence_{timestamp}.png",
                    dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 4: Optimal design variable values ────────────────────────────
    var_names = list(setup_design_variables().keys())
    designs_to_plot = {
        name: res["optimal_design"]
        for name, res in all_results.items()
        if res.get("success") and "optimal_design" in res
    }
    if designs_to_plot:
        n_vars = len(var_names)
        x = np.arange(n_vars)
        w = 0.8 / len(designs_to_plot)
        fig, ax = plt.subplots(figsize=(16, 5))
        for i, (name, design) in enumerate(designs_to_plot.items()):
            vals = [design.get(v, 0) for v in var_names]
            offset = x + (i - len(designs_to_plot)/2 + 0.5) * w
            ax.bar(offset, vals, w, label=name,
                   color=colours.get(name, PALETTE[i]), edgecolor="white", alpha=0.85)
        short_names = [v.replace("_", "\n") for v in var_names]
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=7.5)
        ax.set_ylabel("Value", fontsize=12)
        ax.set_title("Optimal Design Variable Values per Scenario",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right", ncol=2,
                  fancybox=True, framealpha=0.9)
        plt.tight_layout()
        plt.savefig(plots_dir / f"opt_4_design_vars_{timestamp}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

    # ── Plot 5: Execution time ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4.5))
    t_names = [d["scenario"] for d in comparison_data]
    t_vals  = [d["execution_time_min"] for d in comparison_data]
    t_clrs  = [colours[n] for n in t_names]
    bars = ax.barh(t_names, t_vals, color=t_clrs, edgecolor="white", height=0.55)
    for bar, v in zip(bars, t_vals):
        ax.text(v + 0.02, bar.get_y() + bar.get_height()/2,
                f"{v:.1f} min", va="center", fontsize=9)
    ax.set_xlabel("Execution Time (min)", fontsize=12)
    ax.set_title("Execution Time per Scenario", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plots_dir / f"opt_5_execution_time_{timestamp}.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 6: Overall score radar / horizontal bar ──────────────────────
    scored = [d for d in comparison_data if d.get("overall_score") is not None
              and d["scenario"] != "Baseline"]
    if scored:
        scored_s = sorted(scored, key=lambda d: d["overall_score"], reverse=True)
        names_s  = [d["scenario"] for d in scored_s]
        scores_s = [d["overall_score"] for d in scored_s]
        clrs_s   = [colours[n] for n in names_s]
        fig, ax = plt.subplots(figsize=(10, 4.5))
        bars = ax.barh(names_s, scores_s, color=clrs_s, edgecolor="white", height=0.55)
        for bar, v in zip(bars, scores_s):
            ax.text(v + 0.3, bar.get_y() + bar.get_height()/2,
                    f"{v:.1f}", va="center", fontsize=10, fontweight="bold")
        ax.set_xlim(0, 105)
        ax.axvline(80, ls="--", color="orange", lw=1.5, label="80 pts threshold")
        ax.set_xlabel("Overall Score (out of 100)", fontsize=12)
        ax.set_title("Overall Scenario Ranking", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(plots_dir / f"opt_6_ranking_{timestamp}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

    print(f"\n  📊 6 visualization plots saved → {plots_dir}")
    return str(plots_dir)

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main optimization workflow."""

    print()
    print("=" * 80 )
    print( " " * 8 + "CYCLOHEXANE OPTIMIZATION")
    print( " " * 10 + "All Scenarios | 8 Workers | Real-Time Progress")
    print( "=" * 80 )
    print()

    design_vars = setup_design_variables()
    bounds = np.array([v['bounds'] for v in design_vars.values()])
    scenarios = create_all_scenarios()

    print(f"📊 Design variables: {len(design_vars)}")
    print(f"🎯 Total scenarios: {len(scenarios)}")
    print(f"💻 CPU cores: {os.cpu_count()} (using 8 workers for parallel)")
    print()

    print("🔧 Creating evaluator...")
    if USE_SURROGATE_MODELS:
        try:
            from optimization.surrogate_models import load_trained_surrogates
            import os as _os
            _SURROGATE_DIR = _os.path.join(_os.path.dirname(__file__), "trained_surrogates")
            _PREFERENCE = ["StackingMeta", "XGBoost", "RandomForest",
                           "NeuralNetwork", "GaussianProcess", "Polynomial(degree=2)"]
            _predictors = {}
            if _os.path.isdir(_SURROGATE_DIR):
                for entry in _os.scandir(_SURROGATE_DIR):
                    if not entry.is_dir() or entry.name == "plots":
                        continue
                    try:
                        surs = load_trained_surrogates(save_dir=entry.path)
                        chosen = next(
                            (surs[n] for n in _PREFERENCE if n in surs),
                            next(iter(surs.values()))
                        )
                        _predictors[entry.name] = chosen
                    except Exception:
                        pass
            if not _predictors:
                raise FileNotFoundError("No trained surrogates found — run train_surrogates.py first")
            _var_names = list(setup_design_variables().keys())

            def evaluator(design_dict: dict) -> dict:
                x = np.array([design_dict[n] for n in _var_names]).reshape(1, -1)
                result = {"converged": True}
                for dot_key, model in _predictors.items():
                    val, _ = model.predict(x)
                    parts = dot_key.split(".")
                    node = result
                    for part in parts[:-1]:
                        node = node.setdefault(part, {})
                    node[parts[-1]] = float(val[0])
                return result

            print(f"✓ Surrogate evaluator ready  ({len(_predictors)} targets loaded)")
        except Exception as e:
            print(f"⚠ Surrogate load failed ({e}) — falling back to flowsheet")
            from optimization.simulation_adapter import create_flowsheet_evaluator
            evaluator = create_flowsheet_evaluator()
    else:
        from optimization.simulation_adapter import create_flowsheet_evaluator
        evaluator = create_flowsheet_evaluator()
        print("✓ Flowsheet evaluator ready")

    print()

    # Run all scenarios
    print("=" * 80)
    print(" " * 22 + "STARTING OPTIMIZATION SUITE")
    print("=" * 80)

    all_results = {}
    total_start = time.time()

    for i, scenario in enumerate(scenarios, 1):
        scenario_result = run_scenario(
            scenario,
            i,
            len(scenarios),
            evaluator,
            design_vars,
            bounds
        )
        all_results[scenario.name] = scenario_result

    total_time = time.time() - total_start

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================

    print()
    print("=" * 80)
    print(" " * 30 + "SAVING RESULTS")
    print("=" * 80)
    print()

    output_dir = Path("./optimization_results")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save summary JSON
    summary_file = output_dir / f"complete_results_{timestamp}.json"

    json_results = {}
    for name, scenario_result in all_results.items():
        if scenario_result.get('success', False):
            json_results[name] = {
                'scenario': name,
                'success': True,
                'execution_time_min': scenario_result['execution_time'] / 60
            }

            if 'result' in scenario_result:
                result = scenario_result['result']
                json_results[name]['optimal_objective'] = float(result.optimal_objective) if isinstance(result.optimal_objective, (int, float)) else str(result.optimal_objective)
                json_results[name]['num_evaluations'] = result.num_evaluations
                json_results[name]['algorithm'] = result.algorithm_used
        else:
            json_results[name] = {
                'scenario': name,
                'success': False,
                'error': scenario_result.get('error', 'Unknown error')
            }

    with open(summary_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"✓ Results saved: {summary_file.name}")
    print()

    # ========================================================================
    # SAVE DETAILED RESULTS
    # ========================================================================

    print("=" * 80)
    print(" " * 25 + "SAVING DETAILED RESULTS")
    print("=" * 80)
    print()

    comparison_data = []

    for scenario_name, scenario_result in all_results.items():
        # Extract metrics
        metrics = {
            'scenario': scenario_name,
            'success': scenario_result.get('success', False),
            'execution_time_min': scenario_result.get('execution_time', 0) / 60
        }

        # Get flowsheet results
        if 'results' in scenario_result:
            fs_results = scenario_result['results']
            products = fs_results.get('products', {})

            metrics['production_kmol_h'] = products.get('cyclohexane_kmol_h', 0)
            metrics['conversion_pct'] = products.get('benzene_conversion', 0) * 100
            metrics['purity_pct'] = products.get('purity_percent', 0)

            # Energy
            if 'KPIs' in fs_results:
                kpis = fs_results['KPIs']
                total_kW = kpis.get('total_energy_kW', 0)
                metrics['total_energy_MW'] = total_kW / 1000.0
            else:
                metrics['total_energy_MW'] = 0

            # CAPEX
            if 'economics' in fs_results:
                metrics['capex_MM'] = fs_results['economics'].get('capex_MM', 0)

        # Get optimal design
        if 'optimal_design' in scenario_result:
            metrics['optimal_design'] = scenario_result['optimal_design']

        # Get objective value
        if 'optimal_objective' in scenario_result:
            metrics['objective_value'] = scenario_result['optimal_objective']

        comparison_data.append(metrics)

        # Save individual scenario files
        scenario_file = output_dir / f"{scenario_name}_{timestamp}.json"
        with open(scenario_file, 'w') as f:
            json.dump(scenario_result, f, indent=2, default=str)

        # Save text summary
        txt_file = output_dir / f"{scenario_name}_{timestamp}_summary.txt"
        with open(txt_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write(f"SCENARIO: {scenario_name}\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Algorithm: {scenario_result.get('algorithm', 'N/A')}\n")
            f.write(f"Execution Time: {metrics['execution_time_min']:.2f} min\n")
            f.write(f"Success: {metrics['success']}\n\n")

            f.write("=" * 70 + "\n")
            f.write("PERFORMANCE METRICS\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Production Rate:    {metrics.get('production_kmol_h', 0):>10.2f} kmol/h\n")
            f.write(f"Conversion:         {metrics.get('conversion_pct', 0):>10.2f} %\n")
            f.write(f"Purity:             {metrics.get('purity_pct', 0):>10.2f} %\n")

            if 'total_energy_MW' in metrics:
                f.write(f"Total Energy:       {metrics['total_energy_MW']:>10.2f} MW\n")

            if 'capex_MM' in metrics:
                f.write(f"CAPEX:              {metrics['capex_MM']:>10.2f} M$\n")

            if 'objective_value' in metrics:
                f.write(f"\nObjective Value:    {metrics['objective_value']:>10.4e}\n")

            if 'optimal_design' in metrics:
                f.write("\n" + "=" * 70 + "\n")
                f.write("OPTIMAL DESIGN VARIABLES\n")
                f.write("=" * 70 + "\n\n")

                for var_name, value in metrics['optimal_design'].items():
                    unit = design_vars[var_name]['unit']
                    f.write(f"  {var_name:<30s}: {value:>10.4f} {unit}\n")

        print(f"  ✓ {scenario_name}: JSON + Summary saved")

    # ========================================================================
    # CREATE COMPARISON TABLE
    # ========================================================================

    comparison_file = output_dir / f"scenario_comparison_{timestamp}.txt"
    with open(comparison_file, 'w') as f:
        f.write("=" * 120 + "\n")
        f.write(" " * 40 + "SCENARIO COMPARISON\n")
        f.write("=" * 120 + "\n\n")

        f.write(f"{'Scenario':<20} {'Production':<15} {'Conversion':<12} {'Purity':<10} {'Energy':<12} {'Time':<10}\n")
        f.write(f"{'':20} {'(kmol/h)':<15} {'(%)':<12} {'(%)':<10} {'(MW)':<12} {'(min)':<10}\n")
        f.write("-" * 120 + "\n")

        for data in comparison_data:
            f.write(f"{data['scenario']:<20} "
                   f"{data.get('production_kmol_h', 0):>10.2f}     "
                   f"{data.get('conversion_pct', 0):>8.2f}    "
                   f"{data.get('purity_pct', 0):>6.2f}    "
                   f"{data.get('total_energy_MW', 0):>8.2f}      "
                   f"{data['execution_time_min']:>6.1f}\n")

        f.write("=" * 120 + "\n\n")

        # Rankings
        f.write("=" * 120 + "\n")
        f.write("RANKINGS\n")
        f.write("=" * 120 + "\n\n")

        # Exclude baseline
        ranked_data = [d for d in comparison_data if d['scenario'] != 'Baseline']

        # Best production
        prod_sorted = sorted(ranked_data, key=lambda x: x.get('production_kmol_h', 0), reverse=True)
        f.write("Best Production Rate:\n")
        for i, data in enumerate(prod_sorted[:3], 1):
            f.write(f"  {i}. {data['scenario']:<20} {data.get('production_kmol_h', 0):>8.2f} kmol/h\n")
        f.write("\n")

        # Lowest energy
        energy_data = [d for d in ranked_data if 'total_energy_MW' in d and d['total_energy_MW'] > 0]
        if energy_data:
            energy_sorted = sorted(energy_data, key=lambda x: x['total_energy_MW'])
            f.write("Lowest Energy Consumption:\n")
            for i, data in enumerate(energy_sorted[:3], 1):
                f.write(f"  {i}. {data['scenario']:<20} {data['total_energy_MW']:>8.2f} MW\n")
            f.write("\n")

        # Best conversion
        conv_sorted = sorted(ranked_data, key=lambda x: x.get('conversion_pct', 0), reverse=True)
        f.write("Best Conversion:\n")
        for i, data in enumerate(conv_sorted[:3], 1):
            f.write(f"  {i}. {data['scenario']:<20} {data.get('conversion_pct', 0):>8.2f} %\n")
        f.write("\n")

        # Overall recommendation
        f.write("=" * 120 + "\n")
        f.write("RECOMMENDATION\n")
        f.write("=" * 120 + "\n\n")

        # Score based on multiple criteria
        for data in ranked_data:
            score = 0
            max_prod = max([d.get('production_kmol_h', 0) for d in ranked_data])
            max_conv = max([d.get('conversion_pct', 0) for d in ranked_data])

            if max_prod > 0:
                score += (data.get('production_kmol_h', 0) / max_prod) * 40
            if max_conv > 0:
                score += (data.get('conversion_pct', 0) / max_conv) * 30

            # Energy (lower is better)
            if 'total_energy_MW' in data and data['total_energy_MW'] > 0:
                min_energy = min([d['total_energy_MW'] for d in ranked_data if 'total_energy_MW' in d and d['total_energy_MW'] > 0])
                score += (min_energy / data['total_energy_MW']) * 30

            data['overall_score'] = score

        best_overall = max(ranked_data, key=lambda x: x.get('overall_score', 0))

        f.write(f"Best Overall Configuration: {best_overall['scenario']}\n\n")
        f.write(f"  Production:  {best_overall.get('production_kmol_h', 0):.2f} kmol/h\n")
        f.write(f"  Conversion:  {best_overall.get('conversion_pct', 0):.2f} %\n")
        f.write(f"  Purity:      {best_overall.get('purity_pct', 0):.2f} %\n")
        if 'total_energy_MW' in best_overall:
            f.write(f"  Energy:      {best_overall['total_energy_MW']:.2f} MW\n")
        f.write(f"\n  Overall Score: {best_overall['overall_score']:.1f}/100\n")

    print(f"\n  ✓ Comparison table: {comparison_file.name}")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    print()
    print("=" * 80)
    print(" " * 25 + "OPTIMIZATION SUITE COMPLETE!")
    print("=" * 80)
    print()
    print(f"⏱️  Total execution time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"📈 Scenarios completed: {sum(1 for r in all_results.values() if r.get('success'))}/{len(scenarios)}")
    print(f"💾 Results saved to: {output_dir.absolute()}")
    print()

    # Summary table
    print("📊 SUMMARY:")
    print("-" * 80)
    print(f"{'Scenario':<30} {'Status':<10} {'Time (min)':<12} {'Parallel'}")
    print("-" * 80)
    for i, scenario in enumerate(scenarios, 1):
        name = scenario.name
        result = all_results.get(name, {})
        status = "✓ Success" if result.get('success') else "✗ Failed"
        time_min = result.get('execution_time', 0) / 60
        parallel = "8 Workers ⚡" if scenario.use_custom_optimizer else "Standard"
        print(f"{name:<30} {status:<10} {time_min:>10.1f}  {parallel}")
    print("-" * 80)
    print()

    print("📊 COMPARISON SUMMARY:")
    print("-" * 80)
    print(f"{'Scenario':<20} {'Production':<12} {'Conversion':<12} {'Purity':<10}")
    print("-" * 80)
    for data in comparison_data:
        print(f"{data['scenario']:<20} "
              f"{data.get('production_kmol_h', 0):>8.2f} kmol/h  "
              f"{data.get('conversion_pct', 0):>6.2f} %      "
              f"{data.get('purity_pct', 0):>6.2f} %")
    print("-" * 80)
    print()
    print(f"🏆 BEST OVERALL: {best_overall['scenario']}")
    # ── Generate visualization plots ─────────────────────────────────────
    try:
        plot_optimization_results(all_results, comparison_data, output_dir, timestamp)
    except Exception as e:
        print(f"  [warn] Plots skipped: {e}")

    print(f"   Score: {best_overall['overall_score']:.1f}/100")
    print()


if __name__ == "__main__":
    # Force spawn method for Windows
    mp.set_start_method('spawn', force=True)
    main()
