"""
optimization/algorithms.py - PRODUCTION VERSION v1.4.0

Date: 2026-02-07
Version: 1.4.0
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, Bounds

from optimization.constraints import Constraint

logger = logging.getLogger(__name__)


# ============================================================================
# BASE ALGORITHM
# ============================================================================

@dataclass
class OptimizationResult:
    """Result from optimization algorithm."""
    success: bool
    message: str
    optimal_design: np.ndarray
    optimal_objective: float
    constraint_violations: Dict[str, float]
    num_iterations: int
    num_function_evaluations: int
    convergence_history: Dict[str, List[float]]


class OptimizationAlgorithm(ABC):
    """Abstract base class for optimization algorithms."""

    def __init__(self, config: dict):
        """Initialize algorithm."""
        self.config = config
        self.max_iterations = config.get("max_iterations", 100)
        self.verbose = config.get("verbose", True)
        self.callback_func = None
        self.num_evaluations = 0
        self.convergence_history = {
            "iteration": [],
            "best_objective": [],
            "average_objective": [],
            "constraint_violation": []
        }

    def set_callback(self, callback: Callable):
        """Set callback function for progress tracking."""
        self.callback_func = callback

    @abstractmethod
    def optimize(
        self,
        objective: Callable,
        constraints: List[Constraint],
        bounds: np.ndarray
    ) -> OptimizationResult:
        """Run optimization."""
        pass

    def _evaluate_objective(self, x: np.ndarray, objective: Callable) -> float:
        """Evaluate objective with error handling."""
        try:
            self.num_evaluations += 1
            value = objective(x)
            if not np.isfinite(value):
                return 1e10
            return float(value)
        except Exception as e:
            logger.warning(f"Objective evaluation failed: {e}")
            return 1e10

    def _evaluate_constraints(self, x: np.ndarray, constraints: List[Constraint]) -> Dict[str, float]:
        """Evaluate all constraints."""
        violations = {}
        for constraint in constraints:
            try:
                value = constraint.evaluate(x, {})
                violations[constraint.get_name()] = max(0, value)
            except Exception as e:
                logger.warning(f"Constraint {constraint.get_name()} evaluation failed: {e}")
                violations[constraint.get_name()] = 1e10
        return violations

    def _check_bounds(self, x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        """Enforce bounds on design vector."""
        return np.clip(x, bounds[:, 0], bounds[:, 1])


# ============================================================================
# PARTICLE SWARM OPTIMIZATION (PSO)
# ============================================================================

class ParticleSwarmOptimization(OptimizationAlgorithm):

    def __init__(self, config: dict):
        """Initialize PSO."""
        super().__init__(config)

        pso_config = config.get("pso", {})
        self.population_size = pso_config.get("population_size", 24)
        self.inertia_weight = pso_config.get("inertia_weight", 0.7)
        self.cognitive_coeff = pso_config.get("cognitive_coeff", 1.5)
        self.social_coeff = pso_config.get("social_coeff", 1.5)
        self.velocity_limit = pso_config.get("velocity_limit", 0.5)

        # Parallelization
        self.parallel = config.get("parallel", True)
        self.num_workers = config.get("num_workers", 16)

        logger.info(f"PSO initialized: population={self.population_size}, "
                   f"parallel={self.parallel}, workers={self.num_workers}")

    def optimize(
        self,
        objective: Callable,
        constraints: List[Constraint],
        bounds: np.ndarray
    ) -> OptimizationResult:

        start_time = time.time()
        n_dim = bounds.shape[0]

        if self.verbose:
            logger.info("=" * 60)
            logger.info(f"PSO Optimization Starting")
            logger.info(f"Population: {self.population_size}, Iterations: {self.max_iterations}")
            logger.info(f"Parallel: {self.parallel}, Workers: {self.num_workers}")
            logger.info("=" * 60)

        # Initialize swarm
        particles = np.random.uniform(
            bounds[:, 0], bounds[:, 1],
            size=(self.population_size, n_dim)
        )

        velocities = np.random.uniform(
            -self.velocity_limit, self.velocity_limit,
            size=(self.population_size, n_dim)
        )

        if self.parallel and self.num_workers > 1:
            particle_fitness = self._evaluate_parallel(particles, objective)
        else:
            particle_fitness = np.array([
                self._evaluate_objective(p, objective) for p in particles
            ])

        # Initialize personal and global bests
        personal_best_positions = particles.copy()
        personal_best_fitness = particle_fitness.copy()

        global_best_idx = np.argmin(particle_fitness)
        global_best_position = particles[global_best_idx].copy()
        global_best_fitness = particle_fitness[global_best_idx]

        # Track convergence
        iteration_history = []

        # Main PSO loop
        for iteration in range(self.max_iterations):
            iter_start = time.time()

            # Update velocities and positions
            r1 = np.random.random((self.population_size, n_dim))
            r2 = np.random.random((self.population_size, n_dim))

            velocities = (
                self.inertia_weight * velocities +
                self.cognitive_coeff * r1 * (personal_best_positions - particles) +
                self.social_coeff * r2 * (global_best_position - particles)
            )

            # Limit velocities
            velocity_magnitude = np.linalg.norm(velocities, axis=1, keepdims=True)
            max_velocity = self.velocity_limit * (bounds[:, 1] - bounds[:, 0])
            scale = np.where(velocity_magnitude > np.linalg.norm(max_velocity),
                           np.linalg.norm(max_velocity) / velocity_magnitude, 1.0)
            velocities *= scale

            # Update positions
            particles += velocities
            particles = self._check_bounds(particles, bounds)

            if self.parallel and self.num_workers > 1:
                particle_fitness = self._evaluate_parallel(particles, objective)
            else:
                particle_fitness = np.array([
                    self._evaluate_objective(p, objective) for p in particles
                ])

            # Update personal bests
            improved = particle_fitness < personal_best_fitness
            personal_best_positions[improved] = particles[improved]
            personal_best_fitness[improved] = particle_fitness[improved]

            # Update global best
            current_best_idx = np.argmin(particle_fitness)
            current_best_fitness = particle_fitness[current_best_idx]

            if current_best_fitness < global_best_fitness:
                global_best_fitness = current_best_fitness
                global_best_position = particles[current_best_idx].copy()

            # Track convergence
            avg_fitness = np.mean(particle_fitness)
            self.convergence_history["iteration"].append(iteration)
            self.convergence_history["best_objective"].append(global_best_fitness)
            self.convergence_history["average_objective"].append(avg_fitness)

            iteration_history.append({
                "iteration": iteration,
                "best": global_best_fitness,
                "avg": avg_fitness,
                "time": time.time() - iter_start
            })

            # Callback
            if self.callback_func is not None:
                constraint_violations = self._evaluate_constraints(global_best_position, constraints)
                should_continue = self.callback_func(
                    iteration=iteration,
                    best_position=global_best_position,
                    best_value=global_best_fitness,
                    population_values=particle_fitness
                )

                if not should_continue:
                    logger.info("Optimization stopped by callback")
                    break

            # Progress logging
            if self.verbose and (iteration % max(1, self.max_iterations // 10) == 0 or
                                iteration == self.max_iterations - 1):
                elapsed = time.time() - start_time
                logger.info(
                    f"Iteration {iteration+1}/{self.max_iterations} | "
                    f"Best: {global_best_fitness:.6e} | "
                    f"Avg: {avg_fitness:.6e} | "
                    f"Time: {elapsed:.1f}s"
                )

        # Final evaluation
        final_constraints = self._evaluate_constraints(global_best_position, constraints)
        elapsed_time = time.time() - start_time

        if self.verbose:
            logger.info("=" * 60)
            logger.info(f"PSO Complete: Best = {global_best_fitness:.6e} in {elapsed_time:.2f}s")
            logger.info(f"Evaluations: {self.num_evaluations}")
            logger.info("=" * 60)

        return OptimizationResult(
            success=True,
            message="PSO optimization complete",
            optimal_design=global_best_position,
            optimal_objective=global_best_fitness,
            constraint_violations=final_constraints,
            num_iterations=len(iteration_history),
            num_function_evaluations=self.num_evaluations,
            convergence_history=self.convergence_history
        )

    def _evaluate_parallel(self, particles: np.ndarray, objective: Callable) -> np.ndarray:

        if not self.parallel or self.num_workers <= 1:
            # Serial fallback
            return np.array([self._evaluate_objective(p, objective) for p in particles])

        try:
            with mp.Pool(self.num_workers) as pool:
                # Call objective DIRECTLY (it's already module-level from optimization_main.py)
                fitness = pool.map(objective, particles)

            # Update evaluation counter
            self.num_evaluations += len(particles)

            return np.array(fitness)

        except Exception as e:
            logger.warning(f"Parallel execution failed: {e}, falling back to serial")
            # Serial fallback
            return np.array([self._evaluate_objective(p, objective) for p in particles])


# ============================================================================
# GENETIC ALGORITHM (GA)
# ============================================================================

class GeneticAlgorithm(OptimizationAlgorithm):
    """Genetic Algorithm for optimization."""

    def __init__(self, config: dict):
        """Initialize GA."""
        super().__init__(config)

        ga_config = config.get("ga", {})
        self.population_size = ga_config.get("population_size", 50)
        self.mutation_rate = ga_config.get("mutation_rate", 0.1)
        self.crossover_rate = ga_config.get("crossover_rate", 0.8)
        self.tournament_size = ga_config.get("tournament_size", 3)
        self.elitism_size = ga_config.get("elitism_size", 2)

        logger.info(f"GA initialized: population={self.population_size}")

    def optimize(
        self,
        objective: Callable,
        constraints: List[Constraint],
        bounds: np.ndarray
    ) -> OptimizationResult:
        """Run GA optimization."""
        start_time = time.time()
        n_dim = bounds.shape[0]

        # Initialize population
        population = np.random.uniform(
            bounds[:, 0], bounds[:, 1],
            size=(self.population_size, n_dim)
        )

        fitness = np.array([self._evaluate_objective(ind, objective) for ind in population])

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        for generation in range(self.max_iterations):
            # Selection (tournament)
            parents = self._tournament_selection(population, fitness)

            # Crossover
            offspring = self._crossover(parents, bounds)

            # Mutation
            offspring = self._mutate(offspring, bounds)

            # Evaluate offspring
            offspring_fitness = np.array([self._evaluate_objective(ind, objective) for ind in offspring])

            # Elitism: keep best individuals
            elite_idx = np.argsort(fitness)[:self.elitism_size]
            elites = population[elite_idx]
            elite_fitness = fitness[elite_idx]

            # Form new population
            population = np.vstack([elites, offspring[:(self.population_size - self.elitism_size)]])
            fitness = np.hstack([elite_fitness, offspring_fitness[:(self.population_size - self.elitism_size)]])

            # Update best
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_fitness = fitness[current_best_idx]
                best_solution = population[current_best_idx].copy()

            # Track convergence
            self.convergence_history["iteration"].append(generation)
            self.convergence_history["best_objective"].append(best_fitness)
            self.convergence_history["average_objective"].append(np.mean(fitness))

            if self.verbose and generation % max(1, self.max_iterations // 10) == 0:
                logger.info(f"Generation {generation+1}/{self.max_iterations} | Best: {best_fitness:.6e}")

        final_constraints = self._evaluate_constraints(best_solution, constraints)

        return OptimizationResult(
            success=True,
            message="GA optimization complete",
            optimal_design=best_solution,
            optimal_objective=best_fitness,
            constraint_violations=final_constraints,
            num_iterations=self.max_iterations,
            num_function_evaluations=self.num_evaluations,
            convergence_history=self.convergence_history
        )

    def _tournament_selection(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """Select parents using tournament selection."""
        n_parents = len(population)
        parents = []

        for _ in range(n_parents):
            tournament_idx = np.random.choice(len(population), self.tournament_size, replace=False)
            tournament_fitness = fitness[tournament_idx]
            winner_idx = tournament_idx[np.argmin(tournament_fitness)]
            parents.append(population[winner_idx])

        return np.array(parents)

    def _crossover(self, parents: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        """Perform crossover (single-point)."""
        n_offspring = len(parents)
        n_dim = parents.shape[1]
        offspring = parents.copy()

        for i in range(0, n_offspring - 1, 2):
            if np.random.random() < self.crossover_rate:
                parent1 = parents[i]
                parent2 = parents[i + 1]

                # Single-point crossover
                crossover_point = np.random.randint(1, n_dim)
                offspring[i, :crossover_point] = parent2[:crossover_point]
                offspring[i + 1, :crossover_point] = parent1[:crossover_point]

        return offspring

    def _mutate(self, offspring: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        """Perform mutation (uniform random)."""
        for i in range(len(offspring)):
            if np.random.random() < self.mutation_rate:
                mutation_idx = np.random.randint(0, offspring.shape[1])
                offspring[i, mutation_idx] = np.random.uniform(
                    bounds[mutation_idx, 0], bounds[mutation_idx, 1]
                )

        return self._check_bounds(offspring, bounds)


# ============================================================================
# DIFFERENTIAL EVOLUTION (DE)
# ============================================================================

class DifferentialEvolution(OptimizationAlgorithm):
    """Differential Evolution algorithm."""

    def __init__(self, config: dict):
        """Initialize DE."""
        super().__init__(config)

        de_config = config.get("de", {})
        self.population_size = de_config.get("population_size", 30)
        self.mutation_factor = de_config.get("mutation_factor", 0.8)
        self.crossover_probability = de_config.get("crossover_probability", 0.9)

        # Parallelization
        self.parallel = config.get("parallel", True)
        self.num_workers = config.get("num_workers", 16)

        logger.info(f"DE initialized: population={self.population_size}, "
                   f"parallel={self.parallel}, workers={self.num_workers}")

    def optimize(
        self,
        objective: Callable,
        constraints: List[Constraint],
        bounds: np.ndarray
    ) -> OptimizationResult:
        """Run DE optimization with parallel support."""
        start_time = time.time()
        n_dim = bounds.shape[0]

        if self.verbose:
            logger.info("=" * 60)
            logger.info(f"DE Optimization Starting")
            logger.info(f"Population: {self.population_size}, Iterations: {self.max_iterations}")
            logger.info("=" * 60)

        # Initialize population
        population = np.random.uniform(
            bounds[:, 0], bounds[:, 1],
            size=(self.population_size, n_dim)
        )

        # Evaluate initial population
        if self.parallel and self.num_workers > 1:
            fitness = self._evaluate_parallel_de(population, objective)
        else:
            fitness = np.array([self._evaluate_objective(ind, objective) for ind in population])

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        for generation in range(self.max_iterations):
            # Generate trial population
            trial_population = []

            for i in range(self.population_size):
                # Mutation: DE/rand/1
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = a + self.mutation_factor * (b - c)
                mutant = self._check_bounds(mutant, bounds)

                # Crossover
                trial = population[i].copy()
                crossover_mask = np.random.random(n_dim) < self.crossover_probability
                trial[crossover_mask] = mutant[crossover_mask]

                # Ensure at least one parameter is from mutant
                if not np.any(crossover_mask):
                    trial[np.random.randint(n_dim)] = mutant[np.random.randint(n_dim)]

                trial_population.append(trial)

            trial_population = np.array(trial_population)

            # Evaluate trial population in parallel
            if self.parallel and self.num_workers > 1:
                trial_fitness = self._evaluate_parallel_de(trial_population, objective)
            else:
                trial_fitness = np.array([self._evaluate_objective(t, objective) for t in trial_population])

            # Selection
            for i in range(self.population_size):
                if trial_fitness[i] < fitness[i]:
                    population[i] = trial_population[i]
                    fitness[i] = trial_fitness[i]

                    if trial_fitness[i] < best_fitness:
                        best_fitness = trial_fitness[i]
                        best_solution = trial_population[i].copy()

            # Track convergence
            self.convergence_history["iteration"].append(generation)
            self.convergence_history["best_objective"].append(best_fitness)
            self.convergence_history["average_objective"].append(np.mean(fitness))

            if self.verbose and generation % max(1, self.max_iterations // 10) == 0:
                logger.info(f"Generation {generation+1}/{self.max_iterations} | Best: {best_fitness:.6e}")

        final_constraints = self._evaluate_constraints(best_solution, constraints)
        elapsed_time = time.time() - start_time

        if self.verbose:
            logger.info("=" * 60)
            logger.info(f"DE Complete: Best = {best_fitness:.6e} in {elapsed_time:.2f}s")
            logger.info(f"Evaluations: {self.num_evaluations}")
            logger.info("=" * 60)

        return OptimizationResult(
            success=True,
            message="DE optimization complete",
            optimal_design=best_solution,
            optimal_objective=best_fitness,
            constraint_violations=final_constraints,
            num_iterations=self.max_iterations,
            num_function_evaluations=self.num_evaluations,
            convergence_history=self.convergence_history
        )

    def _evaluate_parallel_de(self, population: np.ndarray, objective: Callable) -> np.ndarray:
        """Evaluate population in parallel (same as PSO)."""
        if not self.parallel or self.num_workers <= 1:
            return np.array([self._evaluate_objective(p, objective) for p in population])

        try:
            with mp.Pool(self.num_workers) as pool:
                fitness = pool.map(objective, population)

            self.num_evaluations += len(population)
            return np.array(fitness)

        except Exception as e:
            logger.warning(f"Parallel execution failed: {e}, falling back to serial")
            return np.array([self._evaluate_objective(p, objective) for p in population])


# ============================================================================
# SEQUENTIAL QUADRATIC PROGRAMMING (SQP)
# ============================================================================

class SequentialQuadraticProgramming(OptimizationAlgorithm):
    """SQP using scipy.optimize.minimize."""

    def __init__(self, config: dict):
        """Initialize SQP."""
        super().__init__(config)
        self.method = config.get("method", "SLSQP")

        logger.info(f"SQP initialized: method={self.method}")

    def optimize(
        self,
        objective: Callable,
        constraints: List[Constraint],
        bounds: np.ndarray
    ) -> OptimizationResult:
        """Run SQP optimization."""
        start_time = time.time()

        # Initial guess (midpoint)
        x0 = np.mean(bounds, axis=1)

        # Convert constraints to scipy format
        scipy_constraints = []
        for constraint in constraints:
            def con_func(x, c=constraint):
                return -c.evaluate(x, {})  # Scipy uses g(x) >= 0

            scipy_constraints.append({
                'type': 'ineq',
                'fun': con_func
            })

        # Scipy bounds
        scipy_bounds = Bounds(bounds[:, 0], bounds[:, 1])

        # Optimize
        result = minimize(
            objective,
            x0,
            method=self.method,
            bounds=scipy_bounds,
            constraints=scipy_constraints,
            options={'maxiter': self.max_iterations, 'disp': self.verbose}
        )

        self.num_evaluations = result.nfev
        final_constraints = self._evaluate_constraints(result.x, constraints)

        return OptimizationResult(
            success=result.success,
            message=result.message,
            optimal_design=result.x,
            optimal_objective=result.fun,
            constraint_violations=final_constraints,
            num_iterations=result.nit,
            num_function_evaluations=result.nfev,
            convergence_history=self.convergence_history
        )


# ============================================================================
# ALGORITHM FACTORY
# ============================================================================

def create_algorithm(algorithm_type: str, config: dict) -> OptimizationAlgorithm:
    """Factory function to create optimization algorithms."""
    algorithms = {
        "pso": ParticleSwarmOptimization,
        "ga": GeneticAlgorithm,
        "de": DifferentialEvolution,
        "differential_evolution": DifferentialEvolution,  # Alias
        "sqp": SequentialQuadraticProgramming
    }

    if algorithm_type.lower() not in algorithms:
        raise ValueError(f"Unknown algorithm: {algorithm_type}. Choose from {list(algorithms.keys())}")

    return algorithms[algorithm_type.lower()](config)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def check_convergence(
    history: Dict[str, List[float]],
    tolerance: float = 1e-6,
    window: int = 10
) -> bool:
    """Check if optimization has converged."""
    if len(history["best_objective"]) < window:
        return False

    recent_values = history["best_objective"][-window:]
    improvement = abs(recent_values[0] - recent_values[-1])

    return improvement < tolerance
