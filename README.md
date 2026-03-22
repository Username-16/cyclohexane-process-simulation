# Cyclohexane Process Simulation & AI Optimisation

A production-grade Python process simulation engine for industrial benzene 
hydrogenation to cyclohexane (55,455 t/yr), built without any commercial 
simulator. Developed as a BSc graduation project at King Saud University (CHE 497).

**Supervisors:** Prof. Oualid Hamdaoui · Dr. Yousef Alanazi  
**Authors:** Hamad Almousa · Saud Alrumaih  

---

## Key Results

| Metric | Value |
|---|---|
| Cyclohexane purity | 99.84 mol% |
| Benzene conversion | >81% |
| Energy reduction (vs. baseline) | 10.2% (10.61 → 9.53 MW) |
| Surrogate model mean R² | 0.978 across 10 KPIs |
| Computational speedup | ~10⁵× vs. full simulation |
| Recycle convergence | 46 iterations, tolerance 1×10⁻³ |
| Payback period | 3.73 years |
| Gross profit | 55.46M SAR/year |

---

## Repository Structure

```
cyclohexane_project/
├── config/
│   └── process_parameters.json
├── simulation/
│   ├── streams.py
│   ├── thermodynamics.py
│   ├── flowsheet.py
│   └── main.py
├── reaction/
│   ├── kinetics.py
│   └── reactor.py
├── separation/
│   ├── flash.py
│   ├── distillation.py
│   └── membrane.py
├── heat_transfer/
│   └── heat_exchanger.py
├── utilities/
│   ├── pump.py
│   ├── compressor.py
│   ├── valve.py
│   ├── mixer.py
│   └── splitter.py
├── optimization/
│   ├── surrogate_models.py
│   ├── algorithms.py
│   ├── objective_functions.py
│   ├── constraints.py
│   ├── train_surrogates.py
│   ├── optimizer.py
│   ├── optimization_main.py
│   ├── multiobjective.py
│   ├── sensitivity.py
│   └── uncertainty.py
└── reports/
```

## Simulation Engine

The engine uses a **Sequential-Modular (SM)** architecture — identical in 
philosophy to Aspen HYSYS and ChemCAD — where modules communicate exclusively 
through `Stream` objects and are solved in topological (DAG) order.

**Thermodynamics:** Peng-Robinson EOS with literature-validated binary 
interaction parameters for all H₂–benzene–cyclohexane pairs.

**Kinetics:** Langmuir-Hinshelwood-Hougen-Watson (LHHW) rate expression 
(Saeys et al., 2004) on Pt-based catalyst, with 6 reactions tracked.

**Recycle convergence:** Wegstein acceleration method on 3 tear streams 
(H₂ recycle, liquid recycle, distillate recycle).

---

## AI & ML Optimisation

### Surrogate Model Benchmark

| Model | Algorithm | R² (best target) |
|---|---|---|
| `GaussianProcessSurrogate` | GPR, RBF/Matérn kernel | 0.991 |
| `XGBoostSurrogate` | Gradient-boosted trees | 0.997 |
| `RandomForestSurrogate` | Tree ensemble | 0.985 |
| `NeuralNetworkSurrogate` | MLP 128→64→32 | 0.976 |
| `PolynomialSurrogate` | Response surface (degree 2) | 0.941 |
| `StackingSurrogate` | Level-2 MLP meta-learner | **0.978 mean** |

Training set: **2,000 Latin Hypercube samples** across a 15-dimensional 
design space.

### Optimisation Algorithms

- **Particle Swarm Optimisation (PSO)** — inertia-weight variant
- **Differential Evolution (DE)** — rand/bin mutation with adaptive crossover
- **Bayesian Optimisation** — GP-guided acquisition function

Parallel execution across **8 CPU cores** via `ProcessPoolExecutor`.

### Analysis

- **Sobol global sensitivity indices** — variance decomposition across all 15 design variables
- **Monte Carlo uncertainty propagation** — 10,000 samples over feed and kinetic parameter distributions
- **Multi-objective Pareto front** — purity vs. energy trade-off via NSGA-II

---

## Quick Start

```bash
git clone https://github.com/hamad-almousa/cyclohexane-process-simulation.git
cd cyclohexane-process-simulation
pip install -r requirements.txt

# Run full simulation
python simulation/main.py

# Run optimisation (all 5 scenarios, parallel)
python optimization/optimization_main.py

Process Overview

Feed: 310 kmol/h benzene + 980 kmol/h H₂
Reactor: 6-stage adiabatic fixed-bed (R-101), 90°C inlet → 240°C outlet
Separation: Flash (V-101) → Distillation (T-101) → Pervaporation membrane (M-101)
Product: 310 kmol/h cyclohexane at 99.84 mol% purity
