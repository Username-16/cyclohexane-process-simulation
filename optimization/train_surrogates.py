"""
optimization/train_surrogates.py

Run this script ONCE before optimization to train and save surrogate models
on real cyclohexane process simulation data.

Usage:
    python -m optimization.train_surrogates              # 200 samples, all models
    python -m optimization.train_surrogates --samples 500
    python -m optimization.train_surrogates --models gpr randomforest

Author: King Saud University – Chemical Engineering
Version: 1.0.0
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np
import pickle
import warnings
from optimization.surrogate_models import (
    create_surrogate,
    evaluate_surrogate_accuracy,
    cross_validate_surrogate,
    save_trained_surrogates,
    load_trained_surrogates,
    plot_training_data,
    plot_model_accuracy,
    plot_model_comparison,
    plot_stacking_comparison,
    plot_learning_curve_from_summary,
    plot_r2_heatmap,
    plot_best_model_bars,
    plot_stacking_gain_all,
    plot_learning_curve_per_target,
    StackingSurrogate,
    HybridSurrogate,
    XGBOOST_AVAILABLE,
    DEFAULT_SAVE_DIR,
)


# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Also silence sklearn, surrogate_models, and simulation_adapter loggers
for _noisy in ["sklearn", "optimization.surrogate_models",
               "optimization.simulation_adapter", "__main__"]:
    logging.getLogger(_noisy).setLevel(logging.WARNING)


# =============================================================================
# PARALLEL WORKER  (module-level so ProcessPoolExecutor can pickle it)
# =============================================================================
def _evaluate_single(args):
    """
    Worker function — must be module-level for pickling.
    Fully silences all output from the flowsheet simulator.
    """
    import sys
    import io

    # ── Redirect BEFORE importing or calling anything ─────────────────────────
    _old_stdout = sys.stdout
    _old_stderr = sys.stderr
    sys.stdout  = io.StringIO()
    sys.stderr  = io.StringIO()

    try:
        idx, x, var_names = args
        design_dict = {name: float(x[j]) for j, name in enumerate(var_names)}

        from optimization.simulation_adapter import create_flowsheet_evaluator
        evaluator = create_flowsheet_evaluator()
        result    = evaluator(design_dict)

        if not result.get("converged", False):
            return idx, None, "not converged"
        return idx, result, None

    except Exception as e:
        return idx, None, str(e)

    finally:
        # ── Always restore stdout so the progress bar still works ─────────────
        sys.stdout = _old_stdout
        sys.stderr = _old_stderr

# =============================================================================
# DATA GENERATION  (parallel)
# =============================================================================

def generate_training_data(
    n_samples: int = 2500,
    seed: int = 42,
    cache_path: Optional[str] = None,
    n_workers: int = None,
) -> tuple:

    # ── Cache hit ─────────────────────────────────────────────────────────────
    if cache_path and os.path.exists(cache_path):
        print(f"  Loading cached data from {cache_path}")
        data = np.load(cache_path, allow_pickle=True).item()
        print(f"  {data['X'].shape[0]} samples  "
              f"({len(data['y_dict'])} output targets)\n")
        return data["X"], data["y_dict"]

    # ── LHS sampling ──────────────────────────────────────────────────────────
    X_all = latin_hypercube_sample(BOUNDS, n_samples, seed=seed)

    # ── NO serial probe phase — use hardcoded TARGETS directly ───────────────
    targets = TARGETS

    # ── Parallel evaluation ───────────────────────────────────────────────────
    if n_workers is None:
        n_workers = max(1, (mp.cpu_count() or 4) - 1)

    print(f"  Parallel: {n_samples} samples × {n_workers} workers\n")

    args_list   = [(i, X_all[i], VAR_NAMES) for i in range(n_samples)]
    raw_results = [None] * n_samples
    failed = 0
    done   = 0
    t0     = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_evaluate_single, a): a[0] for a in args_list}
        for future in as_completed(futures):
            idx, result, err = future.result()
            done += 1
            if result is None:
                failed += 1
            else:
                raw_results[idx] = result

            if done % 5 == 0 or done == n_samples:
                elapsed = time.time() - t0
                rate    = done / elapsed if elapsed > 0 else 1
                eta     = (n_samples - done) / rate
                pct     = done / n_samples * 100
                filled  = int(40 * done / n_samples)
                bar     = "█" * filled + "░" * (40 - filled)
                print(
                    f"\r  [{bar}] {pct:5.1f}%  "
                    f"done={done}/{n_samples}  valid={done-failed}  "
                    f"failed={failed}  ETA={eta/60:.1f}min",
                    end="", flush=True,
                )
    print()

    # ── Collect valid results ─────────────────────────────────────────────────
    X_valid = []
    y_raw   = {k: [] for k in targets}

    for i, result in enumerate(raw_results):
        if result is None:
            continue
        X_valid.append(X_all[i])
        for tname, extractor in targets.items():
            y_raw[tname].append(extractor(result))

    if len(X_valid) < 10:
        raise RuntimeError(f"Only {len(X_valid)} valid samples — check your flowsheet.")

    X      = np.array(X_valid)
    y_dict = {k: np.array(v) for k, v in y_raw.items()}

    # ── Cache ─────────────────────────────────────────────────────────────────
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        np.save(cache_path, {"X": X, "y_dict": y_dict})
        print(f"\n  Cached → {cache_path}")

    return X, y_dict

# =============================================================================
# ACTIVE LEARNING — multi-criterion stopping
# =============================================================================

# Stop reason codes (returned in the info dict for logging)
_STOP_TARGET_R2    = "target_r2_achieved"
_STOP_BUDGET       = "sample_budget_exhausted"
_STOP_TIME         = "wall_time_exceeded"
_STOP_ROUNDS       = "max_rounds_reached"
_STOP_CONVERGED    = "disagreement_converged"
_STOP_NO_VALID     = "no_valid_simulations"
_STOP_NO_MODELS    = "no_surrogates_found"

def active_learning_loop(
    X_base:              np.ndarray,
    y_dict_base:         dict,
    save_dir:            str,
    # ── How many rounds / samples ─────────────────────────────────────────
    max_rounds:          int   = 5,       # hard cap on AL iterations
    samples_per_round:   int   = 150,     # new simulations per round
    # ── Stopping criteria (first triggered wins) ──────────────────────────
    target_mean_r2:      float = 0.97,    # stop early if already excellent
    max_total_samples:   int   = 3500,    # budget: initial + all AL combined
    max_wall_time_min:   float = 180.0,   # hard wall-clock limit (minutes)
    min_disagree_drop:   float = 0.05,    # convergence: disagree must fall
    patience:            int   = 2,       # rounds without disagree drop
    # ── Misc ─────────────────────────────────────────────────────────────
    n_workers:           int   = None,
    seed:                int   = 42,
) -> Tuple[np.ndarray, dict]:
    """
    Iteratively improve the dataset by targeting uncertain design regions.

    Stopping criteria checked in this order every round:
      1. mean_r2 >= target_mean_r2           → success, stop early
      2. len(X)  >= max_total_samples        → budget exhausted
      3. elapsed >= max_wall_time_min        → wall-clock limit
      4. round   >= max_rounds               → hard cap (loop condition)
      5. disagree not dropping for `patience` rounds → converged

    NOTE: R² is re-evaluated via quick cross-validation on a 300-sample
    subsample after each round — NOT read from the stale summary JSON.
    """
    import warnings

    X      = X_base.copy()
    y_dict = {k: v.copy() for k, v in y_dict_base.items()}

    # ── Tracking state ────────────────────────────────────────────────────
    r2_history       = []       # mean R² after each round (quick CV)
    disagree_history = []       # mean disagreement score per round
    no_drop_count    = 0        # consecutive rounds without disagree drop
    stop_reason      = None
    wall_t0          = time.time()

    # ── Header ────────────────────────────────────────────────────────────
    print(f"\n  {'─'*66}")
    print(f"  ACTIVE LEARNING  |  max_rounds={max_rounds}  "
          f"samples/round={samples_per_round}")
    print(f"  Stops when ANY of:")
    print(f"    • mean R²     ≥ {target_mean_r2}     (target achieved)")
    print(f"    • total samples ≥ {max_total_samples}     (budget exhausted)")
    print(f"    • wall time   ≥ {max_wall_time_min:.0f} min    (time limit)")
    print(f"    • {max_rounds} rounds reached               (hard cap)")
    print(f"    • disagreement not dropping for {patience} rounds (converged)")
    print(f"  {'─'*66}\n")

    for round_idx in range(max_rounds):
        elapsed_min = (time.time() - wall_t0) / 60.0

        print(f"  Round {round_idx + 1}/{max_rounds}  "
              f"| samples: {len(X)}  "
              f"| elapsed: {elapsed_min:.1f} min")

        # ── Criterion 2: sample budget ────────────────────────────────────
        if len(X) >= max_total_samples:
            stop_reason = _STOP_BUDGET
            print(f"  ✓ STOP: sample budget reached "
                  f"({len(X)} ≥ {max_total_samples})")
            break

        # ── Criterion 3: wall-clock time ──────────────────────────────────
        if elapsed_min >= max_wall_time_min:
            stop_reason = _STOP_TIME
            print(f"  ✓ STOP: wall-clock limit reached "
                  f"({elapsed_min:.1f} ≥ {max_wall_time_min} min)")
            break

        # ── Load current surrogates ───────────────────────────────────────
        target_names   = list(y_dict.keys())
        all_surrogates = {}
        for tname in target_names:
            tdir = os.path.join(save_dir, tname)
            if os.path.isdir(tdir):
                try:
                    all_surrogates[tname] = list(
                        load_trained_surrogates(tdir).values()
                    )
                except Exception:
                    pass

        if not all_surrogates:
            stop_reason = _STOP_NO_MODELS
            print("  ✗ STOP: no surrogates found — run base training first")
            break

        # ── Score candidates by inter-model disagreement ──────────────────
        n_cand = samples_per_round * 30
        X_cand = latin_hypercube_sample(
            BOUNDS, n_cand, seed=seed + round_idx * 7
        )
        disagree_scores = np.zeros(n_cand)

        for tname, sur_list in all_surrogates.items():
            if len(sur_list) < 2:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                preds = np.column_stack([
                    sur.predict(X_cand)[0] for sur in sur_list
                ])
            y_range = float(np.ptp(y_dict[tname])) or 1.0
            disagree_scores += preds.std(axis=1) / y_range

        mean_disagree = float(disagree_scores.mean())
        disagree_history.append(mean_disagree)
        print(f"    Mean disagreement: {mean_disagree:.5f}")

        # ── Criterion 5: disagreement convergence ─────────────────────────
        if len(disagree_history) >= 2:
            drop = disagree_history[-2] - disagree_history[-1]
            print(f"    Disagreement drop: {drop:+.5f}  "
                  f"(threshold: {min_disagree_drop:.5f})")
            if drop < min_disagree_drop:
                no_drop_count += 1
                print(f"    No-drop count: {no_drop_count}/{patience}")
                if no_drop_count >= patience:
                    stop_reason = _STOP_CONVERGED
                    print(f"\n  ✓ STOP: disagreement converged "
                          f"(no drop ≥ {patience} rounds)")
                    break
            else:
                no_drop_count = 0

        # ── Select top uncertain points ───────────────────────────────────
        # Respect remaining sample budget
        remaining = max_total_samples - len(X)
        n_query   = min(samples_per_round, remaining)
        top_idx   = np.argsort(disagree_scores)[::-1][:n_query]
        X_query   = X_cand[top_idx]
        print(f"    Querying {n_query} uncertain points …")

        # ── Run real flowsheet on uncertain points ────────────────────────
        _n_workers  = n_workers or max(1, (mp.cpu_count() or 4) - 1)
        args_list   = [(i, X_query[i], VAR_NAMES) for i in range(len(X_query))]
        raw_results = [None] * len(X_query)

        with ProcessPoolExecutor(max_workers=_n_workers) as pool:
            futures = {pool.submit(_evaluate_single, a): a[0]
                       for a in args_list}
            for future in as_completed(futures):
                idx, result, _ = future.result()
                if result is not None:
                    raw_results[idx] = result

        # ── Collect valid results ─────────────────────────────────────────
        X_new, y_new = [], {k: [] for k in target_names}
        for i, result in enumerate(raw_results):
            if result is None:
                continue
            X_new.append(X_query[i])
            for tname in target_names:
                extractor = TARGETS.get(tname, lambda r: 0.0)
                y_new[tname].append(extractor(result))

        n_valid = len(X_new)
        if n_valid == 0:
            stop_reason = _STOP_NO_VALID
            print("  ✗ STOP: no valid simulations returned")
            break

        # ── Append to dataset ─────────────────────────────────────────────
        X = np.vstack([X, np.array(X_new)])
        for tname in target_names:
            if y_new[tname]:
                y_dict[tname] = np.concatenate(
                    [y_dict[tname], np.array(y_new[tname])]
                )
        print(f"    Added {n_valid} valid samples → total: {len(X)}")

        # ── Criterion 1: quick R² check on a subsample (300 pts) ─────────
        # Uses current surrogates on NEW data only — no retraining needed
        sub_n   = min(300, n_valid)
        sub_idx = np.random.choice(n_valid, sub_n, replace=False)
        X_sub   = np.array(X_new)[sub_idx]
        r2_vals = []
        for tname, sur_list in all_surrogates.items():
            # Use the best (last) surrogate from preference order
            best_sur = sur_list[-1]
            y_sub = np.array(y_new[tname])[sub_idx] if y_new[tname] else None
            if y_sub is None or len(y_sub) < 3:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p, _ = best_sur.predict(X_sub)
                from sklearn.metrics import r2_score as _r2
                r2_vals.append(float(_r2(y_sub, p)))
            except Exception:
                pass

        if r2_vals:
            mean_r2 = float(np.mean(r2_vals))
            r2_history.append(mean_r2)
            print(f"    Quick R² on new samples: {mean_r2:.4f}  "
                  f"(target: {target_mean_r2})")

            # ── Criterion 1: target R² achieved ──────────────────────────
            if mean_r2 >= target_mean_r2:
                stop_reason = _STOP_TARGET_R2
                print(f"\n  ✓ STOP: target R² achieved "
                      f"({mean_r2:.4f} ≥ {target_mean_r2})")
                break

        # ── Save expanded cache ───────────────────────────────────────────
        cache_path = os.path.join(save_dir, "process_training_data.npy")
        np.save(cache_path, {"X": X, "y_dict": y_dict})
        print(f"    Cache saved → {cache_path}\n")

    else:
        stop_reason = _STOP_ROUNDS

    # ── Final summary ─────────────────────────────────────────────────────
    elapsed_min = (time.time() - wall_t0) / 60.0
    print(f"\n  Active learning finished")
    print(f"  Stop reason  : {stop_reason}")
    print(f"  Final samples: {len(X)}")
    print(f"  Elapsed      : {elapsed_min:.1f} min")
    if r2_history:
        print(f"  R² history   : {[round(v,4) for v in r2_history]}")
    print()

    return X, y_dict

# =============================================================================
# DESIGN VARIABLE SPACE  (must match optimization_main.py exactly)
# =============================================================================

DESIGN_VARIABLES = {
    "stage_volume_1":              (2.0,   4.0),
    "stage_volume_2":              (2.0,   4.0),
    "stage_volume_3":              (2.0,   4.0),
    "stage_volume_4":              (2.0,   4.0),
    "stage_volume_5":              (2.0,   4.0),
    "stage_volume_6":              (2.0,   4.0),
    "h2_recycle_fraction":         (0.60,  0.99),
    "liquid_recycle_fraction":     (0.20,  0.65),
    "distillate_recycle_fraction": (0.90,  0.995),
    "h2_benzene_feed_ratio":       (3.0,   5.0),
    "distillate_LK_mole_frac":     (0.20,  0.95),
    "distillate_HK_mole_frac":     (0.05,  0.60),
    "bottoms_LK_mole_frac":        (0.001, 0.30),
    "bottoms_HK_mole_frac":        (0.70,  0.999),
    "reflux_ratio_factor":         (1.5,   7.0),
}

VAR_NAMES  = list(DESIGN_VARIABLES.keys())
BOUNDS     = np.array(list(DESIGN_VARIABLES.values()))   # shape (15, 2)
N_VARS     = len(VAR_NAMES)

# Target outputs to build a surrogate for each
# Each target maps to one surrogate set saved in its own subdirectory
# ── Remove the hardcoded TARGETS dict entirely ────────────────────────────────

# ── REMOVE discover_outputs() entirely ───────────────────────────────────────
# ── REPLACE with this hardcoded dict (keys confirmed from debug_outputs.py) ──

def _make_extractor(dot_key: str):
    """Build a safe nested-dict extractor from a dot-notation key."""
    parts = dot_key.split(".")
    def extractor(r):
        v = r
        for part in parts:
            if not isinstance(v, dict):
                return 0.0
            v = v.get(part, 0.0)
        return float(v) if isinstance(v, (int, float)) else 0.0
    return extractor


# Keys confirmed to VARY from debug_outputs.py  (constants excluded)
_VARYING_KEYS = [
    # ── Main objectives (all confirmed varying) ───────────────────────────
    "economics.capex_USD",
    "economics.opex_annual_USD",
    "economics.steam_cost_USD_yr",

    # ── Energy ────────────────────────────────────────────────────────────
    "KPIs.total_energy_kW",
    "KPIs.heating_duty_kW",

    # ── Electricity/compressor — log-transform applied below ─────────────
    "KPIs.compressor_power_kW",
    "utilities.electricity_kW",

    # ── Constraints ───────────────────────────────────────────────────────
    "products.purity_percent",
    "equipment.distillation_total_duty_kW",
    "equipment.total_reactor_volume_m3",
]


TARGETS = {key: _make_extractor(key) for key in _VARYING_KEYS}

import math

def _log_extractor(dot_key: str):
    """Extract value then log-transform for wide-range targets."""
    parts = dot_key.split(".")
    def extractor(r):
        v = r
        for part in parts:
            if not isinstance(v, dict): return 0.0
            v = v.get(part, 0.0)
        val = float(v) if isinstance(v, (int, float)) else 0.0
        return math.log1p(max(0.0, val))   # log(1+x) safe for val≥0
    return extractor

TARGETS["KPIs.compressor_power_kW"] = _log_extractor("KPIs.compressor_power_kW")
TARGETS["utilities.electricity_kW"]  = _log_extractor("utilities.electricity_kW")

# =============================================================================
# LATIN HYPERCUBE SAMPLING
# =============================================================================

def latin_hypercube_sample(bounds: np.ndarray, n_samples: int,
                            seed: int = 42) -> np.ndarray:
    """Generate n_samples points using Latin Hypercube Sampling."""
    np.random.seed(seed)
    n_vars  = bounds.shape[0]
    samples = np.zeros((n_samples, n_vars))
    for i in range(n_vars):
        perm       = np.random.permutation(n_samples)
        points     = (perm + np.random.rand(n_samples)) / n_samples
        lo, hi     = bounds[i, 0], bounds[i, 1]
        samples[:, i] = lo + points * (hi - lo)
    return samples

# =============================================================================
# MODEL CONFIGS
# =============================================================================

MODEL_CONFIGS = {
    "gpr": {
        "kernel": "Matern", "normalize_y": True,
        "n_restarts_optimizer": 5,
    },
    "randomforest": {
        "n_estimators": 2000, "max_depth": None,
        "min_samples_leaf": 2,
    },
    "polynomial": {"degree": 2},
    "neuralnetwork": {
        "architecture": [128, 64, 32],
        "epochs": 1000, "patience": 30,
        "learning_rate": 0.001,
    },
    "xgboost": {
        "n_estimators": 1000,
        "max_depth": 6,
        "learning_rate": 0.01,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },
}


# =============================================================================
# TRAINING ENTRY POINT
# =============================================================================

def train_all_surrogates(
    n_samples: int = 2000,
    models_to_train: Optional[List[str]] = None,
    save_dir: str = DEFAULT_SAVE_DIR,
    test_fraction: float = 0.20,
    seed: int = 42,
    n_workers: int = None,
    do_stacking: bool = True,
    do_active_learning: bool = False,
    active_rounds: int = 5,
    active_samples: int = 150,
    active_target_r2: float = 0.97,
    active_max_samples: int = 3500,
    active_max_time_min: float = 180.0,
) -> Dict:

    # ── Header ────────────────────────────────────────────────────────────
    print(f"\n  {'─'*70}")
    print(f"  SURROGATE TRAINING — Cyclohexane Process")
    print(f"  Samples: {n_samples}   Workers: {n_workers or 'auto'}")
    print(f"  Save dir: {save_dir}")
    print(f"  {'─'*70}")

    if models_to_train is None:
        models_to_train = ["gpr", "randomforest", "polynomial", "neuralnetwork"]
        if XGBOOST_AVAILABLE:
            models_to_train.append("xgboost")
    print(f"  Models: {models_to_train}\n")

    # ── Output directories ────────────────────────────────────────────────
    plots_dir = os.path.join(save_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # ── Generate / load data ──────────────────────────────────────────────
    cache_path = os.path.join(save_dir, "process_training_data.npy")
    X, y_dict = generate_training_data(
        n_samples=n_samples,
        seed=seed,
        cache_path=cache_path,
        n_workers=n_workers,
    )

    # ── Active learning (optional) ────────────────────────────────────────
    if do_active_learning:
        X, y_dict = active_learning_loop(
            X, y_dict,
            save_dir=save_dir,
            max_rounds=active_rounds,
            samples_per_round=active_samples,
            target_mean_r2=active_target_r2,
            max_total_samples=active_max_samples,
            max_wall_time_min=active_max_time_min,
            n_workers=n_workers,
            seed=seed,
        )
        # Re-run training on the expanded dataset
        return train_all_surrogates(
            n_samples=n_samples,
            models_to_train=models_to_train,
            save_dir=save_dir,
            test_fraction=test_fraction,
            seed=seed,
            n_workers=n_workers,
            do_stacking=do_stacking,
            do_active_learning=False,
        )

    # ── Train / test split ────────────────────────────────────────────────
    n = X.shape[0]
    n_test = max(5, int(n * test_fraction))
    n_train = n - n_test
    np.random.seed(seed)
    idx = np.random.permutation(n)
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    X_train, X_test = X[train_idx], X[test_idx]
    print(f"  Data split: {n_train} train / {n_test} test\n")

    # ── Plot training data overview ───────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_training_data(
            X_train, y_dict[list(y_dict.keys())[0]][train_idx],
            feature_names=VAR_NAMES,
            save_path=os.path.join(plots_dir, "training_data_overview.png"),
        )

    # ── Per-target training loop ──────────────────────────────────────────
    full_summary = {}

    for target_name, y_all in y_dict.items():
        print(f"  {'─'*70}")
        print(f"  Target: {target_name.upper()}"
              f"   range [{y_all.min():.4g}, {y_all.max():.4g}]")
        print(f"  {'─'*70}")

        y_train = y_all[train_idx]
        y_test  = y_all[test_idx]

        target_dir = os.path.join(save_dir, target_name)
        os.makedirs(target_dir, exist_ok=True)

        trained  = {}
        accuracy = {}

        # ── Train each model ──────────────────────────────────────────────
        for key in models_to_train:
            if key not in MODEL_CONFIGS:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cfg = MODEL_CONFIGS[key]
                    sur = create_surrogate(key, cfg)
                    t0  = time.time()
                    sur.fit(X_train, y_train)
                    dt  = time.time() - t0
                    acc = evaluate_surrogate_accuracy(sur, X_test, y_test)
                    trained[sur.get_name()]  = sur
                    accuracy[sur.get_name()] = acc
                    status = "✓" if acc["r2_score"] >= 0.8 else "⚠"
                    print(f"  {status} {sur.get_name():<35}"
                          f"  R²={acc['r2_score']:.4f}"
                          f"  RMSE={acc['rmse']:.4g}"
                          f"  ({dt:.1f}s)")
            except Exception as e:
                print(f"  ✗ {key:<36}  FAILED: {e}")

        if not trained:
            print(f"  ERROR: No models trained for {target_name}")
            continue

        # ── Cross-validate best model ─────────────────────────────────────
        best_name = max(accuracy, key=lambda k: accuracy[k]["r2_score"])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv = cross_validate_surrogate(
                    trained[best_name], X_train, y_train, num_folds=5
                )
            print(f"    CV {best_name}: R²={cv['mean_r2']:.4f} ± {cv['std_r2']:.4f}")
        except Exception:
            pass

        # ── Stacking meta-NN ──────────────────────────────────────────────
        if do_stacking and len(trained) >= 2:
            try:
                stacking = StackingSurrogate({})
                stacking.set_base_models(list(trained.values()))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    t0 = time.time()
                    stacking.fit(X_train, y_train)
                    dt = time.time() - t0
                accs = evaluate_surrogate_accuracy(stacking, X_test, y_test)
                trained["StackingMeta"]  = stacking
                accuracy["StackingMeta"] = accs
                best_base_r2 = max(
                    accuracy[k]["r2_score"]
                    for k in accuracy if k != "StackingMeta"
                )
                gain   = accs["r2_score"] - best_base_r2
                status = "✓" if accs["r2_score"] >= 0.8 else "⚠"
                print(f"  {status} {'StackingMeta':<35}"
                      f"  R²={accs['r2_score']:.4f}"
                      f"  RMSE={accs['rmse']:.4g}"
                      f"  gain={gain:+.4f}  ({dt:.1f}s)")
            except Exception as e:
                print(f"  ⚠ Stacking FAILED: {e}")

        # ── Save models ───────────────────────────────────────────────────
        import io as _io, contextlib
        with contextlib.redirect_stdout(_io.StringIO()):
            save_trained_surrogates(trained, accuracy, save_dir=target_dir)

        # ── Per-target plots ──────────────────────────────────────────────
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plot_model_accuracy(
                accuracy,
                save_path=os.path.join(plots_dir, f"accuracy_{target_name}.png"),
            )
            plot_model_comparison(
                accuracy,
                save_path=os.path.join(plots_dir, f"comparison_{target_name}.png"),
            )
            if do_stacking and "StackingMeta" in accuracy:
                plot_stacking_comparison(
                    {k: v for k, v in accuracy.items() if k != "StackingMeta"},
                    accuracy["StackingMeta"],
                    target_name,
                    save_path=os.path.join(plots_dir, f"stacking_{target_name}.png"),
                )

        # ── Store in full_summary ─────────────────────────────────────────
        full_summary[target_name] = {
            k: {
                "r2_score": float(v["r2_score"]),
                "rmse":     float(v["rmse"]),
                "mae":      float(v["mae"]),
            }
            for k, v in accuracy.items()
        }

    # ── Save master summary JSON ──────────────────────────────────────────
    summary_path = os.path.join(save_dir, "full_training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(full_summary, f, indent=2)

    # ── Print final table ─────────────────────────────────────────────────
    print(f"\n  {'─'*70}")
    print("  TRAINING COMPLETE")
    print(f"  Models saved → {save_dir}")
    print(f"  Plots  saved → {plots_dir}")
    print()
    print(f"  {'Target':<15}  {'Best Model':<28}  {'R²':>8}  {'RMSE':>10}")
    print(f"  {'─'*65}")
    for tname, models in full_summary.items():
        best  = max(models, key=lambda k: models[k]["r2_score"])
        r2    = models[best]["r2_score"]
        rmse  = models[best]["rmse"]
        flag  = "⚠ " if r2 < 0.8 else ""
        print(f"  {flag}{tname:<14}  {best:<28}  {r2:>8.4f}  {rmse:>10.4g}")
    print(f"  {'─'*70}")

    # ── Build and save HybridSurrogate ────────────────────────────────────
    try:
        hybrid = HybridSurrogate.from_save_dir(save_dir, VAR_NAMES)
        hybrid_path = os.path.join(save_dir, "hybrid_surrogate.pkl")
        with open(hybrid_path, "wb") as f:
            pickle.dump(hybrid, f)
        print(f"\n  HybridSurrogate saved → {hybrid_path}")
        print(f"  Usage in optimization_main.py:")
        print(f"    import pickle")
        print(f"    hybrid = pickle.load(open('{hybrid_path}', 'rb'))")
        print(f"    result = hybrid.predict(design_dict)")
    except Exception as e:
        print(f"\n  [warn] HybridSurrogate build failed: {e}")

    # ── Showcase plots (plots_dir, full_summary, X all defined above) ─────
    print("\n  ── Showcase Plots ──────────────────────────────────────────")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plot_r2_heatmap(full_summary,
                save_path=os.path.join(plots_dir, "showcase_1_heatmap.png"))
            plot_best_model_bars(full_summary,
                save_path=os.path.join(plots_dir, "showcase_2_best_model.png"))
            plot_stacking_gain_all(full_summary,
                save_path=os.path.join(plots_dir, "showcase_3_stacking_gain.png"))
            plot_learning_curve_per_target(full_summary, n_train=len(X),
                save_path=os.path.join(plots_dir, "showcase_4_learning_curves.png"))
    except Exception as e:
        print(f"  [warn] Showcase plots skipped: {e}")

    return full_summary

# =============================================================================
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)   # ← required on Windows
    parser = argparse.ArgumentParser(
        description="Train surrogate models on cyclohexane process data"
    )
    parser.add_argument("--samples",  type=int,   default=2500)
    parser.add_argument("--workers",  type=int,   default=None,
                        help="Parallel workers for data generation (default: ncpu-1)")
    parser.add_argument("--models",   nargs="*",  default=None)
    parser.add_argument("--save-dir", type=str,   default=DEFAULT_SAVE_DIR)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--no-stacking", action="store_true")
    parser.add_argument("--active-learning", action="store_true")
    parser.add_argument("--active-rounds", type=int, default=3)
    parser.add_argument("--active-samples", type=int, default=150)
    parser.add_argument("--active-target-r2", type=float, default=0.97)
    parser.add_argument("--active-max-samples", type=int, default=3500)
    parser.add_argument("--active-max-time", type=float, default=180.0)
    args = parser.parse_args()

    train_all_surrogates(
        n_samples=args.samples,
        models_to_train=args.models,
        save_dir=args.save_dir,
        seed=args.seed,
        n_workers=args.workers,
        do_stacking=not args.no_stacking,
        do_active_learning=args.active_learning,
        active_rounds=args.active_rounds,
        active_samples=args.active_samples,
        active_target_r2    = args.active_target_r2,
        active_max_samples  = args.active_max_samples,
        active_max_time_min = args.active_max_time,
    )
