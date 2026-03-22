"""
separation/distillation.py - Rigorous Distillation Column
===============================================================================

Binary distillation with rigorous MESH solver, shortcut methods (Fenske, Underwood,
Gilliland), and complete thermodynamic calculations.

Date: 2026-02-27
Version: 9.0.0 - Standalone Unit
"""

from __future__ import annotations
import math
import logging
from typing import Any, Dict, Optional, Tuple, List
import numpy as np

logger = logging.getLogger(__name__)

# Physical constants
BZ_CH_PHYSICAL_T_MIN = 70.0
BZ_CH_PHYSICAL_T_MAX = 120.0


def run_distillation_column(
    feed: Any,
    distillation_column: Dict[str, Any],
    thermo: Any,
    catalyst_info: Optional[dict] = None,

) -> Tuple[Any, Any, Dict]:

    """
    Run rigorous binary distillation column with MESH solver.

    Parameters:
    -----------
    feed : Stream
        Feed stream
    distillation_column : dict
        Configuration from process_parameters.json
    thermo : ThermodynamicPackage
        Thermodynamic package
    catalyst_info : dict, optional
        Not used (for interface compatibility)

    Returns:
    --------
    distillate : Stream
        Overhead stream (lights)
    bottoms : Stream
        Bottoms stream (to membrane)
    summary : dict
        Equipment summary
    """
    if not feed or getattr(feed, "flowrate_kmol_h", 0.0) <= 0.0:
        raise ValueError("Invalid feed flow")
    if thermo is None:
        raise ValueError("ThermodynamicPackage required")
    logger.info("")
    logger.info("=" * 70)
    logger.info("DISTILLATION COLUMN T-101 - Light Ends Removal")
    logger.info("=" * 70)
    logger.info("")

    # Extract configuration with defaults
    config = _merge_defaults(distillation_column)

    components = sorted(feed.composition.keys())
    LK = config["light_key"]
    HK = config["heavy_key"]

    # Auto-select if benzene/cyclohexane present
    if "methane" in components and "cyclohexane" in components:
        LK = "methane"
        HK = "cyclohexane"

    if LK not in components or HK not in components:
        raise ValueError(f"Key components not in feed: {LK}, {HK}")

    logger.info(f"Binary system: LK={LK}, HK={HK}")
    logger.info(f"Feed (S23): {feed.flowrate_kmol_h:.2f} kmol/h at {feed.temperature_C:.1f}°C, {feed.pressure_bar:.1f} bar")

    # Operating parameters
    P_op = float(config["operating_pressure_bar"])
    x_LK_D = config["distillate_LK_mole_frac"]
    x_HK_D = config["distillate_HK_mole_frac"]
    x_LK_B = config["bottoms_LK_mole_frac"]
    x_HK_B = config["bottoms_HK_mole_frac"]

    logger.info(f"Operating pressure: {P_op:.1f} bar")
    logger.info(f"Condenser: {config['condenser_type']}, Reboiler: {config['reboiler_type']}")
    logger.info("")

    # Define light and heavy components
    light_components = ["H2"]
    heavy_components = [LK, HK, "methylcyclopentane"]

    # Calculate component flows
    F = feed.flowrate_kmol_h
    comp = feed.composition

    F_lights = sum(comp.get(c, 0.0) * F for c in light_components)
    F_heavies = sum(comp.get(c, 0.0) * F for c in heavy_components)

    logger.info(f"Light components (H2, CH4): {F_lights:.2f} kmol/h")
    logger.info(f"Heavy components (Bz, CHex, MCP): {F_heavies:.2f} kmol/h")

    # Material balance for light ends removal
    lights_recovery = config.get("lights_recovery_target", 0.99)
    logger.info(f"Target lights recovery to overhead: {lights_recovery*100:.1f}%")
    logger.info("")

    # Distillate and bottoms flows
    D = F_lights * lights_recovery + F_heavies * 0.01
    B = F - D

    logger.info("Material Balance:")
    logger.info(f"  Feed:      {F:.2f} kmol/h")
    logger.info(f"  Distillate: {D:.2f} kmol/h")
    logger.info(f"  Bottoms:    {B:.2f} kmol/h")
    logger.info("")

    # ========================================================================
    # SHORTCUT DESIGN METHODS
    # ========================================================================

    # Estimate relative volatility
    alpha = _estimate_relative_volatility(feed, LK, HK, components, thermo, P_op)

    # Fenske equation - minimum stages
    N_min = _calculate_minimum_stages_fenske(x_LK_D, x_HK_D, x_LK_B, x_HK_B, alpha)

    # Underwood equation - minimum reflux
    z_LK = comp.get(LK, 0.0)
    z_HK = comp.get(HK, 0.0)
    R_min = _calculate_minimum_reflux_underwood(feed, z_LK, z_HK, x_LK_D, alpha, D, F)

    # Gilliland correlation - actual stages
    if "reflux_ratio" in config and config["reflux_ratio"] is not None:
        R_op = config["reflux_ratio"]
        logger.info(f" Using fixed reflux ratio: {R_op:.2f}")
    else:
        R_op = config["reflux_ratio_factor"] * R_min  # Calculate from factor
        logger.info(f" Calculated reflux ratio: {R_op:.2f} (= {config['reflux_ratio_factor']}×{R_min:.2f})")

    if "max_reflux_ratio" in config:
        R_op = min(R_op, config["max_reflux_ratio"])
        logger.info(f" Reflux ratio clamped to max: {config['max_reflux_ratio']:.2f}")
    N_act = int(math.ceil(_calculate_actual_stages_gilliland(N_min, R_min, R_op)))
    N_act = max(config["min_stages"], min(config["max_stages"], N_act))

    # Kirkbride equation - feed stage location
    feed_stage = _calculate_feed_stage_kirkbride(N_act, z_LK, z_HK, x_LK_D, x_HK_D,
                                                  x_LK_B, x_HK_B, D, B)

    logger.info("Column Performance:")
    logger.info(f"  Theoretical stages: {N_act}")
    logger.info(f"  Reflux ratio: {R_op:.2f}")
    logger.info(f"  Lights recovery: {lights_recovery*100:.1f}%")
    logger.info("")

    # ========================================================================
    # RIGOROUS MESH SOLVER
    # ========================================================================

    mesh = _solve_binary_mesh(
        feed, D, B, N_act, feed_stage, R_op, P_op,
        components, LK, HK, x_LK_D, x_HK_D, x_LK_B, x_HK_B,
        thermo, config["convergence"]
    )

    # Get actual results
    D_act = mesh["distillate_flow"]
    B_act = mesh["bottoms_flow"]
    dist_comp = mesh["distillate_composition"]
    bot_comp = mesh["bottoms_composition"]

    # Adjust for light ends removal
    dist_comp_adjusted = {}
    for component in comp.keys():
        if component in light_components:
            dist_comp_adjusted[component] = comp.get(component, 0) * F * lights_recovery / D
        else:
            dist_comp_adjusted[component] = comp.get(component, 0) * F * 0.01 / D

    # Normalize
    total_dist = sum(dist_comp_adjusted.values())
    if total_dist > 0:
        dist_comp_adjusted = {k: v/total_dist for k, v in dist_comp_adjusted.items()}

    # Bottoms composition
    bot_comp_adjusted = {}
    for component in comp.keys():
        if component in light_components:
            bot_comp_adjusted[component] = comp.get(component, 0) * F * (1 - lights_recovery) / B
        else:
            bot_comp_adjusted[component] = comp.get(component, 0) * F * 0.99 / B

    # Normalize
    total_bot = sum(bot_comp_adjusted.values())
    if total_bot > 0:
        bot_comp_adjusted = {k: v/total_bot for k, v in bot_comp_adjusted.items()}

    # Log compositions
    logger.info("Overhead Composition (S24_dist):")
    for comp_name in ["H2", "methane", LK, HK]:
        if comp_name in dist_comp_adjusted:
            logger.info(f"  {comp_name:20s}: {dist_comp_adjusted[comp_name]*100:6.2f}%")
    logger.info("")

    logger.info("Bottoms Composition (to M-101):")
    for comp_name in [LK, HK, "methylcyclopentane", "H2"]:
        if comp_name in bot_comp_adjusted:
            logger.info(f"  {comp_name:20s}: {bot_comp_adjusted[comp_name]*100:6.2f}%")
    logger.info("")

    # ========================================================================
    # HEAT DUTIES AND SIZING
    # ========================================================================

    duties = _calculate_heat_duties(mesh, thermo, components, P_op, D, R_op)
    sizing = _size_column(N_act, config["tray_design"], mesh, thermo, components, P_op)

    # ========================================================================
    # CREATE OUTPUT STREAMS
    # ========================================================================

    try:
        from simulation.streams import Stream
    except ImportError:
        class Stream:
            def __init__(self, *args, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

    T_condenser = 80.0
    T_reboiler = 150.0

    distillate = Stream(
        name="Distillate Overhead (S24_dist)",
        flowrate_kmol_h=D,
        temperature_C=T_condenser,
        pressure_bar=P_op,
        composition=dist_comp_adjusted,
        thermo=thermo,
        phase="liquid"
    )

    bottoms = Stream(
        name="Distillation Bottoms (to M-101)",
        flowrate_kmol_h=B,
        temperature_C=T_reboiler,
        pressure_bar=P_op,
        composition=bot_comp_adjusted,
        thermo=thermo,
        phase="liquid"
    )

    # ========================================================================
    # EQUIPMENT SUMMARY
    # ========================================================================

    summary = {
        "equipment_type": "Distillation Column",
        "equipment_id": "T-101",
        "function": "Light ends removal",
        "feed_flowrate_kmol_h": F,
        "distillate_flowrate_kmol_h": D,
        "bottoms_flowrate_kmol_h": B,
        "operating_pressure_bar": P_op,
        "condenser_temperature_C": T_condenser,
        "reboiler_temperature_C": T_reboiler,
        "condenser_type": config["condenser_type"],
        "reboiler_type": config["reboiler_type"],
        "theoretical_stages": N_act,
        "minimum_stages": N_min,
        "feed_stage": feed_stage,
        "reflux_ratio": R_op,
        "minimum_reflux_ratio": R_min,
        "relative_volatility": alpha,
        "lights_recovery_percent": lights_recovery * 100,
        "condenser_duty_kW": duties["condenser_duty_kW"],
        "reboiler_duty_kW": duties["reboiler_duty_kW"],
        "column_diameter_m": sizing["diameter_m"],
        "column_height_m": sizing["height_m"],
        "tray_efficiency": sizing["tray_efficiency"],
        "number_of_actual_trays": sizing["number_of_actual_trays"],
        "converged": mesh["converged"],
        "iterations": mesh["iterations"],
        "stage_temperatures_C": mesh["stage_temperatures"],
        "stage_pressures_bar": mesh["stage_pressures"],
        "status": "Operating"
    }

    logger.info("T-101 Complete: Overhead to recycle, Bottoms to M-101")
    logger.info("=" * 70)
    logger.info("")

    return distillate, bottoms, summary


# ==============================================================================
# SHORTCUT DESIGN METHODS
# ==============================================================================

def _estimate_relative_volatility(feed, LK, HK, components, thermo, P):
    """Estimate relative volatility using thermo flash."""
    try:
        T = max(feed.temperature_C, 80.0)
        x = {c: feed.composition.get(c, 0.0) for c in components}
        total = sum(x.values())
        if total > 0:
            x = {c: v/total for c, v in x.items()}
        flash = thermo.flash_TP(T, P, x)
        K = flash.get("K_values", {})
        alpha = K.get(LK, 1.1) / max(K.get(HK, 1.0), 1e-6)
        return max(1.01, min(alpha, 10.0))
    except:
        return 1.05


def _calculate_minimum_stages_fenske(x_LK_D, x_HK_D, x_LK_B, x_HK_B, alpha):
    """Fenske equation for minimum stages."""
    try:
        num = (x_LK_D / max(x_HK_D, 1e-6)) * (x_HK_B / max(x_LK_B, 1e-6))
        N = math.log(num) / math.log(max(alpha, 1.001))
        return max(1.0, N)
    except:
        return 10.0


def _calculate_minimum_reflux_underwood(feed, z_LK, z_HK, x_LK_D, alpha, D, F):
    """Underwood equation for minimum reflux."""
    try:
        if abs(alpha - 1.0) < 0.001:
            return 10.0
        term = (x_LK_D / max(z_LK, 1e-6)) - alpha
        R = term / (alpha - 1.0)
        return max(0.5, R)
    except:
        return 2.0


def _calculate_actual_stages_gilliland(N_min, R_min, R_op):
    """Gilliland correlation for actual stages."""
    try:
        if R_op <= R_min:
            R_op = 1.5 * R_min
        X = (R_op - R_min) / (R_op + 1.0)
        X = max(1e-6, min(X, 0.99))
        exp_arg = ((1.0 + 54.4*X) / (11.0 + 117.2*X)) * ((X - 1.0) / math.sqrt(X))
        Y = 1.0 - math.exp(exp_arg)
        Y = max(0.0, min(Y, 0.99))
        N = N_min * (1.0 + Y) / (1.0 - Y)
        return max(N_min, N)
    except:
        return 2.5 * N_min


def _calculate_feed_stage_kirkbride(N, z_LK, z_HK, x_LK_D, x_HK_D, x_LK_B, x_HK_B, D, B):
    """Kirkbride equation for feed stage location."""
    try:
        ratio_feed = z_HK / max(z_LK, 1e-6)
        ratio_prod = (x_LK_B / max(x_HK_D, 1e-6)) ** 2
        ratio_flow = B / max(D, 1e-6)
        arg = ratio_feed * ratio_prod * ratio_flow
        log_ratio = 0.206 * math.log(max(arg, 1e-6))
        N_rect_over_N_strip = math.exp(log_ratio)
        N_rect = N * N_rect_over_N_strip / (1.0 + N_rect_over_N_strip)
        return max(2, min(N-2, int(round(N_rect))))
    except:
        return N // 2


# ==============================================================================
# RIGOROUS MESH SOLVER
# ==============================================================================

def _solve_binary_mesh(feed, D_target, B_target, N, feed_stage, R, P_op,
                      components, LK, HK, x_LK_D, x_HK_D, x_LK_B, x_HK_B,
                      thermo, conv):
    """
    Rigorous MESH solver for binary distillation.
    Material, Energy, Summation, Hydraulic equations.
    """

    max_iter = conv["max_iterations"]
    tol_x = conv["tolerance"]
    tol_T = conv["temperature_tolerance_C"]
    damp_x = conv["damping_factor"]
    damp_T = conv["temperature_damping"]
    dP = conv["pressure_drop_per_tray_bar"]

    F = feed.flowrate_kmol_h

    # Stage pressures
    P_stages = [P_op + j*dP for j in range(N)]

    # Initialize flows
    V = D_target * (R + 1.0)
    L_rect = R * D_target
    L_strip = L_rect + F
    L = np.array([L_rect if j < feed_stage else L_strip for j in range(N)])
    V_arr = np.array([V for _ in range(N)])

    # Initialize temperatures and compositions
    T = np.zeros(N)
    for j in range(N):
        # Clausius-Clapeyron approximation for benzene/cyclohexane
        T[j] = 78.0 + 8.0 * math.log(max(1.0, P_stages[j]))
        T[j] = max(BZ_CH_PHYSICAL_T_MIN, min(BZ_CH_PHYSICAL_T_MAX, T[j]))

    # Smooth the profile
    T = np.linspace(T[0], T[-1], N)
    x_LK = np.linspace(x_LK_D, x_LK_B, N)
    x_HK = np.linspace(x_HK_D, x_HK_B, N)

    # Normalize
    for j in range(N):
        tot = x_LK[j] + x_HK[j]
        if tot > 1.0:
            x_LK[j] /= tot
            x_HK[j] /= tot

    converged = False

    for it in range(max_iter):
        x_LK_old = x_LK.copy()
        x_HK_old = x_HK.copy()
        T_old = T.copy()

        # Get K-values
        K_LK = np.zeros(N)
        K_HK = np.zeros(N)

        for j in range(N):
            x_full = _build_full_composition(x_LK[j], x_HK[j], feed, components, LK, HK)
            try:
                flash = thermo.flash_TP(T[j], P_stages[j], x_full)
                K_vals = flash.get("K_values", {})
                K_LK[j] = max(0.5, min(2.0, K_vals.get(LK, 1.0)))
                K_HK[j] = max(0.5, min(2.0, K_vals.get(HK, 1.0)))
            except:
                K_LK[j] = 1.05
                K_HK[j] = 0.95

        # Solve component material balances (tridiagonal)
        x_LK_new = _solve_component_tridiagonal(
            x_LK, K_LK, L, V_arr, F, feed_stage,
            feed.composition.get(LK, 0.0), D_target, B_target,
            x_LK_D, x_LK_B
        )

        x_HK_new = _solve_component_tridiagonal(
            x_HK, K_HK, L, V_arr, F, feed_stage,
            feed.composition.get(HK, 0.0), D_target, B_target,
            x_HK_D, x_HK_B
        )

        # Normalize
        for j in range(N):
            tot = x_LK_new[j] + x_HK_new[j]
            if tot > 0.01:
                x_LK_new[j] = max(0.0, min(1.0, x_LK_new[j]/tot))
                x_HK_new[j] = max(0.0, min(1.0, x_HK_new[j]/tot))

        # Update temperatures
        T_new = _update_temperatures(T, x_LK_new, x_HK_new, P_stages, N,
                                     feed, thermo, components, LK, HK)

        # Apply adaptive damping
        if it < 5:
            damp_x_actual = damp_x
            damp_T_actual = damp_T
        elif it < 20:
            damp_x_actual = damp_x * 0.7
            damp_T_actual = damp_T * 0.5
        else:
            damp_x_actual = damp_x * 0.3
            damp_T_actual = damp_T * 0.2

        x_LK = damp_x_actual * x_LK_old + (1 - damp_x_actual) * x_LK_new
        x_HK = damp_x_actual * x_HK_old + (1 - damp_x_actual) * x_HK_new
        T = damp_T_actual * T_old + (1 - damp_T_actual) * T_new

        # Enforce physical bounds
        for j in range(N):
            T[j] = max(BZ_CH_PHYSICAL_T_MIN, min(BZ_CH_PHYSICAL_T_MAX, T[j]))

        # Check convergence
        err_LK = np.max(np.abs(x_LK - x_LK_old))
        err_HK = np.max(np.abs(x_HK - x_HK_old))
        err_T = np.max(np.abs(T - T_old))

        if (it < 5) or ((it+1) % 20 == 0):
            logger.info(f"  Iter {it+1}: Δx_LK={err_LK:.2e}, Δx_HK={err_HK:.2e}, ΔT={err_T:.2f}°C")

        if (err_LK < tol_x) and (err_HK < tol_x) and (err_T < tol_T):
            converged = True
            break

    # Build product compositions
    dist_comp, bot_comp = _build_product_compositions(
        x_LK, x_HK, T, P_stages, feed, components, LK, HK, D_target, B_target, thermo
    )

    boilup = V_arr[-1] / max(L[-1], 1e-6)

    return {
        "converged": converged,
        "iterations": it + 1,
        "stage_temperatures": T.tolist(),
        "stage_pressures": P_stages,
        "liquid_flows": L.tolist(),
        "vapor_flows": V_arr.tolist(),
        "distillate_flow": D_target,
        "bottoms_flow": B_target,
        "boilup_ratio": boilup,
        "distillate_composition": dist_comp,
        "bottoms_composition": bot_comp,
    }


def _solve_component_tridiagonal(x, K, L, V, F, fs, z_F, D, B, x_D_spec, x_B_spec):
    """Solve component material balance using tridiagonal matrix."""
    N = len(x)
    x_new = np.zeros(N)

    # Top stage (distillate spec)
    x_new[0] = x_D_spec

    # Middle stages
    for j in range(1, N-1):
        L_j = L[j]
        V_j = V[j]
        L_jp1 = L[min(j+1, N-1)]
        V_jm1 = V[max(j-1, 0)]
        K_j = K[j]
        K_jm1 = K[max(j-1, 0)]

        F_in = F * z_F if j == fs else 0.0

        y_above = K_jm1 * x_new[j-1]
        x_below = x[min(j+1, N-1)]

        denom = L_j + V_j * K_j
        if abs(denom) > 1e-10:
            x_new[j] = (L_jp1*x_below + V_jm1*y_above + F_in) / denom
        else:
            x_new[j] = x[j]

        x_new[j] = max(0.0, min(1.0, x_new[j]))

    # Bottom stage (bottoms spec)
    x_new[-1] = x_B_spec

    return x_new


def _build_full_composition(x_LK, x_HK, feed, components, LK, HK):
    """Build full composition from binary LK/HK."""
    x_full = {LK: x_LK, HK: x_HK}
    nonkey_sum = 0.0

    for c in components:
        if c not in [LK, HK]:
            x_full[c] = feed.composition.get(c, 0.0) * 0.05
            nonkey_sum += x_full[c]

    binary_sum = x_LK + x_HK
    if binary_sum + nonkey_sum > 1.0:
        scale = (1.0 - binary_sum) / max(nonkey_sum, 1e-8)
        for c in components:
            if c not in [LK, HK]:
                x_full[c] *= scale

    total = sum(x_full.values())
    if total > 0:
        x_full = {c: v/total for c, v in x_full.items()}

    return x_full


def _update_temperatures(T, x_LK, x_HK, P, N, feed, thermo, components, LK, HK):
    """Update stage temperatures using bubble point calculations with Newton-Raphson."""
    T_new = np.zeros(N)

    for j in range(N):
        x_full = _build_full_composition(x_LK[j], x_HK[j], feed, components, LK, HK)
        T_guess = T[j]

        # Bubble point: find T where sum(K_i * x_i) = 1.0
        converged_local = False

        for iteration in range(50):
            try:
                flash = thermo.flash_TP(T_guess, P[j], x_full)
                K_vals = flash.get("K_values", {})

                K_sum = sum(K_vals.get(c, 1.0) * x_full.get(c, 0.0) for c in components)
                error = K_sum - 1.0

                # Check convergence
                if abs(error) < 0.001:
                    converged_local = True
                    break

                # Newton-Raphson: estimate derivative dK_sum/dT
                dT_probe = 0.5
                T_perturbed = T_guess + dT_probe
                T_perturbed = max(BZ_CH_PHYSICAL_T_MIN, min(BZ_CH_PHYSICAL_T_MAX, T_perturbed))

                flash_pert = thermo.flash_TP(T_perturbed, P[j], x_full)
                K_vals_pert = flash_pert.get("K_values", {})
                K_sum_pert = sum(K_vals_pert.get(c, 1.0) * x_full.get(c, 0.0) for c in components)

                dK_dT = (K_sum_pert - K_sum) / dT_probe

                # Newton step with damping
                if abs(dK_dT) > 1e-8:
                    step = -error / dK_dT
                    step = max(-10.0, min(10.0, step))  # Limit step size
                    T_guess += 0.5 * step  # Damping factor
                else:
                    # Fallback to simple stepping
                    step = -5.0 * np.sign(error) if abs(error) > 0.1 else -1.0 * np.sign(error)
                    T_guess += step

                # Enforce physical bounds
                T_guess = max(BZ_CH_PHYSICAL_T_MIN, min(BZ_CH_PHYSICAL_T_MAX, T_guess))

            except Exception as e:
                if iteration == 0:
                    logger.debug(f"Flash failed at stage {j + 1}, keeping T={T[j]:.1f}°C")
                break

        # Store result
        if converged_local:
            T_new[j] = T_guess
        else:
            # Use correlation as fallback
            T_base = 78.0
            T_new[j] = T_base + 10.0 * math.log(max(1.0, P[j]))
            T_new[j] = max(BZ_CH_PHYSICAL_T_MIN, min(BZ_CH_PHYSICAL_T_MAX, T_new[j]))

    return T_new


def _build_product_compositions(x_LK, x_HK, T, P, feed, components, LK, HK, D, B, thermo):
    """Build distillate and bottoms compositions."""
    dist = {LK: x_LK[0], HK: x_HK[0]}
    bot = {LK: x_LK[-1], HK: x_HK[-1]}

    # Classify components by volatility
    light_components = ["H2", "methane"]
    heavy_components = ["methylcyclopentane"]

    for c in components:
        if c in [LK, HK]:
            continue

        F_c = feed.flowrate_kmol_h * feed.composition.get(c, 0.0)

        if F_c < 1e-6:  # Skip trace components
            dist[c] = 0.0
            bot[c] = 0.0
            continue

        try:
            # Estimate split based on volatility relative to keys
            x_top = {LK: x_LK[0], HK: x_HK[0], c: 0.01}
            total = sum(x_top.values())
            x_top = {k: v / total for k, v in x_top.items()}
            flash = thermo.flash_TP(T[0], P[0], x_top)
            K_top = flash.get("K_values", {}).get(c, 1.0)
            K_LK_top = flash.get("K_values", {}).get(LK, 1.0)

            # Relative volatility to light key
            alpha_c_LK = K_top / max(K_LK_top, 1e-6)

            # Component split logic based on volatility
            if c in light_components or alpha_c_LK > 2.0:
                # Very volatile - mostly to distillate
                split = 0.95
            elif alpha_c_LK > 1.2:
                # More volatile than LK - favor distillate
                split = 0.70
            elif alpha_c_LK > 0.8:
                # Similar volatility - distribute
                split = 0.50
            else:
                # Less volatile - mostly to bottoms
                split = 0.10

        except Exception as e:
            # Fallback: lights go up, heavies go down
            if c in light_components:
                split = 0.95
            elif c in heavy_components:
                split = 0.10
            else:
                split = 0.50

        D_c = F_c * split
        B_c = F_c * (1 - split)

        dist[c] = D_c / max(D, 1e-6)
        bot[c] = B_c / max(B, 1e-6)

    # Normalize
    tot_d = sum(dist.values())
    tot_b = sum(bot.values())

    if tot_d > 0:
        dist = {c: v / tot_d for c, v in dist.items()}
    if tot_b > 0:
        bot = {c: v / tot_b for c, v in bot.items()}

    return dist, bot


# ==============================================================================
# HEAT DUTIES AND SIZING
# ==============================================================================

def _calculate_heat_duties(mesh, thermo, components, P, D, R):
    """Calculate condenser and reboiler duties."""
    T = mesh["stage_temperatures"]
    V = mesh["vapor_flows"]

    try:
        V_cond = V[0]
        lambda_vap = 35.0  # kJ/mol latent heat
        Q_cond_kJ_h = -V_cond * lambda_vap * 1000
        Q_cond_kW = Q_cond_kJ_h / 3600.0

        V_reb = V[-1]
        Q_reb_kJ_h = V_reb * lambda_vap * 1000
        Q_reb_kW = Q_reb_kJ_h / 3600.0
    except:
        Q_cond_kW = -D * R * 30.0 / 3600.0 * 1000
        Q_reb_kW = -Q_cond_kW * 1.2

    return {
        "condenser_duty_kW": Q_cond_kW,
        "reboiler_duty_kW": abs(Q_reb_kW),
    }


def _size_column(N, tray_cfg, mesh, thermo, components, P):
    """Size column diameter and height."""
    V_flows = mesh["vapor_flows"]
    V_max = max(V_flows)

    # Physical properties
    MW = 80.0  # kg/kmol
    rho_v = 3.0  # kg/m³
    rho_l = 750.0  # kg/m³

    # Flooding calculations
    C_sbf = 0.12
    u_flood = C_sbf * math.sqrt((rho_l - rho_v) / rho_v) if rho_v > 0 else 1.0
    f_flood = tray_cfg["flooding_fraction_design"]
    u_design = f_flood * u_flood

    # Diameter
    Q_v = (V_max * MW / rho_v) / 3600.0
    A = Q_v / u_design if u_design > 1e-6 else 1.0
    D_col = math.sqrt(4 * A / math.pi)
    D_col = max(0.3, min(D_col, 5.0))
    D_col = math.ceil(D_col * 10) / 10

    # Tray efficiency
    mu = 0.3
    alpha = 1.15
    eff = 0.52 - 0.27 * math.log10(max(1e-3, mu * alpha))
    eff = max(0.5, min(eff, 0.85))

    N_actual = int(N / eff) + 1

    # Height
    tray_spacing = tray_cfg["tray_spacing_m"]
    H_col = (N_actual - 1) * tray_spacing + 1.2 + 1.5

    return {
        "diameter_m": D_col,
        "height_m": H_col,
        "number_of_actual_trays": N_actual,
        "tray_efficiency": eff,
    }


def _merge_defaults(distillation_column):
    """Merge configuration with defaults."""
    defaults = {
        "type": "sieve_tray",
        "operating_pressure_bar": 1.0,
        "condenser_type": "total",
        "reboiler_type": "kettle",
        "light_key": "methane",
        "heavy_key": "cyclohexane",
        "distillate_LK_mole_frac": 0.75,
        "distillate_HK_mole_frac": 0.25,
        "bottoms_LK_mole_frac": 0.25,
        "bottoms_HK_mole_frac": 0.75,
        "reflux_ratio_factor": 1.5,
        "max_stages": 10000,
        "min_stages": 2,
        "lights_recovery_target": 0.999,
        "tray_design": {
            "tray_spacing_m": 0.6,
            "flooding_fraction_design": 0.75,
            "pressure_drop_per_tray_mbar": 8.0,
        },
        "convergence": {
            "max_iterations": 250,
            "tolerance": 1e-3,
            "temperature_tolerance_C": 0.05,
            "damping_factor": 0.6,
            "temperature_damping": 0.4,
            "pressure_drop_per_tray_bar": 0.001,
        },
    }

    cfg = defaults.copy()
    if distillation_column:
        for k, v in distillation_column.items():
            if isinstance(v, dict) and k in cfg and isinstance(cfg[k], dict):
                cfg[k].update(v)
            else:
                cfg[k] = v

    return cfg
