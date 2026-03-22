"""
separation/membrane.py
Membrane Separator — Pervaporation membrane unit for final cyclohexane purification.

Membrane:  PS/PAA concentrated-emulsion membrane
Reference: Sun, H., Ruckenstein, E. (1995). J. Membrane Sci. 99, 273-284.
Review:    Garcia Villaluenga, J.P., Tabe-Mohammadi, A. (2000).
           J. Membrane Sci. 169, 159-174.

Separation is driven by high vacuum on permeate side (0.01 torr).
Benzene (smaller, aromatic) permeates preferentially via sorption-diffusion.
Cyclohexane (larger, aliphatic) is retained as high-purity product.

Key improvements over v2.0.0 (PEMA-EGDM):
  - PS/PAA flux Q1 = 48.4e-6 kg.m/(m2.h) - 5.6x higher than PEMA-EGDM
  - alpha = 9.6 at 50 wt% Bz reference; rises to ~14-15 at dilute 2 wt% feed
    due to reduced membrane swelling (Garcia Villaluenga 2000, Fig. 8)
  - theta is now SOLVED automatically via bisection to meet target_purity

Date:    2026-02-27
Version: 3.2.0 — PS/PAA + theta bisection + correct Stream API (flowrate_kmol_h)
"""

import math
import logging
from typing import Dict, Tuple, Optional, Any
from simulation.streams import Stream

logger = logging.getLogger(__name__)

# ── Physical constants ────────────────────────────────────────────────────────
R_J_PER_MOL_K = 8.314          # Universal gas constant  J/(mol.K)
TORR_TO_BAR   = 1.0 / 750.062  # 1 torr = 1/750.062 bar
BAR_TO_PA     = 1.0e5          # 1 bar  = 100,000 Pa

# ── Membrane: PS/PAA concentrated-emulsion (Sun & Ruckenstein 1995) ──────────
PV_MEMBRANE_TYPE     = "PS/PAA concentrated-emulsion membrane"
PV_MEMBRANE_REF      = "Sun, H., Ruckenstein, E. (1995), J. Membrane Sci. 99, 273-284"
PV_ALPHA_DEFAULT     = 9.6     # alpha at 50 wt% Bz, 20 C (reference condition)
PV_ALPHA_REF_WT      = 0.50   # wt fraction Bz at which alpha_default was measured
PV_ALPHA_SLOPE       = -10.5  # dalpha/d(xBz_wt): selectivity rises as Bz dilutes
PV_Q1_DEFAULT        = 48.4e-6 # kg.m/(m2.h) thickness-normalised flux
PV_THICKNESS_DEFAULT = 50.0   # um film thickness
PV_TEMPERATURE_C     = 20.0   # optimal operating temperature C
PV_PERMEATE_TORR     = 0.01   # downstream vacuum torr

# ── Molecular weights  kg/kmol ────────────────────────────────────────────────
MW = {
    "benzene":            78.0,
    "cyclohexane":        84.0,
    "H2":                  2.0,
    "methane":            16.0,
    "methylcyclopentane": 84.0,
    "H2O":                18.0,
    "nitrogen":           28.0,
    "coke":               12.0,
}
MW_DEFAULT = 80.0   # fallback kg/kmol


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def get_mw(component: str) -> float:
    return MW.get(component, MW_DEFAULT)


def alpha_corrected(alpha_base: float, xbenzene_wt: float,
                    ref_wt: float = PV_ALPHA_REF_WT,
                    slope: float  = PV_ALPHA_SLOPE) -> float:
    """
    Correct separation factor for feed benzene weight fraction.
    At our feed (xBz_wt ~ 0.019): alpha_eff ~ 9.6 + (-10.5)*(0.019 - 0.50) ~ 14.6
    """
    alpha = alpha_base + slope * (xbenzene_wt - ref_wt)
    return max(1.05, alpha)


def pvcompositions(xB: float, xC: float,
                   alpha: float, theta: float) -> Tuple[float, float, float, float]:
    """
    PV separation-factor model for Bz/Chx binary subspace.
    Definition:  yB/yC = alpha * xB/xC
    Balance:     F*xi = P*yi + R*zi,  P = theta*F,  R = (1-theta)*F
    Returns yB, yC, zB, zC.
    """
    total = xB + xC
    if total == 0.0:
        return xB, xC, xB, xC
    xnB = xB / total
    xnC = xC / total
    yB  = alpha * xnB / (xnC + alpha * xnB)
    yC  = 1.0 - yB
    theta = min(theta, 0.9999)
    zB  = (xnB - theta * yB) / (1.0 - theta)
    zC  = 1.0 - zB
    zB  = max(0.0, min(1.0, zB))
    zC  = max(0.0, min(1.0, zC))
    return yB, yC, zB, zC


def solve_theta_for_purity(alpha_eff: float, xBz: float, xChx: float,
                           target_purity: float = 0.995,
                           theta_min: float     = 0.001,
                           theta_max: float     = 0.60) -> Tuple[float, bool]:
    """
    Bisection solver: find stage-cut theta so retentate Chx purity = target_purity.
    Returns (theta_solved, converged: bool).
    """
    def purity_at_theta(th: float) -> float:
        _, _, zB, zC = pvcompositions(xBz, xChx, alpha_eff, th)
        denom = zB + zC
        return zC / denom if denom > 0 else 0.0

    p_max = purity_at_theta(theta_max)
    if p_max < target_purity:
        logger.warning(
            f"solve_theta_for_purity: cannot reach {target_purity*100:.1f}% "
            f"even at theta_max={theta_max:.3f} (achieved {p_max*100:.2f}%). "
            f"Consider a two-stage PV design or higher-alpha membrane."
        )
        return theta_max, False

    lo, hi = theta_min, theta_max
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if purity_at_theta(mid) >= target_purity:
            hi = mid
        else:
            lo = mid
    theta_solved = (lo + hi) / 2.0
    logger.info(
        f"solve_theta_for_purity: converged theta={theta_solved:.6f} "
        f"-> purity={purity_at_theta(theta_solved)*100:.4f}%"
    )
    return theta_solved, True


def vacuum_power_kw(permeate_flow_kmolh: float, permeate_pressure_bar: float,
                    temperature_C: float, vent_pressure_bar: float = 1.0,
                    eta: float = 0.50) -> float:
    """Isothermal vacuum pump power in kW."""
    if permeate_flow_kmolh <= 0.0 or permeate_pressure_bar <= 0.0:
        return 0.0
    if vent_pressure_bar <= permeate_pressure_bar:
        return 0.0
    T_K    = temperature_C + 273.15
    n_mols = permeate_flow_kmolh * 1000.0 / 3600.0
    W_ideal = n_mols * R_J_PER_MOL_K * T_K * math.log(vent_pressure_bar / permeate_pressure_bar)
    return W_ideal / max(eta, 0.01) / 1000.0


# ─────────────────────────────────────────────────────────────────────────────
# Main unit operation
# ─────────────────────────────────────────────────────────────────────────────

def run_membrane_separator(feed_stream: Stream,
                           config: Dict[str, Any],
                           thermo,
                           unit_id: str = "M-101") -> Tuple[Stream, Stream, Dict]:
    """
    Run pervaporation membrane separator for cyclohexane purification (M-101).

    Returns
    -------
    retentate : Stream   S25  - high-purity cyclohexane product
    permeate  : Stream   S24_perm - benzene-rich vapour for recycle
    summary   : Dict     full equipment summary
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"MEMBRANE SEPARATOR {unit_id} - Cyclohexane Purification PV")
    logger.info("=" * 70)

    # ── 1. Read config ────────────────────────────────────────────────────────
    membrane_type          = config.get("membrane_type",          PV_MEMBRANE_TYPE)
    target_purity          = config.get("target_purity",          0.995)
    alpha_sep              = config.get("alpha_sep",               PV_ALPHA_DEFAULT)
    theta_override         = config.get("theta",                   None)
    solve_theta_auto       = config.get("solve_theta",             True)
    Q1                     = config.get("Q1_kg_m_per_m2_h",        PV_Q1_DEFAULT)
    thickness_um           = config.get("thickness_um",             50.0)
    permeate_pressure_torr = config.get("permeate_pressure_torr",  PV_PERMEATE_TORR)
    vacuum_eta             = config.get("vacuum_eta",               0.50)
    P_permeate_bar         = permeate_pressure_torr * TORR_TO_BAR

    # ── 2. Unpack feed stream ─────────────────────────────────────────────────
    F     = feed_stream.flowrate_kmol_h
    comp  = feed_stream.composition
    Tfeed = feed_stream.temperature_C
    Pfeed = feed_stream.pressure_bar

    # ── 3. Feed composition — compute BEFORE calling solver ──────────────────
    xBz      = comp.get("benzene",     0.0)
    xChx     = comp.get("cyclohexane", 0.0)
    denom_mw = sum(v * get_mw(k) for k, v in comp.items())
    wtfrac_Bz = xBz * get_mw("benzene") / denom_mw if denom_mw > 0.0 else 0.0
    alpha_eff  = alpha_corrected(alpha_sep, wtfrac_Bz)

    # ── 4. Solve theta to hit target purity ───────────────────────────────────
    theta_converged = True
    if solve_theta_auto and theta_override is None:
        theta, theta_converged = solve_theta_for_purity(
            alpha_eff, xBz, xChx,
            target_purity=target_purity,
            theta_min=0.001,
            theta_max=0.60,
        )
    else:
        theta = float(theta_override) if theta_override is not None else 0.033
        theta = max(0.001, min(theta, 0.999))

    # ── 5. Log parameters ─────────────────────────────────────────────────────
    logger.info(f"  Feed:          {F:.2f} kmol/h | {Tfeed:.1f} C | {Pfeed:.3f} bar")
    logger.info(f"  Membrane:      {membrane_type}")
    logger.info(f"  Base alpha:    {alpha_sep:.2f}")
    logger.info(f"  Feed Bz wt%:   {wtfrac_Bz*100:.3f}%")
    logger.info(f"  alpha_eff:     {alpha_eff:.3f}")
    logger.info(f"  theta:         {theta:.6f}  "
                f"({'auto-solved' if solve_theta_auto and theta_override is None else 'manual'})")
    logger.info(f"  Target purity: {target_purity*100:.2f}% Chx")
    logger.info(f"  Permeate vac:  {permeate_pressure_torr:.4f} torr ({P_permeate_bar:.4e} bar)")
    logger.info("")

    # ── 6. PV separation model ────────────────────────────────────────────────
    yB, yC, zB, zC = pvcompositions(xBz, xChx, alpha_eff, theta)
    F_BC  = (xBz + xChx) * F
    P_BC  = F_BC * theta
    R_BC  = F_BC - P_BC

    # ── 7. Full component material balance ────────────────────────────────────
    permeate_flows  = {}
    retentate_flows = {}

    for component, xi in comp.items():
        n_in = xi * F
        if component == "benzene":
            permeate_flows[component]  = P_BC * yB
            retentate_flows[component] = R_BC * zB
        elif component == "cyclohexane":
            permeate_flows[component]  = P_BC * yC
            retentate_flows[component] = R_BC * zC
        elif component in ("H2", "methane"):
            permeate_flows[component]  = n_in * 0.99
            retentate_flows[component] = n_in * 0.01
        elif component == "methylcyclopentane":
            permeate_flows[component]  = n_in * 0.10
            retentate_flows[component] = n_in * 0.90
        elif component in ("H2O", "nitrogen"):
            permeate_flows[component]  = n_in * 0.90
            retentate_flows[component] = n_in * 0.10
        else:
            permeate_flows[component]  = n_in * theta
            retentate_flows[component] = n_in * (1.0 - theta)

    F_permeate  = sum(permeate_flows.values())
    F_retentate = sum(retentate_flows.values())

    permeate_comp  = {k: v / F_permeate  for k, v in permeate_flows.items()}  if F_permeate  > 0 else {}
    retentate_comp = {k: v / F_retentate for k, v in retentate_flows.items()} if F_retentate > 0 else {}

    # ── 8. Performance metrics ────────────────────────────────────────────────
    actual_purity    = retentate_comp.get("cyclohexane", 0.0) * 100.0
    actual_bz_prod   = retentate_comp.get("benzene",     0.0) * 100.0
    chx_in_feed      = comp.get("cyclohexane", 0.0) * F
    chx_in_product   = retentate_flows.get("cyclohexane", 0.0)
    actual_recovery  = (chx_in_product / chx_in_feed * 100.0) if chx_in_feed > 0 else 0.0

    # ── 9. Membrane sizing ────────────────────────────────────────────────────
    thickness_m       = thickness_um * 1.0e-6
    J_kg_per_m2_h     = Q1 / thickness_m
    MW_perm           = sum(permeate_comp.get(k, 0.0) * get_mw(k) for k in permeate_comp)
    MW_perm           = MW_perm if MW_perm > 0 else MW_DEFAULT
    permeate_mass_kgh = F_permeate * MW_perm
    membrane_area_m2  = permeate_mass_kgh / J_kg_per_m2_h

    # ── 10. Vacuum system ─────────────────────────────────────────────────────
    vac_power_kw     = vacuum_power_kw(F_permeate, P_permeate_bar, Tfeed, 1.0, vacuum_eta)
    vac_energy_kwh_yr = vac_power_kw * 8000.0

    # ── 11. Log results ───────────────────────────────────────────────────────
    logger.info(f"  Retentate (S25):  {F_retentate:.2f} kmol/h  PRODUCT")
    logger.info(f"  Permeate (S24p):  {F_permeate:.2f} kmol/h")
    logger.info(f"  Chx purity:       {actual_purity:.4f}%  (target {target_purity*100:.2f}%)")
    logger.info(f"  Bz in product:    {actual_bz_prod:.6f}%")
    logger.info(f"  Chx recovery:     {actual_recovery:.3f}%")
    logger.info(f"  Membrane area:    {membrane_area_m2:.2f} m2")
    logger.info(f"  Pump power:       {vac_power_kw:.2f} kW")
    logger.info("")

    # ── 12. Build output streams ──────────────────────────────────────────────
    retentate = Stream(
        name        = "Cyclohexane Product S25",
        flowrate_kmol_h = F_retentate,
        temperature_C   = Tfeed,
        pressure_bar    = Pfeed - 0.1,
        composition  = retentate_comp,
        thermo       = thermo,
        phase        = "liquid",
    )
    permeate = Stream(
        name        = "Membrane Permeate S24_perm",
        flowrate_kmol_h = F_permeate,
        temperature_C   = Tfeed,
        pressure_bar    = P_permeate_bar,
        composition  = permeate_comp,
        thermo       = thermo,
        phase        = "vapor",
    )

    # ── 13. Summary dict ──────────────────────────────────────────────────────
    summary = {
        "equipment_type":               "PV Membrane Separator",
        "equipment_id":                 unit_id,
        "function":                     "Cyclohexane purification via pervaporation",
        "membrane_type":                membrane_type,
        "membrane_reference":           PV_MEMBRANE_REF,
        "feed_flowrate_kmol_h":         F,
        "operating_temperature_C":      Tfeed,
        "feed_pressure_bar":            Pfeed,
        "permeate_pressure_bar":        P_permeate_bar,
        "permeate_pressure_torr":       permeate_pressure_torr,
        "retentate_flowrate_kmol_h":    F_retentate,
        "permeate_flowrate_kmol_h":     F_permeate,
        "permeate_fraction_theta":      theta,
        "theta_solved_auto":            solve_theta_auto and theta_override is None,
        "theta_converged":              theta_converged,
        "alpha_sep_input":              alpha_sep,
        "alpha_sep_effective":          alpha_eff,
        "feed_benzene_wt_fraction":     wtfrac_Bz,
        "cyclohexane_purity_percent":   actual_purity,
        "cyclohexane_recovery_percent": actual_recovery,
        "benzene_in_product_percent":   actual_bz_prod,
        "target_purity_percent":        target_purity * 100.0,
        "purity_margin_pct":            round(actual_purity - target_purity * 100.0, 4),
        "Q1_kg_m_per_m2_h":            Q1,
        "membrane_thickness_um":        thickness_um,
        "membrane_flux_kg_per_m2_h":   J_kg_per_m2_h,
        "membrane_area_m2":             membrane_area_m2,
        "vacuum_pump_power_kw":         vac_power_kw,
        "vacuum_pump_eta":              vacuum_eta,
        "annual_vacuum_energy_kwh":     vac_energy_kwh_yr,
        "two_stage_pv_recommended":     not theta_converged,
        "status":                       "Operating",
    }

    logger.info(f"{unit_id} complete - S25 -> storage | S24_perm -> cold trap")
    logger.info("=" * 70)
    return retentate, permeate, summary


# ─────────────────────────────────────────────────────────────────────────────
# Utility helper
# ─────────────────────────────────────────────────────────────────────────────

def calculate_membrane_area(flowrate_kmolh: float,
                            permeance: float = 1.86e-3,
                            pressure_difference_bar: float = None) -> float:
    """Estimate required membrane area for PS/PAA pervaporation membrane."""
    if pressure_difference_bar is None:
        pressure_difference_bar = 0.8 - PV_PERMEATE_TORR * TORR_TO_BAR
    if permeance <= 0.0 or pressure_difference_bar <= 0.0:
        return 1000.0
    area = flowrate_kmolh / (permeance * pressure_difference_bar)
    return max(10.0, min(50000.0, area))
