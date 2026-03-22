"""

simulation/flowsheet.py - COMPLETE WITH MEMBRANE + WEGSTEIN + REPORTING

FEATURES:
✓ Membrane separator M-101 (T-101 bottoms → S25 product)
✓ Wegstein acceleration for fast convergence
✓ Periodic progress summaries every 10 iterations
✓ Temperature warnings (non-blocking)
✓ Correct flow pattern: T-101 → M-101 → MIX-103 → recycle
✓ All 3 verbose modes (silent, normal, detailed)
✓ generate_pfd() method included

Version 10.1.0
Date: 2026-02-02

"""

from __future__ import annotations
import json
import logging
from typing import Any, Dict, Optional
from pathlib import Path

from simulation.streams import Stream
from reaction.kinetics import ReactionSystem
from reaction.reactor import run_multibed_reactor
from separation.flash import run_flash
from separation.distillation import run_distillation_column
from separation.membrane import run_membrane_separator
from utilities.pump import pump_liquid
from utilities.compressor import compress_gas
from utilities.mixer import mix_streams_adiabatic
from utilities.splitter import split_stream_by_fractions
from heat_transfer.heat_exchanger import run_heat_exchanger
logger = logging.getLogger(__name__)


def load_process_parameters(config_file: str = None) -> Dict[str, Any]:
    """Load process parameters from JSON."""
    if config_file is None:
        config_dir = Path(__file__).parent.parent / "config"
        config_file = config_dir / "process_parameters.json"

    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing process_parameters.json at {config_path}")

    with open(config_path, 'r') as f:
        params = json.load(f)
    return params


PROCESS_PARAMS = load_process_parameters()


class WegsteinAccelerator:
    """Wegstein acceleration for recycle convergence."""

    def __init__(self, config: Dict[str, Any]):
        self.q_min = config.get('q_bounds', [-5.0, 0.0])[0]
        self.q_max = config.get('q_bounds', [-5.0, 0.0])[1]
        self.safety_max = config.get('safety_factor_max', 3.0)
        self.safety_min = config.get('safety_factor_min', 0.3)
        self.history: Dict[str, Dict[str, list]] = {}

    def register_stream(self, stream_name: str):
        if stream_name not in self.history:
            self.history[stream_name] = {'old': [], 'new': []}

    def record(self, stream_name: str, x_old: float, x_new: float):
        if stream_name not in self.history:
            self.register_stream(stream_name)

        self.history[stream_name]['old'].append(x_old)
        self.history[stream_name]['new'].append(x_new)

        if len(self.history[stream_name]['old']) > 10:
            self.history[stream_name]['old'] = self.history[stream_name]['old'][-10:]
            self.history[stream_name]['new'] = self.history[stream_name]['new'][-10:]

    def can_accelerate(self, stream_name: str) -> bool:
        if stream_name not in self.history:
            return False
        return len(self.history[stream_name]['old']) >= 2

    def accelerate(self, stream_name: str, x_old: float, x_new: float,
                   verbose: bool = False) -> float:
        if not self.can_accelerate(stream_name):
            return x_new

        hist = self.history[stream_name]
        x_prev_old = hist['old'][-1]
        x_prev_new = hist['new'][-1]

        denominator = x_old - x_prev_old
        if abs(denominator) < 1e-10:
            return x_new

        slope = (x_new - x_prev_new) / denominator

        if abs(slope - 1.0) < 1e-10:
            q = -0.5
        else:
            q = slope / (slope - 1.0)

        q = max(self.q_min, min(self.q_max, q))
        x_accelerated = q * x_old + (1 - q) * x_new

        if x_accelerated < 0:
            x_accelerated = 0.5 * x_new

        if x_old > 1e-6:
            ratio = x_accelerated / x_old
            if ratio > self.safety_max:
                x_accelerated = x_old * self.safety_max
            elif ratio < self.safety_min:
                x_accelerated = x_old * self.safety_min

        if verbose:
            print(f"  {stream_name}:")
            print(f"    Slope: {slope:.4f}, q: {q:.4f}")
            print(f"    {x_old:.2f} → {x_new:.2f} → {x_accelerated:.2f} kmol/h")

        return x_accelerated

    def reset(self):
        self.history.clear()


class Flowsheet:
    """
    Cyclohexane production flowsheet with:
    - Membrane separator M-101 for product polishing
    - Wegstein acceleration
    - Periodic progress reporting
    """

    def __init__(self, thermo=None, process_parameters=None, **kwargs):
        self.process_params = process_parameters or PROCESS_PARAMS
        self.thermo = thermo

        if self.thermo is None:
            raise ValueError("ThermodynamicPackage required")

        if hasattr(self.thermo, "components"):
            self.component_list = list(self.thermo.components.keys())
        else:
            raise ValueError("ThermodynamicPackage must have 'components' attribute")

        # Extract convergence settings
        conv_settings = self.process_params["simulation_settings"]
        self.max_iterations = conv_settings["max_iterations"]
        self.tolerance = conv_settings["convergence_tolerance"]
        self.damping = conv_settings.get("damping_factor", 0.3)
        self.verbose_mode = conv_settings.get("verbose_mode", "normal").lower()
        self.report_interval = conv_settings.get("report_interval", 10)

        # Initialize Wegstein accelerator
        accel_config = conv_settings.get("acceleration", {})
        self.acceleration_method = accel_config.get("method", "none")
        self.accel_start_iter = accel_config.get("enable_after_iteration", 3)

        if self.acceleration_method == "wegstein":
            self.accelerator = WegsteinAccelerator(accel_config)
            self.accelerator.register_stream('H2')
            self.accelerator.register_stream('LIQ')
            self.accelerator.register_stream('DIST')
        else:
            self.accelerator = None

        self._configure_logging()

        self.kinetics = ReactionSystem()
        self.feed_specs = self.process_params["feed_specifications"]
        self.reactor_config = self.process_params["reactor_configuration"]
        self.separation_config = self.process_params["separation_configuration"]
        self.hx_config = self.process_params["heat_exchangers"]
        self.pump_comp_config = self.process_params["compressors_and_pumps"]
        self.design_vector = dict(self.process_params["design_parameters"])

        self.recycle_factors = {
            "h2_recycle_factor": 1.2,
            "liquid_recycle_factor": 0.4,
            "distillate_recycle_factor": 0.6,
        }

        self.streams: Dict[str, Stream] = {}
        self.equipment_summaries: Dict[str, dict] = {}
        self.heat_duties: Dict[str, float] = {}
        self.KPIs: Dict[str, Any] = {}
        self.iteration = 0
        self.converged = False
        self.initial_error = None

        self._print_header()

    def _configure_logging(self):
        if self.verbose_mode == "silent":
            level = logging.ERROR
        elif self.verbose_mode == "normal":
            level = logging.WARNING
        elif self.verbose_mode == "detailed":
            level = logging.INFO
        else:
            level = logging.WARNING

        for module in ["reaction.reactor", "reaction.kinetics", "separation.flash",
                      "separation.distillation", "separation.membrane",
                      "utilities.pump", "utilities.compressor",
                      "utilities.mixer", "utilities.splitter"]:
            logging.getLogger(module).setLevel(level)

        logging.getLogger().setLevel(level)

    def _print_header(self):
        if self.verbose_mode == "silent":
            print("\n" + "="*60)
            print("  CYCLOHEXANE PROCESS SIMULATION")
            print("="*60)
            print(f"  Mode: Silent | Tolerance: {self.tolerance:.4f} kmol/h")
            print(f"  Max iterations: {self.max_iterations}")
            if self.acceleration_method == "wegstein":
                print(f"  Acceleration: Wegstein (starts iter {self.accel_start_iter})")
            print("="*60)
        else:
            print("\n" + "="*70)
            print("🚀 CYCLOHEXANE PROCESS SIMULATION")
            print("   with Membrane Separator M-101")
            print("="*70)
            print(f"Verbose mode: {self.verbose_mode.upper()}")
            print(f"Convergence tolerance: {self.tolerance:.4f} kmol/h")
            print(f"Damping factor: {self.damping:.2f}")
            if self.acceleration_method == "wegstein":
                print(f"Acceleration: Wegstein (starts at iteration {self.accel_start_iter})")
            print(f"Max iterations: {self.max_iterations}")
            print(f"Report interval: Every {self.report_interval} iterations")
            print("="*70 + "\n")

    def _get_zero_composition(self):
        return {comp: 0.0 for comp in self.component_list}

    def run_simulation(self, design_vector=None):
        if design_vector:
            self.design_vector.update(design_vector)

        converged = self.solve()

        return {
            "converged": converged,
            "iterations": self.iteration,
            "streams": self.streams,
            "equipment_summaries": self.equipment_summaries,
            "heat_duties": self.heat_duties,
            "KPIs": self.KPIs
        }

    def solve(self):
        return self._solve_reactor_first()

    def _calculate_convergence_progress(self, current_error: float) -> float:
        if self.initial_error is None or self.initial_error == 0:
            return 0.0

        progress = 100.0 * (1.0 - current_error / self.initial_error)
        progress = max(-999.0, min(100.0, progress))
        return progress

    def _print_periodic_summary(self, iteration: int, max_error: float,
                               recycle_flows: dict, product_flow: float):
        """Print periodic progress summary."""
        print("\n" + "-"*70)
        print(f"📊 PROGRESS SUMMARY - Iteration {iteration}")
        print("-"*70)
        print(f"Convergence Error: {max_error:.4f} kmol/h (target: {self.tolerance:.4f})")
        print(f"Progress: {self._calculate_convergence_progress(max_error):.1f}%")
        print(f"\nRecycle Flows:")
        print(f"  H2 recycle:      {recycle_flows['H2']:.2f} kmol/h")
        print(f"  Liquid recycle:  {recycle_flows['LIQ']:.2f} kmol/h")
        print(f"  Benzene recycle: {recycle_flows['DIST']:.2f} kmol/h")
        print(f"\nProduct Flow (S25): {product_flow:.2f} kmol/h")
        if self.acceleration_method == "wegstein" and iteration >= self.accel_start_iter:
            print(f"\nAcceleration: Wegstein ACTIVE")
        else:
            print(f"\nAcceleration: Damping (α={self.damping:.2f})")
        print("-"*70)

    def _solve_reactor_first(self):
        """Solve flowsheet with Wegstein acceleration and periodic reporting."""
        self._initialize_feeds()
        self._prepare_fresh_feed_splits()
        self._initialize_recycles_smart()

        if self.verbose_mode == "normal" or self.verbose_mode == "detailed":
            print("\nIter | Max Error (kmol/h) | Product (kmol/h) | Progress | Status")
            print("-" * 80)
        elif self.verbose_mode == "silent":
            print("\nConverging", end="", flush=True)

        for iteration in range(self.max_iterations):
            self.iteration = iteration + 1

            if self.verbose_mode == "detailed":
                print(f"\n{'='*70}")
                print(f"ITERATION {self.iteration}")
                print(f"{'='*70}")

            recycle_H2_old = self.streams["R_H2"].flowrate_kmol_h
            recycle_liq_old = self.streams["R_LIQ"].flowrate_kmol_h
            recycle_bz_old = self.streams["R_DIST"].flowrate_kmol_h

            H2_comp_old = dict(self.streams["R_H2"].composition)
            liq_comp_old = dict(self.streams["R_LIQ"].composition)
            bz_comp_old = dict(self.streams["R_DIST"].composition)

            try:
                self._execute_from_reactor()
            except Exception as e:
                error_msg = str(e)
                if "temperature" in error_msg.lower() or "exceeds maximum" in error_msg.lower():
                    logger.warning(f"Temperature warning (continuing): {e}")
                    if self.verbose_mode == "detailed":
                        print(f"⚠ WARNING: {e} (continuing...)")
                else:
                    if self.verbose_mode != "silent":
                        print(f"\n❌ ERROR at iteration {self.iteration}: {e}")
                    else:
                        print(f" ERROR!")
                    if self.verbose_mode == "detailed":
                        import traceback
                        traceback.print_exc()
                    break

            recycle_H2_new = self.streams["R_H2"].flowrate_kmol_h
            recycle_liq_new = self.streams["R_LIQ"].flowrate_kmol_h
            recycle_bz_new = self.streams["R_DIST"].flowrate_kmol_h

            error_H2 = abs(recycle_H2_new - recycle_H2_old)
            error_liq = abs(recycle_liq_new - recycle_liq_old)
            error_bz = abs(recycle_bz_new - recycle_bz_old)
            max_error = max(error_H2, error_liq, error_bz)

            if self.initial_error is None:
                self.initial_error = max_error

            progress_pct = self._calculate_convergence_progress(max_error)

            product = self.streams.get("S25")
            product_flow = product.flowrate_kmol_h if product else 0.0

            if self.verbose_mode == "silent":
                if max_error < self.tolerance:
                    print(f"\r✓ Converged in {self.iteration} iterations (100.0% complete)  ", flush=True)
                else:
                    if self.iteration % 5 == 0:
                        progress_str = f"{progress_pct:.1f}%" if progress_pct >= 0 else f"{progress_pct:.1f}% (diverging!)"
                        print(f"\rConverging... Iteration {self.iteration} ({progress_str})", end="", flush=True)
                    else:
                        print(".", end="", flush=True)

            elif self.verbose_mode == "normal":
                status = "✓ CONVERGED" if max_error < self.tolerance else ""
                prog_str = f"{progress_pct:.1f}%" if progress_pct >= 0 else f"{progress_pct:.0f}%"
                print(f"{self.iteration:4d} | {max_error:18.4f} | {product_flow:16.2f} | {prog_str:>8} | {status}")

                # PERIODIC SUMMARY
                if self.iteration % self.report_interval == 0 and max_error >= self.tolerance:
                    self._print_periodic_summary(
                        self.iteration, max_error,
                        {'H2': recycle_H2_new, 'LIQ': recycle_liq_new, 'DIST': recycle_bz_new},
                        product_flow
                    )

            elif self.verbose_mode == "detailed":
                print(f"\nRecycle Convergence Check:")
                print(f"  H2 error:  {error_H2:12.4f} kmol/h")
                print(f"  Liq error: {error_liq:12.4f} kmol/h")
                print(f"  Bz error:  {error_bz:12.4f} kmol/h")
                print(f"  Max error: {max_error:12.4f} kmol/h (tolerance: {self.tolerance:.4f})")
                print(f"  Progress:  {progress_pct:6.1f}%")

            if max_error < self.tolerance:
                self.converged = True
                if self.verbose_mode == "normal":
                    print("\n" + "="*70)
                    print(" CONVERGENCE ACHIEVED!")
                    print("="*70)
                elif self.verbose_mode == "detailed":
                    print("\n" + "="*70)
                    print(" CONVERGENCE ACHIEVED!")
                    print("="*70)
                break

            # Apply Wegstein or damping
            if (self.accelerator is not None and
                self.acceleration_method == "wegstein" and
                self.iteration >= self.accel_start_iter):

                if self.verbose_mode == "detailed":
                    print(f"\nApplying Wegstein acceleration:")

                self.accelerator.record('H2', recycle_H2_old, recycle_H2_new)
                self.accelerator.record('LIQ', recycle_liq_old, recycle_liq_new)
                self.accelerator.record('DIST', recycle_bz_old, recycle_bz_new)

                verbose = (self.verbose_mode == "detailed")
                updated_H2_flow = self.accelerator.accelerate('H2', recycle_H2_old, recycle_H2_new, verbose)
                updated_liq_flow = self.accelerator.accelerate('LIQ', recycle_liq_old, recycle_liq_new, verbose)
                updated_bz_flow = self.accelerator.accelerate('DIST', recycle_bz_old, recycle_bz_new, verbose)

            else:
                alpha = self.damping
                if self.verbose_mode == "detailed":
                    print(f"\nApplying damping (α={alpha}):")

                updated_H2_flow = alpha * recycle_H2_new + (1 - alpha) * recycle_H2_old
                updated_liq_flow = alpha * recycle_liq_new + (1 - alpha) * recycle_liq_old
                updated_bz_flow = alpha * recycle_bz_new + (1 - alpha) * recycle_bz_old

            # Update compositions
            alpha = self.damping
            updated_H2_comp = {}
            updated_liq_comp = {}
            updated_bz_comp = {}

            for comp in self.component_list:
                H2_new = self.streams["R_H2"].composition.get(comp, 0.0)
                H2_old = H2_comp_old.get(comp, 0.0)
                updated_H2_comp[comp] = alpha * H2_new + (1 - alpha) * H2_old

                liq_new = self.streams["R_LIQ"].composition.get(comp, 0.0)
                liq_old = liq_comp_old.get(comp, 0.0)
                updated_liq_comp[comp] = alpha * liq_new + (1 - alpha) * liq_old

                bz_new = self.streams["R_DIST"].composition.get(comp, 0.0)
                bz_old = bz_comp_old.get(comp, 0.0)
                updated_bz_comp[comp] = alpha * bz_new + (1 - alpha) * bz_old

            self.streams["R_H2"].flowrate_kmol_h = updated_H2_flow
            self.streams["R_H2"].composition = updated_H2_comp

            self.streams["R_LIQ"].flowrate_kmol_h = updated_liq_flow
            self.streams["R_LIQ"].composition = updated_liq_comp

            self.streams["R_DIST"].flowrate_kmol_h = updated_bz_flow
            self.streams["R_DIST"].composition = updated_bz_comp

        if not self.converged:
            if self.verbose_mode == "silent":
                print(f"\r⚠️  Not converged after {self.max_iterations} iterations  ")
            else:
                print("\n" + "="*70)
                print(f"⚠️  NOT CONVERGED after {self.max_iterations} iterations")
                print("="*70)

        self.calculate_KPIs()

        if self.converged:
            self._validate_separation_section()

        self._print_final_results()

        return self.converged

    def _validate_separation_section(self):
        """Verify material balance across T-101 + M-101."""
        S23 = self.streams.get("S23")
        S24_dist = self.streams.get("S24_dist")
        S24_bottoms = self.streams.get("S24_bottoms")
        S25 = self.streams.get("S25")
        S24_permeate = self.streams.get("S24_permeate")

        if all([S23, S24_dist, S24_bottoms, S25, S24_permeate]):
            F_in = S23.flowrate_kmol_h
            F_out = S24_dist.flowrate_kmol_h + S25.flowrate_kmol_h + S24_permeate.flowrate_kmol_h
            balance_error = abs(F_in - F_out) / F_in * 100 if F_in > 0 else 0

            if self.verbose_mode != "silent":
                print(f"\n{'='*70}")
                print("SEPARATION SECTION VALIDATION")
                print(f"{'='*70}")
                print(f"  Feed to T-101 (S23):         {F_in:.2f} kmol/h")
                print(f"  T-101 overhead (S24_dist):   {S24_dist.flowrate_kmol_h:.2f} kmol/h")
                print(f"  T-101 bottoms (S24_bottoms): {S24_bottoms.flowrate_kmol_h:.2f} kmol/h")
                print(f"  M-101 product (S25):         {S25.flowrate_kmol_h:.2f} kmol/h")
                print(f"  M-101 permeate (S24_perm):   {S24_permeate.flowrate_kmol_h:.2f} kmol/h")
                print(f"  Balance closure error:       {balance_error:.4f}%")

                chex_purity = S25.composition.get("cyclohexane", 0) * 100
                bz_in_product = S25.composition.get("benzene", 0) * 100
                print(f"\n  Cyclohexane purity (S25): {chex_purity:.2f}%")
                print(f"  Benzene in product (S25):  {bz_in_product:.4f}%")

                if balance_error > 1.0:
                    logger.warning(f"⚠️  Material balance error {balance_error:.2f}% exceeds 1%")
                else:
                    print(f"  ✓ Material balance verified")
                print(f"{'='*70}\n")

    def _print_final_results(self):
        """Print final results."""
        if self.verbose_mode == "silent":
            print("\n" + "="*60)
            print("  RESULTS")
            print("="*60)
            product = self.streams.get("S25")
            if product:
                chex_purity = product.composition.get("cyclohexane", 0) * 100
                print(f"  Product: {product.flowrate_kmol_h:.2f} kmol/h")
                print(f"  Purity:  {chex_purity:.2f}% cyclohexane")
            print(f"  Converged: {' Yes' if self.converged else ' No'}")
            print(f"  Iterations: {self.iteration}")
            print("="*60 + "\n")
        else:
            print("\n" + "="*70)
            print("📊 FINAL RESULTS")
            print("="*70)

            product = self.streams.get("S25")
            if product:
                print(f"\nProduct Stream (S25 - from M-101):")
                print(f"  Flowrate:    {product.flowrate_kmol_h:.2f} kmol/h")
                print(f"  Temperature: {product.temperature_C:.1f} °C")
                print(f"  Pressure:    {product.pressure_bar:.1f} bar")

                if self.verbose_mode == "detailed":
                    print(f"\n  Composition:")
                    for comp, frac in sorted(product.composition.items(),
                                            key=lambda x: x[1], reverse=True):
                        if frac > 0.001:
                            print(f"    {comp:20s}: {frac*100:6.2f}%")
                else:
                    print(f"\n  Main components:")
                    for comp, frac in sorted(product.composition.items(),
                                            key=lambda x: x[1], reverse=True)[:3]:
                        if frac > 0.001:
                            print(f"    {comp:20s}: {frac*100:6.2f}%")

            print(f"\nConvergence:")
            print(f"  Status:     {' YES' if self.converged else ' NO'}")
            print(f"  Iterations: {self.iteration}")
            print("="*70 + "\n")

    def _initialize_feeds(self):
        """Initialize fresh feeds."""
        bz_params = self.feed_specs["benzene_feed"]
        bz_comp = self._get_zero_composition()
        bz_comp["benzene"] = 1.0

        self.streams["S1"] = Stream(
            "Benzene Feed", bz_params["flowrate_kmol_h"],
            bz_params["temperature_C"], bz_params["pressure_bar"],
            bz_comp, self.thermo
        )

        h2_params = self.feed_specs["hydrogen_feed"]
        h2_comp = self._get_zero_composition()
        h2_comp["H2"] = 1.0

        self.streams["S2"] = Stream(
            "Hydrogen Feed", h2_params["flowrate_kmol_h"],
            h2_params["temperature_C"], h2_params["pressure_bar"],
            h2_comp, self.thermo
        )

    def _prepare_fresh_feed_splits(self):
        """Prepare fresh feed splits."""
        reactor_P = self.reactor_config["operating_pressure_bar"]
        pump_eff = self.pump_comp_config["pump_efficiency"]

        S3, _ = pump_liquid(self.streams["S1"], reactor_P,
                           {"efficiency": pump_eff}, self.thermo, "P-101")
        self.streams["S3"] = S3

        S4, _ = mix_streams_adiabatic([self.streams["S3"], self.streams["S2"]],
                                      self.thermo, "MIX-101", "min")
        self.streams["S4"] = S4

        splits_main, _ = split_stream_by_fractions(S4, {"S5": 0.5, "S6": 0.5}, self.thermo)
        self.streams["S5"] = splits_main["S5"]
        self.streams["S6"] = splits_main["S6"]

    def _initialize_recycles_smart(self):
        """Initialize recycles."""
        reactor_P = self.reactor_config["operating_pressure_bar"]
        F_bz_fresh = self.feed_specs["benzene_feed"]["flowrate_kmol_h"]
        F_h2_fresh = self.feed_specs["hydrogen_feed"]["flowrate_kmol_h"]
        F_total_fresh = F_bz_fresh + F_h2_fresh

        F_h2_recycle = F_h2_fresh * self.recycle_factors["h2_recycle_factor"]
        F_liq_recycle = F_total_fresh * self.recycle_factors["liquid_recycle_factor"]
        F_dist_recycle = F_bz_fresh * self.recycle_factors["distillate_recycle_factor"]

        r_h2_comp = self._get_zero_composition()
        r_h2_comp["H2"] = 0.98
        r_h2_comp["methane"] = 0.015
        r_h2_comp["cyclohexane"] = 0.005

        self.streams["R_H2"] = Stream(
            "Hydrogen Recycle", F_h2_recycle, 40.0, reactor_P, r_h2_comp, self.thermo
        )

        r_liq_comp = self._get_zero_composition()
        r_liq_comp["benzene"] = 0.45
        r_liq_comp["H2"] = 0.30
        r_liq_comp["cyclohexane"] = 0.20
        r_liq_comp["methylcyclopentane"] = 0.03
        r_liq_comp["methane"] = 0.02

        self.streams["R_LIQ"] = Stream(
            "Liquid Recycle", F_liq_recycle, 40.0, reactor_P, r_liq_comp, self.thermo
        )

        r_dist_comp = self._get_zero_composition()
        r_dist_comp["benzene"] = 0.98
        r_dist_comp["cyclohexane"] = 0.015
        r_dist_comp["methylcyclopentane"] = 0.005

        self.streams["R_DIST"] = Stream(
            "Distillate Recycle", F_dist_recycle, 80.0, reactor_P, r_dist_comp, self.thermo
        )

    def calculate_KPIs(self):
        """Calculate KPIs including TOTAL ENERGY with thermal corrections."""
        product = self.streams.get("S25")
        chex_product = product.flowrate_kmol_h if product else 0.0
        bz_feed = self.feed_specs["benzene_feed"]["flowrate_kmol_h"]

        conversion = ((bz_feed - chex_product * 0.001) / bz_feed) * 100 if bz_feed > 0 else 0.0

        # ═══════════════════════════════════════════════════════════════
        # ENERGY ACCOUNTING - Read from equipment summaries
        # ═══════════════════════════════════════════════════════════════

        total_heating_kW = 0.0
        total_cooling_kW = 0.0
        total_compressor_kW = 0.0
        total_pump_kW = 0.0
        equipment_heat_to_cooling_kW = 0.0

        # ───────────────────────────────────────────────────────────────
        # 1. Heat exchangers (E-*)
        # ───────────────────────────────────────────────────────────────
        for name, summary in self.equipment_summaries.items():
            if name.startswith("E-"):
                duty_kW = summary.get("duty_kW", 0)
                if duty_kW > 0:
                    total_heating_kW += duty_kW
                else:
                    total_cooling_kW += abs(duty_kW)

        # ───────────────────────────────────────────────────────────────
        # 2. Compressors (C-*) - Just READ heat from equipment
        # ───────────────────────────────────────────────────────────────
        for name, summary in self.equipment_summaries.items():
            if name.startswith("C-"):
                driver_power_kW = summary.get("driver_power_kW", 0)
                total_compressor_kW += driver_power_kW

                # Equipment calculated its own heat to remove
                heat_kW = summary.get("heat_to_remove_kW", 0)
                equipment_heat_to_cooling_kW += heat_kW

        # ───────────────────────────────────────────────────────────────
        # 3. Pumps (P-*) - Just READ heat from equipment
        # ───────────────────────────────────────────────────────────────
        for name, summary in self.equipment_summaries.items():
            if name.startswith("P-"):
                motor_power_kW = summary.get("motor_power_kW", 0)
                total_pump_kW += motor_power_kW

                # Equipment calculated its own heat to remove
                heat_kW = summary.get("heat_to_remove_kW", 0)
                equipment_heat_to_cooling_kW += heat_kW

        # ───────────────────────────────────────────────────────────────
        # 4. Total cooling (includes equipment heat)
        # ───────────────────────────────────────────────────────────────
        total_cooling_kW += equipment_heat_to_cooling_kW

        # ───────────────────────────────────────────────────────────────
        # 5. Total energy
        # ───────────────────────────────────────────────────────────────
        total_energy_kW = (
                total_heating_kW +
                total_cooling_kW +
                total_compressor_kW +
                total_pump_kW
        )

        # ═══════════════════════════════════════════════════════════════
        # BUILD KPI DICTIONARY
        # ═══════════════════════════════════════════════════════════════
        self.KPIs = {
            "fresh_benzene_flow_kmol_h": bz_feed,
            "fresh_hydrogen_flow_kmol_h": self.feed_specs["hydrogen_feed"]["flowrate_kmol_h"],
            "cyclohexane_product_kmol_h": chex_product,
            "conversion_percent": conversion,
            "converged": self.converged,
            "iterations": self.iteration,

            # Energy breakdown
            "total_energy_kW": total_energy_kW,
            "heating_duty_kW": total_heating_kW,
            "cooling_duty_kW": total_cooling_kW,
            "compressor_power_kW": total_compressor_kW,
            "pump_power_kW": total_pump_kW,

            # Thermal correction (diagnostic)
            "equipment_heat_to_cooling_kW": equipment_heat_to_cooling_kW,
        }

    def _execute_from_reactor(self):
        # Mix recycles
        S11, _ = mix_streams_adiabatic(
            [self.streams["R_DIST"], self.streams["R_H2"], self.streams["R_LIQ"]],
            self.thermo, "MIX-102", "min"
        )
        self.streams["S11"] = S11

        # Heat Exchanger E-12 (Recycle Heater)
        reactor_inlet_T = self.reactor_config["inlet_temperature_C"]
        heating_steam = Stream("Steam-E12", 5000, 150, 5, {"H2O": 1.0}, self.thermo, phase="vapor")
        hx_config_E12 = {
            "name": "E-12",
            "service": "process_heating",
            "cold_outlet_temperature_C": reactor_inlet_T,
            "pressure_drop_hot_bar": 0.2,
            "pressure_drop_cold_bar": 0.0,
        }
        _, S12, e12_summary = run_heat_exchanger(
            hot_inlet=heating_steam,
            cold_inlet=S11,
            exchanger_config=hx_config_E12,
            thermo=self.thermo,
            mode="design"
        )
        self.streams["S12"] = S12
        self.equipment_summaries["E-12"] = e12_summary
        self.heat_duties["E-12"] = e12_summary["duty_kW"]

        # Reactor R-101
        S13, r1_summary = run_multibed_reactor(
            recycle_stream=S12, fresh_feed_stream=self.streams["S5"],
            reactor_config=self.reactor_config, kinetics=self.kinetics,
            thermo=self.thermo, catalyst_activity=1.0, reactor_name="R-101"
        )
        self.streams["S13"] = S13
        self.equipment_summaries["R-101"] = r1_summary

        # Heat Exchanger E-02 (Reactor R-101 Cooler)
        target_T_E02 = self.hx_config["cooler_E02_outlet_target_C"]
        cooling_water_E02 = Stream("CW-E02", 5000, 25, 3, {"H2O": 1.0}, self.thermo, phase="liquid")
        hx_config_E02 = {
            "name": "E-02",
            "service": "process_cooling",
            "hot_outlet_temperature_C": target_T_E02,
            "pressure_drop_hot_bar": 0.3,
            "pressure_drop_cold_bar": 0.1,
        }
        S14, _, e02_summary = run_heat_exchanger(
            hot_inlet=S13,
            cold_inlet=cooling_water_E02,
            exchanger_config=hx_config_E02,
            thermo=self.thermo,
            mode="design"
        )
        self.streams["S14"] = S14
        self.equipment_summaries["E-02"] = e02_summary
        self.heat_duties["E-02"] = e02_summary["duty_kW"]

        # Reactor R-102
        S16, r2_summary = run_multibed_reactor(
            recycle_stream=S14, fresh_feed_stream=self.streams["S6"],
            reactor_config=self.reactor_config, kinetics=self.kinetics,
            thermo=self.thermo, catalyst_activity=0.95, reactor_name="R-102"
        )
        self.streams["S16"] = S16
        self.equipment_summaries["R-102"] = r2_summary

        # Heat Exchanger E-04 (Reactor R-102 Cooler)
        target_T_E04 = self.hx_config["cooler_E04_outlet_target_C"]
        cooling_water_E04 = Stream("CW-E04", 8000, 25, 3, {"H2O": 1.0}, self.thermo, phase="liquid")
        hx_config_E04 = {
            "name": "E-04",
            "service": "process_cooling",
            "hot_outlet_temperature_C": target_T_E04,
            "pressure_drop_hot_bar": 0.3,
            "pressure_drop_cold_bar": 0.1,
        }
        S17, _, e04_summary = run_heat_exchanger(
            hot_inlet=S16,
            cold_inlet=cooling_water_E04,
            exchanger_config=hx_config_E04,
            thermo=self.thermo,
            mode="design"
        )
        self.streams["S17"] = S17
        self.equipment_summaries["E-04"] = e04_summary
        self.heat_duties["E-04"] = e04_summary["duty_kW"]

        # Flash separator
        flash_cfg = self.separation_config["flash_separator"]
        S18, S19, flash_summary = run_flash(S17, flash_cfg, self.thermo, None)
        self.streams["S18"] = S18
        self.streams["S19"] = S19
        self.equipment_summaries["V-101"] = flash_summary

        # H2 vapor split FIRST (S18 → S20 to compress + S21 purge)
        h2_recycle_frac_raw = self.design_vector["h2_recycle_fraction"]
        h2_purge_frac_raw = self.design_vector["h2_purge_fraction"]

        # Normalize to ensure they sum to 1.0
        h2_total = h2_recycle_frac_raw + h2_purge_frac_raw
        if abs(h2_total - 1.0) > 1e-6:
            h2_recycle_frac = h2_recycle_frac_raw / h2_total
            h2_purge_frac = h2_purge_frac_raw / h2_total
            if self.verbose_mode == "detailed":
                logger.info(f"Normalizing H2 split fractions: {h2_total:.6f} → 1.0")
        else:
            h2_recycle_frac = h2_recycle_frac_raw
            h2_purge_frac = h2_purge_frac_raw

        # Validate
        if h2_recycle_frac < 0 or h2_recycle_frac > 1:
            raise ValueError(f"h2_recycle_fraction must be in [0,1], got {h2_recycle_frac}")
        if h2_purge_frac < 0 or h2_purge_frac > 1:
            raise ValueError(f"h2_purge_fraction must be in [0,1], got {h2_purge_frac}")

        h2_splits, _ = split_stream_by_fractions(S18, {
            "S20": h2_recycle_frac,
            "S21": h2_purge_frac
        }, self.thermo)

        self.streams["S20"] = h2_splits["S20"]
        self.streams["S21"] = h2_splits["S21"]

        # H2 compressor
        comp_P = self.pump_comp_config["H2_compressor_discharge_pressure_bar"]
        comp_eff = self.pump_comp_config["H2_compressor_efficiency"]

        try:
            R_H2, comp_summary = compress_gas(
                h2_splits["S20"], comp_P, {"efficiency": comp_eff}, self.thermo, "C-101"
            )
        except Exception as e:
            if "temperature" in str(e).lower() or "exceeds" in str(e).lower():
                logger.warning(f"Compressor temperature warning: {e} - using approximate outlet")
                R_H2 = Stream(
                    "H2 Recycle Compressed",
                    h2_splits["S20"].flowrate_kmol_h,
                    150.0,
                    comp_P,
                    h2_splits["S20"].composition,
                    self.thermo
                )
                comp_summary = {"warning": "Temperature approximated"}
            else:
                raise

        self.streams["R_H2"] = R_H2
        self.equipment_summaries["C-101"] = comp_summary

        # Liquid split FIRST (S19 → S22 to pump + S23 to distillation)
        liq_splits, _ = split_stream_by_fractions(S19, {
            "S22": self.design_vector["liquid_recycle_fraction"],
            "S23": 1.0 - self.design_vector["liquid_recycle_fraction"]
        }, self.thermo)

        self.streams["S22"] = liq_splits["S22"]
        self.streams["S23"] = liq_splits["S23"]

        # Liquid recycle pump
        reactor_P = self.reactor_config["operating_pressure_bar"]
        pump_eff = self.pump_comp_config["pump_efficiency"]

        R_LIQ, pump_summary = pump_liquid(
            liq_splits["S22"], reactor_P, {"efficiency": pump_eff}, self.thermo, "P-102"
        )
        self.streams["R_LIQ"] = R_LIQ
        self.equipment_summaries["P-102"] = pump_summary

        # === SEPARATION SECTION: T-101 → M-101 → MIX-103 ===

        # Distillation T-101 (Light Ends Removal)
        dist_cfg = self.separation_config["distillation_column"]
        S24_dist, S24_bottoms, dist_summary = run_distillation_column(
            liq_splits["S23"], dist_cfg, self.thermo, None
        )
        self.streams["S24_dist"] = S24_dist
        self.streams["S24_bottoms"] = S24_bottoms
        self.equipment_summaries["T-101"] = dist_summary

        # Heat Exchanger E-05 (Distillation Bottoms Cooler)
        target_T_E05 = self.hx_config["cooler_E05_outlet_target_C"]
        cooling_water_E05 = Stream("CW-E05", 8000, 25, 3, {"H2O": 1.0},
                                   self.thermo, phase="liquid")

        hx_config_E05 = {
            "name": "E-05",
            "service": "process_cooling",
            "hot_outlet_temperature_C": target_T_E05,
            "pressure_drop_hot_bar": 0.2,
            "pressure_drop_cold_bar": 0.1,
        }

        S24B_cooled, _, e05_summary = run_heat_exchanger(
            hot_inlet=S24_bottoms,
            cold_inlet=cooling_water_E05,
            exchanger_config=hx_config_E05,
            thermo=self.thermo,
            mode="design"
        )

        self.streams["S24B_cooled"] = S24B_cooled
        self.equipment_summaries["E-05"] = e05_summary
        self.heat_duties["E-05"] = e05_summary["duty_kW"]

        # Membrane Separator M-101 (Cyclohexane Polishing)
        mem_cfg = self.separation_config.get("membrane_separator", {
            "cyclohexane_recovery": 0.995,
            "benzene_rejection": 0.95,
            "target_purity": 0.998,
            "membrane_type": "Selective permeation"
        })

        S25, S24_permeate, mem_summary = run_membrane_separator(
            S24B_cooled,
            mem_cfg,
            self.thermo,
            "M-101"
        )

        self.streams["S25"] = S25  # PRODUCT
        self.streams["S24_permeate"] = S24_permeate
        self.equipment_summaries["M-101"] = mem_summary

        # Mixer MIX-103 (Combine Distillate + Permeate)
        S24, _ = mix_streams_adiabatic(
            [self.streams["S24_dist"], self.streams["S24_permeate"]],
            self.thermo, "MIX-103", "min"
        )
        self.streams["S24"] = S24
        # Benzene purge split BEFORE pump
        dist_recycle_frac_raw = self.design_vector["distillate_recycle_fraction"]
        bz_purge_frac_raw = self.design_vector["benzene_purge_fraction"]

        # Normalize to ensure they sum to 1.0
        dist_total = dist_recycle_frac_raw + bz_purge_frac_raw
        if abs(dist_total - 1.0) > 1e-6:
            dist_recycle_frac = dist_recycle_frac_raw / dist_total
            bz_purge_frac = bz_purge_frac_raw / dist_total
            if self.verbose_mode == "detailed":
                logger.info(f"Normalizing distillate split fractions: {dist_total:.6f} → 1.0")
        else:
            dist_recycle_frac = dist_recycle_frac_raw
            bz_purge_frac = bz_purge_frac_raw

        # Validate
        if dist_recycle_frac < 0 or dist_recycle_frac > 1:
            raise ValueError(f"distillate_recycle_fraction must be in [0,1], got {dist_recycle_frac}")
        if bz_purge_frac < 0 or bz_purge_frac > 1:
            raise ValueError(f"benzene_purge_fraction must be in [0,1], got {bz_purge_frac}")

        dist_splits, _ = split_stream_by_fractions(S24, {
            "S26": dist_recycle_frac,
            "S27": bz_purge_frac
        }, self.thermo)

        self.streams["S26"] = dist_splits["S26"]
        self.streams["S27"] = dist_splits["S27"]

        # Distillate recycle pump
        R_DIST, pump_summary2 = pump_liquid(
            dist_splits["S26"], reactor_P, {"efficiency": pump_eff}, self.thermo, "P-103"
        )
        self.streams["R_DIST"] = R_DIST
        self.equipment_summaries["P-103"] = pump_summary2

    def generate_pfd(self, output_file: str = "pfd_diagram", format: str = "png",
                     dpi: int = 300) -> Optional[str]:
        """
        Generate comprehensive Process Flow Diagram with equipment and streams.
        Includes heat exchangers, detailed labels, and KPIs.
        """
        try:
            from graphviz import Digraph
        except ImportError:
            logger.error("graphviz not installed")
            return None

        # Determine status for title
        status = "CONVERGED" if self.converged else "PARTIAL"

        dot = Digraph(comment='Cyclohexane Production Process', format=format)
        dot.attr(rankdir='LR', dpi=str(dpi), splines='ortho')
        dot.attr('graph', fontsize='14', labelloc='t',
                 label=f'CYCLOHEXANE PRODUCTION - {status}\nBenzene Hydrogenation Process')
        dot.attr('node', shape='box', style='rounded,filled', fontsize='10')
        dot.attr('edge', fontsize='8')

        # ═══════════════════════════════════════════════════════════════
        # HELPER FUNCTION: Get stream label
        # ═══════════════════════════════════════════════════════════════
        def get_stream_label(stream_name):
            """Create label with stream conditions."""
            stream = self.streams.get(stream_name)
            if stream:
                T = stream.temperature_C
                P = stream.pressure_bar
                F = stream.flowrate_kmol_h
                return f"{stream_name}\n{F:.0f} kmol/h\n{T:.0f}°C, {P:.1f} bar"
            return stream_name

        # ═══════════════════════════════════════════════════════════════
        # FEED STREAMS (gold)
        # ═══════════════════════════════════════════════════════════════
        dot.node('BZ_FEED', get_stream_label('S1'),
                 shape='ellipse', fillcolor='gold', penwidth='2')
        dot.node('H2_FEED', get_stream_label('S2'),
                 shape='ellipse', fillcolor='gold', penwidth='2')

        # ═══════════════════════════════════════════════════════════════
        # PUMPS (green ellipse)
        # ═══════════════════════════════════════════════════════════════
        p101_summary = self.equipment_summaries.get('P-101', {})
        p101_label = f"P-101\nBenzene Pump\n{p101_summary.get('motor_power_kW', 0):.1f} kW"
        dot.node('P101', p101_label, shape='ellipse', fillcolor='lightgreen')

        p102_summary = self.equipment_summaries.get('P-102', {})
        p102_label = f"P-102\nLiquid Recycle\n{p102_summary.get('motor_power_kW', 0):.1f} kW"
        dot.node('P102', p102_label, shape='ellipse', fillcolor='lightgreen')

        p103_summary = self.equipment_summaries.get('P-103', {})
        p103_label = f"P-103\nOH Recycle\n{p103_summary.get('motor_power_kW', 0):.1f} kW"
        dot.node('P103', p103_label, shape='ellipse', fillcolor='lightgreen')

        # ═══════════════════════════════════════════════════════════════
        # COMPRESSOR (green ellipse)
        # ═══════════════════════════════════════════════════════════════
        c101_summary = self.equipment_summaries.get('C-101', {})
        c101_label = f"C-101\nH2 Compressor\n{c101_summary.get('driver_power_kW', 0):.0f} kW"
        dot.node('C101', c101_label, shape='ellipse', fillcolor='lightgreen')

        # ═══════════════════════════════════════════════════════════════
        # HEAT EXCHANGERS (light coral rectangles)
        # ═══════════════════════════════════════════════════════════════
        # E-02: R-101 outlet cooler
        e02_summary = self.equipment_summaries.get('E-02', {})
        e02_label = f"E-02\nR-101 Cooler\n{e02_summary.get('duty_kW', 0):.0f} kW"
        dot.node('E02', e02_label, fillcolor='lightcoral')

        # E-04: R-102 outlet cooler
        e04_summary = self.equipment_summaries.get('E-04', {})
        e04_label = f"E-04\nR-102 Cooler\n{e04_summary.get('duty_kW', 0):.0f} kW"
        dot.node('E04', e04_label, fillcolor='lightcoral')

        # E-12: Reactor feed heater
        e12_summary = self.equipment_summaries.get('E-12', {})
        e12_label = f"E-12\nFeed Heater\n{e12_summary.get('duty_kW', 0):.0f} kW"
        dot.node('E12', e12_label, fillcolor='orange')

        # E-05: Distillation bottoms cooler
        e05_summary = self.equipment_summaries.get('E-05', {})
        e05_label = f"E-05\nBottoms Cooler\n{e05_summary.get('duty_kW', 0):.0f} kW"
        dot.node('E05', e05_label, fillcolor='lightcoral')

        # ═══════════════════════════════════════════════════════════════
        # MIXERS (yellow diamond)
        # ═══════════════════════════════════════════════════════════════
        dot.node('MIX101', 'MIX-101\nFresh Feed',
                 shape='diamond', fillcolor='lightyellow')
        dot.node('MIX102', 'MIX-102\nRecycle Mix',
                 shape='diamond', fillcolor='lightyellow')
        dot.node('MIX103', 'MIX-103\nOH + Perm',
                 shape='diamond', fillcolor='lightyellow')

        # ═══════════════════════════════════════════════════════════════
        # SPLITTERS (yellow triangle)
        # ═══════════════════════════════════════════════════════════════
        dot.node('SPL101', 'SPL-101\nFeed Split',
                 shape='triangle', fillcolor='lightyellow')
        dot.node('SPL_VAPOR', 'SPL-102\nVapor',
                 shape='triangle', fillcolor='lightyellow', fontsize='8')
        dot.node('SPL_LIQUID', 'SPL-103\nLiquid',
                 shape='triangle', fillcolor='lightyellow', fontsize='8')
        dot.node('SPL_DIST', 'SPL-104\nDist',
                 shape='triangle', fillcolor='lightyellow', fontsize='8')

        # ═══════════════════════════════════════════════════════════════
        # REACTORS (light coral)
        # ═══════════════════════════════════════════════════════════════
        r101_summary = self.equipment_summaries.get('R-101', {})
        r101_temp = r101_summary.get('outlet_temperature_C', 0)
        r101_label = f"R-101\nReactor 1\n{r101_temp:.0f}°C"
        dot.node('R101', r101_label, fillcolor='lightcoral', penwidth='2')

        r102_summary = self.equipment_summaries.get('R-102', {})
        r102_temp = r102_summary.get('outlet_temperature_C', 0)
        r102_label = f"R-102\nReactor 2\n{r102_temp:.0f}°C"
        dot.node('R102', r102_label, fillcolor='lightcoral', penwidth='2')

        # ═══════════════════════════════════════════════════════════════
        # FLASH SEPARATOR (wheat)
        # ═══════════════════════════════════════════════════════════════
        v101_summary = self.equipment_summaries.get('V-101', {})
        v101_vf = v101_summary.get('vapor_fraction', 0)
        v101_label = f"V-101\nFlash\nβ={v101_vf:.2f}"
        dot.node('V101', v101_label, fillcolor='wheat')

        # ═══════════════════════════════════════════════════════════════
        # DISTILLATION COLUMN (wheat, tall)
        # ═══════════════════════════════════════════════════════════════
        t101_summary = self.equipment_summaries.get('T-101', {})
        t101_stages = t101_summary.get('theoretical_stages', 0)
        t101_label = f"T-101\nDistillation\n{t101_stages} stages"
        dot.node('T101', t101_label, fillcolor='wheat', penwidth='2')

        # ═══════════════════════════════════════════════════════════════
        # MEMBRANE SEPARATOR (purple)
        # ═══════════════════════════════════════════════════════════════
        m101_summary = self.equipment_summaries.get('M-101', {})
        m101_recovery = m101_summary.get('cyclohexane_recovery_percent', 0)
        m101_label = f"M-101\nMembrane\nRecovery: {m101_recovery:.1f}%"
        dot.node('M101', m101_label, fillcolor='mediumpurple', penwidth='2')

        # ═══════════════════════════════════════════════════════════════
        # PRODUCT AND PURGES (lime green and orange)
        # ═══════════════════════════════════════════════════════════════
        product_stream = self.streams.get('S25')
        if product_stream:
            product_flow = product_stream.flowrate_kmol_h
            product_purity = product_stream.composition.get('cyclohexane', 0) * 100
            product_label = f"PRODUCT\nCyclohexane\n{product_flow:.1f} kmol/h\n{product_purity:.1f}%"
        else:
            product_label = "PRODUCT\nS25"
        dot.node('PRODUCT', product_label,
                 shape='ellipse', fillcolor='lime', penwidth='3')

        # Purge streams
        dot.node('H2_PURGE', 'H2 Purge\nS21',
                 shape='ellipse', fillcolor='orange')
        dot.node('BZ_PURGE', 'Benzene Purge\nS27',
                 shape='ellipse', fillcolor='orange')

        # ═══════════════════════════════════════════════════════════════
        # CONNECTIONS (with stream labels) - CORRECTED FLOW
        # ═══════════════════════════════════════════════════════════════

        # === FRESH FEED PREPARATION ===
        dot.edge('BZ_FEED', 'P101', label='S1')
        dot.edge('P101', 'MIX101', label='S3')
        dot.edge('H2_FEED', 'MIX101', label='S2')
        dot.edge('MIX101', 'SPL101', label='S4')

        # Fresh feeds go DIRECTLY to reactors
        dot.edge('SPL101', 'R101', label='S5\nfresh', color='brown')
        dot.edge('SPL101', 'R102', label='S6\nfresh', color='brown')

        # === RECYCLE MIXING AND HEATING ===
        # Three recycles ONLY go to MIX102
        dot.edge('C101', 'MIX102', label='RH2',
                 style='dashed', color='blue', penwidth='1.5')
        dot.edge('P102', 'MIX102', label='RLIQ',
                 style='dashed', color='blue', penwidth='1.5')
        dot.edge('P103', 'MIX102', label='RDIST',
                 style='dashed', color='blue', penwidth='1.5')

        dot.edge('MIX102', 'E12', label='S11\nrecycle mix')
        dot.edge('E12', 'R101', label='S12\nheated recycle', penwidth='2')

        # === REACTOR TRAIN WITH COOLERS ===
        dot.edge('R101', 'E02', label='S13', penwidth='2')
        dot.edge('E02', 'R102', label='S14', penwidth='2')
        dot.edge('R102', 'E04', label='S16', penwidth='2')
        dot.edge('E04', 'V101', label='S17', penwidth='2')

        # === FLASH SEPARATOR AND SPLITS ===
        dot.edge('V101', 'SPL_VAPOR', label='S18\nvapor')
        dot.edge('V101', 'SPL_LIQUID', label='S19\nliquid')

        # Vapor split: to compressor and purge
        dot.edge('SPL_VAPOR', 'C101', label='S20')
        dot.edge('SPL_VAPOR', 'H2_PURGE', label='S21', style='dashed')

        # Liquid split: to recycle pump and distillation
        dot.edge('SPL_LIQUID', 'P102', label='S22')
        dot.edge('SPL_LIQUID', 'T101', label='S23', penwidth='2')

        # === DISTILLATION AND MEMBRANE ===
        dot.edge('T101', 'MIX103', label='S24_dist\nOverhead')
        dot.edge('T101', 'E05', label='S24_bottoms\nhot', penwidth='2')
        dot.edge('E05', 'M101', label='S24B_cooled\ncooled', penwidth='2', color='blue')

        # === MEMBRANE SEPARATOR ===
        dot.edge('M101', 'PRODUCT', label='S25\nRetentate',
                 color='green', penwidth='3')
        dot.edge('M101', 'MIX103', label='S24_perm\nPermeate',
                 style='dashed', color='purple')

        # === OVERHEAD RECYCLE ===
        dot.edge('MIX103', 'SPL_DIST', label='S24')
        dot.edge('SPL_DIST', 'P103', label='S26')
        dot.edge('SPL_DIST', 'BZ_PURGE', label='S27', style='dashed')

        # ═══════════════════════════════════════════════════════════════
        # LEGEND (in a subgraph)
        # ═══════════════════════════════════════════════════════════════
        with dot.subgraph(name='cluster_legend') as legend:
            legend.attr(label='Legend', fontsize='10')
            legend.attr('node', shape='plaintext', fillcolor='white')

            legend_text = (
                "Equipment:\n"
                "• Pumps/Compressors (green ellipse)\n"
                "• Heat Exchangers (coral/orange)\n"
                "• Reactors (coral)\n"
                "• Separators (wheat/purple)\n"
                "• Mixers (yellow diamond)\n"
                "• Splitters (yellow triangle)\n\n"
                "Streams:\n"
                "• Feed (gold, bold)\n"
                "• Product (lime, bold)\n"
                "• Recycle (dashed blue)\n"
                "• Purge (dashed, orange node)\n"
            )
            legend.node('legend_text', legend_text, fontsize='8')

        # ═══════════════════════════════════════════════════════════════
        # KPIs BOX (in a subgraph)
        # ═══════════════════════════════════════════════════════════════
        if hasattr(self, 'KPIs') and self.KPIs:
            kpis = self.KPIs
            with dot.subgraph(name='cluster_kpis') as kpi_box:
                kpi_box.attr(label='Key Performance Indicators', fontsize='10')
                kpi_box.attr('node', shape='plaintext', fillcolor='lightyellow')

                kpi_text = (
                    f"Production: {kpis.get('cyclohexane_product_kmol_h', 0):.1f} kmol/h\n"
                    f"Conversion: {kpis.get('conversion_percent', 0):.1f}%\n"
                    f"Total Energy: {kpis.get('total_energy_kW', 0):.0f} kW\n"
                    f"  Heating: {kpis.get('heating_duty_kW', 0):.0f} kW\n"
                    f"  Cooling: {kpis.get('cooling_duty_kW', 0):.0f} kW\n"
                    f"  Power: {kpis.get('compressor_power_kW', 0) + kpis.get('pump_power_kW', 0):.0f} kW\n"
                    f"Status: {'Converged' if kpis.get('converged') else 'Not Converged'}\n"
                    f"Iterations: {kpis.get('iterations', 0)}"
                )
                kpi_box.node('kpi_text', kpi_text, fontsize='8')

        # ═══════════════════════════════════════════════════════════════
        # RENDER
        # ═══════════════════════════════════════════════════════════════
        try:
            output_path = dot.render(output_file, cleanup=True)
            logger.info(f"PFD generated: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to generate PFD: {e}")
            return None


def create_flowsheet(thermo, process_parameters=None, design_vector=None):
    """Factory function."""
    flowsheet = Flowsheet(thermo=thermo, process_parameters=process_parameters)
    if design_vector:
        flowsheet.design_vector.update(design_vector)
    return flowsheet
