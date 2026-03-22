"""
simulation/main.py - CORRECTED VERSION with Optimization Support

Supports both:
- Normal operation: Load from process_parameters.json
- Optimization: Accept custom process_parameters dict

Author: King Saud University - Chemical Engineering Department
Date: 2026-02-02
Version: 6.0.0 - OPTIMIZATION-READY
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import numpy as np
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.flowsheet import Flowsheet
from simulation.thermodynamics import ThermodynamicPackage
from simulation.streams import Stream

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('simulation.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def initialize_thermodynamics() -> ThermodynamicPackage:
    """Initialize thermodynamic package."""
    logger.info("Initializing thermodynamic package...")
    thermo = ThermodynamicPackage()
    component_list = list(thermo.components.keys())
    logger.info(f"✓ Thermodynamics initialized with {len(thermo.components)} components")
    logger.info(f"  Components: {', '.join(component_list)}")
    return thermo


def create_flowsheet(
    thermo: ThermodynamicPackage,
    process_parameters: Dict[str, Any] = None,
    design_vector: Dict[str, float] = None
) -> Flowsheet:
    """
    Create flowsheet with optional custom process parameters.

    Args:
        thermo: Thermodynamic package
        process_parameters: Optional dict of process parameters (overrides file)
        design_vector: Optional design vector for optimization (future use)

    Returns:
        Flowsheet instance
    """
    logger.info("Creating flowsheet...")

    if process_parameters is not None:
        # Use provided parameters (for optimization)
        flowsheet = Flowsheet(thermo=thermo, process_parameters=process_parameters)
        logger.info("✓ Flowsheet created with custom parameters")
    else:
        # Load from file (normal operation)
        flowsheet = Flowsheet(thermo=thermo)
        logger.info("✓ Flowsheet created from config file")

    return flowsheet


def make_json_serializable(obj):
    """Recursively convert objects to JSON-serializable format."""
    if isinstance(obj, Stream):
        return {
            "name": obj.name,
            "flowrate_kmol_h": obj.flowrate_kmol_h,
            "temperature_C": obj.temperature_C,
            "pressure_bar": obj.pressure_bar,
            "composition": obj.composition
        }

    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()
                if not k.startswith('_') and k != 'thermo'}

    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]

    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()

    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)

    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj

    if hasattr(obj, '__dict__'):
        return {k: make_json_serializable(v) for k, v in obj.__dict__.items()
                if not k.startswith('_') and k != 'thermo'}

    return str(obj)


def run_simulation(flowsheet: Flowsheet) -> Dict[str, Any]:
    """
    Run simulation with fail-safe error handling.
    ALWAYS returns results, even on failure!
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("RUNNING SIMULATION")
    logger.info("=" * 80)

    try:
        results = flowsheet.run_simulation()
        converged = results.get("converged", False)

        if converged:
            logger.info("")
            logger.info("=" * 80)
            logger.info("✓ SIMULATION SUCCESSFUL")
            logger.info("=" * 80)
        else:
            logger.warning("")
            logger.warning("=" * 80)
            logger.warning("⚠ SIMULATION DID NOT CONVERGE")
            logger.warning("=" * 80)
            logger.warning("Partial results will be reported for diagnosis")

        return results

    except Exception as e:
        logger.error("")
        logger.error("=" * 80)
        logger.error("✗ SIMULATION ERROR OCCURRED")
        logger.error("=" * 80)
        logger.error(f"Error: {e}", exc_info=True)
        logger.error("")
        logger.error("Extracting partial results for diagnosis...")

        # Extract whatever data exists
        partial_results = {
            "converged": False,
            "iterations": flowsheet.iteration if hasattr(flowsheet, 'iteration') else 0,
            "streams": flowsheet.streams if hasattr(flowsheet, 'streams') else {},
            "equipment_summaries": flowsheet.equipment_summaries if hasattr(flowsheet, 'equipment_summaries') else {},
            "heat_duties": flowsheet.heat_duties if hasattr(flowsheet, 'heat_duties') else {},
            "KPIs": {},
            "max_error": 999.0,
            "error_message": str(e),
            "message": f"Simulation failed: {e}"
        }

        # Try to calculate KPIs from partial data
        try:
            flowsheet.calculate_KPIs()
            partial_results["KPIs"] = flowsheet.KPIs
        except:
            partial_results["KPIs"] = {"converged": False, "error": str(e)}

        logger.warning(f"Extracted {len(partial_results['streams'])} streams")
        logger.warning(f"Extracted {len(partial_results['equipment_summaries'])} equipment summaries")

        return partial_results


def generate_reports(results: Dict[str, Any], flowsheet: Flowsheet) -> None:
    """Generate reports - ALWAYS runs, even with partial data!"""
    logger.info("")
    logger.info("=" * 80)
    logger.info("GENERATING REPORTS (FAIL-SAFE MODE)")
    logger.info("=" * 80)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    converged = results.get("converged", False)
    status_suffix = "CONVERGED" if converged else "PARTIAL"

    # 1. KPI Report
    kpi_report_file = reports_dir / f"kpi_report_{status_suffix}_{timestamp}.csv"
    kpis = results.get("KPIs", {})
    try:
        with open(kpi_report_file, 'w') as f:
            f.write("KPI,Value,Unit,Status\n")
            f.write(f"Converged,{converged},Yes/No,{status_suffix}\n")
            f.write(f"Fresh Benzene Feed,{kpis.get('fresh_benzene_flow_kmol_h', 0):.2f},kmol/h,\n")
            f.write(f"Fresh Hydrogen Feed,{kpis.get('fresh_hydrogen_flow_kmol_h', 0):.2f},kmol/h,\n")
            f.write(f"Cyclohexane Production,{kpis.get('cyclohexane_product_flow_kmol_h', 0):.2f},kmol/h,\n")
            f.write(f"Benzene Conversion,{kpis.get('conversion_percent', 0):.2f},%,\n")
            f.write(f"Yield,{kpis.get('yield_percent', 0):.2f},%,\n")
            f.write(f"Iterations,{kpis.get('iterations', 0)},count,\n")

            if not converged and "error_message" in results:
                f.write(f"Error,{results['error_message']},text,FAILED\n")

        logger.info(f"✓ KPI report: {kpi_report_file}")
    except Exception as e:
        logger.error(f"Failed to generate KPI report: {e}")

    # 2. Stream Table
    stream_table_file = reports_dir / f"stream_table_{status_suffix}_{timestamp}.csv"
    streams = results.get("streams", {})
    try:
        with open(stream_table_file, 'w') as f:
            f.write("Stream ID,Name,Flowrate (kmol/h),Temperature (C),Pressure (bar),Phase,")

            if streams:
                first_stream = next(iter(streams.values()))
                components = sorted(list(first_stream.composition.keys()))
                f.write(",".join(components))
                f.write("\n")

                for stream_id in sorted(streams.keys()):
                    stream = streams[stream_id]
                    f.write(f"{stream_id},{stream.name},{stream.flowrate_kmol_h:.2f},")
                    f.write(f"{stream.temperature_C:.2f},{stream.pressure_bar:.3f},")
                    f.write(f"{getattr(stream, 'phase', 'unknown')},")
                    f.write(",".join(f"{stream.composition.get(c, 0):.6f}" for c in components))
                    f.write("\n")
            else:
                f.write("\nNO STREAMS AVAILABLE\n")

        logger.info(f"✓ Stream table: {stream_table_file} ({len(streams)} streams)")
    except Exception as e:
        logger.error(f"Failed to generate stream table: {e}")

    # 3. Equipment Summary
    equipment_file = reports_dir / f"equipment_summary_{status_suffix}_{timestamp}.csv"
    equipment = results.get("equipment_summaries", {})
    try:
        with open(equipment_file, 'w') as f:
            f.write("Equipment ID,Type,Description,Status,Inlet Flow (kmol/h),Outlet Flow (kmol/h),Duty (kW)\n")

            if equipment:
                for eq_id in sorted(equipment.keys()):
                    eq_data = equipment[eq_id]
                    f.write(f"{eq_id},{eq_data.get('type', '')},{eq_data.get('description', '')},")
                    f.write(f"{eq_data.get('status', '')},{eq_data.get('inlet_flowrate_kmol_h', 0):.2f},")
                    f.write(f"{eq_data.get('outlet_flowrate_kmol_h', 0):.2f},{eq_data.get('duty_kW', 0):.2f}\n")
            else:
                f.write("NO EQUIPMENT DATA AVAILABLE\n")

        logger.info(f"✓ Equipment summary: {equipment_file} ({len(equipment)} items)")
    except Exception as e:
        logger.error(f"Failed to generate equipment summary: {e}")

    # 4. Heat Duties
    heat_duties_file = reports_dir / f"heat_duties_{status_suffix}_{timestamp}.csv"
    heat_duties = results.get("heat_duties", {})
    try:
        with open(heat_duties_file, 'w') as f:
            f.write("Equipment ID,Heat Duty (kW),Type\n")

            if heat_duties:
                for eq_id in sorted(heat_duties.keys()):
                    duty = heat_duties[eq_id]
                    duty_type = "Heating" if duty > 0 else "Cooling"
                    f.write(f"{eq_id},{duty:.2f},{duty_type}\n")

                total_heating = sum(d for d in heat_duties.values() if d > 0)
                total_cooling = sum(abs(d) for d in heat_duties.values() if d < 0)
                f.write(f"TOTAL HEATING,{total_heating:.2f},kW\n")
                f.write(f"TOTAL COOLING,{total_cooling:.2f},kW\n")
            else:
                f.write("NO HEAT DUTY DATA AVAILABLE\n")

        logger.info(f"✓ Heat duties: {heat_duties_file}")
    except Exception as e:
        logger.error(f"Failed to generate heat duties report: {e}")

    # 5. Excel Report (if pandas available)
    try:
        import pandas as pd
        excel_file = reports_dir / f"simulation_report_{status_suffix}_{timestamp}.xlsx"

        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Status sheet
            status_df = pd.DataFrame([
                {"Parameter": "Converged", "Value": converged},
                {"Parameter": "Iterations", "Value": results.get("iterations", 0)},
                {"Parameter": "Streams Calculated", "Value": len(streams)},
                {"Parameter": "Equipment Items", "Value": len(equipment)},
                {"Parameter": "Status", "Value": status_suffix},
            ])

            if not converged and "error_message" in results:
                status_df = pd.concat([status_df, pd.DataFrame([
                    {"Parameter": "Error Message", "Value": results["error_message"]}
                ])], ignore_index=True)

            status_df.to_excel(writer, sheet_name='Status', index=False)

            # KPIs sheet
            if kpis:
                kpi_df = pd.DataFrame([{"KPI": k, "Value": v} for k, v in kpis.items()])
                kpi_df.to_excel(writer, sheet_name='KPIs', index=False)

            # Streams sheet
            if streams:
                stream_data = []
                for sid in sorted(streams.keys()):
                    s = streams[sid]
                    row = {
                        "Stream ID": sid,
                        "Name": s.name,
                        "Flowrate (kmol/h)": s.flowrate_kmol_h,
                        "Temperature (C)": s.temperature_C,
                        "Pressure (bar)": s.pressure_bar,
                        "Phase": getattr(s, 'phase', 'unknown')
                    }
                    row.update(s.composition)
                    stream_data.append(row)

                stream_df = pd.DataFrame(stream_data)
                stream_df.to_excel(writer, sheet_name='Streams', index=False)

            # Equipment sheet
            if equipment:
                eq_df = pd.DataFrame([{
                    "Equipment ID": eq_id,
                    "Type": eq_data.get('type', ''),
                    "Description": eq_data.get('description', ''),
                    "Status": eq_data.get('status', ''),
                    "Inlet Flow (kmol/h)": eq_data.get('inlet_flowrate_kmol_h', 0),
                    "Outlet Flow (kmol/h)": eq_data.get('outlet_flowrate_kmol_h', 0),
                    "Duty (kW)": eq_data.get('duty_kW', 0)
                } for eq_id in sorted(equipment.keys()) for eq_data in [equipment[eq_id]]])
                eq_df.to_excel(writer, sheet_name='Equipment', index=False)

        logger.info(f"✓ Excel report: {excel_file}")
    except ImportError:
        logger.warning("pandas not available - Excel report skipped")
    except Exception as e:
        logger.error(f"Failed to generate Excel report: {e}")

    # 6. JSON Results (ALWAYS save, even partial)
    json_file = reports_dir / f"simulation_results_{status_suffix}_{timestamp}.json"
    try:
        results_json = make_json_serializable(results)
        with open(json_file, 'w') as f:
            json.dump(results_json, f, indent=2)
        logger.info(f"✓ JSON results: {json_file}")
    except Exception as e:
        logger.error(f"Failed to save JSON results: {e}")

    logger.info("=" * 80)


def generate_pfd(flowsheet: Flowsheet, converged: bool, output_dir: str = "reports") -> None:
    """Generate Process Flow Diagram after flowsheet activation."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("GENERATING PROCESS FLOW DIAGRAM")
    logger.info("=" * 80)

    try:
        # Check if graphviz is available
        try:
            from graphviz import Digraph
            logger.info("✓ Graphviz Python package found")
        except ImportError:
            logger.error("✗ Graphviz Python package NOT installed")
            logger.error("  Install with: pip install graphviz")
            logger.error("")
            logger.error("  Also install system graphviz binary:")
            logger.error("  - Windows: https://graphviz.org/download/ (add to PATH)")
            logger.error("  - Linux: sudo apt-get install graphviz")
            logger.error("  - Mac: brew install graphviz")
            return

        # Check if flowsheet has generatepfd method
        if not hasattr(flowsheet, 'generate_pfd'):
            logger.error("✗ generate_pfd() method NOT found in flowsheet")
            logger.error("  Your flowsheet.py needs to have a generate_pfd() method")
            return

        logger.info("✓ generate_pfd() method found in flowsheet")

        status_suffix = "CONVERGED" if converged else "PARTIAL"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        logger.info(f"Status: {status_suffix}")
        logger.info(f"Generating high-resolution PFD...")

        # Generate PNG
        pfd_path = flowsheet.generate_pfd(
            output_file=f"{output_dir}/pfd_diagram_{status_suffix}_{timestamp}",
            format='png',
            dpi=300
        )

        if pfd_path:
            logger.info(f"✓ PFD saved successfully: {pfd_path}")

            # Try to generate PDF version
            try:
                pdf_path = flowsheet.generate_pfd(
                    output_file=f"{output_dir}/pfd_diagram_{status_suffix}_{timestamp}_print",
                    format='pdf',
                    dpi=300
                )
                if pdf_path:
                    logger.info(f"✓ PDF version saved: {pdf_path}")
            except Exception as e:
                logger.warning(f"PDF generation failed: {e}")
        else:
            logger.warning("PFD generation returned None")
            logger.warning("This usually means graphviz system binary is not installed")
            logger.warning("or not in PATH. Install it:")
            logger.warning("  - Windows: https://graphviz.org/download/ (add to PATH)")
            logger.warning("  - Linux: sudo apt-get install graphviz")
            logger.warning("  - Mac: brew install graphviz")
            logger.warning("Then restart your terminal/IDE")

    except Exception as e:
        logger.error(f"PFD generation error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("=" * 80)


def print_kpi_summary(kpis: Dict[str, Any], converged: bool) -> None:
    """Print KPI summary to console."""
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"KEY PERFORMANCE INDICATORS {'(PARTIAL RESULTS)' if not converged else ''}")
    logger.info("=" * 80)
    logger.info("")
    logger.info("CONVERGENCE STATUS")
    logger.info(f"  {'✓ CONVERGED' if converged else '✗ NOT CONVERGED / FAILED'}")
    logger.info(f"  Iterations: {kpis.get('iterations', 0)}")
    logger.info("")
    logger.info("FEED STREAMS")
    logger.info(f"  Fresh Benzene Feed:  {kpis.get('fresh_benzene_flow_kmol_h', 0):10.2f} kmol/h")
    logger.info(f"  Fresh Hydrogen Feed: {kpis.get('fresh_hydrogen_flow_kmol_h', 0):10.2f} kmol/h")
    logger.info("")
    logger.info("PRODUCTION")
    logger.info(f"  Cyclohexane Production: {kpis.get('cyclohexane_product_flow_kmol_h', 0):10.2f} kmol/h")
    logger.info("")
    logger.info("PROCESS PERFORMANCE")
    logger.info(f"  Benzene Conversion: {kpis.get('conversion_percent', 0):10.2f} %")
    logger.info(f"  Overall Yield:      {kpis.get('yield_percent', 0):10.2f} %")
    logger.info("")
    logger.info("=" * 80)


def main() -> None:
    """Main entry point - FAIL-SAFE version with PFD generation."""
    results = None
    flowsheet = None
    converged = False

    try:
        logger.info("")
        logger.info("=" * 78)
        logger.info(" " * 15 + "CYCLOHEXANE PRODUCTION SIMULATION" + " " * 30)
        logger.info(" " * 20 + "WITH FAIL-SAFE REPORTING & PFD" + " " * 28)
        logger.info("=" * 78)
        logger.info("")

        # Initialize thermodynamics
        thermo = initialize_thermodynamics()

        # Create flowsheet
        flowsheet = create_flowsheet(thermo)

        # Run simulation (ALWAYS returns results)
        results = run_simulation(flowsheet)
        converged = results.get("converged", False)

    except Exception as e:
        logger.error(f"Fatal error during initialization: {e}", exc_info=True)

        # Create minimal results
        results = {
            "converged": False,
            "iterations": 0,
            "streams": {},
            "equipment_summaries": {},
            "heat_duties": {},
            "KPIs": {"error": str(e)},
            "error_message": str(e)
        }

    # ALWAYS try to print KPIs
    try:
        print_kpi_summary(results.get("KPIs", {}), converged)
    except Exception as e:
        logger.error(f"Could not print KPI summary: {e}")

    # ALWAYS try to generate reports
    try:
        if flowsheet:
            generate_reports(results, flowsheet)
        else:
            logger.error("No flowsheet object - skipping reports")
    except Exception as e:
        logger.error(f"Report generation failed: {e}", exc_info=True)

    # GENERATE PFD AFTER FLOWSHEET ACTIVATION
    try:
        if flowsheet:
            generate_pfd(flowsheet, converged, "reports")
        else:
            logger.warning("No flowsheet object - skipping PFD generation")
    except Exception as e:
        logger.error(f"PFD generation failed: {e}", exc_info=True)

    # Final summary
    logger.info("")
    logger.info("=" * 80)
    if converged:
        logger.info("✓ SIMULATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info("Reports saved in reports/")
        logger.info("  - Stream tables (CSV)")
        logger.info("  - Equipment summaries (CSV)")
        logger.info("  - Heat duties (CSV)")
        logger.info("  - KPI report (CSV)")
        logger.info("  - Excel workbook")
        logger.info("  - JSON data dump")
        logger.info("  - Process Flow Diagram (PNG/PDF)")
        sys.exit(0)
    else:
        logger.warning("⚠ SIMULATION DID NOT CONVERGE OR FAILED")
        logger.warning("=" * 80)
        logger.warning("PARTIAL RESULTS EXPORTED FOR DIAGNOSIS")
        logger.warning("")
        logger.warning("Check these files in reports/ directory:")
        logger.warning("  - stream_table_PARTIAL*.csv (shows all calculated streams)")
        logger.warning("  - equipment_summary_PARTIAL*.csv (shows equipment status)")
        logger.warning("  - simulation_results_PARTIAL*.json (full data dump)")
        logger.warning("  - pfd_diagram_PARTIAL*.png (process flow diagram)")
        logger.warning("")
        if results and "error_message" in results:
            logger.warning(f"Error: {results['error_message']}")
        logger.warning("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
