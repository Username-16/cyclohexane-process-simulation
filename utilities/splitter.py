"""
utilities/splitter.py
===============================================================================

PURPOSE:
Implement stream splitting for dividing a single inlet stream into multiple outlet streams.
Support  fraction splits (recycle/purge, product/recycle) and flow splits.
All outlets have same temperature, pressure, and composition as inlet (material split only).

Date: 2026-01-01
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

FRACTION_SUM_TOLERANCE = 1e-4  # Tolerance for fraction sum = 1.0
FLOW_BALANCE_TOLERANCE_PERCENT = 0.1  # 0.1% tolerance for flow balance
SMALL_FRACTION_WARNING = 0.01  # Warn if fraction < 1%
LARGE_FRACTION_WARNING = 0.99  # Warn if fraction > 99%


# ============================================================================
# FRACTION-BASED SPLIT
# ============================================================================

def split_stream_by_fractions(
    inlet: Any,  # Stream
    fractions: Dict[str, float],
    thermo: Any,  # ThermodynamicPackage
) -> Tuple[Dict[str, Any], dict]:
    """
    Split inlet stream into multiple outlets based on specified fractions.

    All outlets have same T, P, and composition as inlet. Only flowrate differs.
    Fractions must sum to 1.0 within tolerance.

    Args:
        inlet: Inlet stream to split
        fractions: Dict of {outlet_name: fraction}, must sum to 1.0
        thermo: ThermodynamicPackage for outlet stream creation

    Returns:
        (outlet_streams_dict, summary_dict)

    Example:
        fractions = {"recycle": 0.95, "purge": 0.05}
        outlets, summary = split_stream_by_fractions(flash_vapor, fractions, thermo)

    Raises:
        ValueError: Invalid fractions or inlet
    """
    logger.info(f"Splitting stream by fractions: {inlet.name}")

    # Validate inlet
    _validate_inlet_stream(inlet)

    # Validate fractions
    _validate_fractions(fractions)

    # Check fraction sum
    fraction_sum = sum(fractions.values())
    if abs(fraction_sum - 1.0) > FRACTION_SUM_TOLERANCE:
        raise ValueError(
            f"Fractions must sum to 1.0, got {fraction_sum:.6f} "
            f"(error = {abs(fraction_sum - 1.0):.2e})"
        )

    # Log warnings for extreme fractions
    for name, frac in fractions.items():
        if 0 < frac < SMALL_FRACTION_WARNING:
            logger.warning(
                f"Small fraction for '{name}': {frac:.4f} < {SMALL_FRACTION_WARNING}"
            )
        elif frac > LARGE_FRACTION_WARNING:
            logger.warning(
                f"Large fraction for '{name}': {frac:.4f} > {LARGE_FRACTION_WARNING}. "
                f"Consider if splitter is needed."
            )

    # Calculate outlet flows
    outlet_flows = {}
    for name, frac in fractions.items():
        outlet_flows[name] = frac * inlet.flowrate_kmol_h

    logger.debug(
        f"  Inlet flow: {inlet.flowrate_kmol_h:.2f} kmol/h, "
        f"Outlets: {len(fractions)}"
    )

    # Create outlet streams
    outlet_streams = _create_outlet_streams(
        inlet=inlet,
        outlet_flows=outlet_flows,
        thermo=thermo,
    )

    # Verify mass balance
    total_outlet_flow = sum(outlet_flows.values())
    mass_balance_error = total_outlet_flow - inlet.flowrate_kmol_h
    mass_balance_error_pct = (
        abs(mass_balance_error) / inlet.flowrate_kmol_h * 100
        if inlet.flowrate_kmol_h > 1e-10
        else 0.0
    )

    if abs(mass_balance_error) > 1e-6:
        logger.warning(
            f"Mass balance error: {mass_balance_error:.2e} kmol/h "
            f"({mass_balance_error_pct:.4f}%)"
        )

    # Build summary
    summary = {
        "splitter_type": "fraction",
        "inlet_name": inlet.name,
        "inlet_flowrate_kmol_h": inlet.flowrate_kmol_h,
        "number_of_outlets": len(fractions),
        "outlet_flows_kmol_h": outlet_flows,
        "outlet_fractions": dict(fractions),
        "mass_balance_error_kmol_h": mass_balance_error,
        "mass_balance_error_percent": mass_balance_error_pct,
        "temperature_C": inlet.temperature_C,
        "pressure_bar": inlet.pressure_bar,
        "composition": dict(inlet.composition),
    }

    logger.info(
        f"Split complete: {len(outlet_streams)} outlets, "
        f"balance error = {mass_balance_error_pct:.4f}%"
    )

    return outlet_streams, summary


# ============================================================================
# FLOW-BASED SPLIT
# ============================================================================

def split_stream_by_flows(
    inlet: Any,  # Stream
    outlet_flows_kmol_h: Dict[str, float],
    thermo: Any,  # ThermodynamicPackage
) -> Tuple[Dict[str, Any], dict]:
    """
    Split inlet stream into multiple outlets based on specified absolute flows.

    All outlets have same T, P, and composition as inlet. Only flowrate differs.
    Sum of outlet flows must equal inlet flow within tolerance.

    Args:
        inlet: Inlet stream to split
        outlet_flows_kmol_h: Dict of {outlet_name: flow_kmol_h}
        thermo: ThermodynamicPackage for outlet stream creation

    Returns:
        (outlet_streams_dict, summary_dict)

    Example:
        flows = {"stream_a": 50.0, "stream_b": 30.0, "stream_c": 20.0}
        outlets, summary = split_stream_by_flows(inlet, flows, thermo)

    Raises:
        ValueError: Invalid flows or inlet
    """
    logger.info(f"Splitting stream by flows: {inlet.name}")

    # Validate inlet
    _validate_inlet_stream(inlet)

    # Validate flows
    _validate_flows(outlet_flows_kmol_h)

    # Check flow balance
    total_outlet_flow = sum(outlet_flows_kmol_h.values())
    flow_error = abs(total_outlet_flow - inlet.flowrate_kmol_h)
    flow_error_pct = (
        flow_error / inlet.flowrate_kmol_h * 100
        if inlet.flowrate_kmol_h > 1e-10
        else 0.0
    )

    if flow_error_pct > FLOW_BALANCE_TOLERANCE_PERCENT:
        raise ValueError(
            f"Sum of outlet flows ({total_outlet_flow:.2f} kmol/h) does not match "
            f"inlet flow ({inlet.flowrate_kmol_h:.2f} kmol/h). "
            f"Error = {flow_error_pct:.2f}% > {FLOW_BALANCE_TOLERANCE_PERCENT}%"
        )

    # Calculate implied fractions
    outlet_fractions = {}
    for name, flow in outlet_flows_kmol_h.items():
        if inlet.flowrate_kmol_h > 1e-10:
            outlet_fractions[name] = flow / inlet.flowrate_kmol_h
        else:
            outlet_fractions[name] = 0.0

    logger.debug(
        f"  Inlet flow: {inlet.flowrate_kmol_h:.2f} kmol/h, "
        f"Outlets: {len(outlet_flows_kmol_h)}"
    )

    # Create outlet streams
    outlet_streams = _create_outlet_streams(
        inlet=inlet,
        outlet_flows=outlet_flows_kmol_h,
        thermo=thermo,
    )

    # Mass balance error
    mass_balance_error = total_outlet_flow - inlet.flowrate_kmol_h

    # Build summary
    summary = {
        "splitter_type": "flow",
        "inlet_name": inlet.name,
        "inlet_flowrate_kmol_h": inlet.flowrate_kmol_h,
        "number_of_outlets": len(outlet_flows_kmol_h),
        "outlet_flows_kmol_h": dict(outlet_flows_kmol_h),
        "outlet_fractions": outlet_fractions,
        "mass_balance_error_kmol_h": mass_balance_error,
        "mass_balance_error_percent": flow_error_pct,
        "temperature_C": inlet.temperature_C,
        "pressure_bar": inlet.pressure_bar,
        "composition": dict(inlet.composition),
    }

    logger.info(
        f"Split complete: {len(outlet_streams)} outlets, "
        f"balance error = {flow_error_pct:.4f}%"
    )

    return outlet_streams, summary


# ============================================================================
# BINARY SPLIT (SPECIAL CASE)
# ============================================================================

def split_stream_binary(
    inlet: Any,  # Stream
    fraction_to_outlet1: float,
    outlet1_name: str,
    outlet2_name: str,
    thermo: Any,  # ThermodynamicPackage
) -> Tuple[Any, Any, dict]:
    """
    Split inlet stream into exactly two outlets.

    This is a convenience function for the common case of binary splitting
    (e.g., recycle/purge, product/recycle).

    Args:
        inlet: Inlet stream to split
        fraction_to_outlet1: Fraction of inlet flow to outlet1 (0 to 1)
        outlet1_name: Name for first outlet stream
        outlet2_name: Name for second outlet stream
        thermo: ThermodynamicPackage for outlet stream creation

    Returns:
        (outlet1_stream, outlet2_stream, summary_dict)

    Example:
        recycle, purge, summary = split_stream_binary(
            inlet=flash_vapor,
            fraction_to_outlet1=0.95,
            outlet1_name="h2_recycle",
            outlet2_name="h2_purge",
            thermo=thermo
        )

    Raises:
        ValueError: Invalid fraction or inlet
    """
    logger.info(
        f"Binary split: {inlet.name} → {outlet1_name}({fraction_to_outlet1:.2f}), "
        f"{outlet2_name}({1-fraction_to_outlet1:.2f})"
    )

    # Validate inlet
    _validate_inlet_stream(inlet)

    # Validate fraction
    if not (0.0 <= fraction_to_outlet1 <= 1.0):
        raise ValueError(
            f"Fraction must be in [0, 1], got {fraction_to_outlet1:.4f}"
        )

    # Check for duplicate names
    if outlet1_name == outlet2_name:
        raise ValueError(
            f"Outlet names must be unique, got duplicate: '{outlet1_name}'"
        )

    # Calculate fractions
    fraction_to_outlet2 = 1.0 - fraction_to_outlet1

    # Create fractions dict
    fractions = {
        outlet1_name: fraction_to_outlet1,
        outlet2_name: fraction_to_outlet2,
    }

    # Use the general fraction splitter
    outlet_streams, summary = split_stream_by_fractions(
        inlet=inlet,
        fractions=fractions,
        thermo=thermo,
    )

    # Extract the two outlets
    outlet1 = outlet_streams[outlet1_name]
    outlet2 = outlet_streams[outlet2_name]

    # Update summary type
    summary["splitter_type"] = "binary"
    summary["outlet1_name"] = outlet1_name
    summary["outlet2_name"] = outlet2_name
    summary["fraction_to_outlet1"] = fraction_to_outlet1
    summary["fraction_to_outlet2"] = fraction_to_outlet2

    logger.info(
        f"Binary split complete: "
        f"{outlet1_name}={outlet1.flowrate_kmol_h:.2f} kmol/h, "
        f"{outlet2_name}={outlet2.flowrate_kmol_h:.2f} kmol/h"
    )

    return outlet1, outlet2, summary


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _validate_inlet_stream(inlet: Any) -> None:
    """Validate inlet stream"""

    if inlet is None:
        raise ValueError("Inlet stream cannot be None")

    if not hasattr(inlet, "flowrate_kmol_h"):
        raise ValueError("Inlet must have 'flowrate_kmol_h' attribute")

    if inlet.flowrate_kmol_h < 0:
        raise ValueError(
            f"Inlet flowrate cannot be negative: {inlet.flowrate_kmol_h:.2f} kmol/h"
        )

    if inlet.flowrate_kmol_h < 1e-10:
        logger.warning(
            f"Inlet flowrate is zero or very small: {inlet.flowrate_kmol_h:.2e} kmol/h. "
            f"Splitting nothing."
        )


def _validate_fractions(fractions: Dict[str, float]) -> None:
    """Validate fraction specifications"""

    if not fractions:
        raise ValueError("Fractions dict cannot be empty")

    if len(fractions) < 1:
        raise ValueError("Must have at least one outlet")

    # Check for duplicate names
    if len(fractions) != len(set(fractions.keys())):
        raise ValueError("Outlet names must be unique (duplicates found)")

    # Validate each fraction
    for name, frac in fractions.items():
        if not isinstance(frac, (int, float)):
            raise ValueError(
                f"Fraction for '{name}' must be numeric, got {type(frac)}"
            )

        if frac < 0:
            raise ValueError(
                f"Fraction for '{name}' cannot be negative: {frac:.4f}"
            )

        if frac > 1:
            raise ValueError(
                f"Fraction for '{name}' cannot exceed 1.0: {frac:.4f}"
            )


def _validate_flows(flows: Dict[str, float]) -> None:
    """Validate flow specifications"""

    if not flows:
        raise ValueError("Flows dict cannot be empty")

    if len(flows) < 1:
        raise ValueError("Must have at least one outlet")

    # Check for duplicate names
    if len(flows) != len(set(flows.keys())):
        raise ValueError("Outlet names must be unique (duplicates found)")

    # Validate each flow
    for name, flow in flows.items():
        if not isinstance(flow, (int, float)):
            raise ValueError(
                f"Flow for '{name}' must be numeric, got {type(flow)}"
            )

        if flow < 0:
            raise ValueError(
                f"Flow for '{name}' cannot be negative: {flow:.2f} kmol/h"
            )


def _create_outlet_streams(
    inlet: Any,
    outlet_flows: Dict[str, float],
    thermo: Any,
) -> Dict[str, Any]:
    """
    Create outlet streams from inlet and flow specifications.

    All outlets have same T, P, composition as inlet.
    """
    from simulation.streams import Stream

    outlet_streams = {}

    for name, flow in outlet_flows.items():
        try:
            # Create stream with same properties as inlet
            outlet = Stream(
                name=name,
                temperature_C=inlet.temperature_C,
                pressure_bar=inlet.pressure_bar,
                flowrate_kmol_h=flow,
                composition=dict(inlet.composition),  # Copy composition
                thermo=thermo,
                phase=getattr(inlet, "phase", "unknown"),
            )

            outlet_streams[name] = outlet

            logger.debug(
                f"  Created outlet '{name}': {flow:.2f} kmol/h "
                f"@ {inlet.temperature_C:.1f}°C, {inlet.pressure_bar:.1f} bar"
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to create outlet stream '{name}': {e}"
            )

    return outlet_streams


# ============================================================================
# SMOKE TEST
# ============================================================================

def _smoke_test_splitter():
    """
    Test all splitter functions.
    Must run without external files.
    """
    print("="*70)
    print("STREAM SPLITTER SMOKE TEST")
    print("="*70)

    # Fake thermo
    class FakeThermo:
        def flash_TP(self, T_C, P_bar, z):
            return {"vapor_fraction": 1.0, "phase": "vapor"}
        def enthalpy_TP(self, T_C, P_bar, z, phase):
            return 50.0 * T_C
        def molecular_weight(self, z):
            return sum(z.get(comp, 0) * MW for comp, MW in [
                ("H2", 2.016), ("benzene", 78.11), ("cyclohexane", 84.16)
            ])

    thermo = FakeThermo()

    # Import Stream
    from simulation.streams import Stream

    # ========================================================================
    # Test 1: Binary split (H2 recycle/purge)
    # ========================================================================
    print("\n✓ Test 1: Binary split (H2 recycle/purge)...")

    inlet1 = Stream(
        name="flash_vapor",
        temperature_C=40.0,
        pressure_bar=28.0,
        flowrate_kmol_h=100.0,
        composition={"H2": 0.9, "benzene": 0.08, "cyclohexane": 0.02},
        thermo=thermo,
        phase="vapor"
    )

    try:
        recycle, purge, summary1 = split_stream_binary(
            inlet=inlet1,
            fraction_to_outlet1=0.95,
            outlet1_name="h2_recycle",
            outlet2_name="h2_purge",
            thermo=thermo
        )

        print(f"  Inlet: {inlet1.flowrate_kmol_h:.1f} kmol/h")
        print(f"  Recycle: {recycle.flowrate_kmol_h:.1f} kmol/h (95%)")
        print(f"  Purge: {purge.flowrate_kmol_h:.1f} kmol/h (5%)")
        print(f"  Balance error: {summary1['mass_balance_error_percent']:.6f}%")
        print(f"  Temperature same: {recycle.temperature_C == inlet1.temperature_C}")
        print(f"  Pressure same: {recycle.pressure_bar == inlet1.pressure_bar}")

        # Assertions
        assert abs(recycle.flowrate_kmol_h - 95.0) < 0.01, "Recycle flow incorrect"
        assert abs(purge.flowrate_kmol_h - 5.0) < 0.01, "Purge flow incorrect"
        assert recycle.temperature_C == inlet1.temperature_C, "Temperature not preserved"
        assert recycle.pressure_bar == inlet1.pressure_bar, "Pressure not preserved"
        assert recycle.composition == inlet1.composition, "Composition not preserved"
        assert summary1['mass_balance_error_percent'] < 0.01, "Mass balance error too large"

        print("  ✓ Test 1 passed")

    except Exception as e:
        print(f"  ✗ Test 1 failed: {e}")
        raise

    # ========================================================================
    # Test 2: Multi-outlet fraction split
    # ========================================================================
    print("\n✓ Test 2: Multi-outlet fraction split...")

    inlet2 = Stream(
        name="product_stream",
        temperature_C=80.0,
        pressure_bar=1.5,
        flowrate_kmol_h=200.0,
        composition={"cyclohexane": 0.9995, "benzene": 0.0005},
        thermo=thermo,
        phase="liquid"
    )

    fractions = {
        "to_storage": 0.90,
        "to_recycle": 0.08,
        "to_quality_control": 0.02
    }

    try:
        outlets, summary2 = split_stream_by_fractions(
            inlet=inlet2,
            fractions=fractions,
            thermo=thermo
        )

        print(f"  Inlet: {inlet2.flowrate_kmol_h:.1f} kmol/h")
        for name, stream in outlets.items():
            frac = fractions[name]
            print(f"  {name}: {stream.flowrate_kmol_h:.1f} kmol/h ({frac*100:.0f}%)")
        print(f"  Balance error: {summary2['mass_balance_error_percent']:.6f}%")
        print(f"  Number of outlets: {len(outlets)}")

        # Assertions
        assert abs(outlets["to_storage"].flowrate_kmol_h - 180.0) < 0.01, "Storage flow incorrect"
        assert abs(outlets["to_recycle"].flowrate_kmol_h - 16.0) < 0.01, "Recycle flow incorrect"
        assert abs(outlets["to_quality_control"].flowrate_kmol_h - 4.0) < 0.01, "QC flow incorrect"
        assert len(outlets) == 3, "Wrong number of outlets"
        assert summary2['number_of_outlets'] == 3, "Summary incorrect"

        # Check all outlets have same properties
        for stream in outlets.values():
            assert stream.temperature_C == inlet2.temperature_C
            assert stream.pressure_bar == inlet2.pressure_bar
            assert stream.composition == inlet2.composition

        print("  ✓ Test 2 passed")

    except Exception as e:
        print(f"  ✗ Test 2 failed: {e}")
        raise

    # ========================================================================
    # Test 3: Flow-based split
    # ========================================================================
    print("\n✓ Test 3: Flow-based split...")

    flows = {
        "stream_a": 50.0,
        "stream_b": 30.0,
        "stream_c": 20.0
    }

    try:
        outlets3, summary3 = split_stream_by_flows(
            inlet=inlet1,
            outlet_flows_kmol_h=flows,
            thermo=thermo
        )

        total_out = sum(s.flowrate_kmol_h for s in outlets3.values())
        print(f"  Inlet: {inlet1.flowrate_kmol_h:.1f} kmol/h")
        print(f"  Sum of outlets: {total_out:.1f} kmol/h")
        print(f"  Balance error: {summary3['mass_balance_error_percent']:.6f}%")

        for name, stream in outlets3.items():
            print(f"  {name}: {stream.flowrate_kmol_h:.1f} kmol/h")

        # Assertions
        assert abs(total_out - inlet1.flowrate_kmol_h) < 0.01, "Flow balance incorrect"
        assert abs(outlets3["stream_a"].flowrate_kmol_h - 50.0) < 0.01
        assert abs(outlets3["stream_b"].flowrate_kmol_h - 30.0) < 0.01
        assert abs(outlets3["stream_c"].flowrate_kmol_h - 20.0) < 0.01

        print("  ✓ Test 3 passed")

    except Exception as e:
        print(f"  ✗ Test 3 failed: {e}")
        raise

    # ========================================================================
    # Test 4: Edge case - zero fraction
    # ========================================================================
    print("\n✓ Test 4: Edge case - zero fraction...")

    try:
        out1, out2, summary4 = split_stream_binary(
            inlet=inlet1,
            fraction_to_outlet1=0.0,  # All to outlet2
            outlet1_name="zero_stream",
            outlet2_name="full_stream",
            thermo=thermo
        )

        print(f"  Zero stream: {out1.flowrate_kmol_h:.1f} kmol/h")
        print(f"  Full stream: {out2.flowrate_kmol_h:.1f} kmol/h")

        assert abs(out1.flowrate_kmol_h) < 0.01, "Zero stream should be zero"
        assert abs(out2.flowrate_kmol_h - inlet1.flowrate_kmol_h) < 0.01, "Full stream should equal inlet"

        print("  ✓ Test 4 passed")

    except Exception as e:
        print(f"  ✗ Test 4 failed: {e}")
        raise

    # ========================================================================
    # Test 5: Error handling - invalid fraction sum
    # ========================================================================
    print("\n✓ Test 5: Error handling - invalid fraction sum...")

    try:
        bad_fractions = {"out1": 0.5, "out2": 0.6}  # Sum = 1.1 > 1.0
        outlets, summary = split_stream_by_fractions(
            inlet=inlet1,
            fractions=bad_fractions,
            thermo=thermo
        )
        print("  ✗ Should have raised ValueError for invalid fraction sum")
        assert False, "Should have raised ValueError"

    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")

    except Exception as e:
        print(f"  ✗ Wrong exception type: {e}")
        raise

    print("\n" + "="*70)
    print("✓ ALL STREAM SPLITTER SMOKE TESTS PASSED")
    print("="*70)


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    _smoke_test_splitter()
