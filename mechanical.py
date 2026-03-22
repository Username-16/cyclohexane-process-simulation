"""
utilities/mechanical.py
===============================================================================

PURPOSE: Implement mechanical design utilities for process equipment sizing and costing.
Calculate vessel dimensions, wall thickness, weight, and material requirements.
Support ASME pressure vessel code calculations for reactors, separators, columns.

Date: 2026-01-01
Version: 1.0.0
"""

# =============================================================================
# PARAMETERS (Global constants for mechanical design)
# =============================================================================

# Material Properties Database
MATERIAL_PROPERTIES = {
    "carbon_steel": {
        "allowable_stress_MPa": {  # ASME SA-516 Grade 70
            20: 138,
            100: 138,
            200: 126,
            300: 111,
            400: 85,
            500: 36
        },
        "density_kg_m3": 7850,
        "modulus_GPa": 200,
        "thermal_expansion_per_C": 1.2e-5,
        "cost_factor": 1.0
    },
    "stainless_steel_304": {
        "allowable_stress_MPa": {  # ASME SA-240 Type 304
            20: 138,
            100: 129,
            200: 108,
            300: 95,
            400: 85,
            500: 74
        },
        "density_kg_m3": 8000,
        "modulus_GPa": 193,
        "thermal_expansion_per_C": 1.7e-5,
        "cost_factor": 2.5
    },
    "stainless_steel_316": {
        "allowable_stress_MPa": {  # ASME SA-240 Type 316
            20: 138,
            100: 129,
            200: 108,
            300: 95,
            400: 85,
            500: 74
        },
        "density_kg_m3": 8000,
        "modulus_GPa": 193,
        "thermal_expansion_per_C": 1.6e-5,
        "cost_factor": 3.0
    }
}

# Standard plate thicknesses [mm] per ASTM/ASME standards
STANDARD_THICKNESS_MM = [
    6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 28, 32, 36, 40,
    45, 50, 56, 63, 71, 80, 90, 100
]

# Design Limits
MIN_DIAMETER_M = 0.3
MAX_DIAMETER_M = 10.0
MIN_HEIGHT_M = 0.5
MAX_HEIGHT_M = 100.0
MIN_DESIGN_PRESSURE_BAR = 0.1
MAX_DESIGN_PRESSURE_BAR = 200.0
MIN_DESIGN_TEMP_C = -50
MAX_DESIGN_TEMP_C = 600
DEFAULT_CORROSION_ALLOWANCE_MM = 3.0
DEFAULT_JOINT_EFFICIENCY = 1.0
MIN_JOINT_EFFICIENCY = 0.7
MAX_JOINT_EFFICIENCY = 1.0
MAX_WALL_THICKNESS_MM = 100.0
SLENDERNESS_WARNING_THRESHOLD = 10.0

# Tray Design Parameters
DEFAULT_TRAY_THICKNESS_MM = 5.0
DEFAULT_TRAY_SPACING_M = 0.5
DEFAULT_DOWNCOMER_FRACTION = 0.12
DEFAULT_ACTIVE_AREA_FRACTION = 0.88
PLATFORM_SPACING_M = 7.0
MIN_HEIGHT_FOR_PLATFORMS_M = 5.0

# Wind Load Parameters
DESIGN_WIND_SPEED_MS = 45.0  # For Saudi Arabia
AIR_DENSITY_KG_M3 = 1.225
WIND_DRAG_COEFFICIENT = 0.7

# =============================================================================
# IMPORTS
# =============================================================================

from __future__ import annotations

import math
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def size_vertical_vessel(
    diameter_m: float,
    height_m: float,
    design_pressure_bar: float,
    design_temperature_C: float,
    material: str,
    vessel_config: dict,
) -> dict:
    """
    Size a vertical pressure vessel per ASME VIII Division 1.

    Calculates shell and head thickness, weights, dimensions, and foundation loads
    for vertical vessels (reactors, flash drums, absorbers).

    Process:
    1. Validate inputs against ASME code requirements
    2. Calculate shell thickness using cylindrical formula
    3. Calculate head thickness based on head type
    4. Estimate weights (shell, heads, internals)
    5. Calculate operating and test loads
    6. Check slenderness ratio and wind loads

    Args:
        diameter_m: Internal diameter [m]
        height_m: Tangent-to-tangent height [m]
        design_pressure_bar: Maximum Allowable Working Pressure
        design_temperature_C: Design temperature
        material: Material name from database
        vessel_config: Configuration dict with design parameters

    Returns:
        Comprehensive sizing summary dict

    Example:
        sizing = size_vertical_vessel(
            diameter_m=3.0,
            height_m=20.0,
            design_pressure_bar=35.0,
            design_temperature_C=450.0,
            material="carbon_steel",
            vessel_config={
                "joint_efficiency": 1.0,
                "corrosion_allowance_mm": 3.0,
                "head_type": "ellipsoidal"
            }
        )
        weight = sizing["empty_weight_kg"]

    Raises:
        ValueError: Invalid inputs
        RuntimeError: Calculation failures
    """
    logger.info(f"Sizing vertical vessel D={diameter_m:.2f}m, H={height_m:.1f}m")

    # Validate inputs
    validate_vessel_inputs(diameter_m, height_m, design_pressure_bar, design_temperature_C, material)

    # Extract configuration
    joint_efficiency = vessel_config.get("joint_efficiency", DEFAULT_JOINT_EFFICIENCY)
    corrosion_allowance_mm = vessel_config.get("corrosion_allowance_mm", DEFAULT_CORROSION_ALLOWANCE_MM)
    head_type = vessel_config.get("head_type", "ellipsoidal")
    skirt_height_m = vessel_config.get("skirt_height_m", 2.0)
    include_wind = vessel_config.get("include_wind_load", True)

    # Get material properties
    mat_props = get_material_properties(material, design_temperature_C)
    allowable_stress_MPa = mat_props["allowable_stress_MPa"]
    density_kg_m3 = mat_props["density_kg_m3"]

    # Calculate shell thickness
    shell_thickness = calculate_wall_thickness(
        diameter_m=diameter_m,
        design_pressure_bar=design_pressure_bar,
        material=material,
        joint_efficiency=joint_efficiency,
        corrosion_allowance_mm=corrosion_allowance_mm
    )

    # Calculate head thickness
    head_thickness = calculate_head_thickness(
        diameter_m=diameter_m,
        design_pressure_bar=design_pressure_bar,
        head_type=head_type,
        material=material,
        joint_efficiency=joint_efficiency,
        corrosion_allowance_mm=corrosion_allowance_mm
    )

    # Calculate head depth
    head_depth_m = calculate_head_depth(diameter_m, head_type)

    # Total height with heads
    total_height_m = height_m + 2 * head_depth_m

    # Outer diameter
    shell_thickness_m = shell_thickness["actual_thickness_mm"] / 1000.0
    outer_diameter_m = diameter_m + 2 * shell_thickness_m
    outer_diameter_mm = outer_diameter_m * 1000.0

    # Check slenderness ratio
    slenderness = height_m / diameter_m
    if slenderness > SLENDERNESS_WARNING_THRESHOLD:
        logger.warning(f"High slenderness ratio H/D={slenderness:.1f} > {SLENDERNESS_WARNING_THRESHOLD}. "
                      f"Consider stability analysis.")

    # Calculate weights
    weight_data = estimate_vessel_weight(
        diameter_m=diameter_m,
        height_or_length_m=height_m,
        wall_thickness_mm=shell_thickness["actual_thickness_mm"],
        material_density_kg_m3=density_kg_m3,
        vessel_type="vertical",
        include_internals=True
    )

    # Update with head thickness
    weight_data = update_weight_with_head_thickness(
        weight_data, diameter_m, head_thickness["actual_thickness_mm"], head_type, density_kg_m3
    )

    # Calculate operating weight (assume 60% liquid holdup)
    liquid_volume_m3 = math.pi * (diameter_m**2) / 4 * height_m * 0.6
    liquid_density_kg_m3 = 800  # Typical hydrocarbon
    operating_liquid_weight_kg = liquid_volume_m3 * liquid_density_kg_m3
    operating_weight_kg = weight_data["empty_weight_kg"] + operating_liquid_weight_kg

    # Test weight (full of water)
    test_volume_m3 = math.pi * (diameter_m**2) / 4 * height_m
    test_water_weight_kg = test_volume_m3 * 1000  # Water density
    test_weight_kg = weight_data["empty_weight_kg"] + test_water_weight_kg

    # Surface areas
    shell_area_m2 = math.pi * diameter_m * height_m
    head_area_m2 = calculate_head_surface_area(diameter_m, head_type)
    internal_surface_m2 = shell_area_m2 + 2 * head_area_m2
    external_surface_m2 = math.pi * outer_diameter_m * height_m + 2 * head_area_m2

    # Wind load for tall vessels
    wind_moment_kNm = None
    if include_wind and height_m > 10:
        wind_moment_kNm = calculate_wind_moment(
            diameter_m=outer_diameter_m,
            height_m=total_height_m,
            wind_speed_ms=DESIGN_WIND_SPEED_MS
        )

    # Material volume
    shell_material_volume_m3 = math.pi * ((outer_diameter_m/2)**2 - (diameter_m/2)**2) * height_m
    head_material_volume_m3 = 2 * head_area_m2 * head_thickness["actual_thickness_mm"] / 1000.0
    total_material_volume_m3 = shell_material_volume_m3 + head_material_volume_m3

    # Design margin
    design_margin_percent = ((shell_thickness["actual_thickness_mm"] - shell_thickness["required_thickness_mm"]) /
                            shell_thickness["required_thickness_mm"]) * 100.0
    if design_margin_percent < 5:
        logger.warning(f"Low design margin: {design_margin_percent:.1f}%")
    elif design_margin_percent > 50:
        logger.warning(f"High design margin: {design_margin_percent:.1f}% - overdesigned")

    # Build comprehensive summary
    sizing_summary = {
        # Basic info
        "vessel_type": "pressure_vessel",
        "orientation": "vertical",
        "internal_diameter_m": diameter_m,
        "internal_diameter_mm": diameter_m * 1000,
        "height_or_length_m": height_m,
        "design_pressure_bar": design_pressure_bar,
        "design_pressure_MPa": design_pressure_bar / 10.0,
        "design_temperature_C": design_temperature_C,
        "material": material,
        "material_density_kg_m3": density_kg_m3,
        "allowable_stress_MPa": allowable_stress_MPa,

        # Shell thickness
        "shell_thickness_required_mm": shell_thickness["required_thickness_mm"],
        "shell_thickness_nominal_mm": shell_thickness["nominal_thickness_mm"],
        "shell_thickness_actual_mm": shell_thickness["actual_thickness_mm"],
        "corrosion_allowance_mm": corrosion_allowance_mm,

        # Head thickness
        "head_type": head_type,
        "head_thickness_required_mm": head_thickness["required_thickness_mm"],
        "head_thickness_nominal_mm": head_thickness["nominal_thickness_mm"],
        "head_thickness_actual_mm": head_thickness["actual_thickness_mm"],

        # Dimensions
        "outer_diameter_mm": outer_diameter_mm,
        "tangent_to_tangent_length_m": height_m,
        "total_height_with_heads_m": total_height_m,
        "head_depth_m": head_depth_m,
        "slenderness_ratio": slenderness,

        # Weight
        "shell_weight_kg": weight_data["shell_weight_kg"],
        "head_weight_kg": weight_data["heads_weight_kg"],
        "internals_weight_kg": weight_data["internals_weight_kg"],
        "empty_weight_kg": weight_data["empty_weight_kg"],
        "operating_weight_kg": operating_weight_kg,
        "test_weight_kg": test_weight_kg,

        # Material volume
        "shell_material_volume_m3": shell_material_volume_m3,
        "total_material_volume_m3": total_material_volume_m3,

        # Surface area
        "internal_surface_area_m2": internal_surface_m2,
        "external_surface_area_m2": external_surface_m2,

        # Foundation loads
        "empty_load_kg": weight_data["empty_weight_kg"],
        "operating_load_kg": operating_weight_kg,
        "test_load_kg": test_weight_kg,
        "wind_moment_kNm": wind_moment_kNm,

        # Code compliance
        "code": vessel_config.get("code", "ASME_VIII_Div1"),
        "joint_efficiency": joint_efficiency,
        "design_margin_percent": design_margin_percent,
    }

    logger.info(f"Vessel sized: t_shell={shell_thickness['actual_thickness_mm']:.0f}mm, "
               f"W_empty={weight_data['empty_weight_kg']:.0f}kg")

    return sizing_summary


def size_horizontal_vessel(
    diameter_m: float,
    length_m: float,
    design_pressure_bar: float,
    design_temperature_C: float,
    material: str,
    vessel_config: dict,
) -> dict:
    """
    Size a horizontal pressure vessel per ASME VIII Division 1.

    Similar to vertical vessel but horizontal orientation (typical for heat exchangers,
    knockout drums, horizontal separators).

    Args:
        diameter_m: Internal diameter [m]
        length_m: Tangent-to-tangent length [m]
        design_pressure_bar: MAWP
        design_temperature_C: Design temperature
        material: Material from database
        vessel_config: Configuration dict

    Returns:
        Sizing summary dict
    """
    logger.info(f"Sizing horizontal vessel D={diameter_m:.2f}m, L={length_m:.1f}m")

    # Use size_vertical_vessel logic but mark as horizontal
    sizing = size_vertical_vessel(
        diameter_m=diameter_m,
        height_m=length_m,  # Length is analogous to height
        design_pressure_bar=design_pressure_bar,
        design_temperature_C=design_temperature_C,
        material=material,
        vessel_config=vessel_config
    )

    # Update orientation-specific fields
    sizing["orientation"] = "horizontal"
    sizing["length_m"] = length_m
    sizing["tangent_to_tangent_length_m"] = length_m
    sizing["support_type"] = "saddle"  # Horizontal vessels typically have saddle supports

    return sizing


def calculate_wall_thickness(
    diameter_m: float,
    design_pressure_bar: float,
    material: str,
    joint_efficiency: float = DEFAULT_JOINT_EFFICIENCY,
    corrosion_allowance_mm: float = DEFAULT_CORROSION_ALLOWANCE_MM,
) -> dict:
    """
    Calculate required wall thickness per ASME VIII Division 1.

    Uses cylindrical shell formula: t = (P * R) / (S * E - 0.6 * P) + CA

    Args:
        diameter_m: Internal diameter [m]
        design_pressure_bar: Design pressure
        material: Material name
        joint_efficiency: Weld joint efficiency [0.7-1.0]
        corrosion_allowance_mm: Corrosion allowance

    Returns:
        Thickness data dict with required, nominal, and actual thickness

    Example:
        thickness = calculate_wall_thickness(
            diameter_m=3.0,
            design_pressure_bar=35.0,
            material="carbon_steel",
            joint_efficiency=1.0,
            corrosion_allowance_mm=3.0
        )
        t_actual = thickness["actual_thickness_mm"]
    """
    # Validate
    if not (MIN_JOINT_EFFICIENCY <= joint_efficiency <= MAX_JOINT_EFFICIENCY):
        raise ValueError(f"Joint efficiency {joint_efficiency} must be {MIN_JOINT_EFFICIENCY}-{MAX_JOINT_EFFICIENCY}")

    # Get material properties (use conservative temperature)
    mat_props = get_material_properties(material, 400.0)
    allowable_stress_MPa = mat_props["allowable_stress_MPa"]

    # Convert units
    P_MPa = design_pressure_bar / 10.0
    R_mm = (diameter_m * 1000) / 2  # Radius in mm
    S_MPa = allowable_stress_MPa
    E = joint_efficiency
    CA_mm = corrosion_allowance_mm

    # Calculate required thickness: t = (P * R) / (S * E - 0.6 * P) + CA
    denominator = (S_MPa * E - 0.6 * P_MPa)
    if denominator <= 0:
        raise RuntimeError(f"Invalid wall thickness calculation - pressure too high for material. "
                          f"P={P_MPa:.1f} MPa, S*E={S_MPa*E:.1f} MPa")

    t_required_mm = (P_MPa * R_mm) / denominator

    if t_required_mm < 0:
        raise RuntimeError(f"Negative thickness calculated: {t_required_mm:.2f} mm")

    if t_required_mm > MAX_WALL_THICKNESS_MM:
        raise RuntimeError(f"Required thickness {t_required_mm:.1f} mm > {MAX_WALL_THICKNESS_MM} mm. "
                          f"Consider different material or multi-layer construction.")

    # Round to standard thickness
    t_nominal_mm = round_to_standard_thickness(t_required_mm)

    # Actual thickness includes nominal + corrosion allowance
    t_actual_mm = t_nominal_mm + CA_mm

    return {
        "required_thickness_mm": t_required_mm,
        "nominal_thickness_mm": t_nominal_mm,
        "actual_thickness_mm": t_actual_mm,
        "thickness_margin_mm": t_nominal_mm - t_required_mm,
        "corrosion_allowance_mm": CA_mm,
        "calculation_method": "ASME_VIII_Div1_Cylindrical_Shell"
    }


def calculate_head_thickness(
    diameter_m: float,
    design_pressure_bar: float,
    head_type: str,
    material: str,
    joint_efficiency: float = DEFAULT_JOINT_EFFICIENCY,
    corrosion_allowance_mm: float = DEFAULT_CORROSION_ALLOWANCE_MM,
) -> dict:
    """
    Calculate head thickness per ASME VIII Division 1.

    Supports ellipsoidal (2:1), hemispherical, and torispherical heads.

    Args:
        diameter_m: Internal diameter [m]
        design_pressure_bar: Design pressure
        head_type: "ellipsoidal", "hemispherical", or "torispherical"
        material: Material name
        joint_efficiency: Weld joint efficiency
        corrosion_allowance_mm: Corrosion allowance

    Returns:
        Head thickness data dict
    """
    # Get material properties
    mat_props = get_material_properties(material, 400.0)
    S_MPa = mat_props["allowable_stress_MPa"]
    E = joint_efficiency
    P_MPa = design_pressure_bar / 10.0
    D_mm = diameter_m * 1000
    CA_mm = corrosion_allowance_mm

    # Calculate based on head type
    if head_type == "ellipsoidal":
        # 2:1 Ellipsoidal: t = (P * D) / (2 * S * E - 0.2 * P)
        t_required_mm = (P_MPa * D_mm) / (2 * S_MPa * E - 0.2 * P_MPa)
    elif head_type == "hemispherical":
        # Hemispherical: t = (P * R) / (2 * S * E - 0.2 * P)
        R_mm = D_mm / 2
        t_required_mm = (P_MPa * R_mm) / (2 * S_MPa * E - 0.2 * P_MPa)
    elif head_type == "torispherical":
        # Torispherical (F&D): t = (0.885 * P * L) / (S * E - 0.1 * P)
        # Crown radius L ≈ D for standard F&D
        L_mm = D_mm
        t_required_mm = (0.885 * P_MPa * L_mm) / (S_MPa * E - 0.1 * P_MPa)
    else:
        raise ValueError(f"Unknown head type: {head_type}")

    # Round to standard thickness
    t_nominal_mm = round_to_standard_thickness(t_required_mm)
    t_actual_mm = t_nominal_mm + CA_mm

    return {
        "required_thickness_mm": t_required_mm,
        "nominal_thickness_mm": t_nominal_mm,
        "actual_thickness_mm": t_actual_mm,
        "thickness_margin_mm": t_nominal_mm - t_required_mm,
        "head_type": head_type
    }


def size_distillation_column(
    diameter_m: float,
    height_m: float,
    number_of_trays: int,
    tray_spacing_m: float,
    design_pressure_bar: float,
    design_temperature_C: float,
    material: str,
    column_config: dict,
) -> dict:
    """
    Size a distillation column including trays, platforms, and ladders.

    Extends vertical vessel sizing with column internals.

    Args:
        diameter_m: Column internal diameter [m]
        height_m: Column height [m]
        number_of_trays: Number of trays
        tray_spacing_m: Tray spacing [m]
        design_pressure_bar: Design pressure
        design_temperature_C: Design temperature
        material: Material of construction
        column_config: Configuration dict

    Returns:
        Column sizing dict (extends vessel sizing)

    Example:
        column = size_distillation_column(
            diameter_m=1.5,
            height_m=18.0,
            number_of_trays=36,
            tray_spacing_m=0.5,
            design_pressure_bar=2.0,
            design_temperature_C=120.0,
            material="carbon_steel",
            column_config={...}
        )
    """
    logger.info(f"Sizing distillation column D={diameter_m:.2f}m, N_trays={number_of_trays}")

    # Basic vessel sizing
    vessel_sizing = size_vertical_vessel(
        diameter_m=diameter_m,
        height_m=height_m,
        design_pressure_bar=design_pressure_bar,
        design_temperature_C=design_temperature_C,
        material=material,
        vessel_config=column_config
    )

    # Tray design
    tray_area_m2 = math.pi * (diameter_m**2) / 4
    downcomer_fraction = column_config.get("downcomer_fraction", DEFAULT_DOWNCOMER_FRACTION)
    active_fraction = 1.0 - downcomer_fraction

    # Tray weight (typical tray: 3-6 mm thick steel plate with holes)
    tray_thickness_mm = DEFAULT_TRAY_THICKNESS_MM
    tray_density_kg_m3 = vessel_sizing["material_density_kg_m3"]
    weight_per_tray_kg = tray_area_m2 * (tray_thickness_mm / 1000.0) * tray_density_kg_m3
    total_tray_weight_kg = number_of_trays * weight_per_tray_kg

    # Platforms and ladders (for access)
    platform_levels = max(0, int((height_m - MIN_HEIGHT_FOR_PLATFORMS_M) / PLATFORM_SPACING_M))
    platform_weight_kg = platform_levels * 500  # Approximate
    ladder_length_m = height_m
    ladder_weight_kg = ladder_length_m * 15  # kg/m

    # Update empty weight
    updated_empty_weight_kg = (vessel_sizing["empty_weight_kg"] +
                               total_tray_weight_kg +
                               platform_weight_kg +
                               ladder_weight_kg)

    # Build column sizing dict (extends vessel sizing)
    column_sizing = vessel_sizing.copy()
    column_sizing.update({
        "vessel_type": "distillation_column",
        "number_of_trays": number_of_trays,
        "tray_spacing_m": tray_spacing_m,
        "tray_area_m2": tray_area_m2,
        "tray_weight_kg": total_tray_weight_kg,
        "downcomer_area_fraction": downcomer_fraction,
        "active_area_fraction": active_fraction,
        "platform_levels": platform_levels,
        "platform_weight_kg": platform_weight_kg,
        "ladder_length_m": ladder_length_m,
        "ladder_weight_kg": ladder_weight_kg,
        "empty_weight_kg": updated_empty_weight_kg,
        "column_height_with_skirt_m": height_m + column_config.get("skirt_height_m", 2.0),
    })

    logger.info(f"Column sized: {number_of_trays} trays, W_empty={updated_empty_weight_kg:.0f} kg")

    return column_sizing


def estimate_vessel_weight(
    diameter_m: float,
    height_or_length_m: float,
    wall_thickness_mm: float,
    material_density_kg_m3: float,
    vessel_type: str = "vertical",
    include_internals: bool = True
) -> dict:
    """
    Estimate vessel weight components.

    Calculates shell, heads, nozzles, internals, platforms/ladders weights.

    Args:
        diameter_m: Internal diameter [m]
        height_or_length_m: Height or length [m]
        wall_thickness_mm: Wall thickness [mm]
        material_density_kg_m3: Material density [kg/m3]
        vessel_type: "vertical" or "horizontal"
        include_internals: Include internals weight estimate

    Returns:
        Weight data dict with components
    """
    # Shell weight
    t_m = wall_thickness_mm / 1000.0
    D_outer_m = diameter_m + 2 * t_m
    shell_volume_m3 = math.pi * ((D_outer_m/2)**2 - (diameter_m/2)**2) * height_or_length_m
    shell_weight_kg = shell_volume_m3 * material_density_kg_m3

    # Heads weight (approximate)
    head_area_m2 = math.pi * (diameter_m**2) / 4
    head_volume_m3 = 2 * head_area_m2 * t_m * 1.1  # Factor for head shape
    heads_weight_kg = head_volume_m3 * material_density_kg_m3

    # Nozzles weight (estimate 5% of shell weight)
    nozzles_weight_kg = 0.05 * shell_weight_kg

    # Internals weight (if requested)
    internals_weight_kg = 0.0
    if include_internals:
        vessel_volume_m3 = math.pi * (diameter_m**2) / 4 * height_or_length_m
        internals_weight_kg = vessel_volume_m3 * 50  # kg/m3 estimate

    # Platforms and ladders (for vertical vessels)
    platforms_ladders_kg = 0.0
    if vessel_type == "vertical" and height_or_length_m > MIN_HEIGHT_FOR_PLATFORMS_M:
        platforms_ladders_kg = height_or_length_m * 20  # kg/m

    # Insulation weight (if applicable)
    insulation_weight_kg = 0.0

    # Total empty weight
    empty_weight_kg = (shell_weight_kg + heads_weight_kg + nozzles_weight_kg +
                      internals_weight_kg + platforms_ladders_kg + insulation_weight_kg)

    # Test weight (full of water)
    test_volume_m3 = math.pi * (diameter_m**2) / 4 * height_or_length_m
    test_weight_kg = empty_weight_kg + test_volume_m3 * 1000

    return {
        "shell_weight_kg": shell_weight_kg,
        "heads_weight_kg": heads_weight_kg,
        "nozzles_weight_kg": nozzles_weight_kg,
        "internals_weight_kg": internals_weight_kg,
        "platforms_ladders_kg": platforms_ladders_kg,
        "insulation_weight_kg": insulation_weight_kg,
        "empty_weight_kg": empty_weight_kg,
        "test_weight_kg": test_weight_kg
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_material_properties(material: str, temperature_C: float) -> dict:
    """
    Get material properties at specified temperature.

    Interpolates allowable stress from temperature-dependent data.
    """
    if material not in MATERIAL_PROPERTIES:
        raise ValueError(f"Material '{material}' not found")

    mat_data = MATERIAL_PROPERTIES[material]
    stress_table = mat_data["allowable_stress_MPa"]
    temps = sorted(stress_table.keys())
    stresses = [stress_table[t] for t in temps]

    # Interpolate allowable stress
    if temperature_C <= temps[0]:
        allowable_stress = stresses[0]
    elif temperature_C >= temps[-1]:
        allowable_stress = stresses[-1]
    else:
        # Linear interpolation
        for i in range(len(temps) - 1):
            if temps[i] <= temperature_C <= temps[i+1]:
                t1, t2 = temps[i], temps[i+1]
                s1, s2 = stresses[i], stresses[i+1]
                allowable_stress = s1 + (s2 - s1) * (temperature_C - t1) / (t2 - t1)
                break
        else:
            allowable_stress = stresses[0]

    return {
        "allowable_stress_MPa": allowable_stress,
        "density_kg_m3": mat_data["density_kg_m3"],
        "modulus_GPa": mat_data["modulus_GPa"],
        "thermal_expansion_per_C": mat_data["thermal_expansion_per_C"],
        "cost_factor": mat_data["cost_factor"],
    }


def round_to_standard_thickness(thickness_mm: float) -> float:
    """
    Round up to next standard plate thickness.
    """
    for standard in STANDARD_THICKNESS_MM:
        if standard >= thickness_mm:
            return float(standard)

    # If exceeds largest standard, return next increment
    return float(STANDARD_THICKNESS_MM[-1]) + 10


def calculate_head_depth(diameter_m: float, head_type: str) -> float:
    """
    Calculate head depth based on type.
    """
    D = diameter_m

    if head_type == "ellipsoidal":
        # 2:1 Ellipsoidal head
        return D / 4
    elif head_type == "hemispherical":
        # Hemisphere
        return D / 2
    elif head_type == "torispherical":
        # F&D head (approximate)
        return 0.17 * D
    else:
        # Default
        return D / 4


def calculate_head_surface_area(diameter_m: float, head_type: str) -> float:
    """
    Calculate head surface area.
    """
    D = diameter_m

    if head_type == "ellipsoidal":
        # Approximate for standard 2:1 F&D
        return 1.09 * math.pi * (D/2)**2
    elif head_type == "hemispherical":
        # Hemisphere
        return 2 * math.pi * (D/2)**2
    else:
        # Default (approximate)
        return math.pi * (D/2)**2


def update_weight_with_head_thickness(
    weight_data: dict,
    diameter_m: float,
    head_thickness_mm: float,
    head_type: str,
    density_kg_m3: float,
) -> dict:
    """
    Update weight data with actual head thickness.
    """
    # Recalculate head weight
    head_area_m2 = calculate_head_surface_area(diameter_m, head_type)
    t_m = head_thickness_mm / 1000.0
    heads_weight_kg = 2 * head_area_m2 * t_m * density_kg_m3 * 1.1  # Shape factor

    # Update
    weight_data["heads_weight_kg"] = heads_weight_kg
    weight_data["empty_weight_kg"] = (weight_data["shell_weight_kg"] +
                                     heads_weight_kg +
                                     weight_data["internals_weight_kg"] +
                                     weight_data["nozzles_weight_kg"] +
                                     weight_data["platforms_ladders_kg"] +
                                     weight_data["insulation_weight_kg"])

    return weight_data


def calculate_wind_moment(
    diameter_m: float,
    height_m: float,
    wind_speed_ms: float,
) -> float:
    """
    Calculate wind overturning moment (simplified).

    Returns moment in kN·m
    """
    # Wind pressure: q = 0.5 * rho * V^2 * Cd
    q_Pa = 0.5 * AIR_DENSITY_KG_M3 * wind_speed_ms**2 * WIND_DRAG_COEFFICIENT

    # Projected area
    A_m2 = diameter_m * height_m

    # Wind force
    F_wind_N = q_Pa * A_m2

    # Moment arm (at mid-height)
    moment_arm_m = height_m / 2

    # Moment
    M_wind_Nm = F_wind_N * moment_arm_m
    M_wind_kNm = M_wind_Nm / 1000.0

    return M_wind_kNm


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_vessel_inputs(
    diameter_m: float,
    height_m: float,
    design_pressure_bar: float,
    design_temperature_C: float,
    material: str,
) -> None:
    """
    Validate vessel sizing inputs.
    """
    if diameter_m <= 0 or diameter_m > MAX_DIAMETER_M:
        raise ValueError(f"Diameter {diameter_m:.2f} m outside valid range (0, {MAX_DIAMETER_M} m)")

    if height_m <= 0 or height_m > MAX_HEIGHT_M:
        raise ValueError(f"Height {height_m:.1f} m outside valid range (0, {MAX_HEIGHT_M} m)")

    if design_pressure_bar <= 0 or design_pressure_bar > MAX_DESIGN_PRESSURE_BAR:
        raise ValueError(f"Design pressure {design_pressure_bar:.1f} bar outside valid range "
                        f"(0, {MAX_DESIGN_PRESSURE_BAR} bar)")

    if design_temperature_C < MIN_DESIGN_TEMP_C or design_temperature_C > MAX_DESIGN_TEMP_C:
        raise ValueError(f"Design temperature {design_temperature_C:.0f}°C outside valid range "
                        f"({MIN_DESIGN_TEMP_C}, {MAX_DESIGN_TEMP_C}°C)")

    if material not in MATERIAL_PROPERTIES:
        raise ValueError(f"Material '{material}' not in database. "
                        f"Available: {list(MATERIAL_PROPERTIES.keys())}")

    if design_pressure_bar < 1.0:
        logger.warning(f"Low design pressure {design_pressure_bar:.2f} bar. "
                      f"Consider vacuum or external pressure design.")


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def smoke_test_mechanical():
    """Test mechanical design functions."""
    print("=" * 70)
    print("MECHANICAL DESIGN SMOKE TEST")
    print("=" * 70)

    # Test 1: Vertical reactor vessel sizing
    print("\nTest 1: Vertical reactor vessel sizing...")
    vessel_config = {
        "code": "ASME_VIII_Div1",
        "joint_efficiency": 1.0,
        "corrosion_allowance_mm": 3.0,
        "head_type": "ellipsoidal",
        "skirt_height_m": 2.5,
        "internal_pressure_only": True,
        "include_wind_load": True
    }
    try:
        reactor_sizing = size_vertical_vessel(
            diameter_m=3.0,
            height_m=20.0,
            design_pressure_bar=35.0,
            design_temperature_C=450.0,
            material="carbon_steel",
            vessel_config=vessel_config
        )
        print(f"  Diameter: {reactor_sizing['internal_diameter_m']:.2f} m")
        print(f"  Height (T-T): {reactor_sizing['tangent_to_tangent_length_m']:.1f} m")
        print(f"  Total height: {reactor_sizing['total_height_with_heads_m']:.1f} m")
        print(f"  Shell thickness (required): {reactor_sizing['shell_thickness_required_mm']:.2f} mm")
        print(f"  Shell thickness (actual): {reactor_sizing['shell_thickness_actual_mm']:.0f} mm")
        print(f"  Head thickness: {reactor_sizing['head_thickness_actual_mm']:.0f} mm")
        print(f"  Empty weight: {reactor_sizing['empty_weight_kg']:.0f} kg")
        print(f"  Test weight: {reactor_sizing['test_weight_kg']:.0f} kg")
        print(f"  Design margin: {reactor_sizing['design_margin_percent']:.1f}%")
        assert reactor_sizing['shell_thickness_actual_mm'] >= 10
        assert reactor_sizing['empty_weight_kg'] > 10000
        assert reactor_sizing['orientation'] == 'vertical'
        print("  Test 1 passed ✓")
    except Exception as e:
        print(f"  Test 1 failed: {e}")
        raise

    # Test 2: Wall thickness calculation
    print("\nTest 2: Wall thickness calculation...")
    try:
        thickness_data = calculate_wall_thickness(
            diameter_m=3.0,
            design_pressure_bar=35.0,
            material="carbon_steel",
            joint_efficiency=1.0,
            corrosion_allowance_mm=3.0
        )
        print(f"  Required thickness: {thickness_data['required_thickness_mm']:.2f} mm")
        print(f"  Nominal thickness: {thickness_data['nominal_thickness_mm']:.0f} mm")
        print(f"  Actual thickness: {thickness_data['actual_thickness_mm']:.0f} mm")
        print(f"  Margin: {thickness_data['thickness_margin_mm']:.1f} mm")
        print(f"  Method: {thickness_data['calculation_method']}")
        assert thickness_data['nominal_thickness_mm'] >= thickness_data['required_thickness_mm']
        assert thickness_data['nominal_thickness_mm'] in STANDARD_THICKNESS_MM
        print("  Test 2 passed ✓")
    except Exception as e:
        print(f"  Test 2 failed: {e}")
        raise

    # Test 3: Head thickness calculation
    print("\nTest 3: Head thickness calculation...")
    try:
        for head_type in ["ellipsoidal", "hemispherical", "torispherical"]:
            head_thickness = calculate_head_thickness(
                diameter_m=3.0,
                design_pressure_bar=35.0,
                head_type=head_type,
                material="carbon_steel",
                joint_efficiency=1.0,
                corrosion_allowance_mm=3.0
            )
            print(f"  {head_type}: {head_thickness['actual_thickness_mm']:.0f} mm")
            assert head_thickness['actual_thickness_mm'] > 0
        print("  Test 3 passed ✓")
    except Exception as e:
        print(f"  Test 3 failed: {e}")
        raise

    # Test 4: Distillation column sizing
    print("\nTest 4: Distillation column sizing...")
    column_config = vessel_config.copy()
    column_config["platform_levels"] = 3
    try:
        column_sizing = size_distillation_column(
            diameter_m=1.5,
            height_m=18.0,
            number_of_trays=36,
            tray_spacing_m=0.5,
            design_pressure_bar=2.0,
            design_temperature_C=120.0,
            material="carbon_steel",
            column_config=column_config
        )
        print(f"  Diameter: {column_sizing['internal_diameter_m']:.2f} m")
        print(f"  Number of trays: {column_sizing['number_of_trays']}")
        print(f"  Tray area: {column_sizing['tray_area_m2']:.2f} m²")
        print(f"  Tray weight: {column_sizing['tray_weight_kg']:.0f} kg")
        print(f"  Total empty weight: {column_sizing['empty_weight_kg']:.0f} kg")
        print(f"  Platform levels: {column_sizing['platform_levels']}")
        print(f"  Shell thickness: {column_sizing['shell_thickness_actual_mm']:.0f} mm")
        assert column_sizing['number_of_trays'] == 36
        assert column_sizing['tray_weight_kg'] > 0
        assert column_sizing['vessel_type'] == 'distillation_column'
        print("  Test 4 passed ✓")
    except Exception as e:
        print(f"  Test 4 failed: {e}")
        raise

    # Test 5: Weight estimation
    print("\nTest 5: Weight estimation...")
    try:
        weight_data = estimate_vessel_weight(
            diameter_m=2.0,
            height_or_length_m=10.0,
            wall_thickness_mm=20.0,
            material_density_kg_m3=7850,
            vessel_type="vertical",
            include_internals=True
        )
        print(f"  Shell weight: {weight_data['shell_weight_kg']:.0f} kg")
        print(f"  Heads weight: {weight_data['heads_weight_kg']:.0f} kg")
        print(f"  Internals weight: {weight_data['internals_weight_kg']:.0f} kg")
        print(f"  Empty weight: {weight_data['empty_weight_kg']:.0f} kg")
        print(f"  Test weight: {weight_data['test_weight_kg']:.0f} kg")
        assert weight_data['empty_weight_kg'] > 0
        assert weight_data['test_weight_kg'] > weight_data['empty_weight_kg']
        print("  Test 5 passed ✓")
    except Exception as e:
        print(f"  Test 5 failed: {e}")
        raise

    print("\n" + "=" * 70)
    print("ALL MECHANICAL DESIGN SMOKE TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    smoke_test_mechanical()
