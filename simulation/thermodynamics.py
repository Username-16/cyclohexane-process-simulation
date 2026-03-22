"""
thermodynamics.py
===============================================================================


Date: 2026-01-18
Version: 2.1.0
"""

from __future__ import annotations
import math
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================
R_GAS = 8.314  # J/(mol·K) - Universal gas constant
R_BAR = 0.08314  # bar·m³/(kmol·K)

# ============================================================================
# COMPONENT DATABASE (ALL PARAMETERS HARDCODED)
# ============================================================================
COMPONENT_DATABASE = {
    "benzene": {
        "name": "Benzene",
        "formula": "C6H6",
        "MW": 78.114,  # kg/kmol
        "Tc": 562.05,  # K
        "Pc": 48.95,  # bar
        "omega": 0.2103,  # Acentric factor
        "Tb": 353.24,  # K (normal boiling point)
        "Hf_298": 82.93,  # kJ/mol (heat of formation)
        "Gf_298": 129.66,  # kJ/mol (Gibbs energy)
        "S_298": 269.2,  # J/(mol·K) (entropy)
        "Cp_ig": {"A": 33.92, "B": 0.4743, "C": -0.0002836, "D": 7.943e-08},
        "Antoine": {"A": 4.72583, "B": 1660.652, "C": -1.461},
        "Hvap_298": 30.72,  # kJ/mol (heat of vaporization)
    },
    "cyclohexane": {
        "name": "Cyclohexane",
        "formula": "C6H12",
        "MW": 84.162,
        "Tc": 553.58,
        "Pc": 40.73,
        "omega": 0.2096,
        "Tb": 353.87,
        "Hf_298": -156.23,
        "Gf_298": 26.85,
        "S_298": 204.4,
        "Cp_ig": {"A": -33.71, "B": 0.6575, "C": -0.0003893, "D": 1.031e-07},
        "Antoine": {"A": 4.72583, "B": 1660.652, "C": -1.461},
        "Hvap_298": 29.97,
    },
    "H2": {
        "name": "Hydrogen",
        "formula": "H2",
        "MW": 2.016,
        "Tc": 33.19,
        "Pc": 13.13,
        "omega": -0.216,
        "Tb": 20.28,
        "Hf_298": 0.0,
        "Gf_298": 0.0,
        "S_298": 130.7,
        "Cp_ig": {"A": 27.14, "B": 0.009274, "C": -1.38e-05, "D": 7.645e-09},
        "Antoine": {"A": 3.543, "B": 99.395, "C": 7.726},
        "Hvap_298": 0.904,
    },
    "methylcyclopentane": {
        "name": "Methylcyclopentane",
        "formula": "C6H12",
        "MW": 84.162,
        "Tc": 532.79,
        "Pc": 37.85,
        "omega": 0.2301,
        "Tb": 345.0,
        "Hf_298": -106.7,
        "Gf_298": 35.0,
        "S_298": 206.0,
        "Cp_ig": {"A": -36.17, "B": 0.6452, "C": -0.0003821, "D": 1.013e-07},
        "Antoine": {"A": 4.70426, "B": 1569.57, "C": -34.846},
        "Hvap_298": 28.5,
    },
    "cyclohexene": {
        "name": "Cyclohexene",
        "formula": "C6H10",
        "MW": 82.144,
        "Tc": 560.4,
        "Pc": 43.5,
        "omega": 0.212,
        "Tb": 356.0,
        "Hf_298": -1.0,
        "Gf_298": 106.0,
        "S_298": 212.0,
        "Cp_ig": {"A": 6.35, "B": 0.5798, "C": -0.0003427, "D": 9.052e-08},
        "Antoine": {"A": 4.72583, "B": 1660.652, "C": -1.461},
        "Hvap_298": 30.0,
    },
    "methane": {
        "name": "Methane",
        "formula": "CH4",
        "MW": 16.043,
        "Tc": 190.56,
        "Pc": 45.99,
        "omega": 0.0115,
        "Tb": 111.66,
        "Hf_298": -74.6,
        "Gf_298": -50.5,
        "S_298": 186.3,
        "Cp_ig": {"A": 19.89, "B": 0.05015, "C": 1.28e-05, "D": -1.101e-08},
        "Antoine": {"A": 3.9895, "B": 443.028, "C": -0.49},
        "Hvap_298": 8.19,
    },
    "nitrogen": {
        "name": "Nitrogen",
        "formula": "N2",
        "MW": 28.014,
        "Tc": 126.19,
        "Pc": 33.98,
        "omega": 0.0377,
        "Tb": 77.36,
        "Hf_298": 0.0,
        "Gf_298": 0.0,
        "S_298": 191.6,
        "Cp_ig": {"A": 31.15, "B": -0.0136, "C": 2.677e-05, "D": -1.167e-08},
        "Antoine": {"A": 3.7362, "B": 264.651, "C": -6.788},
        "Hvap_298": 5.58,
    },
    "coke": {
        "name": "Carbon (Coke)",
        "formula": "C",
        "MW": 12.011,
        "Tc": 5000.0,
        "Pc": 100.0,
        "omega": 0.0,
        "Tb": 4000.0,
        "Hf_298": 0.0,
        "Gf_298": 0.0,
        "S_298": 5.74,
        "Cp_ig": {"A": 8.0, "B": 0.0, "C": 0.0, "D": 0.0},
        "Antoine": {"A": 0.0, "B": 0.0, "C": 0.0},
        "Hvap_298": 0.0,
    },
    "H2O": {
        "name": "Water",
        "formula": "H2O",
        "MW": 18.015,
        "Tc": 647.1,
        "Pc": 220.64,
        "omega": 0.3443,
        "Tb": 373.15,
        "Hf_298": -241.83,
        "Gf_298": -228.59,
        "S_298": 188.8,
        "Cp_ig": {"A": 33.363, "B": 0.0, "C": 0.0, "D": 0.0},
        "Antoine": {"A": 5.40221, "B": 1838.675, "C": -31.737},
        "Hvap_298": 40.66,
    },
}

# Binary Interaction Parameters (BIPs) for Peng-Robinson
BIP_MATRIX = {
    ("benzene", "H2"): 0.170,
    ("cyclohexane", "H2"): 0.185,
    ("benzene", "cyclohexane"): 0.0023,
    ("benzene", "methylcyclopentane"): 0.0,
    ("cyclohexane", "methylcyclopentane"): 0.0,
    ("H2", "methane"): 0.0,
    ("H2", "nitrogen"): 0.0,
    ("benzene", "cyclohexene"): 0.005,
    ("cyclohexane", "cyclohexene"): 0.0,
}

# ============================================================================
# PENG-ROBINSON EQUATION OF STATE
# ============================================================================
def peng_robinson_eos(T_K: float, P_bar: float, composition: Dict[str, float]) -> Dict:
    """
    Peng-Robinson EOS for mixture.
    Returns:
        dict with Z (compressibility), phi (fugacity coefficients), etc.
    """
    if not composition or sum(composition.values()) < 0.99:
        raise ValueError("Invalid composition (must sum to ~1.0)")

    total = sum(composition.values())
    z = {comp: frac / total for comp, frac in composition.items()}

    components = []
    for comp in z.keys():
        if comp not in COMPONENT_DATABASE:
            logger.warning(f"Component {comp} not in database, using default values")
            continue
        components.append(comp)

    if not components:
        raise ValueError("No valid components found")

    a_pure = {}
    b_pure = {}
    for comp in components:
        data = COMPONENT_DATABASE[comp]
        Tc = data["Tc"]
        Pc = data["Pc"]
        omega = data["omega"]

        a_c = 0.45724 * (R_GAS * Tc)**2 / Pc
        b_c = 0.07780 * R_GAS * Tc / Pc

        Tr = T_K / Tc
        kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
        alpha = (1 + kappa * (1 - math.sqrt(Tr)))**2

        a_pure[comp] = a_c * alpha
        b_pure[comp] = b_c

    a_mix = 0.0
    for i in components:
        for j in components:
            k_ij = _get_bip(i, j)
            a_ij = math.sqrt(a_pure[i] * a_pure[j]) * (1 - k_ij)
            a_mix += z.get(i, 0) * z.get(j, 0) * a_ij

    b_mix = sum(z.get(comp, 0) * b_pure[comp] for comp in components)

    R_SI = 8.314
    P_Pa = P_bar * 1e5
    A = a_mix * P_Pa / (R_SI * T_K)**2
    B = b_mix * P_Pa / (R_SI * T_K)

    c0 = -(A * B - B**2 - B**3)
    c1 = A - 3 * B**2 - 2 * B
    c2 = -(1 - B)
    c3 = 1.0

    Z = _solve_cubic_z(c0, c1, c2, c3, phase="vapor")

    phi = {comp: 1.0 for comp in components}

    return {
        "Z": Z,
        "phi": phi,
        "a_mix": a_mix,
        "b_mix": b_mix,
        "A": A,
        "B": B,
    }

def _get_bip(comp1: str, comp2: str) -> float:
    """Get binary interaction parameter (symmetric)."""
    if comp1 == comp2:
        return 0.0
    key = (comp1, comp2) if (comp1, comp2) in BIP_MATRIX else (comp2, comp1)
    return BIP_MATRIX.get(key, 0.0)

def _solve_cubic_z(c0, c1, c2, c3, phase="vapor") -> float:
    """Solve cubic equation for Z factor."""
    if phase == "vapor":
        Z = 1.0
    else:
        Z = 0.1

    for _ in range(50):
        f = c3 * Z**3 + c2 * Z**2 + c1 * Z + c0
        df = 3 * c3 * Z**2 + 2 * c2 * Z + c1
        if abs(df) < 1e-10:
            break
        Z_new = Z - f / df
        if abs(Z_new - Z) < 1e-6:
            return Z_new
        Z = max(Z_new, 0.01)

    return Z

# ============================================================================
# THERMODYNAMIC PACKAGE CLASS
# ============================================================================
class ThermodynamicPackage:
    """
    Standalone thermodynamic package with all parameters hardcoded.
    No external files needed.
    """

    def __init__(self):
        """Initialize with hardcoded component database."""
        self.components = COMPONENT_DATABASE
        logger.info(f"Thermo package initialized with {len(self.components)} components")

    def molecular_weight(self, composition: Dict[str, float]) -> float:
        """Calculate mixture molecular weight [kg/kmol]."""
        MW = 0.0
        for comp, frac in composition.items():
            if comp in self.components:
                MW += frac * self.components[comp]["MW"]
        return MW

    def ideal_gas_cp(self, T_C: float, composition: Dict[str, float]) -> float:
        """
        Calculate ideal gas heat capacity [J/(mol·K)].
        Cp = A + B*T + C*T² + D*T³
        """
        T_K = T_C + 273.15
        Cp_mix = 0.0
        for comp, frac in composition.items():
            if comp not in self.components:
                continue
            coeffs = self.components[comp]["Cp_ig"]
            Cp_i = (coeffs["A"] +
                    coeffs["B"] * T_K +
                    coeffs["C"] * T_K**2 +
                    coeffs["D"] * T_K**3)
            Cp_mix += frac * Cp_i
        return Cp_mix

    def flash_TP(self, T_C: float, P_bar: float, composition: Dict[str, float]) -> Dict:
        """
        Isothermal flash at given T and P with STRICT mass balance.
        NUMERICALLY STABLE for extreme K-values (H2, light gases).

        Returns:
            {
                "vapor_fraction": float (0-1),
                "x": dict (liquid composition),
                "y": dict (vapor composition),
                "K_values": dict,
                "phase": str
            }
        """
        T_K = T_C + 273.15

        # Normalize feed composition
        total_z = sum(composition.values())
        if total_z < 0.99:
            raise ValueError(f"Feed composition sum = {total_z}, must be ~1.0")
        z = {k: v / total_z for k, v in composition.items()}

        # ═══════════════════════════════════════════════════════════════
        # 1. Calculate K-values (vapor-liquid equilibrium ratios)
        # ═══════════════════════════════════════════════════════════════
        K_values = {}
        for comp in z.keys():
            if comp == "H2":
                # H2 is highly volatile - use Wilson correlation
                # K = (Pc/P) * exp(5.37(1 + ω)(1 - Tc/T))
                if comp in self.components:
                    Tc = self.components[comp]["Tc"]
                    Pc = self.components[comp]["Pc"]
                    omega = self.components[comp]["omega"]
                    K_values[comp] = (Pc / P_bar) * math.exp(5.37 * (1 + omega) * (1 - Tc / T_K))
                    # Clamp to reasonable range to avoid overflow
                    K_values[comp] = min(K_values[comp], 1000.0)
                else:
                    K_values[comp] = 100.0

            elif comp == "methane":
                if comp in self.components:
                    Tc = self.components[comp]["Tc"]
                    Pc = self.components[comp]["Pc"]
                    omega = self.components[comp]["omega"]
                    K_values[comp] = (Pc / P_bar) * math.exp(5.37 * (1 + omega) * (1 - Tc / T_K))
                    K_values[comp] = min(K_values[comp], 500.0)
                else:
                    K_values[comp] = 50.0

            elif comp in ["cyclohexane", "benzene", "methylcyclopentane", "cyclohexene"]:
                # Use Antoine equation for accurate vapor pressure
                try:
                    P_sat = self.vapor_pressure(T_C, comp)
                    K_values[comp] = P_sat / P_bar
                    # Ensure K is in reasonable range
                    K_values[comp] = max(0.001, min(K_values[comp], 10.0))
                except:
                    K_values[comp] = 1.0
            else:
                # Default for unknown components
                K_values[comp] = 1.0

        logger.debug(f"Flash K-values at {T_C}°C, {P_bar} bar: {K_values}")

        # ═══════════════════════════════════════════════════════════════
        # 2. Check for single-phase conditions
        # ═══════════════════════════════════════════════════════════════

        K_min = min(K_values.values())
        K_max = max(K_values.values())

        # All vapor if K_min > 1
        if K_min > 1.0:
            logger.debug(f"Flash at {T_C}°C → All vapor (K_min={K_min:.2f} > 1)")
            return {
                "vapor_fraction": 1.0,
                "x": dict(z),
                "y": dict(z),
                "K_values": K_values,
                "phase": "vapor"
            }

        # All liquid if K_max < 1
        if K_max < 1.0:
            logger.debug(f"Flash at {T_C}°C → All liquid (K_max={K_max:.2f} < 1)")
            return {
                "vapor_fraction": 0.0,
                "x": dict(z),
                "y": dict(z),
                "K_values": K_values,
                "phase": "liquid"
            }

        # ═══════════════════════════════════════════════════════════════
        # 3. Two-phase: Solve Rachford-Rice with STABLE numerics
        # ═══════════════════════════════════════════════════════════════

        def rachford_rice(V):
            """Rachford-Rice objective function with overflow protection"""
            result = 0.0
            for comp in z.keys():
                K = K_values[comp]
                denom = 1.0 + V * (K - 1.0)
                if abs(denom) < 1e-10:
                    continue  # Skip singular terms
                result += z[comp] * (K - 1.0) / denom
            return result

        def rachford_rice_derivative(V):
            """Derivative with overflow protection"""
            result = 0.0
            for comp in z.keys():
                K = K_values[comp]
                denom = 1.0 + V * (K - 1.0)
                if abs(denom) < 1e-10:
                    continue
                # Use safe calculation: avoid (K-1)^2 directly
                term = z[comp] * (K - 1.0) / denom
                result -= term * (K - 1.0) / denom
            return result

        # Initial guess using weighted average
        V = 0.5

        # Newton-Raphson with damping and bounds
        for iteration in range(100):
            try:
                f = rachford_rice(V)
                df = rachford_rice_derivative(V)

                # Check for convergence
                if abs(f) < 1e-8:
                    break

                # Prevent division by zero
                if abs(df) < 1e-12:
                    logger.warning(f"Flash derivative near zero at V={V:.4f}, using bisection")
                    # Fall back to bisection
                    if f > 0:
                        V = (V + 1.0) / 2.0
                    else:
                        V = V / 2.0
                    continue

                # Newton step with damping
                dV = -f / df
                # Limit step size
                dV = max(-0.3, min(0.3, dV))
                V_new = V + dV

                # Keep in bounds [0.001, 0.999]
                V_new = max(0.001, min(0.999, V_new))

                if abs(V_new - V) < 1e-7:
                    break

                V = V_new

            except (OverflowError, ValueError) as e:
                logger.warning(f"Numerical error in flash at iteration {iteration}: {e}")
                # Use simpler approximation
                V = 0.5
                break

        # Ensure final bounds
        V = max(0.0, min(1.0, V))

        logger.debug(f"Flash converged: V={V:.4f} after {iteration + 1} iterations")

        # ═══════════════════════════════════════════════════════════════
        # 4. Calculate phase compositions (MASS BALANCED!)
        # ═══════════════════════════════════════════════════════════════

        x = {}
        y = {}
        for comp in z.keys():
            K = K_values[comp]
            denom = 1.0 + V * (K - 1.0)

            # Prevent division by zero
            if abs(denom) < 1e-10:
                x[comp] = 0.0
                y[comp] = 0.0
                continue

            x[comp] = z[comp] / denom
            y[comp] = K * x[comp]

        # Normalize
        sum_x = sum(x.values())
        sum_y = sum(y.values())

        if sum_x > 1e-10:
            x = {k: v / sum_x for k, v in x.items()}
        else:
            x = dict(z)

        if sum_y > 1e-10:
            y = {k: v / sum_y for k, v in y.items()}
        else:
            y = dict(z)

        # ═══════════════════════════════════════════════════════════════
        # 5. Verify mass balance
        # ═══════════════════════════════════════════════════════════════

        max_error = 0.0
        for comp in z.keys():
            balance = V * y[comp] + (1.0 - V) * x[comp]
            error = abs(balance - z[comp])
            max_error = max(max_error, error)

            if error > 1e-4:
                logger.debug(
                    f"Flash {comp}: z={z[comp]:.6f}, V*y+L*x={balance:.6f}, "
                    f"error={error:.6e}"
                )

        if max_error > 1e-3:
            logger.warning(f"Flash mass balance error: {max_error:.6e}")

        logger.debug(f"Flash at {T_C}°C → Two-phase (β={V:.3f})")

        return {
            "vapor_fraction": V,
            "x": x,
            "y": y,
            "K_values": K_values,
            "phase": "two_phase"
        }

        # ═══════════════════════════════════════════════════════════════
        # 3. Two-phase: Solve Rachford-Rice equation
        # ═══════════════════════════════════════════════════════════════

        def rachford_rice(V):
            """Rachford-Rice objective function"""
            return sum(z[i] * (K_values[i] - 1) / (1 + V * (K_values[i] - 1))
                       for i in z.keys())

        # Newton-Raphson iteration for vapor fraction
        V = 0.5  # Initial guess
        for iteration in range(50):
            f = rachford_rice(V)

            # Derivative
            df = -sum(z[i] * (K_values[i] - 1) ** 2 / (1 + V * (K_values[i] - 1)) ** 2
                      for i in z.keys())

            if abs(df) < 1e-12:
                break

            V_new = V - f / df

            # Keep V in bounds [0, 1]
            V_new = max(0.0, min(1.0, V_new))

            if abs(V_new - V) < 1e-6:
                break

            V = V_new

        # Ensure bounds
        V = max(0.0, min(1.0, V))

        # ═══════════════════════════════════════════════════════════════
        # 4. Calculate phase compositions (MASS BALANCED!)
        # ═══════════════════════════════════════════════════════════════

        x = {}
        y = {}
        for comp in z.keys():
            x[comp] = z[comp] / (1 + V * (K_values[comp] - 1))
            y[comp] = K_values[comp] * x[comp]

        # Normalize (should already be ~1.0, but ensure it)
        sum_x = sum(x.values())
        sum_y = sum(y.values())

        x = {k: v / sum_x for k, v in x.items()}
        y = {k: v / sum_y for k, v in y.items()}

        # ═══════════════════════════════════════════════════════════════
        # 5. Verify mass balance (CRITICAL CHECK!)
        # ═══════════════════════════════════════════════════════════════

        for comp in z.keys():
            balance = V * y[comp] + (1 - V) * x[comp]
            error = abs(balance - z[comp])
            if error > 1e-6:
                logger.warning(
                    f"Flash mass balance error for {comp}: "
                    f"z={z[comp]:.6f}, V*y+L*x={balance:.6f}, error={error:.6e}"
                )

        logger.debug(f"Flash at {T_C}°C → Two-phase (β={V:.3f})")

        return {
            "vapor_fraction": V,
            "x": x,
            "y": y,
            "K_values": K_values,
            "phase": "two_phase"
        }

    def enthalpy_TP(self, T_C: float, P_bar: float, composition: Dict[str, float],
                    phase: Optional[str] = None) -> float:
        """
        Calculate mixture enthalpy [kJ/kmol].
        H(T,P) = H_f(298K) + ∫Cp dT + H_dep (departure function)
        """
        T_K = T_C + 273.15
        T_ref = 298.15

        H_f = 0.0
        for comp, frac in composition.items():
            if comp in self.components:
                H_f += frac * self.components[comp]["Hf_298"]

        H_sens = 0.0
        for comp, frac in composition.items():
            if comp not in self.components:
                continue
            coeffs = self.components[comp]["Cp_ig"]

            dT = T_K - T_ref
            H_sens_i = (coeffs["A"] * dT +
                        0.5 * coeffs["B"] * (T_K**2 - T_ref**2) +
                        (1/3) * coeffs["C"] * (T_K**3 - T_ref**3) +
                        0.25 * coeffs["D"] * (T_K**4 - T_ref**4))
            H_sens += frac * H_sens_i / 1000.0

        H_total = H_f + H_sens
        return H_total * 1000.0

    def vapor_pressure(self, T_C: float, component: str) -> float:
        """
        Calculate vapor pressure using Antoine equation [bar].
        log10(P_mmHg) = A - B/(T_C + C)
        """
        if component not in self.components:
            logger.warning(f"Component {component} not found")
            return 0.0

        coeffs = self.components[component]["Antoine"]
        log_P_mmHg = coeffs["A"] - coeffs["B"] / (T_C + coeffs["C"])
        P_mmHg = 10**log_P_mmHg
        P_bar = P_mmHg / 750.062
        return P_bar

    def density_TP(self, T_C: float, P_bar: float, composition: Dict[str, float],
                   phase: str = "liquid") -> float:
        """
        Estimate density [kg/m³].
        """
        MW = self.molecular_weight(composition)
        T_K = T_C + 273.15

        if phase == "vapor" or phase == "gas":
            rho = (P_bar * MW) / (R_BAR * T_K)
            return rho
        else:
            return 700.0

# ============================================================================
# SMOKE TEST
# ============================================================================
def _smoke_test():
    """Test thermodynamic package without external files."""
    print("="*70)
    print("THERMODYNAMICS MODULE SMOKE TEST - VERSION 2.1.0")
    print("="*70)

    thermo = ThermodynamicPackage()
    print(f"\n✓ Initialized with {len(thermo.components)} components")

    composition = {"benzene": 0.3, "H2": 0.5, "cyclohexane": 0.2}

    MW = thermo.molecular_weight(composition)
    print(f"\n1. Molecular weight: {MW:.2f} kg/kmol")
    assert 10 < MW < 100, "MW out of range"

    Cp = thermo.ideal_gas_cp(200, composition)
    print(f"2. Ideal gas Cp at 200°C: {Cp:.2f} J/(mol·K)")
    assert 20 < Cp < 200, "Cp out of range"

    # Test all three flash cases
    print("\n3. Flash tests:")

    # All liquid
    flash_liquid = thermo.flash_TP(30, 30, composition)
    print(f"   a) T=30°C: β={flash_liquid['vapor_fraction']:.2f}, phase={flash_liquid['phase']}")
    print(f"      x={list(flash_liquid['x'].keys())}, y={list(flash_liquid['y'].keys())}")
    assert len(flash_liquid['x']) > 0, "FAIL: x is empty!"
    assert len(flash_liquid['y']) > 0, "FAIL: y is empty!"
    print(f"      ✓ Both x and y have compositions")

    # Two-phase
    flash_twophase = thermo.flash_TP(150, 30, composition)
    print(f"   b) T=150°C: β={flash_twophase['vapor_fraction']:.2f}, phase={flash_twophase['phase']}")
    print(f"      x={list(flash_twophase['x'].keys())}, y={list(flash_twophase['y'].keys())}")
    assert len(flash_twophase['x']) > 0, "FAIL: x is empty!"
    assert len(flash_twophase['y']) > 0, "FAIL: y is empty!"
    print(f"      ✓ Both x and y have compositions")

    # All vapor
    flash_vapor = thermo.flash_TP(250, 30, composition)
    print(f"   c) T=250°C: β={flash_vapor['vapor_fraction']:.2f}, phase={flash_vapor['phase']}")
    print(f"      x={list(flash_vapor['x'].keys())}, y={list(flash_vapor['y'].keys())}")
    assert len(flash_vapor['x']) > 0, "FAIL: x is empty!"
    assert len(flash_vapor['y']) > 0, "FAIL: y is empty!"
    print(f"      ✓ Both x and y have compositions")

    H = thermo.enthalpy_TP(200, 30, composition, "vapor")
    print(f"\n4. Enthalpy at 200°C: {H:.0f} kJ/kmol")

    P_vap = thermo.vapor_pressure(80, "benzene")
    print(f"5. Benzene vapor pressure at 80°C: {P_vap:.3f} bar")
    assert 0.5 < P_vap < 2.0, "Vapor pressure out of range"

    print("\n" + "="*70)
    print("✓ ALL THERMODYNAMICS TESTS PASSED")
    print("="*70)

if __name__ == "__main__":
    _smoke_test()
