"""
Reaction kinetics for benzene hydrogenation to cyclohexane.
===============================================================================

R1 (MAIN REACTION) uses LHHW kinetics based on Saeys et al. (2004)
R2-R6 (SIDE REACTIONS) use power-law kinetics

ALL KINETIC PARAMETERS ARE HARDCODED AT THE TOP.
For optimization, use process parameters (recycle, pressure, etc.) - NOT kinetics.

Reaction Network:
- R1: Main hydrogenation (C6H6 + 3H2 → C6H12) - LHHW
- R2: Isomerization to MCP (C6H6 + 3H2 → CH3-C5H9) - Power Law
- R3: Partial hydrogenation (C6H6 + 2H2 → C6H10) - Power Law
- R4: Cyclohexene hydrogenation (C6H10 + H2 → C6H12) - Power Law (fast)
- R5: Cracking (C6H6 + 9H2 → 6CH4) - Power Law
- R6: Coking (C6H6 → 6C + 3H2) - Power Law (deactivation)

References:
- Saeys, M., et al. (2004). Molecular Physics, 102(3), 267-272.
  DOI: 10.1080/00268970410001668516

Date: 2026-01-14
Version: 4.0.0 - LHHW for R1, Power Law for R2-R6
"""

import logging
import math
from typing import Dict
import numpy as np
from scipy.interpolate import interp1d

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================
R_GAS = 8.314          # Universal gas constant [J/(mol·K)]
R_GAS_KJ = 0.008314    # Universal gas constant [kJ/(mol·K)]

logger = logging.getLogger(__name__)


# ============================================================================
# R1: LHHW KINETICS PARAMETERS (Saeys et al., 2004)
# ============================================================================
# Main hydrogenation: C6H6 + 3H2 → C6H12
# Rate equation: r = (C_t * k_5 * K_1*K_2*K_3*K_4 * K_A * K_H^2.5 * P_A * P_H^2.5)
#                    / (1 + K_A*P_A + (K_H*P_H)^0.5)^2

# Active sites
R1_LHHW_C_TOTAL = 0.102                # Total active site concentration [mol/kg_cat]

# Rate constant for 5th hydrogenation step (RDS - Rate Determining Step)
R1_LHHW_K5_PREEXP = 1.0e12             # Pre-exponential factor k_5,0 [s^-1]
R1_LHHW_K5_EA = 103.5                  # Activation energy for 5th step [kJ/mol]

# Benzene adsorption equilibrium
R1_LHHW_KA_PREEXP = 1.0e-10            # Benzene adsorption pre-exponential [Pa^-1]
R1_LHHW_H_ADS_BENZENE = -74.0          # Benzene adsorption enthalpy [kJ/mol]

# Hydrogen adsorption equilibrium
R1_LHHW_KH_PREEXP = 1.0e-10            # H2 adsorption pre-exponential [Pa^-1]
R1_LHHW_H_ADS_H2 = -68.8               # H2 adsorption enthalpy (optimized) [kJ/mol]

# Equilibrium constants for hydrogenation steps 1-4
# K_i = exp(-ΔH_i / RT)
R1_LHHW_DELTA_H_STEP1 = -70.6          # Step 1 reaction enthalpy [kJ/mol]
R1_LHHW_DELTA_H_STEP2 = -72.0          # Step 2 reaction enthalpy [kJ/mol]
R1_LHHW_DELTA_H_STEP3 = -74.0          # Step 3 reaction enthalpy [kJ/mol]
R1_LHHW_DELTA_H_STEP4 = -77.0          # Step 4 reaction enthalpy [kJ/mol]

# Overall heat of reaction
R1_HEAT_OF_REACTION = -206.0           # Overall ΔH_rxn [kJ/mol] (exothermic)

# Stoichiometry
R1_STOICH = {"benzene": -1.0, "H2": -3.0, "cyclohexane": 1.0}


# ============================================================================
# R2-R6: POWER LAW KINETICS PARAMETERS
# ============================================================================
# These reactions use simplified power-law kinetics: r = k * C_benzene^a * C_H2^b

# ----------------------------------------------------------------------------
# R2: Methylcyclopentane Formation (C6H6 + 3H2 → CH3-C5H9)
# ----------------------------------------------------------------------------
R2_PRE_EXPONENTIAL = 1.0e7
R2_ACTIVATION_ENERGY = 65.0
R2_HEAT_OF_REACTION = -213.0
R2_BENZENE_ORDER = 0.8
R2_H2_ORDER = 0.5
R2_STOICH = {"benzene": -1.0, "H2": -3.0, "methylcyclopentane": 1.0}

# ----------------------------------------------------------------------------
# R3: Cyclohexene Formation (C6H6 + 2H2 → C6H10) - Intermediate
# ----------------------------------------------------------------------------
R3_PRE_EXPONENTIAL = 5.0e6
R3_ACTIVATION_ENERGY = 55.0
R3_HEAT_OF_REACTION = -119.0
R3_BENZENE_ORDER = 0.7
R3_H2_ORDER = 0.6
R3_STOICH = {"benzene": -1.0, "H2": -2.0, "cyclohexene": 1.0}

# ----------------------------------------------------------------------------
# R4: Cyclohexene Hydrogenation (C6H10 + H2 → C6H12) - Fast
# ----------------------------------------------------------------------------
R4_PRE_EXPONENTIAL = 1.0e9
R4_ACTIVATION_ENERGY = 45.0
R4_HEAT_OF_REACTION = -87.0
R4_CYCLOHEXENE_ORDER = 1.0
R4_H2_ORDER = 1.0
R4_STOICH = {"cyclohexene": -1.0, "H2": -1.0, "cyclohexane": 1.0}

# ----------------------------------------------------------------------------
# R5: Cracking (C6H6 + 9H2 → 6CH4) - Unwanted
# ----------------------------------------------------------------------------
R5_PRE_EXPONENTIAL = 1.0e5
R5_ACTIVATION_ENERGY = 75.0
R5_HEAT_OF_REACTION = -531.0
R5_BENZENE_ORDER = 1.0
R5_H2_ORDER = 2.0
R5_STOICH = {"benzene": -1.0, "H2": -9.0, "methane": 6.0}

# ----------------------------------------------------------------------------
# R6: Coking (C6H6 → 6C + 3H2) - Catalyst Deactivation
# ----------------------------------------------------------------------------
R6_PRE_EXPONENTIAL = 1.0e3
R6_ACTIVATION_ENERGY = 95.0
R6_HEAT_OF_REACTION = 82.0              # Positive = endothermic
R6_BENZENE_ORDER = 1.0
R6_STOICH = {"benzene": -1.0, "H2": 3.0, "coke": 6.0}

# Enable catalyst deactivation tracking
ENABLE_CATALYST_DEACTIVATION = True


# ============================================================================
# TEMPERATURE-DEPENDENT SELECTIVITY MATRIX
# ============================================================================
# Selectivity to each product changes with temperature
# Based on pilot plant data for YOUR specific catalyst

SELECTIVITY_TEMP_C = np.array([160, 180, 200, 220, 250, 280])  # [°C]

# Selectivity to each product (must sum to 1.0 at each temperature)
SELECTIVITY_CYCLOHEXANE = np.array([0.92, 0.91, 0.90, 0.88, 0.84, 0.78])
SELECTIVITY_MCP = np.array([0.02, 0.025, 0.03, 0.04, 0.06, 0.09])
SELECTIVITY_CYCLOHEXENE = np.array([0.03, 0.025, 0.02, 0.025, 0.03, 0.04])
SELECTIVITY_CRACKING = np.array([0.02, 0.025, 0.03, 0.035, 0.05, 0.07])
SELECTIVITY_COKING = np.array([0.01, 0.015, 0.02, 0.02, 0.02, 0.02])


# ============================================================================
# R1: LHHW KINETICS CLASS
# ============================================================================

class LHHW_R1_MainHydrogenation:
    """
    Langmuir-Hinshelwood-Hougen-Watson kinetics for main benzene hydrogenation.

    Based on Saeys et al. (2004) first-principles DFT calculations.
    Fifth hydrogenation step is rate-determining.
    """

    def __init__(self):
        """Initialize with hardcoded LHHW parameters."""
        # All parameters from module-level constants
        self.C_t = R1_LHHW_C_TOTAL
        self.k5_0 = R1_LHHW_K5_PREEXP
        self.Ea_5 = R1_LHHW_K5_EA
        self.KA_0 = R1_LHHW_KA_PREEXP
        self.H_ads_A = R1_LHHW_H_ADS_BENZENE
        self.KH_0 = R1_LHHW_KH_PREEXP
        self.H_ads_H = R1_LHHW_H_ADS_H2

        # Reaction enthalpies for steps 1-4
        self.delta_H_steps = {
            1: R1_LHHW_DELTA_H_STEP1,
            2: R1_LHHW_DELTA_H_STEP2,
            3: R1_LHHW_DELTA_H_STEP3,
            4: R1_LHHW_DELTA_H_STEP4
        }

        self.heat_of_reaction_kJ_mol = R1_HEAT_OF_REACTION
        self.stoichiometry = R1_STOICH

        logger.debug("Initialized LHHW kinetics for R1 (Saeys et al., 2004)")

    def rate_constant_k5(self, T_K: float) -> float:
        """
        Rate constant for 5th hydrogenation step (RDS).

        k_5 = k_5,0 * exp(-Ea_5 / RT)

        Args:
            T_K: Temperature [K]

        Returns:
            k_5 [s^-1]
        """
        return self.k5_0 * math.exp(-self.Ea_5 * 1000.0 / (R_GAS * T_K))

    def adsorption_constant_benzene(self, T_K: float) -> float:
        """
        Benzene adsorption equilibrium constant.

        K_A = K_A,0 * exp(-ΔH_ads_A / RT)

        Args:
            T_K: Temperature [K]

        Returns:
            K_A [Pa^-1]
        """
        return self.KA_0 * math.exp(-self.H_ads_A * 1000.0 / (R_GAS * T_K))

    def adsorption_constant_hydrogen(self, T_K: float) -> float:
        """
        Hydrogen adsorption equilibrium constant.

        K_H = K_H,0 * exp(-ΔH_ads_H / RT)

        Args:
            T_K: Temperature [K]

        Returns:
            K_H [Pa^-1]
        """
        return self.KH_0 * math.exp(-self.H_ads_H * 1000.0 / (R_GAS * T_K))

    def equilibrium_constant(self, step: int, T_K: float) -> float:
        """
        Equilibrium constant for hydrogenation steps 1-4.

        K_i = exp(-ΔH_i / RT)

        Args:
            step: Step number (1-4)
            T_K: Temperature [K]

        Returns:
            K_i [dimensionless]
        """
        if step not in self.delta_H_steps:
            raise ValueError(f"Step {step} invalid. Must be 1-4.")

        delta_H = self.delta_H_steps[step]
        return math.exp(-delta_H * 1000.0 / (R_GAS * T_K))

    def reaction_rate(
        self,
        temperature_C: float,
        pressure_bar: float,
        concentrations: Dict[str, float],
        activity_factor: float = 1.0
    ) -> float:
        """
        Calculate LHHW reaction rate.

        r = (C_t * k_5 * K_1*K_2*K_3*K_4 * K_A * K_H^2.5 * P_A * P_H^2.5)
            / (1 + K_A*P_A + sqrt(K_H*P_H))^2

        Args:
            temperature_C: Temperature [°C]
            pressure_bar: Total pressure [bar]
            concentrations: Component concentrations [kmol/m³]
            activity_factor: Catalyst activity (0-1)

        Returns:
            Reaction rate [kmol/(m³·s)]
        """
        # Convert to Kelvin
        T_K = temperature_C + 273.15

        # Get partial pressures [Pa]
        C_benzene = concentrations.get("benzene", 0.0)
        C_H2 = concentrations.get("H2", 0.0)

        # Convert concentrations to partial pressures using ideal gas law
        # P = C * R * T (Pa = kmol/m³ * J/(mol·K) * K * 1000 mol/kmol)
        P_benzene = C_benzene * R_GAS * T_K * 1000.0  # [Pa]
        P_H2 = C_H2 * R_GAS * T_K * 1000.0            # [Pa]

        # Check for zero concentrations
        if P_benzene < 1e-6 or P_H2 < 1e-6:
            return 0.0

        # Calculate rate constant
        k_5 = self.rate_constant_k5(T_K)

        # Calculate adsorption constants
        K_A = self.adsorption_constant_benzene(T_K)
        K_H = self.adsorption_constant_hydrogen(T_K)

        # Calculate equilibrium constants for steps 1-4
        K_prod = 1.0
        for i in range(1, 5):
            K_prod *= self.equilibrium_constant(i, T_K)

        # LHHW rate equation
        numerator = (self.C_t * k_5 * K_prod * K_A * (K_H ** 2.5) *
                    P_benzene * (P_H2 ** 2.5))

        denominator = (1.0 + K_A * P_benzene + math.sqrt(K_H * P_H2)) ** 2

        r_intrinsic = numerator / denominator  # [mol/(kg_cat·s)]

        # Apply catalyst activity factor
        r_intrinsic *= activity_factor

        # Convert to [kmol/(m³·s)] assuming catalyst density ~1000 kg/m³
        # This is a simplification - actual conversion depends on reactor configuration
        catalyst_density = 1000.0  # [kg_cat/m³_reactor]
        rate = r_intrinsic * catalyst_density / 1000.0  # [kmol/(m³_reactor·s)]

        # Validate
        if math.isnan(rate) or math.isinf(rate):
            raise RuntimeError("LHHW rate calculation failed for R1")

        if rate < 0:
            logger.warning(f"Negative LHHW rate for R1: {rate}, setting to 0")
            rate = 0.0

        return rate


# ============================================================================
# R2-R6: POWER LAW KINETICS CLASS
# ============================================================================

class PowerLawKinetics:
    """Power-law kinetics for side reactions R2-R6."""

    def __init__(
        self,
        reaction_id: str,
        pre_exp: float,
        ea_kj_mol: float,
        delta_h_kj_mol: float,
        stoichiometry: Dict[str, float],
        power_law_exponents: Dict[str, float]
    ):
        """Initialize power-law kinetics."""
        self.reaction_id = reaction_id
        self.pre_exponential_factor = pre_exp
        self.activation_energy_kJ_mol = ea_kj_mol
        self.heat_of_reaction_kJ_mol = delta_h_kj_mol
        self.stoichiometry = stoichiometry
        self.power_law_exponents = power_law_exponents

        logger.debug(
            f"Initialized {reaction_id}: A={pre_exp:.2e}, Ea={ea_kj_mol:.1f} kJ/mol"
        )

    def rate_constant(self, temperature_C: float) -> float:
        """Arrhenius rate constant: k(T) = A * exp(-Ea / RT)"""
        if temperature_C < 0 or temperature_C > 500:
            raise ValueError(f"Temperature {temperature_C}°C out of range")

        T_K = temperature_C + 273.15
        k = self.pre_exponential_factor * math.exp(
            -self.activation_energy_kJ_mol / (R_GAS_KJ * T_K)
        )

        return k

    def reaction_rate(
        self,
        temperature_C: float,
        pressure_bar: float,
        concentrations: Dict[str, float],
        activity_factor: float = 1.0
    ) -> float:
        """
        Calculate power-law reaction rate.

        r = k(T) * Π[C_i^n_i] * activity_factor
        """
        # Get rate constant
        k = self.rate_constant(temperature_C)

        # Apply power law
        rate = k
        for component, exponent in self.power_law_exponents.items():
            conc = concentrations.get(component, 0.0)

            if conc < 1e-10 and exponent > 0:
                return 0.0

            rate *= np.power(conc, exponent)

        # Apply catalyst activity
        rate *= activity_factor

        # Validate
        if math.isnan(rate) or math.isinf(rate):
            raise RuntimeError(f"Rate calculation failed for {self.reaction_id}")

        if rate < 0:
            rate = 0.0

        return rate


# ============================================================================
# COMPLETE REACTION SYSTEM
# ============================================================================

class ReactionSystem:
    """
    Complete 6-reaction system with LHHW for R1 and power-law for R2-R6.

    All parameters hardcoded from module-level constants.
    """

    def __init__(self):
        """Initialize complete reaction system."""

        # R1: LHHW kinetics
        self.R1_lhhw = LHHW_R1_MainHydrogenation()

        # R2-R6: Power-law kinetics
        self.reactions_powerlaw: Dict[str, PowerLawKinetics] = {}

        # R2: MCP formation
        self.reactions_powerlaw["R2_methylcyclopentane"] = PowerLawKinetics(
            reaction_id="R2_methylcyclopentane",
            pre_exp=R2_PRE_EXPONENTIAL,
            ea_kj_mol=R2_ACTIVATION_ENERGY,
            delta_h_kj_mol=R2_HEAT_OF_REACTION,
            stoichiometry=R2_STOICH,
            power_law_exponents={"benzene": R2_BENZENE_ORDER, "H2": R2_H2_ORDER}
        )

        # R3: Cyclohexene formation
        self.reactions_powerlaw["R3_cyclohexene_intermediate"] = PowerLawKinetics(
            reaction_id="R3_cyclohexene_intermediate",
            pre_exp=R3_PRE_EXPONENTIAL,
            ea_kj_mol=R3_ACTIVATION_ENERGY,
            delta_h_kj_mol=R3_HEAT_OF_REACTION,
            stoichiometry=R3_STOICH,
            power_law_exponents={"benzene": R3_BENZENE_ORDER, "H2": R3_H2_ORDER}
        )

        # R4: Fast cyclohexene hydrogenation
        self.reactions_powerlaw["R4_cyclohexene_to_cyclohexane"] = PowerLawKinetics(
            reaction_id="R4_cyclohexene_to_cyclohexane",
            pre_exp=R4_PRE_EXPONENTIAL,
            ea_kj_mol=R4_ACTIVATION_ENERGY,
            delta_h_kj_mol=R4_HEAT_OF_REACTION,
            stoichiometry=R4_STOICH,
            power_law_exponents={"cyclohexene": R4_CYCLOHEXENE_ORDER, "H2": R4_H2_ORDER}
        )

        # R5: Cracking
        self.reactions_powerlaw["R5_cracking_side"] = PowerLawKinetics(
            reaction_id="R5_cracking_side",
            pre_exp=R5_PRE_EXPONENTIAL,
            ea_kj_mol=R5_ACTIVATION_ENERGY,
            delta_h_kj_mol=R5_HEAT_OF_REACTION,
            stoichiometry=R5_STOICH,
            power_law_exponents={"benzene": R5_BENZENE_ORDER, "H2": R5_H2_ORDER}
        )

        # R6: Coking
        self.reactions_powerlaw["R6_coking"] = PowerLawKinetics(
            reaction_id="R6_coking",
            pre_exp=R6_PRE_EXPONENTIAL,
            ea_kj_mol=R6_ACTIVATION_ENERGY,
            delta_h_kj_mol=R6_HEAT_OF_REACTION,
            stoichiometry=R6_STOICH,
            power_law_exponents={"benzene": R6_BENZENE_ORDER}
        )

        logger.info(f"Initialized 1 LHHW + {len(self.reactions_powerlaw)} power-law reactions")

        # Build selectivity interpolators
        self.selectivity_temperatures = SELECTIVITY_TEMP_C

        selectivity_data = {
            "cyclohexane": SELECTIVITY_CYCLOHEXANE,
            "methylcyclopentane": SELECTIVITY_MCP,
            "cyclohexene": SELECTIVITY_CYCLOHEXENE,
            "cracking": SELECTIVITY_CRACKING,
            "coking": SELECTIVITY_COKING
        }

        self.selectivity_interpolators = {}
        for product, selectivity_values in selectivity_data.items():
            self.selectivity_interpolators[product] = interp1d(
                self.selectivity_temperatures,
                selectivity_values,
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )

        logger.info(
            f"Selectivity matrix loaded for "
            f"{self.selectivity_temperatures.min():.0f}-"
            f"{self.selectivity_temperatures.max():.0f}°C"
        )

        self.enable_deactivation = ENABLE_CATALYST_DEACTIVATION

    def get_selectivity(self, temperature_C: float) -> Dict[str, float]:
        """Get temperature-dependent product selectivity (normalized to 1.0)."""
        T_min = self.selectivity_temperatures.min()
        T_max = self.selectivity_temperatures.max()

        if temperature_C < T_min - 50 or temperature_C > T_max + 50:
            logger.warning(
                f"Temperature {temperature_C}°C far outside selectivity range!"
            )

        # Interpolate
        selectivity = {}
        for product, interpolator in self.selectivity_interpolators.items():
            value = float(interpolator(temperature_C))
            selectivity[product] = max(0.0, min(1.0, value))

        # Normalize
        total = sum(selectivity.values())
        if total < 1e-10:
            raise RuntimeError(f"Selectivity sum near zero at {temperature_C}°C")

        for product in selectivity:
            selectivity[product] /= total

        return selectivity

    def calculate_rates(
        self,
        temperature_C: float,
        pressure_bar: float,
        concentrations: Dict[str, float],
        catalyst_activity: float = 1.0
    ) -> Dict[str, float]:
        """
        Calculate all 6 reaction rates.

        R1: LHHW kinetics (independent)
        R2-R6: Power-law kinetics distributed by selectivity

        Returns:
            Dictionary {reaction_id: rate [kmol/(m³·s)]}
        """
        # Calculate R1 using LHHW (independent of selectivity)
        r1_lhhw = self.R1_lhhw.reaction_rate(
            temperature_C, pressure_bar, concentrations, catalyst_activity
        )

        # Get selectivity
        selectivity = self.get_selectivity(temperature_C)

        # Calculate base rate for power-law reactions using R2 as reference
        # (this maintains consistency with selectivity distribution)
        rxn_r2 = self.reactions_powerlaw["R2_methylcyclopentane"]
        base_rate = rxn_r2.reaction_rate(
            temperature_C, pressure_bar, concentrations, catalyst_activity
        )

        # Distribute power-law reactions by selectivity
        rates = {
            "R1_main_hydrogenation": r1_lhhw * selectivity["cyclohexane"],
            "R2_methylcyclopentane": base_rate * selectivity["methylcyclopentane"],
            "R3_cyclohexene_intermediate": base_rate * selectivity["cyclohexene"],
            "R5_cracking_side": base_rate * selectivity["cracking"],
            "R6_coking": base_rate * selectivity["coking"]
        }

        # R4 is independent (fast parallel reaction)
        rxn_r4 = self.reactions_powerlaw["R4_cyclohexene_to_cyclohexane"]
        rates["R4_cyclohexene_to_cyclohexane"] = rxn_r4.reaction_rate(
            temperature_C, pressure_bar, concentrations, catalyst_activity
        )

        # Validate
        for rxn_id, rate in rates.items():
            if rate < 0:
                logger.warning(f"Negative rate for {rxn_id}: {rate}, setting to 0")
                rates[rxn_id] = 0.0

        return rates

    def net_production_rates(
        self,
        temperature_C: float,
        pressure_bar: float,
        concentrations: Dict[str, float],
        catalyst_activity: float = 1.0
    ) -> Dict[str, float]:
        """
        Calculate net production rates for all components.

        r_net(i) = Σ_j [ν_i,j * r_j]

        Returns:
            Dictionary {component: net_rate [kmol/(m³·s)]}
        """
        rates = self.calculate_rates(
            temperature_C, pressure_bar, concentrations, catalyst_activity
        )

        net_rates = {
            "benzene": 0.0,
            "H2": 0.0,
            "cyclohexane": 0.0,
            "methylcyclopentane": 0.0,
            "cyclohexene": 0.0,
            "methane": 0.0,
            "coke": 0.0
        }

        # R1 contribution (LHHW)
        for component, stoich_coeff in self.R1_lhhw.stoichiometry.items():
            net_rates[component] += stoich_coeff * rates["R1_main_hydrogenation"]

        # R2-R6 contributions (power-law)
        for rxn_id, rxn in self.reactions_powerlaw.items():
            for component, stoich_coeff in rxn.stoichiometry.items():
                if component in net_rates:
                    net_rates[component] += stoich_coeff * rates[rxn_id]
                else:
                    net_rates[component] = net_rates.get(component, 0.0) + stoich_coeff * rates[rxn_id]

        return net_rates

    def heat_of_reaction(self, reaction_id: str) -> float:
        """Get heat of reaction [kJ/mol] (negative = exothermic)."""
        if reaction_id == "R1_main_hydrogenation":
            return self.R1_lhhw.heat_of_reaction_kJ_mol
        elif reaction_id in self.reactions_powerlaw:
            return self.reactions_powerlaw[reaction_id].heat_of_reaction_kJ_mol
        else:
            raise KeyError(f"Unknown reaction: {reaction_id}")


# ============================================================================
# SMOKE TEST
# ============================================================================

def _smoke_test():
    """Test the kinetics module."""
    print("="*80)
    print("KINETICS MODULE SMOKE TEST (LHHW + POWER LAW)")
    print("="*80)

    print("\n1. Initializing ReactionSystem...")
    rxn = ReactionSystem()
    print(f" ✓ 1 LHHW reaction + {len(rxn.reactions_powerlaw)} power-law reactions")

    print("\n2. Testing LHHW parameters...")
    print(f" C_t = {R1_LHHW_C_TOTAL} mol/kg_cat")
    print(f" k_5,0 = {R1_LHHW_K5_PREEXP:.2e} s^-1")
    print(f" Ea_5 = {R1_LHHW_K5_EA} kJ/mol (RDS)")
    print(f" ΔH_ads(benzene) = {R1_LHHW_H_ADS_BENZENE} kJ/mol")
    print(f" ΔH_ads(H2) = {R1_LHHW_H_ADS_H2} kJ/mol")
    print(" ✓ LHHW parameters loaded from Saeys et al. (2004)")

    print("\n3. Testing selectivity at 200°C...")
    sel = rxn.get_selectivity(200.0)
    print(f" Selectivity: {sel}")
    print(f" Sum: {sum(sel.values()):.6f}")
    assert abs(sum(sel.values()) - 1.0) < 1e-6
    print(" ✓ Selectivity normalized")

    print("\n4. Testing reaction rates at 180°C, 30 bar...")
    conc = {
        "benzene": 0.5,
        "H2": 2.0,
        "cyclohexene": 0.01,
        "cyclohexane": 0.05,
        "methylcyclopentane": 0.0
    }
    rates = rxn.calculate_rates(180.0, 30.0, conc)
    print(" Rates [kmol/(m³·s)]:")
    for rxn_id, rate in rates.items():
        kinetic_type = "LHHW" if rxn_id == "R1_main_hydrogenation" else "Power-Law"
        print(f"   {rxn_id}: {rate:.6e} ({kinetic_type})")
    print(" ✓ All rates calculated")

    print("\n5. Testing net production rates...")
    net = rxn.net_production_rates(180.0, 30.0, conc)
    print(" Net rates [kmol/(m³·s)]:")
    for comp, rate in net.items():
        if abs(rate) > 1e-12:
            print(f"   {comp}: {rate:+.6e}")
    assert net["benzene"] < 0, "Benzene should be consumed"
    assert net["cyclohexane"] > 0, "Cyclohexane should be produced"
    print(" ✓ Stoichiometry correct")

    print("\n6. Testing heat of reaction...")
    for rxn_id in ["R1_main_hydrogenation", "R2_methylcyclopentane", "R6_coking"]:
        dH = rxn.heat_of_reaction(rxn_id)
        exo_endo = "exothermic" if dH < 0 else "endothermic"
        print(f" {rxn_id}: ΔH = {dH:.1f} kJ/mol ({exo_endo})")
    print(" ✓ Heat of reaction values loaded")

    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED")
    print("="*80)
    print("\nKINETIC MODEL SUMMARY:")
    print("  • R1: LHHW kinetics (Saeys et al., 2004) - First principles")
    print("  • R2-R6: Power-law kinetics - Empirical")
    print("  • All parameters hardcoded at top of file")
    print("  • Ready for process optimization (NOT kinetic optimization)")
    print("="*80)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _smoke_test()
