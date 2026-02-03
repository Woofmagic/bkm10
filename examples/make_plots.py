"""
We demonstrate how one uses this library to make plots of the cross-section 
at a specified kinematic setting.

## Notes:

1. 2025/08/19:
    - Initialized program.
2. 2026/02/03:
    - Removed the configuration of lepton helicity and target polarization
    in accordance with class configuration changes. 
"""

# External Library | NumPy
import numpy as np

# Internal Package | DifferentialCrossSection
from bkm10_lib.core import DifferentialCrossSection

# Internal Package | BKM10Inputs
from bkm10_lib.inputs import BKM10Inputs

# Internal Package | CFFInputs
from bkm10_lib.cff_inputs import CFFInputs

# (X): Specify a value for k (the beam energy):
TEST_BEAM_ENERGY = 5.75

# (X): Specify a Q^{2} value:
TEST_Q_SQUARED = 1.82

# (X): Specify an x_{B} value:
TEST_X_BJORKEN = 0.34

# (X): Specify a t value.
# | [NOTE]: This number is usually negative:
TEST_T_VALUE = -.17

# (X): Specify a starting value for azimuthal phi:
STARTING_PHI_VALUE_IN_RADIANS = 0. * np.pi

# (X): Specify a final value for azimuthal phi:
ENDING_PHI_VALUE_IN_RADIANS = 2. * np.pi

# (X): Specify *how many* values of phi you want to evaluate the cross-section
# | at. [NOTE]: This determines the *length* of the array:
NUMBER_OF_PHI_POINTS = 360

# (X): Specify the CFF H values:
CFF_H = complex(-2.449, 3.482)

# (X): Specify the CFF H-tilde values:
CFF_H_TILDE = complex(1.409, 1.577)

# (X): Specify the CFF E values:
CFF_E = complex(2.217, 0.)

# (X): Specify the CFF E-tilde values:
CFF_E_TILDE = complex(144.410, 0.)

# (X): For all of these, we want phi to range from 0 to 360 degrees:
phi_array = np.linspace(
    start = STARTING_PHI_VALUE_IN_RADIANS,
    stop = ENDING_PHI_VALUE_IN_RADIANS,
    num = NUMBER_OF_PHI_POINTS)

# Example (1):
# | We want to obtain a NumPy array for the value of the
# | differential cross-section using standard plug-and-chug.
example_1_kinematic_inputs = BKM10Inputs(
    lab_kinematics_k = TEST_BEAM_ENERGY,
    squared_Q_momentum_transfer = TEST_Q_SQUARED,
    x_Bjorken = TEST_X_BJORKEN,
    squared_hadronic_momentum_transfer_t = TEST_T_VALUE)

# (X): Make sure to specify the CFF inputs:
# | Note: these CFF values come DIRECTLY from the KM15
# | CFF/GPD models. Two of them are ZERO, and that is how
# | you should know that they come from KM15.
example_1_cff_inputs = CFFInputs(
    compton_form_factor_h = CFF_H,
    compton_form_factor_h_tilde = CFF_H_TILDE,
    compton_form_factor_e = CFF_E,
    compton_form_factor_e_tilde = CFF_E_TILDE)

# (X): We are using the WW relations in this computation:
example_1_ww_setting = True

# (X): Using the setting we wrote earlier, we now need to construct
# | all of it into a dictionary that will be passed into the main
# | class. It's a lot of information, but we need all of it in order
# | for the BKM formalism to evaluate.
example_1_config_dictionary = {
    "kinematics": example_1_kinematic_inputs,
    "cff_inputs": example_1_cff_inputs,
    "using_ww": example_1_ww_setting
}

# (X): Instantiate the class for the total cross-section:
total_cross_section = DifferentialCrossSection(
    configuration = example_1_config_dictionary)

# (X): Make another class for *only* the cross-section due to the BH^{2} term:
bh_only_cross_section = DifferentialCrossSection(
    configuration = example_1_config_dictionary,
    dvcs_setting = False,
    interference_setting = False)

# (X): Make another class for *only* the cross-section due to the DVCS^{2} term:
dvcs_only_cross_section = DifferentialCrossSection(
    configuration = example_1_config_dictionary,
    bh_setting = False,
    interference_setting = False)

# (X): Make another class for *only* the cross-section due to the I(nterference) term:
interference_only_cross_section = DifferentialCrossSection(
    configuration = example_1_config_dictionary,
    bh_setting = False,
    dvcs_setting = False)

# (X): Make another class for *only* the cross-section due to the I(nterference) + DVCS term:
dvcs_and_interference_cross_section = DifferentialCrossSection(
    configuration = example_1_config_dictionary,
    bh_setting = False,
    dvcs_setting = True)

# (X): Save the "total" cross-section plot:
total_cross_section.plot_cross_section(
    phi_array,
    lepton_helicity = 0.0,
    target_polarization = 0.0,
    save_plot_name = "bkm_cross_section_v1.png")

# (X): Save the "total" BSA plot:
total_cross_section.plot_bsa(
    phi_array,
    target_polarization = 0.0,
    save_plot_name = "bkm_bsa_v1.png")

# (X): Save the "total" TSA plot:
total_cross_section.plot_tsa(
    phi_array,
    lepton_helicity = 0.0,
    save_plot_name = "bkm_tsa_v1.png")

# (X): Save the "total" DSA plot:
total_cross_section.plot_dsa(
    phi_array,
    save_plot_name = "bkm_dsa_v1.png")

# (X): Save the BH cross-section plot:
bh_only_cross_section.plot_cross_section(
    phi_array,
    lepton_helicity = 0.0,
    target_polarization = 0.0,
    save_plot_name = "bh_cross_section_v1.png")

# (X): Save the BH BSA plot:
bh_only_cross_section.plot_bsa(
    phi_array,
    target_polarization = 0.0,
    save_plot_name = "bh_bsa_v1.png")

# (X): Save the BH TSA plot:
bh_only_cross_section.plot_tsa(
    phi_array,
    lepton_helicity = 0.0,
    save_plot_name = "bh_tsa_v1.png")

# (X): Save the BH DSA plot:
bh_only_cross_section.plot_dsa(
    phi_array,
    save_plot_name = "bh_dsa_v1.png")

# (X): Save the DVCS cross-section plot:
dvcs_only_cross_section.plot_cross_section(
    phi_array,
    lepton_helicity = 0.0,
    target_polarization = 0.0,
    save_plot_name = "dvcs_cross_section_v1.png")

# (X): Save the DVCS BSA plot:
dvcs_only_cross_section.plot_bsa(
    phi_array,
    target_polarization = 0.0,
    save_plot_name = "dvcs_bsa_v1.png")

# (X): Save the DVCS TSA plot:
dvcs_only_cross_section.plot_tsa(
    phi_array,
    lepton_helicity = 0.0,
    save_plot_name = "dvcs_tsa_v1.png")

# (X): Save the DVCS DSA plot:
dvcs_only_cross_section.plot_dsa(
    phi_array,
    save_plot_name = "dvcs_dsa_v1.png")

# (X): Save the interference plot:
interference_only_cross_section.plot_cross_section(
    phi_array,
    lepton_helicity = 0.0,
    target_polarization = 0.0,
    save_plot_name = "interference_cross_section_v1.png")

# (X): Save the interference BSA plot:
interference_only_cross_section.plot_bsa(
    phi_array,
    target_polarization = 0.0,
    save_plot_name = "interference_bsa_v1.png")

# (X): Save the Interference TSA plot:
interference_only_cross_section.plot_tsa(
    phi_array,
    lepton_helicity = 0.0,
    save_plot_name = "interference_tsa_v1.png")

# (X): Save the DVCS DSA plot:
interference_only_cross_section.plot_dsa(
    phi_array,
    save_plot_name = "dvcs_dsa_v1.png")

# (X): Save the DVCS and Interference cross-section plot:
dvcs_and_interference_cross_section.plot_cross_section(
    phi_array,
    lepton_helicity = 0.0,
    target_polarization = 0.0,
    save_plot_name = "dvcs_interference_cross_section_v1.png")

# (X): Save the DVCS and Interference BSA plot:
dvcs_and_interference_cross_section.plot_bsa(
    phi_array,
    target_polarization = 0.0,
    save_plot_name = "dvcs_interference_bsa_v1.png")

# (X): Save the DVCS and Interference TSA plot:
dvcs_and_interference_cross_section.plot_tsa(
    phi_array,
    lepton_helicity = 0.0,
    save_plot_name = "dvcs_interference_tsa_v1.png")

# (X): Save the DVCS and Interference DSA plot:
interference_only_cross_section.plot_dsa(
    phi_array,
    save_plot_name = "dvcs_dsa_v1.png")
