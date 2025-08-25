"""
We demonstrate how one uses this library to make plots of the cross-section 
at a specified kinematic setting.

## Notes:

1. 2025/08/19:
    Initialized program.
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
STARTING_PHI_VALUE_IN_DEGREES = 0

# (X): Specify a final value for azimuthal phi:
ENDING_PHI_VALUE_IN_DEGREES = 2 * np.pi

# (X): Specify *how many* values of phi you want to evaluate the cross-section
# | at. [NOTE]: This determines the *length* of the array:
NUMBER_OF_PHI_POINTS = 360

# (X): Specify the CFF H values:
CFF_H = complex(-20, 2.421)

# (X): Specify the CFF H-tilde values:
CFF_H_TILDE = complex(2.444, 1.131)

# (X): Specify the CFF E values:
CFF_E = complex(-0.541, 0.903)

# (X): Specify the CFF E-tilde values:
CFF_E_TILDE = complex(2.207, 5.383)

# (X): For all of these, we want phi to range from 0 to 360 degrees:
phi_array = np.linspace(
    start = STARTING_PHI_VALUE_IN_DEGREES,
    stop = ENDING_PHI_VALUE_IN_DEGREES,
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

# (X): Specify the target polarization *as a float*:
example_1_target_polarization = 0.

# (X): Specify the beam polarization *as a float*:
example_1_lepton_polarization = 0.0

# (X): We are using the WW relations in this computation:
example_1_ww_setting = True

# (X): Using the setting we wrote earlier, we now need to construct
# | all of it into a dictionary that will be passed into the main
# | class. It's a lot of information, but we need all of it in order
# | for the BKM formalism to evaluate.
example_1_config_dictionary = {
    "kinematics": example_1_kinematic_inputs,
    "cff_inputs": example_1_cff_inputs,
    "target_polarization": example_1_target_polarization,
    "lepton_beam_polarization": example_1_lepton_polarization,
    "using_ww": example_1_ww_setting
}

# (X): Instantiate the class for the cross-section:
example_1_cross_section = DifferentialCrossSection(
    configuration = example_1_config_dictionary,
    verbose = True,
    debugging = False)

# (X): `plot_cross_section` will *for the time being* just plt.show() the plot:
example_1_cross_section.plot_cross_section(phi_array, save_plot_name = "cross_section_plot_v1.png")

# (X): `plot_bsa` will also *for the time being* just plt.show() the plot:
example_1_cross_section.plot_bsa(phi_array, save_plot_name = "cross_section_plot_v1.png")
