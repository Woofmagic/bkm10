"""
Here is an example of how one might use this library
for analysis. Physicists are mostly interested in: cross sections,
various cross-section asymmetries, CFFs, and GPDs.

Example (1): We obtain the BKM10 predictions for the four-fold differential
cross-section at the "standard kinematic bin" using the WW relations for
an unpolarized target and unpolarized beam.

## Notes:

1. 2025/07/24:
    Program correctly outputs *only* the interference contribution to the four-fold
    differential cross-section --- that is the standard "U"-shape. The plot matches with
    that one derived using the `BKM10_Spin_Polarized` repository. 
2. 2025/08/19:
    This program used to include plotting of the cross-section, but we have moved that
    to a different script: `make_plots.py`.
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
TEST_T_VALUE = -0.17

# (X): Specify a starting value for azimuthal phi:
STARTING_PHI_VALUE_IN_DEGREES = 0

# (X): Specify a final value for azimuthal phi:
ENDING_PHI_VALUE_IN_DEGREES = 360

# (X): Specify *how many* values of phi you want to evaluate the cross-section
# | at. [NOTE]: This determines the *length* of the array:
NUMBER_OF_PHI_POINTS = 15

# (X): Specify the CFF H values:
CFF_H = complex(-2.449, 3.482)

# (X): Specify the CFF H-tilde values:
CFF_H_TILDE = complex(1.409, 1.577)

# (X): Specify the CFF E values:
CFF_E = complex(2.217, 0.)

# (X): Specify the CFF E-tilde values:
CFF_E_TILDE = complex(144.410, 0.0)

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
# | [NOTE]: These CFF values come DIRECTLY from the KM15
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
    verbose = False,
    debugging = False)

# (X): `compute_cross_section` returns an array of cross-section values:
cross_section_values = example_1_cross_section.compute_cross_section(phi_array)

# (X): This will return True:
print(f"> The number of cross-sections should be the same as the number of phi points. Is it? {len(cross_section_values) == len(phi_array)}")

# (X): Cross-section values:
print(f"> Obtained cross-section values for {len(phi_array)} values of phi:\n{cross_section_values}")

# (X): `compute_bsa` computes the beam-spin asymmetry with length = len(phi_array):
bsa_values = example_1_cross_section.compute_bsa(phi_array)

# (X): This will return True:
print(f"> The number of BSA values should be the same as the number of phi points. Is it? {len(bsa_values) == len(phi_array)}")

# (X): BSA values:
print(f"> Obtained BSA values for {len(phi_array)} values of phi:\n{bsa_values}")
