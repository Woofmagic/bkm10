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

# (X): For all of these, we want phi to range from 0 to 360 degrees:
phi_array = np.linspace(
    start = 0,
    stop = 360,
    num = 15)

# Example (1):
# | We want to obtain a NumPy array for the value of the
# | differential cross-section using standard plug-and-chug.
example_1_kinematic_inputs = BKM10Inputs(
    lab_kinematics_k = 5.5,
    squared_Q_momentum_transfer = 1.00,
    x_Bjorken = 0.200,
    squared_hadronic_momentum_transfer_t = -.1)

# (X): Make sure to specify the CFF inputs:
# | Note: these CFF values come DIRECTLY from the KM15
# | CFF/GPD models. Two of them are ZERO, and that is how
# | you should know that they come from KM15.
example_1_cff_inputs = CFFInputs(
    compton_form_factor_h = complex(-1.9844358866394172,0.7496811451932895),
    compton_form_factor_h_tilde = complex(-0.07196088868244656,0.5885369653510624),
    compton_form_factor_e = complex(0.9693397301223567,0.0),
    compton_form_factor_e_tilde = complex(44.771526441606106,0.0))
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

# (X): `compute_cross_section` returns an array of cross-section values:
cross_section_values = example_1_cross_section.compute_cross_section(phi_array)

# (X): This will return True:
print(len(cross_section_values) == len(phi_array))

# (X): Cross-section values:
print(f"> Obtained cross-section values for {len(phi_array)} values of phi:\n{cross_section_values}")

# (X): `compute_bsa` computes the beam-spin asymmetry with length = len(phi_array):
bsa_values = example_1_cross_section.compute_bsa(phi_array)

# (X): This will return True:
print(len(bsa_values) == len(phi_array))

# (X): BSA values:
print(f"> Obtained BSA values for {len(phi_array)} values of phi:\n{bsa_values}")
