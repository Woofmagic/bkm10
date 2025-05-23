"""
"""

from bkm10_lib.core import DifferentialCrossSection
from bkm10_lib.inputs import BKM10Inputs
from bkm10_lib.cff_inputs import CFFInputs

import numpy as np

# (X): For all of these, we want phi to range from 0 to 360 degrees:
phi_array = np.linspace(0, 2 * np.pi, 360)

# Example (1):
# | We want to obtain a NumPy array for the value of the 
# | differential cross-section using standard plug-and-chug.
example_1_kinematic_inputs = BKM10Inputs(
    lab_kinematics_k = 1.0,
    squared_Q_momentum_transfer = 1.2,
    x_Bjorken = 0.3,
    squared_hadronic_momentum_transfer_t = -0.2)

example_1_cff_inputs = CFFInputs(
    compton_form_factor_h = 0,
    compton_form_factor_h_tilde = 0,
    compton_form_factor_e = 0,
    compton_form_factor_e_tilde = 0)

example_1_target_polarization = 0.
example_1_lepton_polarization = 0.

example_1_config_dictionary = {
    "kinematics": example_1_kinematic_inputs,
    "cff_inputs": example_1_cff_inputs,
    "target_polarization": example_1_target_polarization,
    "lepton_beam_polarization": example_1_lepton_polarization,
}

example_1_cross_section = DifferentialCrossSection(
    configuration = example_1_config_dictionary,
    verbose = True,
    debugging = True)

example_1_cross_section.compute_cross_section(phi_array)