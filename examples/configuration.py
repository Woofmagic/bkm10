"""
Here are a few exapmples of how one might use this library
for analysis. Physicists are mostly interested in: cross sections,
vaarious cross-section asymmetries, CFFs, and GPDs.

Example (1): We plot the BKM10 predictions for the four-fold differential
cross-section at the "standard kinematic bin" using the WW relations for
an unpolarized target and unpolarized beam.
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
phi_array = np.linspace(0, 2 * np.pi, 360)

# Example (1):
# | We want to obtain a NumPy array for the value of the 
# | differential cross-section using standard plug-and-chug.
example_1_kinematic_inputs = BKM10Inputs(
    lab_kinematics_k = 5.75,
    squared_Q_momentum_transfer = 1.82,
    x_Bjorken = 0.34,
    squared_hadronic_momentum_transfer_t = -0.17)

# (X): Make sure to specify the CFF inputs:
example_1_cff_inputs = CFFInputs(
    compton_form_factor_h = complex(-0.897, 2.421),
    compton_form_factor_h_tilde = complex(2.444, 1.131),
    compton_form_factor_e = complex(-0.541, 0.903),
    compton_form_factor_e_tilde = complex(2.207, 5.383))

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
    debugging = True)

# (X): `compute_cross_section` returns an array of cross-section values:
cross_section_values = example_1_cross_section.compute_cross_section(phi_array)

# (X): This will return True:
print(len(cross_section_values) == len(phi_array))

# (X): `plot_cross_section` will *for the time being* just plt.show() the plot:
example_1_cross_section.plot_cross_section(phi_array)

# (X): `compute_bsa` computes the beam-spin asymmetry with length = len(phi_array):
bsa_values = example_1_cross_section.compute_bsa(phi_array)

# (X): This will return True:
print(len(bsa_values) == len(phi_array))

# (X): `plot_bsa` will also *for the time being* just plt.show() the plot:
example_1_cross_section.plot_bsa(phi_array)
