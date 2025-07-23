"""
In this example, we want to demonstrate that one can use the
main functions that compute the relevant BKM10 coefficients
with TensorFlow through basic wrapping.
"""

# External Library | NumPy
import numpy as np

# External Library | TensorFlow
import tensorflow as tf

# Internal Package | DifferentialCrossSection
from bkm10_lib.core import DifferentialCrossSection

# Internal Package | BKM10Inputs
from bkm10_lib.inputs import BKM10Inputs

# Internal Package | BKMFormalism
from bkm10_lib.formalism import BKMFormalism

# Internal Package | CFFInputs
from bkm10_lib.cff_inputs import CFFInputs

from bkm10_lib import backend

# (X): For all of these, we want phi to range from 0 to 360 degrees:
phi_array = np.linspace(0, 2 * np.pi, 360)

# (X): Define the BKM10 inputs:
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

# (X): Use the BKM10 formalism to directly compute things:
bkm_formalism = BKMFormalism(
    inputs = example_1_kinematic_inputs,
    cff_values = example_1_cff_inputs,
    lepton_polarization = 1.0,
    target_polarization = 0.0,
    using_ww = True
)

# (X): Compute the first three unpolarized coefficients with Numpy:
c0pp = bkm_formalism.calculate_c_0_plus_plus_unpolarized()
c0ppv = bkm_formalism.calculate_c_0_plus_plus_unpolarized_v()
c0ppa = bkm_formalism.calculate_c_0_plus_plus_unpolarized_a()

# (X): Tell the backend now to compute with TF:
backend.set_backend("tensorflow")

# (X): Compute the first three unpolarized coefficients with TF:
c0pp_tf = bkm_formalism.calculate_c_0_plus_plus_unpolarized()
c0ppv_tf  = bkm_formalism.calculate_c_0_plus_plus_unpolarized_v()
c0ppa_tf  = bkm_formalism.calculate_c_0_plus_plus_unpolarized_a()

# (X): Require a one-liner that converts TF tensors into NumPy arrays:
if tf.is_tensor(c0pp_tf): c0pp_tf = c0pp_tf.numpy()
if tf.is_tensor(c0ppv_tf): c0ppv_tf = c0ppv_tf.numpy()
if tf.is_tensor(c0ppa_tf): c0ppa_tf = c0ppa_tf.numpy()

# (X): We now have to compare it:
print("C(n = 0)++:", np.allclose(c0pp, c0pp_tf, rtol = 1e-5, atol = 1e-7))
print("C^V(N = 0)++:", np.allclose(c0ppv, c0ppv_tf, rtol = 1e-5, atol = 1e-7))
print("C^V(N = 0)++:", np.allclose(c0ppa, c0ppa_tf, rtol = 1e-5, atol = 1e-7))
