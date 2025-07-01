"""
Here is another example of how one might use this library
for analysis. Physicists are mostly interested in: cross sections,
various cross-section asymmetries, CFFs, and GPDs.

Example (2): We run basically the same code as before, except we try 
to verify that computation with NumPy and TensorFlow (by changing the
backend assignment) is the same.
"""

import numpy as np
import tensorflow as tf

from bkm10_lib import backend
from bkm10_lib.core import DifferentialCrossSection
from bkm10_lib.inputs import BKM10Inputs
from bkm10_lib.cff_inputs import CFFInputs

from bkm10_lib.backend import math

def run_cross_section(azimuthal_angles: np.ndarray):
    """
    ## Description:
    Just a function that does the same computation of the cross
    section and the BSA. It is set up to run according to the
    NumPy math computation methods and then the TF math computation
    methods. The results should be the same!
    """

    example_inputs = BKM10Inputs(
        lab_kinematics_k = 5.75,
        squared_Q_momentum_transfer = 1.82,
        x_Bjorken = 0.34,
        squared_hadronic_momentum_transfer_t = -0.17)

    example_cffs = CFFInputs(
        compton_form_factor_h = math.complex(-0.897, 2.421),
        compton_form_factor_h_tilde = math.complex(2.444, 1.131),
        compton_form_factor_e = math.complex(-0.541, 0.903),
        compton_form_factor_e_tilde = math.complex(2.207, 5.383))

    configuration = {
        "kinematics": example_inputs,
        "cff_inputs": example_cffs,
        "target_polarization": 0.0,
        "lepton_beam_polarization": 0.0,
        "using_ww": True
    }

    cross_section = DifferentialCrossSection(
        configuration = configuration,
        verbose = False,
        debugging = False)
    
    sigma = cross_section.compute_cross_section(azimuthal_angles)
    bsa = cross_section.compute_bsa(azimuthal_angles)
    return sigma, bsa


# (X): For all of these, we want phi to range from 0 to 360 degrees:
phi_array = np.linspace(0, 2 * np.pi, 360).astype(np.float32)

# (X): Set the backend to NumPy:
backend.set_backend("numpy")

# (X): Compute the cross-section and BSA:
sigma_np, bsa_np = run_cross_section(phi_array)

# (X): We need to convert the NumPy array to a TF tensor... UGH!
phi_array_tf = tf.convert_to_tensor(phi_array)

# (X): Set the backend tO TensorFlow:
backend.set_backend("tensorflow")

# (X): Compute the cross-section and BSA:
sigma_tf, bsa_tf = run_cross_section(phi_array_tf)

# (X): Require a one-liner that converts TF tensors into NumPy arrays:
if tf.is_tensor(sigma_tf): sigma_tf = sigma_tf.numpy()
if tf.is_tensor(bsa_tf): bsa_tf = bsa_tf.numpy()

# (X): We now have to compare it:
print("Cross-section match:", np.allclose(sigma_np, sigma_tf, rtol = 1e-5, atol = 1e-7))
print("BSA match:", np.allclose(bsa_np, bsa_tf, rtol = 1e-5, atol = 1e-7))
