"""
We will now attempt to provide a reproducible piece of code
that demonstrates a major issue with the library. The main 
issue arises in that one cannot multiply TF tensors of 
different types, and there are 
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from bkm10_lib import backend
from bkm10_lib.core import DifferentialCrossSection
from bkm10_lib.inputs import BKM10Inputs
from bkm10_lib.cff_inputs import CFFInputs

# ====== Generate Fake Data ======
np.random.seed(0)
N = 256

# Kinematic inputs: QQ, x, t, k, phi
QQ = np.random.uniform(1.0, 4.0, N).astype(np.float32)
x  = np.random.uniform(0.1, 0.5, N).astype(np.float32)
t  = np.random.uniform(-1.0, -0.1, N).astype(np.float32)
k  = np.random.uniform(4.0, 6.0, N).astype(np.float32)
phi = np.random.uniform(0.0, 2*np.pi, N).astype(np.float32)

array_of_kinematics = np.stack([QQ, x, t, k, phi], axis = -1)

# CFFs: Re[H], Im[H], Re[E], Im[E], Re[H̃], Im[H̃], Re[Ē], Im[Ē]
array_of_cffs = np.random.uniform(-3, 3, size = (N, 8)).astype(np.float32)

# ====== Compute True Cross Sections ======
def compute_sigma_batch(kinematics, cffs):
    valid_kinematics = []
    valid_cffs = []
    sigmas = []
    for kin, cff in zip(kinematics, cffs):
        try:
            kin_inputs = BKM10Inputs(
                squared_Q_momentum_transfer = kin[0],
                x_Bjorken = kin[1],
                squared_hadronic_momentum_transfer_t = kin[2],
                lab_kinematics_k = kin[3])

            cff_inputs = CFFInputs(
                compton_form_factor_h       = complex(cff[0], cff[1]),
                compton_form_factor_e       = complex(cff[2], cff[3]),
                compton_form_factor_h_tilde = complex(cff[4], cff[5]),
                compton_form_factor_e_tilde = complex(cff[6], cff[7]))

            config = {
                "kinematics": kin_inputs,
                "cff_inputs": cff_inputs,
                "target_polarization": 0.0,
                "lepton_beam_polarization": 0.0,
                "using_ww": True
            }

            ds = DifferentialCrossSection(config)
            sigma = ds.compute_cross_section(kin[4])  # phi

            if np.isfinite(sigma):
                sigmas.append(sigma)
                valid_kinematics.append(kin)
                valid_cffs.append(cff)

        except Exception as e:
            print(f"[Warning] Skipped a sample due to error: {e}")

    return np.array(valid_kinematics, dtype=np.float32), np.array(valid_cffs, dtype=np.float32), np.array(sigmas, dtype=np.float32)

array_of_kinematics, array_of_cffs, array_of_cross_sections = compute_sigma_batch(array_of_kinematics, array_of_cffs)

backend.set_backend("tensorflow")

class BKM10Layer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target_polarization = 0.0
        self.lepton_beam_polarization = 0.0
        self.using_ww = True

    def call(self, inputs):
        
        # (X): Extract the input to the layer:
        kinematics, cffs = inputs

        # (X): Now, unstack each CFF input according to its order:
        real_H, imag_H, real_E, imag_E, real_Ht, imag_Ht, real_Et, imag_Et = tf.unstack(cffs, axis = -1)
    
        # (X): Unstack the kinematics similarly:
        q_squared, x_bjorken, hadron_t, beam_k, phi = tf.unstack(kinematics, axis = -1)

        # (X): Require the BKM10 lib use TF as the backend for math:
        backend.set_backend("tensorflow")

        kin_inputs = BKM10Inputs(
            squared_Q_momentum_transfer = q_squared,
            x_Bjorken = x_bjorken,
            squared_hadronic_momentum_transfer_t = hadron_t,
            lab_kinematics_k = beam_k)

        cff_inputs = CFFInputs(
            compton_form_factor_h = backend.math.complex(real_H, imag_H),
            compton_form_factor_e = backend.math.complex(real_E, imag_E),
            compton_form_factor_h_tilde = backend.math.complex(real_Ht, imag_Ht),
            compton_form_factor_e_tilde = backend.math.complex(real_Et, imag_Et))

        config = {
            "kinematics": kin_inputs,
            "cff_inputs": cff_inputs,
            "target_polarization": self.target_polarization,
            "lepton_beam_polarization": self.lepton_beam_polarization,
            "using_ww": self.using_ww,
        }

        sigma = DifferentialCrossSection(config).compute_cross_section(phi)

        return tf.expand_dims(sigma, axis = -1)
    
    def compute_output_shape(self, input_shape):
        """
        ## Description:
        Safety net for telling TF about the output shape
        of this custom layer.
        """
        return (input_shape[0][0], 1)

# (X): Define the input layer for the 5 kinematic settings:
kin_input = tf.keras.Input(shape = (5,), name = "kinematics")

# (X): Define the input layer for the 8 CFFs, real/imaginary:
cff_input = tf.keras.Input(shape = (8,), name = "cffs")

# (X): Define the network's inputs:
network_inputs = [kin_input, cff_input]

# (X): Concatenate the two Input layers together and pass them to a custom layer
# | that computes the BKM10 cross section:
output = BKM10Layer()(network_inputs)

# (X): Define a Keras model with given input and output layers:
model = tf.keras.Model(
    inputs = network_inputs,
    outputs = output)

# (X): Compile the model and specify optimizers and loss functions:
model.compile(optimizer = "adam", loss = "mse")

# (X): Train the model by passing in the lists of kineamtics and CFFs:
model.fit([array_of_kinematics, array_of_cffs], array_of_cross_sections, epochs = 1, batch_size = 32)

# (X): Predict the cross sections with the model:
preds = model.predict([array_of_kinematics, array_of_cffs])

# (X): Make a figure:
plt.figure(figsize = (6, 6))

# (X): Add a scatter plot:
plt.scatter(array_of_cross_sections, preds, alpha = 0.6)

# (X): Add some lines:
plt.plot(
    [array_of_cross_sections.min(), array_of_cross_sections.max()],
    [array_of_cross_sections.min(), array_of_cross_sections.max()],
    color = "red",
    linestyle = "--")

# (X): Add the x-label:
plt.xlabel("True Cross Section")

# (X): Add the y-label:
plt.ylabel("Predicted Cross Section")

# (X): Add the title:
plt.title("TF-BKM10 Cross Section Layer Test")

# (X): "Underlay" a grid:
plt.grid(True)

# (X): Add a tight layout:
plt.tight_layout()

# (X): Show it for the time being instead of saving:
plt.show()
