import numpy as np

class DifferentialCrossSection:

    def __init__(self, inputs: KinematicInputs):

        self.inputs = inputs

        self.derived_quantities = BKMFormalism(inputs)

    def compute_cross_section(self, phi_values: np.ndarray) -> np.ndarray:
        """
        ## Description:
        We compute the four-fold *differential cross-section* as 
        described with the BKM10 Formalism.

        ## Arguments:
        
        phi: np.ndarray
            A NumPy array that will be plugged-and-chugged into the BKM10 formalism.
        """

        # (1): Verify that the array of angles is at least 1D:
        verified_phi_values = np.atleast_1d(phi_values)

        # (X): Obtain coefficients:
        coefficient_c_0 = 0.
        coefficient_c_1 = 0.

        # (X): Compute the dfferential cross-section:

        differential_cross_section = coefficient_c_0 + coefficient_c_1 * np.cos(verified_phi_values)