"""

"""

import numpy as np

class BKMFormalism:

    def __init__(self, formalism_version: str = "10", inputs: KinematicInputs):

        self.kinematics = inputs

    def c_0_zero_plus_unpolarized(self, phi: np.ndarray) -> np.ndarray:

        return 0.