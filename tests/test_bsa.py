"""
## Description:
Testing the BSA values.

## Notes:
1. 2026/01/19
    - BSA test cannot pass due to early mismatch at index 1 in truth list.
"""

# (X): Native Library | unittest:
import unittest

# (X): External Library | NumPy:
import numpy as np

# (X): Self-Import | BKM10Inputs:
from bkm10_lib.inputs import BKM10Inputs

# (X): Self-Import | CFFInputs:
from bkm10_lib.cff_inputs import CFFInputs

# (X): Self-Import | DifferentialCrossSection
from bkm10_lib.core import DifferentialCrossSection

# (X): Self-Import | BKMFormalism
from bkm10_lib.formalism import BKMFormalism


# (X): Define a class that inherits unittest's TestCase:
class TestBSA(unittest.TestCase):
    """
    ## Description:
    We need to verify that all of the coefficients that go into computation of the 
    BKM10 BSA are correct. There are a LOT of them, so this test is important.

    ## Detailed Description:
    Later!
    """

    # (X): Specify a value for k (the beam energy):
    TEST_LAB_K = 5.75

    # (X): Specify a Q^{2} value:
    TEST_Q_SQUARED = 1.82

    # (X): Specify an x_{B} value:
    TEST_X_BJORKEN = 0.34

    # (X): Specify a t value.
    # | [NOTE]: This number is usually negative:
    TEST_T_VALUE = -.17

    # (X): Specify the CFF H values:
    CFF_H = complex(-0.897, 2.421)

    # (X): Specify the CFF H-tilde values:
    CFF_H_TILDE = complex(2.444, 1.131)

    # (X): Specify the CFF E values:
    CFF_E = complex(-0.541, 0.903)

    # (X): Specify the CFF E-tilde values:
    CFF_E_TILDE = complex(2.207, 5.383)

    # (X): Specify a starting value for azimuthal phi:
    STARTING_PHI_VALUE_IN_RADIANS = 0.0 * np.pi

    # (X): Specify a final value for azimuthal phi:
    ENDING_PHI_VALUE_IN_RADIANS = 2.0 * np.pi

    # (X): Specify *how many* values of phi you want to evaluate the BSA
    # | at. [NOTE]: This determines the *length* of the array:
    NUMBER_OF_PHI_POINTS = 15

    @classmethod
    def setUpClass(cls):

        # (X): Provide the BKM10 inputs to the dataclass:
        cls.test_kinematics = BKM10Inputs(
            lab_kinematics_k = cls.TEST_LAB_K,
            squared_Q_momentum_transfer = cls.TEST_Q_SQUARED,
            x_Bjorken = cls.TEST_X_BJORKEN,
            squared_hadronic_momentum_transfer_t = cls.TEST_T_VALUE)

        # (X): Provide the CFF inputs to the dataclass:
        cls.test_cff_inputs = CFFInputs(
            compton_form_factor_h = cls.CFF_H,
            compton_form_factor_h_tilde = cls.CFF_H_TILDE,
            compton_form_factor_e = cls.CFF_E,
            compton_form_factor_e_tilde = cls.CFF_E_TILDE)
        
        # (X): Specify the target polarization *as a float*:
        cls.target_polarization = 0.0

        # (X): Specify the beam polarization *as a float*:
        cls.lepton_polarization = 0.0

        # (X): We are using the WW relations in this computation:
        cls.ww_setting = True

        # (X): Using the setting we wrote earlier, we now need to construct
        cls.configuration = {
            "kinematics": cls.test_kinematics,
            "cff_inputs": cls.test_cff_inputs,
            "target_polarization": cls.target_polarization,
            "lepton_beam_polarization": cls.lepton_polarization,
            "using_ww": cls.ww_setting
        }
        
        # (X): *Initialize* the cross-section class.
        # | [NOTE]: This does NOT compute the cross-section automatically.
        cls.cross_section = DifferentialCrossSection(
            configuration = cls.configuration)

        # (X): Initialize an array of phi-values in preparation to evaluate the
        # | cross-section at.
        cls.phi_values = np.linspace(
            start = cls.STARTING_PHI_VALUE_IN_RADIANS,
            stop = cls.ENDING_PHI_VALUE_IN_RADIANS,
            num = cls.NUMBER_OF_PHI_POINTS)
        
        # (X): Initialize a `BKMFormalism` class. This enables us to
        # | fully disentangle each of the coefficients.
        cls.bkm_formalism = BKMFormalism(
            inputs = cls.test_kinematics,
            cff_values = cls.test_cff_inputs,
            
            # (X): [NOTE]: All the S-coeffcicients are sensitive to lambda, so
            # | they will be 0 if you do not make this value 1.0.
            lepton_polarization = cls.lepton_polarization,
            target_polarization = cls.target_polarization,
            using_ww = True)
        
    def test_unpolarized_bsa(self):
        """
        ## Description:
        Test the function that computes the beam-spin asymmetry (BSA) observable.
        """

        # (X): Compute the BSA values:
        bsa_library_list = self.cross_section.compute_bsa(phi_values = self.phi_values)

        # (X): We selected 15 phi points within 0 to 2pi (equally-spaced) and evaluated
        # | our Mathematica code at each point to produce a BSA value. That's where this
        # | list comes from.
        _MATHEMATICA_LIST_VALUES = [
                0.0, 0.057749666884978575, 0.1168149401295814, 0.17308539107193663, 0.21323927136841755,
                0.2163168132162337, 0.16491759814369988, 0.06208575102633405, -0.06208575102633405, -0.16491759814369988,
                -0.2163168132162337, -0.21323927136841755, -0.17308539107193663, -0.1168149401295814, -0.057749666884978575
            ]
            
        # (X): Check to see if the list lengths are equal --- just an easy thing first:
        self.assertEqual(
            len(bsa_library_list),
            len(_MATHEMATICA_LIST_VALUES),
            "[ASSERT]: List lengths are not equal.")
        
        # (X): Perform the test:
        for index, (library_bsa_value, mathematica_bsa_value) in enumerate(zip(bsa_library_list, _MATHEMATICA_LIST_VALUES)):

            # (X.1): Pairwise assert almost equal.
            self.assertAlmostEqual(
                library_bsa_value,
                mathematica_bsa_value,
                places = 5,
                msg = f"\nLists differ at index {index}: {library_bsa_value} != {mathematica_bsa_value}")
