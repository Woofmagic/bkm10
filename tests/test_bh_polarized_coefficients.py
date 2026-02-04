"""
## Description:
Testing the BH coefficients for the LP-target case.

## Notes:

1. 2026/01/29:
    - Initalized testing hub.
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
class TestBHLPPolarizedCoefficients(unittest.TestCase):
    """
    ## Description:
    We need to verify that all of the coefficients that go into computation of the 
    BKM10 cross-section are correct. There are a LOT of them, so this test is important.

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
    STARTING_PHI_VALUE_IN_RADIANS = 0

    # (X): Specify a final value for azimuthal phi:
    ENDING_PHI_VALUE_IN_RADIANS = 2. * np.pi

    # (X): Specify *how many* values of phi you want to evaluate the cross-section
    # | at. [NOTE]: This determines the *length* of the array:
    NUMBER_OF_PHI_POINTS = 16

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

        # (X): Using the setting we wrote earlier, we now need to construct
        cls.configuration = {
            "kinematics": cls.test_kinematics,
            "cff_inputs": cls.test_cff_inputs,
            "using_ww": True
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
            lepton_polarization = 1.0,
            target_polarization = +0.5,
            using_ww = True)
    
    def assert_is_finite(self, value):
        """
        ## Description:
        A general test in the suite that verifies that all the
        numbers in an array are *finite* (as opposed to Inf.-type or NaN)

        ## Notes:
        "NaN" means "not a number." Having NaN values in an array causes problems in
        functions that are designed to perform arithmetic.
        """
        self.assertTrue(
            expr = np.isfinite(value).all(),
            msg = "Value contains NaNs or infinities/Inf.")

    def assert_no_nans(self, value):
        """
        ## Description:
        A general test in the suite that determines if an array has NaNs.
        
        ## Notes:
        "NaN" means "not a number." Having NaN values in an array causes problems in
        functions that are designed to perform arithmetic.
        """
        self.assertFalse(
            expr = np.isnan(value).any(),
            msg = "> [ERROR]: Value contains NaNs")

    def assert_no_negatives(self, value):
        """
        ## Description:
        A general test in the suite that determines if an array has negative values
        in it.

        ## Notes:
        There *are* important negative quantities, and several coefficients are indeed
        negative. But cross-sections, for example, should be positive.
        """
        self.assertTrue(
            expr = (value >= 0).all(),
            msg = "> [ERROR]: Value contains negative values")

    def assert_is_real(self, value):
        """
        ## Description:
        A general test in the suite that determines that an array has
        all real values.
        """
        self.assertTrue(
            expr = np.isreal(value).all(),
            msg = "> [ERROR]: Value contains complex components")
        
    def test_calculate_bh_c0_coefficient(self):
        """
        ## Description:
        Test the function that corresponds to the BKM10 coefficient c_{0}^{BH}.
        """
        c0bh = self.bkm_formalism.compute_bh_c0_coefficient()
        
        # (X): Verify that c_{0}^{BH} is a *finite* number:
        self.assert_is_finite(c0bh)
        
        # (X); Verify that c_{0}^{BH} is not a NaN:
        self.assert_no_nans(c0bh)

        # (X): Verify that c_{0}^{BH} is real:
        self.assert_is_real(c0bh)

        _MATHEMATICA_RESULT = 4.196441097163937 + 1.0209385078703184

        self.assertAlmostEqual(c0bh, second = _MATHEMATICA_RESULT)

    def test_calculate_bh_c1_coefficient(self):
        """
        ## Description:
        Test the function that corresponds to the BKM10 coefficient c_{1}^{BH}.
        """
        c1bh = self.bkm_formalism.compute_bh_c1_coefficient()

        # (X): Verify that c_{1}^{BH} is a *finite* number:
        self.assert_is_finite(c1bh)
        
        # (X); Verify that c_{1}^{BH} is not a NaN:
        self.assert_no_nans(c1bh)

        # (X): Verify that c_{1}^{BH} is real:
        self.assert_is_real(c1bh)

        # [NOTE]: We get this result by taking UNPOLARIZED part and adding
        # | to it the sigma(lambda = +1, Lambda = +0.5) contribution because
        # | that is now how the function is calculating this coefficient.
        _MATHEMATICA_RESULT = -1.0718559129262486 - 0.04922677820072388

        self.assertAlmostEqual(c1bh, second = _MATHEMATICA_RESULT)

    def test_calculate_bh_c2_coefficient(self):
        """
        ## Description:
        Test the function that corresponds to the BKM10 coefficient c_{2}^{BH}.
        """
        c2bh = self.bkm_formalism.compute_bh_c2_coefficient()

        # (X): Verify that c_{2}^{BH} is a *finite* number:
        self.assert_is_finite(c2bh)
        
        # (X); Verify that c_{2}^{BH} is not a NaN:
        self.assert_no_nans(c2bh)

        # (X): Verify that c_{2}^{BH} is real:
        self.assert_is_real(c2bh)

        # [NOTE]: We get this result by taking UNPOLARIZED part and adding
        # | to it the sigma(lambda = +1, Lambda = +0.5) contribution because
        # | that is now how the function is calculating this coefficient.
        _MATHEMATICA_RESULT = -0.03281299774352729 + 0.0

        self.assertAlmostEqual(c2bh, second = _MATHEMATICA_RESULT)

    def test_calculate_bh_s1_coefficient(self):
        """
        ## Description:
        Test the function that corresponds to the BKM10 coefficient s_{1}^{BH}.
        """
        s1bh = self.bkm_formalism.compute_bh_s1_coefficient()

        # (X): Verify that c_{2}^{BH} is a *finite* number:
        self.assert_is_finite(s1bh)
        
        # (X); Verify that c_{2}^{BH} is not a NaN:
        self.assert_no_nans(s1bh)

        # (X): Verify that c_{2}^{BH} is real:
        self.assert_is_real(s1bh)

        _MATHEMATICA_RESULT = 0.0 + 0.0

        self.assertAlmostEqual(s1bh, second = _MATHEMATICA_RESULT)