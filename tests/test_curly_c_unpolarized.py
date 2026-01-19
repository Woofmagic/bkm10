"""
## Description:
Testing all of the absolutely incredibly headache coefficients that we call "curly C." These
show up in the computation of the DVCS and the Interference contributions to the cross-section.

## Notes:
1. 2026/01/19:
    - Numbers are close, but still 3/5 can't pass the default tolerance of 1e-7.
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
class TestCurlyCCoefficients(unittest.TestCase):
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
    STARTING_PHI_VALUE_IN_DEGREES = 0

    # (X): Specify a final value for azimuthal phi:
    ENDING_PHI_VALUE_IN_DEGREES = 360

    # (X): Specify *how many* values of phi you want to evaluate the cross-section
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
            start = cls.STARTING_PHI_VALUE_IN_DEGREES,
            stop = cls.ENDING_PHI_VALUE_IN_DEGREES,
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
        
    def test_curly_c_dvcs_no_eff_cff_no_eff_cff_conjugate(self):
        """
        ## Description:
        Test the function that corresponds to the BKM10 coefficient CurlyC_{DVCS}(F | F*).
        """
        curly_c_dvcs_cff_cff_star = self.bkm_formalism.calculate_curly_c_unpolarized_dvcs(
            effective_cffs = False,
            effective_conjugate_cffs = False
        )

        # (X): Verify that CurlyC_{DVCS}(F | F*) is a *finite* number:
        self.assert_is_finite(curly_c_dvcs_cff_cff_star)
        
        # (X); Verify that CurlyC_{DVCS}(F | F*) is not a NaN:
        self.assert_no_nans(curly_c_dvcs_cff_cff_star)

        # (X): Verify that CurlyC_{DVCS}(F | F*) is real:
        self.assert_is_real(curly_c_dvcs_cff_cff_star)

        _MATHEMATICA_RESULT = complex(13.478125253553266, 0.)

        self.assertAlmostEqual(curly_c_dvcs_cff_cff_star, second = _MATHEMATICA_RESULT)

    def test_curly_c_dvcs_eff_cff_eff_cff_conjugate(self):
        """
        ## Description:
        Test the function that corresponds to the BKM10 coefficient CurlyC_{DVCS}(Feff, Feff*).
        """
        curly_c_dvcs_eff_cff_eff_cff_star = self.bkm_formalism.calculate_curly_c_unpolarized_dvcs(
            effective_cffs = True,
            effective_conjugate_cffs = True
        )

        # (X): Verify that CurlyC(Feff, Feff*) is a *finite* number:
        self.assert_is_finite(curly_c_dvcs_eff_cff_eff_cff_star)
        
        # (X); Verify that CurlyC(Feff, Feff*) is not a NaN:
        self.assert_no_nans(curly_c_dvcs_eff_cff_eff_cff_star)

        # (X): Verify that CurlyC(Feff, Feff*) is real:
        self.assert_is_real(curly_c_dvcs_eff_cff_eff_cff_star)

        _MATHEMATICA_RESULT = complex(37.49784250218004, 0.)

        self.assertAlmostEqual(curly_c_dvcs_eff_cff_eff_cff_star, second = _MATHEMATICA_RESULT)

    def test_curly_c_dvcs_eff_cff_no_eff_cff_conjugate(self):
        """
        ## Description:
        Test the function that corresponds to the BKM10 coefficient CurlyC_{DVCS}(Feff, F*).
        """
        curly_c_dvcs_eff_cff_cff_star = self.bkm_formalism.calculate_curly_c_unpolarized_dvcs(
            effective_cffs = True,
            effective_conjugate_cffs = False
        )

        # (X): Verify that CurlyC(Feff, F*) is a *finite* number:
        self.assert_is_finite(curly_c_dvcs_eff_cff_cff_star)
        
        # (X); Verify that CurlyC(Feff, F*) is not a NaN:
        self.assert_no_nans(curly_c_dvcs_eff_cff_cff_star)

        # (X): Verify that CurlyC(Feff, F*) is real:
        self.assert_is_real(curly_c_dvcs_eff_cff_cff_star)

        _MATHEMATICA_RESULT = complex(22.481116920259893, 5.604782843278753e-16)

        self.assertAlmostEqual(curly_c_dvcs_eff_cff_cff_star, second = _MATHEMATICA_RESULT)

    def test_curly_c_interference_no_eff_cff_no_eff_cff_conjugate(self):
        """
        ## Description:
        Test the function that corresponds to the BKM10 coefficient CurlyC_{I}(F).
        """
        curly_c_interference_cff_cff_star = self.bkm_formalism.calculate_curly_c_unpolarized_interference(effective_cffs = False)

        # (X): Verify that CurlyC_{I}(F) is a *finite* number:
        self.assert_is_finite(curly_c_interference_cff_cff_star)
        
        # (X); Verify that CurlyC_{I}(F) is not a NaN:
        self.assert_no_nans(curly_c_interference_cff_cff_star)

        _MATHEMATICA_RESULT = complex(0.266711013189341, 2.1847473098840733)

        self.assertAlmostEqual(curly_c_interference_cff_cff_star, second = _MATHEMATICA_RESULT)

    def test_curly_c_interference_eff_cff(self):
        """
        ## Description:
        Test the function that corresponds to the BKM10 coefficient CurlyC_{I}(Feff).
        """
        curly_c_interference_eff_cff = self.bkm_formalism.calculate_curly_c_unpolarized_interference(effective_cffs = True)

        # (X): Verify that CurlyC_{I}(Feff) is a *finite* number:
        self.assert_is_finite(curly_c_interference_eff_cff)
        
        # (X); Verify that CurlyC_{I}(Feff) is not a NaN:
        self.assert_no_nans(curly_c_interference_eff_cff)

        _MATHEMATICA_RESULT = complex(0.44486613372656025, 3.6440943225229856)

        self.assertAlmostEqual(curly_c_interference_eff_cff, second = _MATHEMATICA_RESULT)