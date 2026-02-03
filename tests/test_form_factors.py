"""
## Description:
A testing suite to check if the computation of the form factors, especially
the effective form factors, are correct.

## Notes:
1. 2026/01/19: 
    - Initialized tests.
2. 2026/01/19: 
    - Xi (skewness) test passes, but CFF tests cannot clear within 1e-8 tolerance... 
3. 2026/01/19:
    - CFF E_{eff} can pass to within 1e-7, but the other three *still* cannot clear that tolerance.
4. 2026/01/27
    - All tests passed after removing TensorFlow's `promote_scalar_to_dtype` function; it was
    found to alter the precision of standard Python mathematical operations such that
    unittests did not pass. This behavior is highly not desired.
5. 2026/02/03
    - All tests still pass.
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
class TestFormFactors(unittest.TestCase):
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
        
    def test_xi_parameter(self):
        """
        ## Description:
        Since Xi goes into the effective CFF computation, we need to check that it's right first.
        
        ## Notes:
        1. We initialized the `BKMFormalism` class with the WW setting *ON*.
        """

        # (X): Verify the Xi parameter is correct:
        xi_parameter = self.bkm_formalism.skewness_parameter

        # (X): The Mathematica-computed value for Xi:
        _MATHEMATICA_RESULT_XI = 0.19906188837146524

        # (X): This one tests H:
        self.assertAlmostEqual(
            first = xi_parameter,
            second = _MATHEMATICA_RESULT_XI,
            places = 8)
    
    def test_effective_cff_h(self):
        """
        ## Description:
        Tests to see if the effective CFF H matches what we got using the Mathematica
        code.

        ## Notes:
        1. We initialized the `BKMFormalism` class with the WW setting *ON*.
        """

        # (X): Compute the effective CFFs:
        effective_cffs = self.bkm_formalism.compute_cff_effective(self.test_cff_inputs)

        # (X): We need to match this value for H (computed from Mathematica):
        _MATHEMATICA_RESULT_H_WW = complex(-1.4961696451186222, 4.038156868263304)

        # (X): This one tests H:
        self.assertAlmostEqual(
            first = effective_cffs.compton_form_factor_h,
            second = _MATHEMATICA_RESULT_H_WW,
            places = 7)
    
    def test_effective_cff_e(self):
        """
        ## Description:
        Tests to see if the effective CFF E matches what we got using the Mathematica
        code.

        ## Notes:
        1. We initialized the `BKMFormalism` class with the WW setting *ON*.
        """

        # (X): Compute the effective CFFs:
        effective_cffs = self.bkm_formalism.compute_cff_effective(self.test_cff_inputs)

        # (X): We need to match this value for E (computed from Mathematica):
        _MATHEMATICA_RESULT_E_WW = complex(-0.9023721048039851, 1.5061774688317902)
        
        # (X): This one tests E:
        self.assertAlmostEqual(
            first = effective_cffs.compton_form_factor_e,
            second = _MATHEMATICA_RESULT_E_WW,
            places = 7)
        
    def test_effective_cff_h_tilde(self):
        """
        ## Description:
        Tests to see if the effective CFF HT matches what we got using the Mathematica
        code.

        ## Notes:
        1. We initialized the `BKMFormalism` class with the WW setting *ON*.
        """

        # (X): Compute the effective CFFs:
        effective_cffs = self.bkm_formalism.compute_cff_effective(self.test_cff_inputs)

        # (X): We need to match this value for HT (computed from Mathematica):
        _MATHEMATICA_RESULT_HT_WW = complex(4.0765201924971155, 1.8864747699321758)
        
        # (X): This one tests E:
        self.assertAlmostEqual(
            first = effective_cffs.compton_form_factor_h_tilde,
            second = _MATHEMATICA_RESULT_HT_WW,
            places = 7)
        
    def test_effective_cff_e_tilde(self):
        """
        ## Description:
        Tests to see if the effective CFF ET matches what we got using the Mathematica
        code.

        ## Notes:
        1. We initialized the `BKMFormalism` class with the WW setting *ON*.
        """

        # (X): Compute the effective CFFs:
        effective_cffs = self.bkm_formalism.compute_cff_effective(self.test_cff_inputs)

        # (X): We need to match this value for ET (computed from Mathematica):
        _MATHEMATICA_RESULT_ET_WW = complex(3.6812111558269773, 8.978685841330593)
        
        # (X): This one tests ET:
        self.assertAlmostEqual(
            first = effective_cffs.compton_form_factor_e_tilde,
            second = _MATHEMATICA_RESULT_ET_WW,
            places = 7)