"""
## Description:
Testing various constants/special numbers in the program.

## Notes:
1. 2026/01/15:
    - Prefactor test passed.
2. 2026/01/19:
    - All derived quantities added and passed tests.
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
class TestDerivedQuantities(unittest.TestCase):
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
    ENDING_PHI_VALUE_IN_RADIANS = 360

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
        
    def test_kinematics_epsilon(self):
        """
        ## Description:
        Test the calculation of epsilon.

        ## Notes:
        """
        # (X): Extract the value for epsilon:
        epsilon = self.bkm_formalism.epsilon

        # (X): Type out the Mathematica result:
        _MATHEMATICA_RESULT = 0.47293561004973345

        # (X): Do the test:
        self.assertAlmostEqual(
            first = epsilon,
            second = _MATHEMATICA_RESULT)
        
    def test_lepton_energy_fraction(self):
        """
        ## Description:
        Test the calculation of y, lepton energy fraction.

        ## Notes:
        """

        # (X): Compute the y value:
        y = self.bkm_formalism.lepton_energy_fraction

        # (X): Type out the Mathematica result:
        _MATHEMATICA_RESULT = 0.49609612355928445

        # (X): Do the test:
        self.assertAlmostEqual(
            first = y,
            second = _MATHEMATICA_RESULT)
    
    def test_skewness(self):
        """
        ## Description:
        Test the calculation of xi, the skewness.

        ## Notes:
        """

        # (X): Extract the Xi value:
        xi = self.bkm_formalism.skewness_parameter

        # (X): Type out the Mathematica result:
        _MATHEMATICA_RESULT = 0.19906188837146524
        
        # (X): Do the test:
        self.assertAlmostEqual(
            first = xi,
            second = _MATHEMATICA_RESULT)
        
    def test_t_minimum(self):
        """
        ## Description:
        Test the calculation of t_min.

        ## Notes:
        1. 2026/01/19:
            - Units are technically GeV^{2}.
        """

        # (X): Extract the Xi value:
        t_min = self.bkm_formalism.t_minimum

        # (X): Type out the Mathematica result:
        _MATHEMATICA_RESULT = -0.13551824472915253
        
        # (X): Do the test:
        self.assertAlmostEqual(
            first = t_min,
            second = _MATHEMATICA_RESULT)
        
    def test_t_prime(self):
        """
        ## Description:
        Test the calculation of t'.

        ## Notes:
        1. 2026/01/19:
            - Units are technically GeV^{2}.
        """

        # (X): Extract the value for t':
        t_prime = self.bkm_formalism.t_prime

        # (X): Type out the Mathematica result:
        _MATHEMATICA_RESULT = -0.034481755270847486
        
        # (X): Do the test:
        self.assertAlmostEqual(
            first = t_prime,
            second = _MATHEMATICA_RESULT)
        
    def test_k_tilde(self):
        """
        ## Description:
        Test the calculation of Ktilde.

        ## Notes:
        1. 2026/01/19:
            - Units are technically GeV^{1}.
        """

        # (X): Extract the value for Ktilde:
        k_tilde = self.bkm_formalism.k_tilde

        # (X): Type out the Mathematica result:
        _MATHEMATICA_RESULT = 0.1592415651944438
        
        # (X): Do the test:
        self.assertAlmostEqual(
            first = k_tilde,
            second = _MATHEMATICA_RESULT)
        
    def test_k_shorthand(self):
        """
        ## Description:
        Test the calculation of K (capital K, not lowercase k).

        ## Notes:
        """

        # (X): Extract the value for Ktilde:
        kinematic_k = self.bkm_formalism.kinematic_k

        # (X): Type out the Mathematica result:
        _MATHEMATICA_RESULT = 0.08492693191323883
        
        # (X): Do the test:
        self.assertAlmostEqual(
            first = kinematic_k,
            second = _MATHEMATICA_RESULT)

    def test_bkm10_prefactor(self):
        """
        ## Description: Test the function computing BKM10 prefactor.
        """
        # (X): Compute the BSA values:
        prefactor = self.cross_section.compute_prefactor()

        # (X): Type out the Mathematica result:
        _MATHEMATICA_RESULT = 3.5309544777485675e-10

        # (X): Do the test:
        self.assertAlmostEqual(
            first = prefactor,
            second = _MATHEMATICA_RESULT)