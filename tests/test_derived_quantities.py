"""
## Description:
Testing various constants/special numbers in the program.

## Notes:
1. 2026/01/15:
    - Prefactor test passed.
2. 2026/01/19:
    - All derived quantities added and passed tests.
3. 2026/01/27:
    - Added KDD and propagator tests, and they passed tests finally.
    Crucially, these tests need to use the *Trento convention* for 
    the phi values.
4. 2026/02/03:
    - Testing the propagators requires that we iterate over phi for a
    certain number of points; we chose 15. That is 0 to 2pi divided uniformly,
    and it gives the points 0 (inclusive), 2pi/15, 2 * 2pi/15, ... So, just
    so the programmer is aware where those numbers come from.
    - All derived quantities pass!
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
        # | cross-section at. [NOTE]: TRENTO CONVENTION!
        cls.phi_values = np.pi - np.linspace(
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

    def assert_approximately_equal(self, value, expected, tolerance = 1e-8):
        """
        ## Description:
        A general test in the suite that determines if a *number* (`value`) is approximately
        equal to what is expected (`expected`). "Approximately equal" is quantified with the 
        parameter `tolerance`.
        """
        self.assertTrue(
            np.allclose(value, expected, rtol = tolerance, atol = tolerance),
            f"> [ERROR]: Expected {expected}, got {value}")
        
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
        
    def test_k_dot_delta(self):
        """
        ## Description:
        Test the propagators that comes as prefactor-ish things in the BH and Interference
        amplitudes.
        """
        kdd = self.bkm_formalism.calculate_k_dot_delta(phi_values = self.phi_values)

        # (X): Verify that KDD is a *finite* number:
        self.assert_is_finite(kdd)
        
        # (X); Verify that KDD is not a NaN:
        self.assert_no_nans(kdd)

        # (X): Verify that KDD is real:
        self.assert_is_real(kdd)

        _MATHEMATICA_LIST_VALUES = [
                -1.403777900600661, -1.4257906946445564, -1.488022864706097, -1.5797139032201262, -1.6850095966406338,
                -1.7857033629938701, -1.864384335303211, -1.9074478586621781, -1.9074478586621781, -1.864384335303211,
                -1.7857033629938701, -1.6850095966406338, -1.5797139032201262, -1.488022864706097, -1.4257906946445564, -1.403777900600661
            ]
            
        # (X): Check to see if the list lengths are equal --- just an easy thing first:
        self.assertEqual(
            len(kdd),
            len(_MATHEMATICA_LIST_VALUES),
            "[ASSERT]: List lengths are not equal.")
        
        # (X): Perform the test:
        for index, (library_kdd_values, mathematica_kdd_value) in enumerate(zip(kdd, _MATHEMATICA_LIST_VALUES)):

            # (X.1): Pairwise assert almost equal.
            self.assertAlmostEqual(
                library_kdd_values,
                mathematica_kdd_value,
                places = 7,
                msg = f"\nLists differ at index {index}: {library_kdd_values} != {mathematica_kdd_value}")
            
    def test_propagators(self):
        """
        ## Description:
        Test the propagators that comes as prefactor-ish things in the BH and Interference
        amplitudes.
        """
        library_propagator_p1 = self.bkm_formalism.calculate_lepton_propagator_p1(phi_values = self.phi_values)

        library_propagator_p2 = self.bkm_formalism.calculate_lepton_propagator_p2(phi_values = self.phi_values)
        
        propagator_product = library_propagator_p1 * library_propagator_p2

        # (X): Verify that the propagator is a *finite* number:
        self.assert_is_finite(propagator_product)
        
        # (X); Verify tthat the propagator is not a NaN:
        self.assert_no_nans(propagator_product)

        # (X): Verify that the propagator is real:
        self.assert_is_real(propagator_product)

        _MATHEMATICA_LIST_VALUES = [
                -0.7863583904324856, -0.8351254241802596, -0.9793253176012227,
                -1.2088282602619738, -1.49743121858906, -1.798468366657066,
                -2.050738480977298, -2.195141545883421, -2.195141545883421,
                -2.050738480977298, -1.798468366657066, -1.49743121858906,
                -1.2088282602619738, -0.9793253176012227, -0.8351254241802596,
                -0.7863583904324856
            ]
            
        # (X): Check to see if the list lengths are equal --- just an easy thing first:
        self.assertEqual(
            len(propagator_product),
            len(_MATHEMATICA_LIST_VALUES),
            "[ASSERT]: List lengths are not equal.")
        
        # (X): Perform the test:
        for index, (library_propagator_values, mathematica_propagator_value) in enumerate(zip(propagator_product, _MATHEMATICA_LIST_VALUES)):

            # (X.1): Pairwise assert almost equal.
            self.assertAlmostEqual(
                library_propagator_values,
                mathematica_propagator_value,
                places = 7,
                msg = f"\nLists differ at index {index}: {library_propagator_values} != {mathematica_propagator_value}")