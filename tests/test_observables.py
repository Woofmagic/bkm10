"""
## Description:
Here, we test the numerical results of the Python library by comparing
them to numerical results we obtained using the Mathematica functions. Remember what
the acronyms stand for:
i): BSA: beam-spin asymmetry,
ii): TSA: target-spin asymmetry,
iii): DSA: double-spin asymmetry

## Notes:
1. 2026/01/19:
    - Cross-section test cannot pass due to early mismatch at index 1 in truth list.
2. 2026/01/19
    - BSA test cannot pass due to early mismatch at index 1 in truth list.
3. 2026/01/28:
    - All unpolarized-target cross-sections officially passed.
4. 2026/02/03:
    -  All configurations of DSA (there's only one), TSA, and BSA successfully passed.
    The last tests are just individual cross section configurations, but since all of the
    other observables are computed in terms of them, it would be strange to see violations
    at the level of individual cross sections. Anyway, we will check them soon.
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


# (X): Define a class that inherits unittest's TestCase:
class TestCrossSections(unittest.TestCase):
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
        
        # (X): Initialize unpolarized cross-section class:
        cls.unpolarized_cross_section = DifferentialCrossSection(
            configuration = {
                "kinematics": cls.test_kinematics,
                "cff_inputs": cls.test_cff_inputs,
                "target_polarization": 0.0, # unpolarized target
                "lepton_beam_polarization": 0.0, # unpolarized beam
                "using_ww": True # using WW
            })
        
        # (X): Initalize [plus-beam unpolarized target] cross-section class:
        cls.plus_beam_unp_target = DifferentialCrossSection(
            configuration = {
                "kinematics": cls.test_kinematics,
                "cff_inputs": cls.test_cff_inputs,
                "target_polarization": 0.0, # unpolarized target
                "lepton_beam_polarization": +1.0, # unpolarized beam
                "using_ww": True # using WW
            })
        
        # (X): Initalize [minus-beam unpolarized target] cross-section class:
        cls.minus_beam_unp_target = DifferentialCrossSection(
            configuration = {
                "kinematics": cls.test_kinematics,
                "cff_inputs": cls.test_cff_inputs,
                "target_polarization": 0.0, # unpolarized target
                "lepton_beam_polarization": -1.0, # unpolarized beam
                "using_ww": True # using WW
            })

        # (X): Initialize an array of phi-values in preparation to evaluate the
        # | cross-section at.
        cls.phi_values = np.linspace(
            start = cls.STARTING_PHI_VALUE_IN_RADIANS,
            stop = cls.ENDING_PHI_VALUE_IN_RADIANS,
            num = cls.NUMBER_OF_PHI_POINTS)
        
    def test_unpolarized_cross_section(self):
        """
        ## Description:
        Test the unpolarized cross-section numerics.
        """

        # (X): Compute the cross-section values:
        cross_section_library_list = self.unpolarized_cross_section.compute_cross_section(
            phi_values = self.phi_values,
            lepton_helicity = 0.0,
            target_polarization = 0.0).real

        # (X): We selected 15 phi points within 0 to 2pi (equally-spaced) and evaluated
        # | our Mathematica code at each point to produce a cross-section value. That's where this
        # | list comes from.
        _MATHEMATICA_LIST_VALUES = [
                0.12847265889847323, 0.1192964714306159, 0.09768651630810463, 0.07433165867901309, 0.05563539461111375,
                0.042978388336577404, 0.03551580060922911, 0.0321170027195132, 0.0321170027195132, 0.03551580060922911,
                0.042978388336577404, 0.05563539461111375, 0.07433165867901309, 0.09768651630810463, 0.1192964714306159, 0.12847265889847323
            ]
            
        # (X): Check to see if the list lengths are equal --- just an easy thing first:
        self.assertEqual(
            len(cross_section_library_list),
            len(_MATHEMATICA_LIST_VALUES),
            "[ASSERT]: List lengths are not equal.")
        
        # (X): Perform the test:
        for index, (library_xsec_value, mathematica_xsec_value) in enumerate(zip(cross_section_library_list, _MATHEMATICA_LIST_VALUES)):

            # (X.1): Pairwise assert almost equal.
            self.assertAlmostEqual(
                library_xsec_value,
                mathematica_xsec_value,
                places = 7,
                msg = f"\nLists differ at index {index}: {library_xsec_value} != {mathematica_xsec_value}")
            
    def test_plus_beam_cross_section(self):
        """
        ## Description:
        Test the "plus-beam" cross-section (lambda = 1) numerics.
        """

        # (X): Compute the cross-section values:
        cross_section_library_list = self.plus_beam_unp_target.compute_cross_section(
            phi_values = self.phi_values,
            lepton_helicity = +1.0,
            target_polarization = 0.0).real

        # (X): We selected 15 phi points within 0 to 2pi (equally-spaced) and evaluated
        # | our Mathematica code at each point to produce a cross-section value. That's where this
        # | list comes from.
        _MATHEMATICA_LIST_VALUES = [
                0.12847265889847323, 0.12618580291628734, 0.10909776086210324, 0.08719738289049578, 0.06749904562028203,
                0.052275336338715575, 0.041372981141853726, 0.03411101095406899, 0.030122994484957408, 0.02965862007660449,
                0.03368144033443923, 0.04377174360194547, 0.06146593446753039, 0.086275271754106, 0.11240713994494447, 0.12847265889847323
            ]
            
        # (X): Check to see if the list lengths are equal --- just an easy thing first:
        self.assertEqual(
            len(cross_section_library_list),
            len(_MATHEMATICA_LIST_VALUES),
            "[ASSERT]: List lengths are not equal.")
        
        # (X): Perform the test:
        for index, (library_xsec_value, mathematica_xsec_value) in enumerate(zip(cross_section_library_list, _MATHEMATICA_LIST_VALUES)):

            # (X.1): Pairwise assert almost equal.
            self.assertAlmostEqual(
                library_xsec_value,
                mathematica_xsec_value,
                places = 7,
                msg = f"\nLists differ at index {index}: {library_xsec_value} != {mathematica_xsec_value}")
            
    def test_minus_beam_cross_section(self):
        """
        ## Description:
        Test the "minus-beam" cross-section (lambda = -1) numerics.
        """

        # (X): Compute the cross-section values:
        cross_section_library_list = self.minus_beam_unp_target.compute_cross_section(
            phi_values = self.phi_values,
            lepton_helicity = -1.0,
            target_polarization = 0.0).real

        # (X): We selected 15 phi points within 0 to 2pi (equally-spaced) and evaluated
        # | our Mathematica code at each point to produce a cross-section value. That's where this
        # | list comes from.
        _MATHEMATICA_LIST_VALUES = [
                0.12847265889847323, 0.11240713994494447, 0.086275271754106, 0.06146593446753039, 0.04377174360194547,
                0.03368144033443923, 0.02965862007660449, 0.030122994484957408, 0.03411101095406899, 0.041372981141853726,
                0.052275336338715575, 0.06749904562028203, 0.08719738289049578, 0.10909776086210324, 0.12618580291628734, 0.12847265889847323
            ]
            
        # (X): Check to see if the list lengths are equal --- just an easy thing first:
        self.assertEqual(
            len(cross_section_library_list),
            len(_MATHEMATICA_LIST_VALUES),
            "[ASSERT]: List lengths are not equal.")
        
        # (X): Perform the test:
        for index, (library_xsec_value, mathematica_xsec_value) in enumerate(zip(cross_section_library_list, _MATHEMATICA_LIST_VALUES)):

            # (X.1): Pairwise assert almost equal.
            self.assertAlmostEqual(
                library_xsec_value,
                mathematica_xsec_value,
                places = 7,
                msg = f"\nLists differ at index {index}: {library_xsec_value} != {mathematica_xsec_value}")

    def test_unpolarized_bsa(self):
        """
        ## Description:
        Test the function that computes the beam-spin asymmetry (BSA) observable.
        """

        # (X): Compute the BSA values:
        bsa_library_list = self.unpolarized_cross_section.compute_bsa(
            phi_values = self.phi_values,
            target_polarization = 0.0).real

        # (X): We selected 15 phi points within 0 to 2pi (equally-spaced) and evaluated
        # | our Mathematica code at each point to produce a BSA value. That's where this
        # | list comes from.
        _MATHEMATICA_LIST_VALUES = [
                0.0, 0.057749666884978575, 0.1168149401295814, 0.17308539107193663, 0.21323927136841755,
                0.2163168132162337, 0.16491759814369988, 0.06208575102633405, -0.06208575102633405, -0.16491759814369988,
                -0.2163168132162337, -0.21323927136841755, -0.17308539107193663, -0.1168149401295814, -0.057749666884978575, 0.0
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
                places = 8,
                msg = f"\nLists differ at index {index}: {library_bsa_value} != {mathematica_bsa_value}")
        
    def test_plus_lp_target_bsa(self):
        """
        ## Description:
        Test the function that computes the beam-spin asymmetry (BSA) observable for
        the BSA(Lambda = +0.5) setting.
        """

        # (X): Compute the BSA values:
        bsa_library_list = self.unpolarized_cross_section.compute_bsa(
            phi_values = self.phi_values,
            target_polarization = 0.5).real

        # (X): We selected 15 phi points within 0 to 2pi (equally-spaced) and evaluated
        # | our Mathematica code at each point to produce a BSA value. That's where this
        # | list comes from.
        _MATHEMATICA_LIST_VALUES = [
                0.25718140674286344, 0.3072768833329253, 0.355440455942344, 0.3957240161410663, 0.4175948580887451,
                0.40929172189663515, 0.3625960757567763, 0.2781682590713304, 0.1721354072906047, 0.07925548996053475,
                0.036082965616844966, 0.049482598183614605, 0.0962500767656345, 0.15203324209571448, 0.20599875367165948, 0.25718140674286344 
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
                places = 8,
                msg = f"\nLists differ at index {index}: {library_bsa_value} != {mathematica_bsa_value}")
            
    def test_minus_lp_target_bsa(self):
        """
        ## Description:
        Test the function that computes the beam-spin asymmetry (BSA) observable for
        the BSA(Lambda = -0.5) setting.
        """

        # (X): Compute the BSA values:
        bsa_library_list = self.unpolarized_cross_section.compute_bsa(
            phi_values = self.phi_values,
            target_polarization = -0.5).real

        # (X): We selected 15 phi points within 0 to 2pi (equally-spaced) and evaluated
        # | our Mathematica code at each point to produce a BSA value. That's where this
        # | list comes from.
        _MATHEMATICA_LIST_VALUES = [
                -0.25718140674286344, -0.20599875367165948, -0.15203324209571448, -0.0962500767656345, -0.049482598183614605,
                -0.036082965616844966, -0.07925548996053475, -0.1721354072906047, -0.2781682590713304, -0.3625960757567763,
                -0.40929172189663515, -0.4175948580887451, -0.3957240161410663, -0.355440455942344, -0.3072768833329253, -0.25718140674286344
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
                places = 8,
                msg = f"\nLists differ at index {index}: {library_bsa_value} != {mathematica_bsa_value}")

    def test_unpolarized_beam_tsa(self):
        """
        ## Description:
        Test the function that computes the target-spin asymmetry (TSA) observable for 
        setting TSA(lambda = 0).
        """

        # (X): Compute the TSA values:
        tsa_library_list = self.unpolarized_cross_section.compute_tsa(
            phi_values = self.phi_values,
            lepton_polarization = 0.0).real

        # (X): We selected 15 phi points within 0 to 2pi (equally-spaced) and evaluated
        # | our Mathematica code at each point to produce a DSA value. That's where this
        # | list comes from.
        _MATHEMATICA_LIST_VALUES = [
            0.0, 0.027706758481046482, 0.05955513858033799, 0.09491728007982221, 0.12496060781334403,
            0.13342668952392653, 0.1052267641412066, 0.04028092957467329, -0.04028092957467329, -0.1052267641412066,
            -0.13342668952392653, -0.12496060781334403, -0.09491728007982221, -0.05955513858033799, -0.027706758481046482, 0.
        ]
            
        # (X): Check to see if the list lengths are equal ---  0. I,just an easy thing first:
        self.assertEqual(
            len(tsa_library_list),
            len(_MATHEMATICA_LIST_VALUES),
            "[ASSERT]: List lengths are not equal.")
        
        # (X): Perform the test:
        for index, (library_tsa_value, mathematica_tsa_value) in enumerate(zip(tsa_library_list, _MATHEMATICA_LIST_VALUES)):

            # (X.1): Pairwise assert almost equal.
            self.assertAlmostEqual(
                library_tsa_value,
                mathematica_tsa_value,
                places = 8,
                msg = f"\nLists differ at index {index}: {library_tsa_value} != {mathematica_tsa_value}")
            
    def test_plus_beam_tsa(self):
        """
        ## Description:
        Test the function that computes the target-spin asymmetry (TSA) observable for
        TSA(lambda = +1.0).
        """

        # (X): Compute the TSA values:
        tsa_library_list = self.unpolarized_cross_section.compute_tsa(
            phi_values = self.phi_values,
            lepton_polarization = 1.0).real

        # (X): We selected 15 phi points within 0 to 2pi (equally-spaced) and evaluated
        # | our Mathematica code at each point to produce a DSA value. That's where this
        # | list comes from.
        _MATHEMATICA_LIST_VALUES = [
            0.25718140674286344, 0.27014673723681726, 0.28594617472322875, 0.3027204627438316, 0.3144467137054291,
            0.31325065440726446, 0.292776119134906, 0.25192722283927643, 0.19938544992628154, 0.15639956598741703,
            0.145669398787535, 0.16724002788285378, 0.19987842817039844, 0.2267233583780376, 0.24445107235859914, 0.25718140674286344 
        ]
            
        # (X): Check to see if the list lengths are equal ---  0. I,just an easy thing first:
        self.assertEqual(
            len(tsa_library_list),
            len(_MATHEMATICA_LIST_VALUES),
            "[ASSERT]: List lengths are not equal.")
        
        # (X): Perform the test:
        for index, (library_tsa_value, mathematica_tsa_value) in enumerate(zip(tsa_library_list, _MATHEMATICA_LIST_VALUES)):

            # (X.1): Pairwise assert almost equal.
            self.assertAlmostEqual(
                library_tsa_value,
                mathematica_tsa_value,
                places = 8,
                msg = f"\nLists differ at index {index}: {library_tsa_value} != {mathematica_tsa_value}")
            
    def test_minus_beam_tsa(self):
        """
        ## Description:
        Test the function that computes the target-spin asymmetry (TSA) observable for
        TSA(lambda = -1.0).
        """

        # (X): Compute the TSA values:
        tsa_library_list = self.unpolarized_cross_section.compute_tsa(
            phi_values = self.phi_values,
            lepton_polarization = -1.0).real

        # (X): We selected 15 phi points within 0 to 2pi (equally-spaced) and evaluated
        # | our Mathematica code at each point to produce a DSA value. That's where this
        # | list comes from.
        _MATHEMATICA_LIST_VALUES = [
            -0.25718140674286344, -0.24445107235859914, -0.2267233583780376, -0.19987842817039844, -0.16724002788285378,
            -0.145669398787535, -0.15639956598741703, -0.19938544992628154 , -0.25192722283927643, -0.292776119134906,
            -0.31325065440726446, -0.3144467137054291, -0.3027204627438316, -0.28594617472322875, -0.27014673723681726, -0.25718140674286344
        ]
            
        # (X): Check to see if the list lengths are equal ---  0. I,just an easy thing first:
        self.assertEqual(
            len(tsa_library_list),
            len(_MATHEMATICA_LIST_VALUES),
            "[ASSERT]: List lengths are not equal.")
        
        # (X): Perform the test:
        for index, (library_tsa_value, mathematica_tsa_value) in enumerate(zip(tsa_library_list, _MATHEMATICA_LIST_VALUES)):

            # (X.1): Pairwise assert almost equal.
            self.assertAlmostEqual(
                library_tsa_value,
                mathematica_tsa_value,
                places = 8,
                msg = f"\nLists differ at index {index}: {library_tsa_value} != {mathematica_tsa_value}")
            
    def test_dsa(self):
        """
        ## Description:
        Test the function that computes the beam-spin asymmetry (BSA) observable.
        """

        # (X): Compute the BSA values:
        dsa_library_list = self.unpolarized_cross_section.compute_dsa(phi_values = self.phi_values).real

        # (X): We selected 15 phi points within 0 to 2pi (equally-spaced) and evaluated
        # | our Mathematica code at each point to produce a DSA value. That's where this
        # | list comes from.
        _MATHEMATICA_LIST_VALUES = [
                0.25718140674286344, 0.25804086284126077, 0.25979382142346746, 0.2601996723435032, 0.25653849400682416,
                0.24758534818261718, 0.23583328935526174, 0.22728738409855817, 0.22728738409855817, 0.23583328935526174,
                0.24758534818261718, 0.25653849400682416, 0.2601996723435032, 0.25979382142346746, 0.25804086284126077, 0.25718140674286344 
            ]
            
        # (X): Check to see if the list lengths are equal --- just an easy thing first:
        self.assertEqual(
            len(dsa_library_list),
            len(_MATHEMATICA_LIST_VALUES),
            "[ASSERT]: List lengths are not equal.")
        
        # (X): Perform the test:
        for index, (library_dsa_value, mathematica_dsa_value) in enumerate(zip(dsa_library_list, _MATHEMATICA_LIST_VALUES)):

            # (X.1): Pairwise assert almost equal.
            self.assertAlmostEqual(
                library_dsa_value,
                mathematica_dsa_value,
                places = 8,
                msg = f"\nLists differ at index {index}: {library_dsa_value} != {mathematica_dsa_value}")
