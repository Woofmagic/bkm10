"""
Entry point for `DifferentialCrossSection` class.

## Description:
This class computes *the* BKM10 four-fold differential cross section according to
the user-provided values of the (i) kinematic settings and (ii) CFFs.

## Notes:
1. 2026/02/02:
    - *Commented out* all of the individual coefficient calculations for two reasons:
    (i) it was jank, (ii) one can show that all you need in the library are four combinations
    of cross section settings, sigma(l = +1, L = +0.5), sigma(l = +1, L = -0.5), sigma(l = -1, L = +0.5),
    and sigma(l = -1, L = -0.5). This represents a *major structural refactor*.
2. 2026/02/03:
    - All tests in `tests/test_obserables.py` passed, which indicates the desired results --- many of which
    are exactly what we had before --- are obtained *without* using the method of individual coefficient
    calculations.
    - Functions computing observales now *take as arguments* Lambda and lambda; the class `DifferentialCrossSection`
    *does not take these arguments anymore*. This represents a *major structural refactor*.
"""

# (1): Import native libraries | shutil
import shutil

# (2): Import native libraries | warnings:
import warnings

# (3) Import 3rd Party Library | NumPy:
import numpy as np

# (4): Import third-party libraries | Matplotlib:
import matplotlib.pyplot as plt

# (5): Import accompanying modules | bkm10_lib > validation > validate_configuration
from bkm10_lib.validation import validate_configuration

# (6): Import accompanying modules | bkm10_lib > formalism > BKMFormalism:
from bkm10_lib.formalism import BKMFormalism

class DifferentialCrossSection:
    """
    Welcome to the `DifferentialCrossSection` class!

    ## Description:
    Compute BKM10 differential cross sections using user-defined inputs.

    ## Detailed Description:
    This class provides the means to compute the (i) four-fold differential cross section and
    (ii) the beam-spin asymmetry as was parameterized in what is called the "BKM10 formalism." 
    One should note that this class actually *calls* another class: `BKMFormalism`. What this class 
    explicitly does is essenentially "gather" together all the results that are computed using 
    `BKMFormalism` and then brings them together in a final computation. More explicitly, the `BKMFormalism`
    class computes ALL of the 100+ coefficients that the BKM10 formalism relies, and this class,
    `DifferentialCrossSection` computes only the "mode expansion" coefficients for the BH, the DVCS, and
    the interference contribution to the total differential cross-section.

    :param dict configuration:
        A dictionary containing the configuration settings with the following keys:
        
    :param BKM10Inputs kinematics:
        Dataclass containing the required kinematic variables.
        
    :param CFFInputs cff_inputs:
        Object or  dictionary containing Compton Form Factor values or parameters.
        
    :param float lepton_polarization:
            The BKM10 formalism uses +1.0, 0.0, or -1.0. Nothing else!

    :param float target_polarization:
        The BKM10 formalism uses +0.5, 0.0, or -0.5. Nothing else!

    :param bool verbose:
        A boolean flag that will tell the class to print out various messages at
        intermediate steps in the calculation. Useful if you want to determine when
        you have, say, calculated a given coefficient, like C_{++}^{LP}(n = 1).
    
    :param bool debugging:
        A boolean flag that will bomb anybody's terminal with output. As the flag is
        entitled, DO NOT USE THIS unless you need to do some serious debugging. We are
        talking about following how the data gets transformed through every calculation.
    """

    def __init__(
            self,
            configuration: dict,
            bh_setting: bool = True,
            dvcs_setting: bool = True,
            interference_setting: bool = True,
            verbose: bool = False,
            debugging: bool = False):
        """
        ## Description:
        Initialize the `DifferentialCrossSection` class.

        :param dict configuration:
            A dictionary of configuration parameters.

        :param bool bh_on:
            `True` to keep the|BH|^{2} term in the computation; `False` otherwise.

        :param bool dvcs_on:
            `True` to keep the |DVCS|^{2} term in the computation; `False` otherwise.

        :param bool interference_on:
            `True` to keep the I(nterference) term in the computation; `False` otherwise.

        :param bool verbose:
            Boolean setting to turn on if you want to see 
            frequent print output that shows you "where" the code  
            is in its execution.

        :param bool debugging:
            Do not turn this on.
        """
        
        # (1): Obtain a True/False to operate the calculation in.
        # | [NOTE]: if the configuration *is* none, then we *must*
        # | configure the class, and that turns `configuration_mode` to
        # | True!
        self.configuration_mode = configuration is not None

        # (X): Will we compute the BH^{2} part of the BKM10 formalism?
        self.bh_setting = bh_setting

        # (X): Will we compute the BH^{2} part of the BKM10 formalism?
        self.dvcs_setting = dvcs_setting

        # (X): Will we compute the BH^{2} part of the BKM10 formalism?
        self.interference_setting = interference_setting

        # (2): Determine verbose mode:
        self.verbose = verbose

        # (3): Determine debugging mode (DO NOT TURN ON!):
        self.debugging = debugging

        # (4): A dictionary of *every coefficient* that we computed:
        self.coefficients = {}

        # (5): The Trento Angle convention basically shifts all phi to pi - phi:
        # | [TODO]: We have made this a private variable for now. We *will* eventually
        # | make this available to the user.
        # | [NOTE]: The "Trento Angle" convention is a transformation of the azimuthal angle
        # | phi that follows: phi -> pi - phi.
        self._using_trento_angle_convention = True

        # (6): Hidden data that says if configuration passed:
        self._passed_configuration = False

        # (8): If the verbose mode flag is True...
        if self.verbose:

            # (8.1): ... inform the user that we will log verbose output:
            print("> [VERBOSE]: Verbose mode on.")

        # (9): If the debugging mode flag is True...
        if self.debugging:

            # (9.1): ... inform the user to turn it off ASAP.
            print("> [DEBUGGING]: Debugging mode is on — DO NOT USE THIS!")

        # (10): If the configuration flag is True....
        if self.configuration_mode:
            
            # (10.1): ... verbose mode says it received it.
            if self.verbose:
                print("> [VERBOSE]: Configuration dictionary received!")

            # (10.2): ... debugging mode will PRINT IT OUT:
            if self.debugging:
                print(f"> [DEBUGGING]:Configuration dictionary received:\n{configuration}")

            # (10.3): Attempt to run initialization of the class using the dictionary:
            try:
                
                # (10.3.1): If we are in debugging mode...
                if self.debugging:

                    # (10.3.1.1): Inform the user that we are now trying to initialize the configuration:
                    print("> [DEBUGGING]: Trying to initialize configuration...")
            
                # (10.3.2): Initialize the class from the dictionary:
                self._initialize_from_config(configuration)

                # (10.3.3): If the line above succeeds and we are in debugging mode...
                if self.debugging:

                    # (10.3.3.1): ... inform the user that the configuration passed!
                    print("> [DEBUGGING]: Configuration passed!")

            # (10.4): If an issue occurs during the initialization of the class from the dictionary...
            except:

                # (10.4.1): ... raise a super general exception (for now):
                raise Exception("> [ERROR]: Unable to initialize configuration!")
            
            # (10.5): If the try/except block passed, then we *now* can inform the class itself
            # | that configuration has passed!
            self._passed_configuration = True

            # (10.6): If we are in verbose output...
            if self.verbose:

                # (10.6.1): ... inform the user that the configuration passed!
                print("> [VERBOSE]: Configuration succeeded!")

            # (10.7): If we are in debugging output...
            if self.debugging:

                # (10.7.1): ... PRINT OUT the internal variable state...
                print(f"> [DEBUGGING]: Configuration succeeded! Corresponding internal attribute now set to: {self._passed_configuration}")

    @staticmethod
    def _set_plot_style():
        """
        ## Description:
        We want the plots to look a particular way. So, let's do that. In particular,
        we check if a LaTeX distribution is installed!
        """

        # (X): Call shutil to find a TeX distribution:
        latex_installed = shutil.which("latex") is not None

        # (X): If TeX was found...
        if latex_installed:

            # (X): ... matplotlib will not crash if we put this in our plots:
            plt.rcParams.update({
                "text.usetex": True,
                "font.family": "serif",
                "text.latex.preamble": r"\usepackage{amsmath}"
            })

        # (X): If TeX was not found...
        else:

            # (X): ... first, tell the user that we recommend a TeX distribution:
            warnings.warn(
                "> LaTeX is not installed. Falling back to Matplotlib's mathtext.",
                UserWarning)
            
            # (X): If we don't have TeX, then we have to set it to false:
            plt.rcParams.update({
                "text.usetex": False,
                "font.family": "serif"
            })
    
        # (X): rcParams for the x-axis tick direction:
        plt.rcParams['xtick.direction'] = 'in'

        # (X): rcParams for the "major" (larger) x-axis vertical size:
        plt.rcParams['xtick.major.size'] = 8.5

        # (X): rcParams for the "major" (larger) x-axis horizonal width:
        plt.rcParams['xtick.major.width'] = 0.5

        # (X): rcParams for the "minor" (smaller) x-axis vertical size:
        plt.rcParams['xtick.minor.size'] = 2.5

        # (X): rcParams for the "minor" (smaller) x-axis horizonal width:
        plt.rcParams['xtick.minor.width'] = 0.5

        # (X): rcParams for the minor ticks to be *shown* versus invisible:
        plt.rcParams['xtick.minor.visible'] = True

        # (X): rcParams dictating that we want ticks along the x-axis on *top* (opposite side) of the bounding box:
        plt.rcParams['xtick.top'] = True

        # (X): rcParams for the y-axis tick direction:
        plt.rcParams['ytick.direction'] = 'in'

        # (X): rcParams for the "major" (larger) y-axis vertical size:
        plt.rcParams['ytick.major.size'] = 8.5

        # (X): rcParams for the "major" (larger) y-axis horizonal width:
        plt.rcParams['ytick.major.width'] = 0.5

        # (X): rcParams for the "minor" (smaller) y-axis vertical size:
        plt.rcParams['ytick.minor.size'] = 2.5

        # (X): rcParams for the "minor" (smaller) y-axis horizonal width:
        plt.rcParams['ytick.minor.width'] = 0.5

        # (X): rcParams for the minor ticks to be *shown* versus invisible:
        plt.rcParams['ytick.minor.visible'] = True

        # (X): rcParams dictating that we want ticks along the y-axis on the *left* of the bounding box:
        plt.rcParams['ytick.right'] = True

    def _initialize_from_config(self, configuration_dictionary: dict)-> None:
        """
        ## Description:
        We demand a dictionary type, extract each of its keys and values, and then
        we perform validation on each of the values. These values *cannot* be anything!
        So, this function is responsible for that.

        :param dict configuration_dictionary: 
            The *required* dictionary for initializing this class. See the class docstring for
            more information.
        """

        # (1): Initialize a try-catch block:
        try:

            # (1.1): Pass the whole dictionary into the validation function:
            validated_configuration_dictionary = validate_configuration(
                configuration_dictionary, self.verbose)

            # (1.2): If validation is passed, we *set* the kinematic inputs using the `kinematics` key.
            # | Should be of type `BKMInputs`!
            self.kinematic_inputs = validated_configuration_dictionary["kinematics"]

            # (1.3): Assuming (1.2) passed, then we continue to extract dictionary keys.
            # | Here, it is `cff_inputs`, and should be of type `CFFInputs`.
            self.cff_inputs = validated_configuration_dictionary["cff_inputs"]

            # (1.6): Extract the boolean value that tells us to evaluate with/out the WW relations:
            self.using_ww = validated_configuration_dictionary["using_ww"]

            # (1.11): Initialize a BKM formalism with beam polarization = 1.0 and target polarization = 0.5
            self.formalism_plus_beam_plus_target = self._build_formalism_beam_target(+1.0, +0.5)
            
            # (1.12): Initialize a BKM formalism with beam polarization = -1.0 and target polarization = 0.5
            self.formalism_minus_beam_plus_target = self._build_formalism_beam_target(-1.0, +0.5)
            
            # (1.13): Initialize a BKM formalism with beam polarization = 1.0 and target polarization = -0.5
            self.formalism_plus_beam_minus_target = self._build_formalism_beam_target(+1.0, -0.5)

            # (1.14): Initialize a BKM formalism with beam polarization = -1.0 and target polarization = -0.5
            self.formalism_minus_beam_minus_target = self._build_formalism_beam_target(-1.0, -0.5)

        # (2): If there are errors in the initialization above...
        except Exception as error:

            # (2.1): ... too general, yes, but not sure what we put here yet:
            raise Exception("> [ERROR]: Error occurred during validation...") from error
        
    def _build_formalism_beam_target(self, lepton_polarization: float, target_polarization: float) -> BKMFormalism:
        """
        ## Description:
        We build a `BKMFormalism` based on a given value of the target polarization, which is
        usually called Lambda (capital "L"). This function enables us to compute the target-spin
        asymmetry (A_{UL}), which is an observable.
        """

        # (1): Initialize a try-catch block to build the formalism:
        try:

            # (1.1): Immediately return a BKMFormalism instance:
            return BKMFormalism(
                inputs = self.kinematic_inputs,
                cff_values = self.cff_inputs,
                lepton_polarization = lepton_polarization,
                target_polarization = target_polarization,
                using_ww = self.using_ww,
                bh_on = self.bh_setting,
                dvcs_on = self.dvcs_setting,
                interference_on = self.interference_setting,
                verbose = self.verbose,
                debugging = self.debugging)
        
        # (2): If there are errors in initializing a BKMFormalism instance...
        except Exception as error:

            # (2.1): ... too general, yes, but not sure what we put here yet:
            raise Exception("> [ERROR]: Error occurred during validation...") from error
        
    def compute_prefactor(self) -> float:
        """
        ## Description:
        Immediately compute the prefactor that multiplies the entire cross section.

        ## Detailed Description:
        This prefactor is separate from the squared amplitude of the whole DVCS process.
        We compute it separately from the squared amplitudes in accordance with the whole
        separation-of-concerns thing.
        """

        # (1): Immediately return the prefactor as computed from the `BKMFormalism`:
        # | [NOTE]: It *does not matter* what (lambda, Lambda) formalism we call here
        # | because they all go with the same prefactor.
        return self.formalism_plus_beam_plus_target.compute_cross_section_prefactor()

    def compute_cross_section(self, phi_values, lepton_helicity, target_polarization):
        """
        ## Description:
        We compute the four-fold *differential cross-section* as 
        described with the BKM10 Formalism.

        :param np.ndarray phi: A NumPy array that will be plugged-and-chugged into the BKM10 formalism.
        """

        # (X): If  the user has not filled in the class inputs...
        if not hasattr(self, 'kinematic_inputs'):

            # (X): ...enforce the class to use this setting:
            raise RuntimeError("> Missing 'kinematic_inputs' configuration before evaluation.")

        # (X): If the user wants some confirmation that stuff is good...
        if self.verbose:

            # (X): ... we simply evaluate the length of the phi array:
            print(f"> [VERBOSE]: Evaluating cross-section at {len(phi_values)} phi points.")

        # (X): If the user wants to see everything...
        if self.debugging:

            # (X): ... we give it to them:
            print(f"> [DEBUGGING]: Evaluating cross-section with phi values of:\n> {phi_values}")

        # (X): Remember what the Trento angle convention is...
        if self._using_trento_angle_convention:

            # (X): ...if it's on, we apply the shift to the angle array:
            verified_phi_values = np.pi - np.atleast_1d(phi_values)

        # (X): Otherwise...
        else:

            # (X): ... just verify that the array of angles is at least 1D:
            verified_phi_values = np.atleast_1d(phi_values)

        # (X): Obtain the cross-section prefactor:
        cross_section_prefactor = self.compute_prefactor()

        # (X): Compute the differential cross-section according to Lambda = +0.5 and lambda = 1.0:
        sigma_plus_beam_plus_target = (
            self.formalism_plus_beam_plus_target.compute_c0_coefficient(verified_phi_values) * np.cos(0. * verified_phi_values)
            + self.formalism_plus_beam_plus_target.compute_c1_coefficient(verified_phi_values) * np.cos(1. * verified_phi_values)
            + self.formalism_plus_beam_plus_target.compute_c2_coefficient(verified_phi_values) * np.cos(2. * verified_phi_values)
            + self.formalism_plus_beam_plus_target.compute_c3_coefficient(verified_phi_values) * np.cos(3. * verified_phi_values)
            + self.formalism_plus_beam_plus_target.compute_s1_coefficient(verified_phi_values) * np.sin(1. * verified_phi_values)
            + self.formalism_plus_beam_plus_target.compute_s2_coefficient(verified_phi_values) * np.sin(2. * verified_phi_values)
            + self.formalism_plus_beam_plus_target.compute_s3_coefficient(verified_phi_values) * np.sin(3. * verified_phi_values)
            )
        
        # (X): Compute the differential cross-section according to Lambda = +0.5 and lambda = -1.0:
        sigma_minus_beam_plus_target = (
            self.formalism_minus_beam_plus_target.compute_c0_coefficient(verified_phi_values) * np.cos(0. * verified_phi_values)
            + self.formalism_minus_beam_plus_target.compute_c1_coefficient(verified_phi_values) * np.cos(1. * verified_phi_values)
            + self.formalism_minus_beam_plus_target.compute_c2_coefficient(verified_phi_values) * np.cos(2. * verified_phi_values)
            + self.formalism_minus_beam_plus_target.compute_c3_coefficient(verified_phi_values) * np.cos(3. * verified_phi_values)
            + self.formalism_minus_beam_plus_target.compute_s1_coefficient(verified_phi_values) * np.sin(1. * verified_phi_values)
            + self.formalism_minus_beam_plus_target.compute_s2_coefficient(verified_phi_values) * np.sin(2. * verified_phi_values)
            + self.formalism_minus_beam_plus_target.compute_s3_coefficient(verified_phi_values) * np.sin(3. * verified_phi_values)
            )
        
        # (X): Compute the differential cross-section according to Lambda = -0.5 and lambda = 1.0:
        sigma_plus_beam_minus_target = (
            self.formalism_plus_beam_minus_target.compute_c0_coefficient(verified_phi_values) * np.cos(0. * verified_phi_values)
            + self.formalism_plus_beam_minus_target.compute_c1_coefficient(verified_phi_values) * np.cos(1. * verified_phi_values)
            + self.formalism_plus_beam_minus_target.compute_c2_coefficient(verified_phi_values) * np.cos(2. * verified_phi_values)
            + self.formalism_plus_beam_minus_target.compute_c3_coefficient(verified_phi_values) * np.cos(3. * verified_phi_values)
            + self.formalism_plus_beam_minus_target.compute_s1_coefficient(verified_phi_values) * np.sin(1. * verified_phi_values)
            + self.formalism_plus_beam_minus_target.compute_s2_coefficient(verified_phi_values) * np.sin(2. * verified_phi_values)
            + self.formalism_plus_beam_minus_target.compute_s3_coefficient(verified_phi_values) * np.sin(3. * verified_phi_values)
            )
        
        # (X): Compute the differential cross-section according to Lambda = -0.5 and lambda = -1.0:
        sigma_minus_beam_minus_target = (
            self.formalism_minus_beam_minus_target.compute_c0_coefficient(verified_phi_values) * np.cos(0. * verified_phi_values)
            + self.formalism_minus_beam_minus_target.compute_c1_coefficient(verified_phi_values) * np.cos(1. * verified_phi_values)
            + self.formalism_minus_beam_minus_target.compute_c2_coefficient(verified_phi_values) * np.cos(2. * verified_phi_values)
            + self.formalism_minus_beam_minus_target.compute_c3_coefficient(verified_phi_values) * np.cos(3. * verified_phi_values)
            + self.formalism_minus_beam_minus_target.compute_s1_coefficient(verified_phi_values) * np.sin(1. * verified_phi_values)
            + self.formalism_minus_beam_minus_target.compute_s2_coefficient(verified_phi_values) * np.sin(2. * verified_phi_values)
            + self.formalism_minus_beam_minus_target.compute_s3_coefficient(verified_phi_values) * np.sin(3. * verified_phi_values)
            )
        
        # (X): Initializing this just in case...
        differential_cross_section = 0.0
    
        if lepton_helicity == 0.0:
            if target_polarization == 0.0:
                differential_cross_section = 0.25 * (
                    .389379 * 1000000. * (
                        cross_section_prefactor * (
                            sigma_plus_beam_plus_target +
                            sigma_minus_beam_plus_target +
                            sigma_plus_beam_minus_target +
                            sigma_minus_beam_minus_target
                        )
                    )
                )
            elif target_polarization == +0.5:
                differential_cross_section = 0.5 * (
                    .389379 * 1000000. * (
                        cross_section_prefactor * (
                            sigma_plus_beam_plus_target +
                            sigma_minus_beam_plus_target
                        )
                    )
                )
            elif target_polarization == -0.5:
                differential_cross_section = 0.5 * (
                    .389379 * 1000000. * (
                        cross_section_prefactor * (
                            sigma_plus_beam_minus_target +
                            sigma_minus_beam_minus_target
                        )
                    )
                )
        elif lepton_helicity == +1.0:
            if target_polarization == 0.0:
                differential_cross_section = 0.5 * (
                    .389379 * 1000000. * (
                        cross_section_prefactor * (
                            sigma_plus_beam_plus_target +
                            sigma_plus_beam_minus_target
                        )
                    )
                )
            elif target_polarization == +0.5:
                differential_cross_section = (
                    .389379 * 1000000. * (
                        cross_section_prefactor * sigma_plus_beam_plus_target
                        )
                    )
            elif target_polarization == -0.5:
                differential_cross_section = (
                    .389379 * 1000000. * (
                        cross_section_prefactor * sigma_plus_beam_minus_target
                        )
                    )
        elif lepton_helicity == -1.0:
            if target_polarization == 0.0:
                differential_cross_section = 0.5 * (
                    .389379 * 1000000. * (
                        cross_section_prefactor * (
                            sigma_minus_beam_plus_target +
                            sigma_minus_beam_minus_target
                        )
                    )
                )
            elif target_polarization == +0.5:
                differential_cross_section = (
                    .389379 * 1000000. * (
                        cross_section_prefactor * sigma_minus_beam_plus_target
                        )
                    )
            elif target_polarization == -0.5:
                differential_cross_section = (
                    .389379 * 1000000. * (
                        cross_section_prefactor * sigma_minus_beam_minus_target
                        )
                    )
        else:
            raise NotImplementedError(f"[ERROR]: Unknown setting of lambda = {lepton_helicity} and Lambda = {target_polarization}")

        # (X): Return the cross section:
        return differential_cross_section
    
    def compute_bsa(self, phi_values, target_polarization):
        """
        ## Description:
        We compute the BKM-predicted BSA.

        :param np.ndarray phi_values: A NumPy array that will be plugged-and-chugged into the BKM10 formalism.
        """

        # (X): If  the user has not filled in the class inputs...
        if not hasattr(self, 'kinematic_inputs'):

            # (X): ...enforce the class to use this key:
            raise RuntimeError("> Missing 'kinematic_inputs' configuration before evaluation.")

        # (X): If the user wants some confirmation that stuff is good...
        if self.verbose:

            # (X): ... we simply evaluate the length of the phi array:
            print(f"> [VERBOSE]: Evaluating cross-section at {len(phi_values)} phi points.")

        # (X): If the user wants to see everything...
        if self.debugging:

            # (X): ... we give it to them:
            print(f"> [DEBUGGING]: Evaluating cross-section with phi values of:\n> {phi_values}")

        # (X): Remember what the Trento angle convention is...
        if self._using_trento_angle_convention:

            # (X): ...if it's on, we apply the shift to the angle array:
            verified_phi_values = np.pi - np.atleast_1d(phi_values)

        # (X): Otherwise...
        else:

            # (X): ... just verify that the array of angles is at least 1D:
            verified_phi_values = np.atleast_1d(phi_values)

        # (X): Compute the differential cross-section according to Lambda = +0.5 and lambda = 1.0:
        sigma_plus_beam_plus_target = (
            self.formalism_plus_beam_plus_target.compute_c0_coefficient(verified_phi_values) * np.cos(0. * verified_phi_values)
            + self.formalism_plus_beam_plus_target.compute_c1_coefficient(verified_phi_values) * np.cos(1. * verified_phi_values)
            + self.formalism_plus_beam_plus_target.compute_c2_coefficient(verified_phi_values) * np.cos(2. * verified_phi_values)
            + self.formalism_plus_beam_plus_target.compute_c3_coefficient(verified_phi_values) * np.cos(3. * verified_phi_values)
            + self.formalism_plus_beam_plus_target.compute_s1_coefficient(verified_phi_values) * np.sin(1. * verified_phi_values)
            + self.formalism_plus_beam_plus_target.compute_s2_coefficient(verified_phi_values) * np.sin(2. * verified_phi_values)
            + self.formalism_plus_beam_plus_target.compute_s3_coefficient(verified_phi_values) * np.sin(3. * verified_phi_values)
            )
        
        # (X): Compute the differential cross-section according to Lambda = +0.5 and lambda = -1.0:
        sigma_minus_beam_plus_target = (
            self.formalism_minus_beam_plus_target.compute_c0_coefficient(verified_phi_values) * np.cos(0. * verified_phi_values)
            + self.formalism_minus_beam_plus_target.compute_c1_coefficient(verified_phi_values) * np.cos(1. * verified_phi_values)
            + self.formalism_minus_beam_plus_target.compute_c2_coefficient(verified_phi_values) * np.cos(2. * verified_phi_values)
            + self.formalism_minus_beam_plus_target.compute_c3_coefficient(verified_phi_values) * np.cos(3. * verified_phi_values)
            + self.formalism_minus_beam_plus_target.compute_s1_coefficient(verified_phi_values) * np.sin(1. * verified_phi_values)
            + self.formalism_minus_beam_plus_target.compute_s2_coefficient(verified_phi_values) * np.sin(2. * verified_phi_values)
            + self.formalism_minus_beam_plus_target.compute_s3_coefficient(verified_phi_values) * np.sin(3. * verified_phi_values)
            )
        
        # (X): Compute the differential cross-section according to Lambda = -0.5 and lambda = 1.0:
        sigma_plus_beam_minus_target = (
            self.formalism_plus_beam_minus_target.compute_c0_coefficient(verified_phi_values) * np.cos(0. * verified_phi_values)
            + self.formalism_plus_beam_minus_target.compute_c1_coefficient(verified_phi_values) * np.cos(1. * verified_phi_values)
            + self.formalism_plus_beam_minus_target.compute_c2_coefficient(verified_phi_values) * np.cos(2. * verified_phi_values)
            + self.formalism_plus_beam_minus_target.compute_c3_coefficient(verified_phi_values) * np.cos(3. * verified_phi_values)
            + self.formalism_plus_beam_minus_target.compute_s1_coefficient(verified_phi_values) * np.sin(1. * verified_phi_values)
            + self.formalism_plus_beam_minus_target.compute_s2_coefficient(verified_phi_values) * np.sin(2. * verified_phi_values)
            + self.formalism_plus_beam_minus_target.compute_s3_coefficient(verified_phi_values) * np.sin(3. * verified_phi_values)
            )
        
        # (X): Compute the differential cross-section according to Lambda = -0.5 and lambda = -1.0:
        sigma_minus_beam_minus_target = (
            self.formalism_minus_beam_minus_target.compute_c0_coefficient(verified_phi_values) * np.cos(0. * verified_phi_values)
            + self.formalism_minus_beam_minus_target.compute_c1_coefficient(verified_phi_values) * np.cos(1. * verified_phi_values)
            + self.formalism_minus_beam_minus_target.compute_c2_coefficient(verified_phi_values) * np.cos(2. * verified_phi_values)
            + self.formalism_minus_beam_minus_target.compute_c3_coefficient(verified_phi_values) * np.cos(3. * verified_phi_values)
            + self.formalism_minus_beam_minus_target.compute_s1_coefficient(verified_phi_values) * np.sin(1. * verified_phi_values)
            + self.formalism_minus_beam_minus_target.compute_s2_coefficient(verified_phi_values) * np.sin(2. * verified_phi_values)
            + self.formalism_minus_beam_minus_target.compute_s3_coefficient(verified_phi_values) * np.sin(3. * verified_phi_values)
            )
        
        if target_polarization == 0.0:

            sigma_plus = 0.5 * (sigma_plus_beam_plus_target + sigma_plus_beam_minus_target)
            sigma_minus = 0.5 * (sigma_minus_beam_plus_target + sigma_minus_beam_minus_target)

        elif target_polarization == 0.5:

            sigma_plus = sigma_plus_beam_plus_target
            sigma_minus = sigma_minus_beam_plus_target

        elif target_polarization == -0.5:

            sigma_plus = sigma_plus_beam_minus_target
            sigma_minus = sigma_minus_beam_minus_target

        else:

            raise NotImplementedError("[ERROR]: Acceptable values for target_polarization are -0.5, 0.0, and +0.5.")

        # (X): Compute the numerator of the BSA: sigma(+) - sigma(-):
        numerator = sigma_plus - sigma_minus

        # (X): Compute the denominator of the BSA: sigma(+) + sigma(-):
        denominator = sigma_plus + sigma_minus

        # (X): Compute the dfferential cross-section:
        bsa_values = numerator / denominator

        # (X): Return the cross section:
        return bsa_values
    
    def compute_tsa(self, phi_values, lepton_polarization):
        """
        ## Description:
        We compute the BKM-predicted TSA.

        :param np.ndarray phi_values: A NumPy array that will be plugged-and-chugged into the BKM10 formalism.
        """

        # (X): If  the user has not filled in the class inputs...
        if not hasattr(self, 'kinematic_inputs'):

            # (X): ...enforce the class to use this key:
            raise RuntimeError("> Missing 'kinematic_inputs' configuration before evaluation.")

        # (X): If the user wants some confirmation that stuff is good...
        if self.verbose:

            # (X): ... we simply evaluate the length of the phi array:
            print(f"> [VERBOSE]: Evaluating cross-section at {len(phi_values)} phi points.")

        # (X): If the user wants to see everything...
        if self.debugging:

            # (X): ... we give it to them:
            print(f"> [DEBUGGING]: Evaluating cross-section with phi values of:\n> {phi_values}")

        # (X): Remember what the Trento angle convention is...
        if self._using_trento_angle_convention:

            # (X): ...if it's on, we apply the shift to the angle array:
            verified_phi_values = np.pi - np.atleast_1d(phi_values)

        # (X): Otherwise...
        else:

            # (X): ... just verify that the array of angles is at least 1D:
            verified_phi_values = np.atleast_1d(phi_values)

        if lepton_polarization == 0.0:

            # (X): If the user wants some confirmation that stuff is good...
            if self.verbose:

                # (X): ... we simply evaluate the length of the phi array:
                print("[VERBOSE]: Lepton polarization corresponding to *unpolarized* detected. Must perform cross-section averaging...")

            # (X): If the user wants to see everything...
            if self.debugging:

                # (X): ... we give it to them:
                print(f"[VERBOSE]: Lepton polarization corresponding to *unpolarized* detected: {lepton_polarization}. Must perform cross-section averaging...")

            # (X): Compute the differential cross-section according to Lambda = +0.5 and lambda = 1.0:
        sigma_plus_beam_plus_target = (
            self.formalism_plus_beam_plus_target.compute_c0_coefficient(verified_phi_values) * np.cos(0. * verified_phi_values)
            + self.formalism_plus_beam_plus_target.compute_c1_coefficient(verified_phi_values) * np.cos(1. * verified_phi_values)
            + self.formalism_plus_beam_plus_target.compute_c2_coefficient(verified_phi_values) * np.cos(2. * verified_phi_values)
            + self.formalism_plus_beam_plus_target.compute_c3_coefficient(verified_phi_values) * np.cos(3. * verified_phi_values)
            + self.formalism_plus_beam_plus_target.compute_s1_coefficient(verified_phi_values) * np.sin(1. * verified_phi_values)
            + self.formalism_plus_beam_plus_target.compute_s2_coefficient(verified_phi_values) * np.sin(2. * verified_phi_values)
            + self.formalism_plus_beam_plus_target.compute_s3_coefficient(verified_phi_values) * np.sin(3. * verified_phi_values)
            )
        
        # (X): Compute the differential cross-section according to Lambda = +0.5 and lambda = -1.0:
        sigma_minus_beam_plus_target = (
            self.formalism_minus_beam_plus_target.compute_c0_coefficient(verified_phi_values) * np.cos(0. * verified_phi_values)
            + self.formalism_minus_beam_plus_target.compute_c1_coefficient(verified_phi_values) * np.cos(1. * verified_phi_values)
            + self.formalism_minus_beam_plus_target.compute_c2_coefficient(verified_phi_values) * np.cos(2. * verified_phi_values)
            + self.formalism_minus_beam_plus_target.compute_c3_coefficient(verified_phi_values) * np.cos(3. * verified_phi_values)
            + self.formalism_minus_beam_plus_target.compute_s1_coefficient(verified_phi_values) * np.sin(1. * verified_phi_values)
            + self.formalism_minus_beam_plus_target.compute_s2_coefficient(verified_phi_values) * np.sin(2. * verified_phi_values)
            + self.formalism_minus_beam_plus_target.compute_s3_coefficient(verified_phi_values) * np.sin(3. * verified_phi_values)
            )
        
        # (X): Compute the differential cross-section according to Lambda = -0.5 and lambda = 1.0:
        sigma_plus_beam_minus_target = (
            self.formalism_plus_beam_minus_target.compute_c0_coefficient(verified_phi_values) * np.cos(0. * verified_phi_values)
            + self.formalism_plus_beam_minus_target.compute_c1_coefficient(verified_phi_values) * np.cos(1. * verified_phi_values)
            + self.formalism_plus_beam_minus_target.compute_c2_coefficient(verified_phi_values) * np.cos(2. * verified_phi_values)
            + self.formalism_plus_beam_minus_target.compute_c3_coefficient(verified_phi_values) * np.cos(3. * verified_phi_values)
            + self.formalism_plus_beam_minus_target.compute_s1_coefficient(verified_phi_values) * np.sin(1. * verified_phi_values)
            + self.formalism_plus_beam_minus_target.compute_s2_coefficient(verified_phi_values) * np.sin(2. * verified_phi_values)
            + self.formalism_plus_beam_minus_target.compute_s3_coefficient(verified_phi_values) * np.sin(3. * verified_phi_values)
            )
        
        # (X): Compute the differential cross-section according to Lambda = -0.5 and lambda = -1.0:
        sigma_minus_beam_minus_target = (
            self.formalism_minus_beam_minus_target.compute_c0_coefficient(verified_phi_values) * np.cos(0. * verified_phi_values)
            + self.formalism_minus_beam_minus_target.compute_c1_coefficient(verified_phi_values) * np.cos(1. * verified_phi_values)
            + self.formalism_minus_beam_minus_target.compute_c2_coefficient(verified_phi_values) * np.cos(2. * verified_phi_values)
            + self.formalism_minus_beam_minus_target.compute_c3_coefficient(verified_phi_values) * np.cos(3. * verified_phi_values)
            + self.formalism_minus_beam_minus_target.compute_s1_coefficient(verified_phi_values) * np.sin(1. * verified_phi_values)
            + self.formalism_minus_beam_minus_target.compute_s2_coefficient(verified_phi_values) * np.sin(2. * verified_phi_values)
            + self.formalism_minus_beam_minus_target.compute_s3_coefficient(verified_phi_values) * np.sin(3. * verified_phi_values)
            )
        
        if lepton_polarization == 0.0:

            sigma_plus = 0.5 * (sigma_plus_beam_plus_target + sigma_minus_beam_plus_target)
            sigma_minus = 0.5 * (sigma_plus_beam_minus_target + sigma_minus_beam_minus_target)

        elif lepton_polarization == 1.0:

            sigma_plus = sigma_plus_beam_plus_target
            sigma_minus = sigma_plus_beam_minus_target

        elif lepton_polarization == -1.0:

            sigma_plus = sigma_minus_beam_plus_target
            sigma_minus = sigma_minus_beam_minus_target

        else:
            
            raise NotImplementedError("[ERROR]: Acceptable values for lepton_polarization are -1.0, 0.0, and +1.0.")
        
        # (X): Compute the numerator of the TSA: sigma(+0.5) - sigma(-0.5):
        numerator = sigma_plus - sigma_minus

        # (X): Compute the denominator of the TSA: sigma(+0.5) + sigma(-0.5):
        denominator = sigma_plus + sigma_minus

        # (X): Compute the dfferential cross-section:
        tsa_values = numerator / denominator

        # (X): Return the cross section:
        return tsa_values
    
    def compute_dsa(self, phi_values):
        """
        ## Description:
        We compute the BKM-predicted DSA (double-spin asymmetry).

        :param np.ndarray phi_values: A NumPy array that will be plugged-and-chugged into the BKM10 formalism.
        """

        # (X): If  the user has not filled in the class inputs...
        if not hasattr(self, 'kinematic_inputs'):

            # (X): ...enforce the class to use this key:
            raise RuntimeError("> Missing 'kinematic_inputs' configuration before evaluation.")

        # (X): If the user wants some confirmation that stuff is good...
        if self.verbose:

            # (X): ... we simply evaluate the length of the phi array:
            print(f"> [VERBOSE]: Evaluating cross-section at {len(phi_values)} phi points.")

        # (X): If the user wants to see everything...
        if self.debugging:

            # (X): ... we give it to them:
            print(f"> [DEBUGGING]: Evaluating cross-section with phi values of:\n> {phi_values}")

        # (X): Remember what the Trento angle convention is...
        if self._using_trento_angle_convention:

            # (X): ...if it's on, we apply the shift to the angle array:
            verified_phi_values = np.pi - np.atleast_1d(phi_values)

        # (X): Otherwise...
        else:

            # (X): ... just verify that the array of angles is at least 1D:
            verified_phi_values = np.atleast_1d(phi_values)

        # (X): Compute the differential cross-section according to Lambda = +0.5 and lambda = 1.0:
        sigma_plus_beam_plus_target = (
            self.formalism_plus_beam_plus_target.compute_c0_coefficient(verified_phi_values) * np.cos(0. * verified_phi_values)
            + self.formalism_plus_beam_plus_target.compute_c1_coefficient(verified_phi_values) * np.cos(1. * verified_phi_values)
            + self.formalism_plus_beam_plus_target.compute_c2_coefficient(verified_phi_values) * np.cos(2. * verified_phi_values)
            + self.formalism_plus_beam_plus_target.compute_c3_coefficient(verified_phi_values) * np.cos(3. * verified_phi_values)
            + self.formalism_plus_beam_plus_target.compute_s1_coefficient(verified_phi_values) * np.sin(1. * verified_phi_values)
            + self.formalism_plus_beam_plus_target.compute_s2_coefficient(verified_phi_values) * np.sin(2. * verified_phi_values)
            + self.formalism_plus_beam_plus_target.compute_s3_coefficient(verified_phi_values) * np.sin(3. * verified_phi_values)
            )
        
        # (X): Compute the differential cross-section according to Lambda = +0.5 and lambda = -1.0:
        sigma_minus_beam_plus_target = (
            self.formalism_minus_beam_plus_target.compute_c0_coefficient(verified_phi_values) * np.cos(0. * verified_phi_values)
            + self.formalism_minus_beam_plus_target.compute_c1_coefficient(verified_phi_values) * np.cos(1. * verified_phi_values)
            + self.formalism_minus_beam_plus_target.compute_c2_coefficient(verified_phi_values) * np.cos(2. * verified_phi_values)
            + self.formalism_minus_beam_plus_target.compute_c3_coefficient(verified_phi_values) * np.cos(3. * verified_phi_values)
            + self.formalism_minus_beam_plus_target.compute_s1_coefficient(verified_phi_values) * np.sin(1. * verified_phi_values)
            + self.formalism_minus_beam_plus_target.compute_s2_coefficient(verified_phi_values) * np.sin(2. * verified_phi_values)
            + self.formalism_minus_beam_plus_target.compute_s3_coefficient(verified_phi_values) * np.sin(3. * verified_phi_values)
            )
        
        # (X): Compute the differential cross-section according to Lambda = -0.5 and lambda = 1.0:
        sigma_plus_beam_minus_target = (
            self.formalism_plus_beam_minus_target.compute_c0_coefficient(verified_phi_values) * np.cos(0. * verified_phi_values)
            + self.formalism_plus_beam_minus_target.compute_c1_coefficient(verified_phi_values) * np.cos(1. * verified_phi_values)
            + self.formalism_plus_beam_minus_target.compute_c2_coefficient(verified_phi_values) * np.cos(2. * verified_phi_values)
            + self.formalism_plus_beam_minus_target.compute_c3_coefficient(verified_phi_values) * np.cos(3. * verified_phi_values)
            + self.formalism_plus_beam_minus_target.compute_s1_coefficient(verified_phi_values) * np.sin(1. * verified_phi_values)
            + self.formalism_plus_beam_minus_target.compute_s2_coefficient(verified_phi_values) * np.sin(2. * verified_phi_values)
            + self.formalism_plus_beam_minus_target.compute_s3_coefficient(verified_phi_values) * np.sin(3. * verified_phi_values)
            )
        
        # (X): Compute the differential cross-section according to Lambda = -0.5 and lambda = -1.0:
        sigma_minus_beam_minus_target = (
            self.formalism_minus_beam_minus_target.compute_c0_coefficient(verified_phi_values) * np.cos(0. * verified_phi_values)
            + self.formalism_minus_beam_minus_target.compute_c1_coefficient(verified_phi_values) * np.cos(1. * verified_phi_values)
            + self.formalism_minus_beam_minus_target.compute_c2_coefficient(verified_phi_values) * np.cos(2. * verified_phi_values)
            + self.formalism_minus_beam_minus_target.compute_c3_coefficient(verified_phi_values) * np.cos(3. * verified_phi_values)
            + self.formalism_minus_beam_minus_target.compute_s1_coefficient(verified_phi_values) * np.sin(1. * verified_phi_values)
            + self.formalism_minus_beam_minus_target.compute_s2_coefficient(verified_phi_values) * np.sin(2. * verified_phi_values)
            + self.formalism_minus_beam_minus_target.compute_s3_coefficient(verified_phi_values) * np.sin(3. * verified_phi_values)
            )

        # (X): Compute the numerator of the DSA:
        numerator = ((sigma_plus_beam_plus_target - sigma_plus_beam_minus_target) - (sigma_minus_beam_plus_target - sigma_minus_beam_minus_target))

        # (X): Compute the denominator of the DSA:
        denominator = sigma_plus_beam_plus_target + sigma_plus_beam_minus_target + sigma_minus_beam_plus_target + sigma_minus_beam_minus_target

        # (X): Compute the dfferential cross-section:
        dsa_values = numerator / denominator

        # (X): Return the cross section:
        return dsa_values
    
    def get_coefficient(self, name: str):
        """
        ## Description:
        An interface to query a given BKM coefficient's value.
        [NOTE]: This function does NOT YET WORK. Do NOT USE IT.
        [TODO]: Make this function.
        """
        
        # (X): In case there is an issue:
        try:
            
            # (X): Return the coefficient:
            return self.coefficients.get(name, None)
        
        # (X): Catch general exceptions:
        except Exception as exception:

            # (X): Raise an error:
            raise NotImplementedError(f"[ERROR]: We haven't written this function yet. See exception here: {exception}")
        
    def plot_cross_section(
            self,
            phi_values,
            lepton_helicity,
            target_polarization,
            save_plot_name: str):
        """
        ## Description:
        Plot the four-fold differential cross-section as a function of azimuthal angle φ.

        :param np.ndarray phi_values: Array of φ values (in degrees) at which to compute and plot the cross-section.

        :param str save_plot_name: If you want to save the plot, provide a non-empty string here.
        """

        # (X): If it has *NOT* been evaluated, we need to actually COMPUTE it
        cross_section_values = self.compute_cross_section(
                phi_values,
                lepton_helicity,
                target_polarization)

        # (X): Set the plot style using this method:
        self._set_plot_style()

        # (X): Initialize a figure for the plotting:
        cross_section_figure_instance = plt.figure(
            figsize = (8, 5))
        
        # (X): Now, add an axis on it:
        cross_section_axis_instance = cross_section_figure_instance.add_subplot(1, 1, 1)

        # (X): Construct the plot:
        cross_section_axis_instance.plot(
            phi_values,
            cross_section_values,
            color = 'black')
        
        # (X): Set the x-label of the plot:
        cross_section_axis_instance.set_xlabel(r"Azimuthal Angle $\phi$ (degrees)", fontsize = 14)

        # (X): Initialize an empty string that *may* be used to specify what *parts* of the cross-section
        # | are actually being computed.
        # | [NOTE]: For example, if the BH setting is `False` but the DVCS setting is `True`, then we will
        # | show d^{4}\sigma^{DVCS} to be super specific. This helps us figure out *what* cross-section
        # | specifically was calculated and plotted on a plot.
        cross_section_contribution_label = ""

        # (X): Only if *not all* of the contributions are on do we get pedantic in this if-nest:
        if not (self.bh_setting and self.dvcs_setting and self.interference_setting):

            # (X): If the BH^{2} setting is on...
            if self.bh_setting:

                # (X): ... add the "BH" label *with a space!*
                cross_section_contribution_label += "BH "

            # (X): If the DVCS^{2} setting is on...
            if self.dvcs_setting:

                # (X): ... add the "DVCS" label *with a space!*
                cross_section_contribution_label += "DVCS "

            # (X): If the I setting is on...
            if self.interference_setting:

                # (X): ... add the "I" label *with a space!*
                cross_section_contribution_label += "I"

            # (X): Set the y-label of the plot:
            cross_section_axis_instance.set_ylabel(
                rf"$\frac{{d^4\sigma^{{\text{{{cross_section_contribution_label}}}}}}}{{dQ^2 dx_B dt d\phi}}$ (nb/GeV$^4$)",
                fontsize = 14)

        # (X): If all the contributions to the cross-section are on...
        else:

            # (X): ... then we do *not* label the cross-section and immediately set the y-label of the plot:
            cross_section_axis_instance.set_ylabel(
                r"$\frac{{d^4\sigma}}{{dQ^2 dx_B dt d\phi}}$ (nb/GeV$^4$)",
                fontsize = 14)

        # (X): Turn the grid on:
        cross_section_axis_instance.grid(visible = True)

        # (X): Attempt to extract the kinematic inputs:
        try:

            # (X): Extract the *numerical* form of the kinematics:
            kinematics = self.kinematic_inputs

            # (X): Coerce them into a string:
            title_string = (
                rf"$Q^2 = {kinematics.squared_Q_momentum_transfer:.2f}$ GeV$^2$, "
                rf"$x_B = {kinematics.x_Bjorken:.2f}$, "
                rf"$t = {kinematics.squared_hadronic_momentum_transfer_t:.2f}$ GeV$^2$, "
                rf"$k = {kinematics.lab_kinematics_k:.2f}$ GeV")
            
            # (X): Obtain the CFF inputs as well to display:
            cff_string = (
                rf"$\mathcal{{H}} = {self.cff_inputs.compton_form_factor_h:.3f}$, "
                rf"$\widetilde{{\mathcal{{H}}}} = {self.cff_inputs.compton_form_factor_h_tilde:.3f}$, "
                rf"$\mathcal{{E}} = {self.cff_inputs.compton_form_factor_e:.3f}$, "
                rf"$\widetilde{{\mathcal{{E}}}} = {self.cff_inputs.compton_form_factor_e_tilde:.3f}$")
            
            # (X): We now use that string to set the plot title:
            cross_section_axis_instance.set_title(f"{title_string}\n{cff_string}", fontsize = 14)

        # (X): If there are errors in extracting the numbers and making the strings...
        except AttributeError:

            if self.verbose:
                print("> [VERBOSE]: Could not find full kinematics for title.")

            # (X): ... we just *don't* specify the numbers, but we still make the plot:
            cross_section_axis_instance.set_title(r"Differential Cross Section vs. $\phi$", fontsize = 14)

        # (X): Use a tight-layout:
        plt.tight_layout()

        # (X): (Using type-coercion!): If the user did not specify this...
        # | [NOTE]: An empty string is falsy: `"" == False` is True:
        if not save_plot_name:

            # (X): ... then just show the plot:
            plt.show()

        # (X): If the user has an idea for the name of the plot...
        else:

            # (X): ... then we save the plot with that name!
            cross_section_figure_instance.savefig(save_plot_name)

    def plot_bsa(
            self,
            phi_values,
            target_polarization,
            save_plot_name: str):
        """
        ## Description:
        Plot the BKM-predicted BSA with azimuthal angle φ.

        :param np.ndarray phi_values: Array of φ values (in degrees) at which to compute and plot the cross-section.

        :param str save_plot_name: If you want to save the plot, provide a non-empty string here.):
        """

        # (X): If it has *NOT* been evaluated, we need to actually COMPUTE it
        bsa_values = self.compute_bsa(phi_values, target_polarization)

        # (X): Set the plot style using our customziation method:
        self._set_plot_style()

        # (X): Initialize a figure for the plotting:
        bsa_figure_instance = plt.figure(
            figsize = (8, 5))
        
        # (X): Now, add an axis on it:
        bsa_axis_instance = bsa_figure_instance.add_subplot(1, 1, 1)

        # (X): Add the BSA curve on the plot:
        bsa_axis_instance.plot(
            phi_values,
            bsa_values,
            color = 'black')
        
        # (X): Add the x-label of the plot:
        bsa_axis_instance.set_xlabel(
            r"Azimuthal Angle $\phi$ (degrees)",
            fontsize = 14)
        
        # (X): Add the y-label of the plot:
        bsa_axis_instance.set_ylabel(
            r"$\frac{d^4\sigma \left( \lambda = +1 \right) - d^4\sigma \left( \lambda = -1 \right)}{d^4\sigma \left( \lambda = +1 \right) + d^4\sigma \left( \lambda = -1 \right)}$ (unitless)",
            fontsize = 14)
        
        # (X): Add a grid to the plot:
        bsa_axis_instance.grid(True)

        # (X): Attempt to extract the kinematic inputs:
        try:

            # (X): Extract the *numerical* form of the kinematics:
            kinematics = self.kinematic_inputs

            # (X): Coerce them into a string:
            title_string = (
                rf"$Q^2 = {kinematics.squared_Q_momentum_transfer:.2f}$ GeV$^2$, "
                rf"$x_B = {kinematics.x_Bjorken:.2f}$, "
                rf"$t = {kinematics.squared_hadronic_momentum_transfer_t:.2f}$ GeV$^2$, "
                rf"$k = {kinematics.lab_kinematics_k:.2f}$ GeV")
            
            # (X): Obtain the CFF inputs as well to display:
            cff_string = (
                rf"$\mathcal{{H}} = {self.cff_inputs.compton_form_factor_h:.3f}$, "
                rf"$\widetilde{{\mathcal{{H}}}} = {self.cff_inputs.compton_form_factor_h_tilde:.3f}$, "
                rf"$\mathcal{{E}} = {self.cff_inputs.compton_form_factor_e:.3f}$, "
                rf"$\widetilde{{\mathcal{{E}}}} = {self.cff_inputs.compton_form_factor_e_tilde:.3f}$")
            
            # (X): We now use that string to set the plot title:
            bsa_axis_instance.set_title(f"{title_string}\n{cff_string}", fontsize = 14)
            
        # (X): If there are errors in extracting the numbers and making the strings...
        except AttributeError:

            if self.verbose:
                print("> [VERBOSE]: Could not find full kinematics for title.")

            # (X): ... we just *don't* specify the numbers, but we still make the plot:
            bsa_axis_instance.set_title(r"Differential Cross Section vs. $\phi$", fontsize = 14)

        # (X): Use a tight-layout:
        plt.tight_layout()

        # (X): (Using type-coercion!): If the user did not specify this...
        # | Note: `"" == False` is True:
        if not save_plot_name:

            # (X): ... then just show the plot:
            plt.show()

        # (X): If the user has an idea for the name of the plot...
        else:

            # (X): ... then we save the plot with that name!
            bsa_figure_instance.savefig(save_plot_name)
        
    def plot_tsa(
            self,
            phi_values,
            lepton_helicity,
            save_plot_name: str):
        """
        ## Description:
        Plot the BKM-predicted TSA with azimuthal angle φ.

        :param np.ndarray phi_values: Array of φ values (in degrees) at which to compute and plot the cross-section.

        :param str save_plot_name: If you want to save the plot, provide a non-empty string here.:
        """


        # (X): If it has *NOT* been evaluated, we need to actually COMPUTE it
        tsa_values = self.compute_tsa(phi_values, lepton_helicity)

        # (X): Set the plot style using our customziation method:
        self._set_plot_style()

        # (X): Initialize a figure for the plotting:
        tsa_figure_instance = plt.figure(
            figsize = (8, 5))
        
        # (X): Now, add an axis on it:
        tsa_axis_instance = tsa_figure_instance.add_subplot(1, 1, 1)

        # (X): Add the BSA curve on the plot:
        tsa_axis_instance.plot(
            phi_values,
            tsa_values,
            color = 'black')
        
        # (X): Add the x-label of the plot:
        tsa_axis_instance.set_xlabel(
            r"Azimuthal Angle $\phi$ (degrees)",
            fontsize = 14)
        
        # (X): Add the y-label of the plot:
        tsa_axis_instance.set_ylabel(
            r"$\frac{d^4\sigma \left( \Lambda = +\frac{1}{2} \right) - d^4\sigma \left( \Lambda = -\frac{1}{2} \right)}{d^4\sigma \left( \Lambda = +\frac{1}{2} \right) + d^4\sigma \left( \Lambda = -\frac{1}{2} \right)}$ (unitless)",
            fontsize = 14)
        
        # (X): Add a grid to the plot:
        tsa_axis_instance.grid(True)

        # (X): Attempt to extract the kinematic inputs:
        try:

            # (X): Extract the *numerical* form of the kinematics:
            kinematics = self.kinematic_inputs

            # (X): Coerce them into a string:
            title_string = (
                rf"$Q^2 = {kinematics.squared_Q_momentum_transfer:.2f}$ GeV$^2$, "
                rf"$x_B = {kinematics.x_Bjorken:.2f}$, "
                rf"$t = {kinematics.squared_hadronic_momentum_transfer_t:.2f}$ GeV$^2$, "
                rf"$k = {kinematics.lab_kinematics_k:.2f}$ GeV")
            
            # (X): Obtain the CFF inputs as well to display:
            cff_string = (
                rf"$\mathcal{{H}} = {self.cff_inputs.compton_form_factor_h:.3f}$, "
                rf"$\widetilde{{\mathcal{{H}}}} = {self.cff_inputs.compton_form_factor_h_tilde:.3f}$, "
                rf"$\mathcal{{E}} = {self.cff_inputs.compton_form_factor_e:.3f}$, "
                rf"$\widetilde{{\mathcal{{E}}}} = {self.cff_inputs.compton_form_factor_e_tilde:.3f}$")
            
            # (X): We now use that string to set the plot title:
            tsa_axis_instance.set_title(f"{title_string}\n{cff_string}", fontsize = 14)
            
        # (X): If there are errors in extracting the numbers and making the strings...
        except AttributeError:

            if self.verbose:
                print("> [VERBOSE]: Could not find full kinematics for title.")

            # (X): ... we just *don't* specify the numbers, but we still make the plot:
            tsa_axis_instance.set_title(r"Target-Spin Asymmetry vs. $\phi$", fontsize = 14)

        # (X): Use a tight-layout:
        plt.tight_layout()

        # (X): (Using type-coercion!): If the user did not specify this...
        # | Note: `"" == False` is True:
        if not save_plot_name:

            # (X): ... then just show the plot:
            plt.show()

        # (X): If the user has an idea for the name of the plot...
        else:

            # (X): ... then we save the plot with that name!
            tsa_figure_instance.savefig(save_plot_name)

    def plot_dsa(
            self,
            phi_values,
            save_plot_name: str):
        """
        ## Description:
        Plot the BKM-predicted DSA with azimuthal angle φ.

        :param np.ndarray phi_values: Array of φ values (in degrees) at which to compute and plot the cross-section.

        :param str save_plot_name: If you want to save the plot, provide a non-empty string here.:
        """


        # (X): If it has *NOT* been evaluated, we need to actually COMPUTE it
        dsa_values = self.compute_dsa(phi_values)

        # (X): Set the plot style using our customziation method:
        self._set_plot_style()

        # (X): Initialize a figure for the plotting:
        dsa_figure_instance = plt.figure(
            figsize = (8, 5))
        
        # (X): Now, add an axis on it:
        dsa_axis_instance = dsa_figure_instance.add_subplot(1, 1, 1)

        # (X): Add the BSA curve on the plot:
        dsa_axis_instance.plot(
            phi_values,
            dsa_values,
            color = 'black')
        
        # (X): Add the x-label of the plot:
        dsa_axis_instance.set_xlabel(
            r"Azimuthal Angle $\phi$ (degrees)",
            fontsize = 14)
        
        # (X): Add the y-label of the plot:
        dsa_axis_instance.set_ylabel(
            r"$\frac{d^4\sigma \left( \Lambda = +\frac{1}{2} \right) - d^4\sigma \left( \Lambda = -\frac{1}{2} \right)}{d^4\sigma \left( \Lambda = +\frac{1}{2} \right) + d^4\sigma \left( \Lambda = -\frac{1}{2} \right)}$ (unitless)",
            fontsize = 14)
        
        # (X): Add a grid to the plot:
        dsa_axis_instance.grid(True)

        # (X): Attempt to extract the kinematic inputs:
        try:

            # (X): Extract the *numerical* form of the kinematics:
            kinematics = self.kinematic_inputs

            # (X): Coerce them into a string:
            title_string = (
                rf"$Q^2 = {kinematics.squared_Q_momentum_transfer:.2f}$ GeV$^2$, "
                rf"$x_B = {kinematics.x_Bjorken:.2f}$, "
                rf"$t = {kinematics.squared_hadronic_momentum_transfer_t:.2f}$ GeV$^2$, "
                rf"$k = {kinematics.lab_kinematics_k:.2f}$ GeV")
            
            # (X): Obtain the CFF inputs as well to display:
            cff_string = (
                rf"$\mathcal{{H}} = {self.cff_inputs.compton_form_factor_h:.3f}$, "
                rf"$\widetilde{{\mathcal{{H}}}} = {self.cff_inputs.compton_form_factor_h_tilde:.3f}$, "
                rf"$\mathcal{{E}} = {self.cff_inputs.compton_form_factor_e:.3f}$, "
                rf"$\widetilde{{\mathcal{{E}}}} = {self.cff_inputs.compton_form_factor_e_tilde:.3f}$")
            
            # (X): We now use that string to set the plot title:
            dsa_axis_instance.set_title(f"{title_string}\n{cff_string}", fontsize = 14)
            
        # (X): If there are errors in extracting the numbers and making the strings...
        except AttributeError:

            if self.verbose:
                print("> [VERBOSE]: Could not find full kinematics for title.")

            # (X): ... we just *don't* specify the numbers, but we still make the plot:
            dsa_axis_instance.set_title(r"Double-Spin Asymmetry vs. $\phi$", fontsize = 14)

        # (X): Use a tight-layout:
        plt.tight_layout()

        # (X): (Using type-coercion!): If the user did not specify this...
        # | Note: `"" == False` is True:
        if not save_plot_name:

            # (X): ... then just show the plot:
            plt.show()

        # (X): If the user has an idea for the name of the plot...
        else:

            # (X): ... then we save the plot with that name!
            dsa_figure_instance.savefig(save_plot_name)
