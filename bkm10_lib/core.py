# core.py

from bkm10_lib.validation import validate_configuration

from bkm10_lib.formalism import BKMFormalism

import numpy as np

import matplotlib.pyplot as plt

class DifferentialCrossSection:
    """
    Welcome to the `DifferentialCrossSection` class!

    ## Description:
    Compute BKM10 differential cross sections using user-defined inputs.

    ## Parameters
    configuration : dict
        A dictionary containing the configuration settings with the following keys:
        
        - "kinematics" : BKM10Inputs
            Dataclass containing the required kinematic variables.
        
        - "cff_inputs" : Any
            Object or dictionary containing Compton Form Factor values or parameters.
        
        - "target_polarization" : float
            Polarization value for the target (e.g., 0 for unpolarized).
        
        - "lepton_beam_polarization" : float
            Polarization of the lepton beam (e.g., +1 or -1).

    verbose : bool
        A boolean flag that will tell the class to print out various messages at
        intermediate steps in the calculation. Useful if you want to determine when
        you have, say, calculated a given coefficient, like C_{++}^{LP}(n = 1).
    
    debugging : bool
        A boolean flag that will bomb anybody's terminal with output. As the flag is
        entitled, DO NOT USE THIS unless you need to do some serious debugging. We are
        talking about following how the data gets transformed through every calculation.
    """

    def __init__(self, configuration = None, verbose = False, debugging = False):
        
        # (X): Obtain a True/False to operate the calculation in:
        self.configuration_mode = configuration is not None

        # (X): Determine verbose mode:
        self.verbose = verbose

        # (X): Determine debugging mode (DO NOT TURN ON!):
        self.debugging = debugging

        # (X): A dictionary of *every coefficient* that we computed:
        self.coefficients = {}

        # (X): Hidden data that says if configuration passed:
        self._passed_configuration = False

        # (X): Hidden data that tells us if the functions executed correctly:
        self._evaluated = False

        if verbose:
            print(f"> [VERBOSE]: Verbose mode on.")
        if debugging:
            print(f"> [DEBUGGING]: Debugging mode is on — DO NOT USE THIS!")

        if configuration:
            if verbose:
                print(f"> [VERBOSE]: Configuration dictionary received!")
            if debugging:
                print(f"> [DEBUGGING]:Configuration dictionary received:\n{configuration}")

            try:
                if debugging:
                    print(f"> [DEBUGGING]: Trying to initialize configuration...")
            
                # (X): Initialize the class from the dictionary:
                self._initialize_from_config(configuration)

                if debugging:
                    print(f"> [DEBUGGING]: Configuration passed!")

            except:
                raise Exception("> Unable to initialize configuration!")
            
            self._passed_configuration = True

            if verbose:
                print(f"> [VERBOSE]: Configuration succeeded!")
            if debugging:
                print(f"> [DEBUGGING]: Configuration succeeded! Now set internal attribute: {self._passed_configuration}")

    @staticmethod
    def _set_plot_style():
        
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
        })

        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['xtick.major.size'] = 5
        plt.rcParams['xtick.major.width'] = 0.5
        plt.rcParams['xtick.minor.size'] = 2.5
        plt.rcParams['xtick.minor.width'] = 0.5
        plt.rcParams['xtick.minor.visible'] = True
        plt.rcParams['xtick.top'] = True    

        # Set y axis
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['ytick.major.size'] = 5
        plt.rcParams['ytick.major.width'] = 0.5
        plt.rcParams['ytick.minor.size'] = 2.5
        plt.rcParams['ytick.minor.width'] = 0.5
        plt.rcParams['ytick.minor.visible'] = True
        plt.rcParams['ytick.right'] = True

    def _initialize_from_config(self, configuration_dictionary: dict):
        try:

            # (X): Pass the dictionary into the validation function:
            validated_configuration_dictionary = validate_configuration(configuration_dictionary, self.verbose)

            self.kinematic_inputs = validated_configuration_dictionary["kinematics"]

            self.cff_inputs = validated_configuration_dictionary["cff_inputs"]

            self.target_polarization = validated_configuration_dictionary["target_polarization"]

            self.lepton_polarization = validated_configuration_dictionary["lepton_beam_polarization"]

            self.formalism = BKMFormalism(self.kinematic_inputs, verbose = self.verbose)

        except Exception as error:

            # (X): Too general, yes, but not sure what we put here yet:
            raise Exception("> Error occurred during validation...") from error
        
    def compute_c0_coefficient(self, phi_values: np.ndarray) -> np.ndarray:
        """
        """
        if not hasattr(self, "formalism"):
            raise RuntimeError("> Formalism not initialized. Make sure configuration is valid.")

        return self.formalism.calculate_c0_coefficinent(phi_values)
    
    def compute_c1_coefficient(self, phi_values: np.ndarray) -> np.ndarray:
        """
        """
        if not hasattr(self, "formalism"):
            raise RuntimeError("> Formalism not initialized. Make sure configuration is valid.")

        return self.formalism.calculate_c1_coefficient(phi_values)
    
    def compute_c2_coefficient(self, phi_values: np.ndarray) -> np.ndarray:
        """
        """
        if not hasattr(self, "formalism"):
            raise RuntimeError("> Formalism not initialized. Make sure configuration is valid.")

        return self.formalism.calculate_c2_coefficinent(phi_values)
    
    def compute_c3_coefficient(self, phi_values: np.ndarray) -> np.ndarray:
        """
        """
        if not hasattr(self, "formalism"):
            raise RuntimeError("> Formalism not initialized. Make sure configuration is valid.")

        return self.formalism.calculate_c3_coefficinent(phi_values)
    
    def compute_s1_coefficient(self, phi_values: np.ndarray) -> np.ndarray:
        """
        """
        if not hasattr(self, "formalism"):
            raise RuntimeError("> Formalism not initialized. Make sure configuration is valid.")

        return self.formalism.calculate_s1_coefficinent(phi_values)
    
    def compute_s2_coefficient(self, phi_values: np.ndarray) -> np.ndarray:
        """
        """
        if not hasattr(self, "formalism"):
            raise RuntimeError("> Formalism not initialized. Make sure configuration is valid.")

        return self.formalism.calculate_s2_coefficinent(phi_values)
    
    def compute_s3_coefficient(self, phi_values: np.ndarray) -> np.ndarray:
        """
        """
        if not hasattr(self, "formalism"):
            raise RuntimeError("> Formalism not initialized. Make sure configuration is valid.")

        return self.formalism.calculate_s3_coefficinent(phi_values)

    def compute_cross_section(self, phi_values: np.ndarray) -> np.ndarray:
        """
        ## Description:
        We compute the four-fold *differential cross-section* as 
        described with the BKM10 Formalism.

        ## Arguments:
        
        phi: np.ndarray
            A NumPy array that will be plugged-and-chugged into the BKM10 formalism.
        """

        # (X): If  the user has not filled in the class inputs...
        if not hasattr(self, 'kinematic_inputs'):

            # (X): ...enforce the class to 
            raise RuntimeError("> Missing 'kinematic_inputs' configuration before evaluation.")

        # (X): If the user wants some confirmation that stuff is good...
        if self.verbose:

            # (X): ... we simply evaluate the length of the phi array:
            print(f"> [VERBOSE]: Evaluating cross-section at {len(phi_values)} phi points.")

        # (X): If the user wants to see everything...
        if self.debugging:

            # (X): ... we give it to them:
            print(f"> [DEBUGGING]: Evaluating cross-section with phi values of:\n> {phi_values}")

        # (X): Verify that the array of angles is at least 1D:
        verified_phi_values = np.atleast_1d(phi_values)

        # (X): Obtain coefficients:
        coefficient_c_0 = self.compute_c0_coefficient()
        coefficient_c_1 = self.compute_c1_coefficient()
        coefficient_c_2 = self.compute_c2_coefficient()
        coefficient_c_3 = self.compute_c3_coefficient()
        coefficient_s_1 = self.compute_s1_coefficient()
        coefficient_s_2 = self.compute_s2_coefficient()
        coefficient_s_3 = self.compute_s3_coefficient()

        # (X): Compute the dfferential cross-section:
        differential_cross_section = (
            coefficient_c_0 + 
            coefficient_c_1 * np.cos(verified_phi_values) +
            coefficient_c_2 * np.cos(verified_phi_values) +
            coefficient_c_3 * np.cos(verified_phi_values) +
            coefficient_s_1 * np.sin(verified_phi_values) + 
            coefficient_s_2 * np.sin(verified_phi_values) +
            coefficient_s_3 * np.sin(verified_phi_values))

        # (X): Store cross-section data as class attribute:
        self.cross_section_values = differential_cross_section

        # (X): The class has now evaluated:
        self._evaluated = True

        # (X): Return the cross section:
        return differential_cross_section
    
    def get_coefficient(self, name: str) -> np.ndarray:
        """
        ## Description:
        An interface to query a given BKM coefficient
        """

        # (X): ...
        if not self._evaluated:

            # (X): ...
            raise RuntimeError("Call `evaluate(phi)` first before accessing coefficients.")
        
        # (X): In case there is an issue:
        try:
            
            # (X): Return the coefficient:
            return self.coefficients.get(name, None)
        
        # (X): Catch general exceptions:
        except Exception as exception:

            # (X): Raise an error:
            raise NotImplementedError(f"> Something bad happened...: {exception}")
        
    def plot_cross_section(self, phi_values: np.ndarray) -> np.ndarray:
        """
        ## Description:
        Plot the four-fold differential cross-section as a function of azimuthal angle φ.

        ## Arguments:
        phi_values : np.ndarray
            Array of φ values (in degrees) at which to compute and plot the cross-section.
        """

        # (X): We need to check if the cross-section has been evaluated yet:
        if not self._evaluated:
            if self.verbose:
                print("> [VERBOSE]: No precomputed cross-section found. Computing now...")
            if self.debugging:
                print("> [DEBUGGING]: No precomputed cross-section found. Computing now...")

            self.cross_section_values = self.compute_cross_section(phi_values)

        else:
            if self.verbose:
                print("> [VERBOSE]: Found cross-section data... Now constructing plots.")

        self._set_plot_style()

        cross_section_figure_instance, cross_section_axis_instance = plt.subplots(figsize = (8, 5))

        cross_section_axis_instance.plot(phi_values, self.cross_section_values, color = 'black')
        cross_section_axis_instance.set_xlabel(r"Azimuthal Angle $\phi$ (degrees)", fontsize = 14)
        cross_section_axis_instance.set_ylabel(r"$\frac{d^4\sigma}{dQ^2 dx_B dt d\phi}$ (nb/GeV$^4$)", fontsize = 14)
        cross_section_axis_instance.grid(True)
        # cross_section_axis_instance.legend(fontsize = 12)

        try:
            kinematics = self.kinematic_inputs

            title_string = (
                rf"$Q^2 = {kinematics.squared_Q_momentum_transfer:.2f}$ GeV$^2$, "
                rf"$x_B = {kinematics.x_Bjorken:.2f}$, "
                rf"$t = {kinematics.squared_hadronic_momentum_transfer_t:.2f}$ GeV$^2$, "
                rf"$k = {kinematics.lab_kinematics_k:.2f}$ GeV"
                )
            
            cross_section_axis_instance.set_title(title_string, fontsize = 14)

        except AttributeError:

            if self.verbose:
                print("> Could not find full kinematics for title.")

            cross_section_axis_instance.set_title("Differential Cross Section vs. $\phi$", fontsize=14)

        plt.tight_layout()
        plt.show()