# core.py

from bkm10_lib.validation import validate_configuration

import numpy as np

class DifferentialCrossSection:
    """
    
    """

    def __init__(self, configuration = None, verbose = False, debugging = False):
        
        # (X): Obtain a True/False to operate the calculation in:
        self.configuration_mode = configuration is not None

        # (X): Determine verbose mode:
        self.verbose = verbose

        # (X): Determine debugging mode (DO NOT TURN ON!):
        self.debugging = debugging

        # (X): Hidden data that tells us if the functions executed correctly:
        self._evaluated = False

        # (X): A dictionary of *every coefficient* that we computed:
        self.coefficients = {}

        if verbose:
            print(f"> Verbose mode on.")

        if debugging:
            print(f"> Debugging mode is on — DO NOT USE THIS!")

        if configuration:
            if verbose:
                print(f"> Configuration dictionary received!")

            if debugging:
                print(f"> Configuration dictionary received:\n{configuration}")
            
            # (X): Initialize the class from the dictionary:
            self._initialize_from_config(configuration)

    def _initialize_from_config(self, configuration_dict: dict):
        try:

            # (X): Pass the dictionary into the validation function:
            validated = validate_configuration(configuration_dict, self.verbose)

        except Exception as error:

            # (X): Too general, yes, but not sure what we put here yet:
            raise Exception("> Error occurred during validation...") from error

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
        if not hasattr(self, 'inputs'):

            # (X): ...enforce the class to 
            raise RuntimeError("> Must initialize with configuration before evaluation.")

        # (X): If the user wants some confirmation that stuff is good...
        if self.verbose:

            # (X): ... we simply evaluate the length of the phi array:
            print(f"> Evaluating cross-section at {len(phi_values)} phi points.")

        # (X): If the user wants to see everything...
        if self.debugging:

            # (X): ... we give it to them:
            print(f"> Evaluating cross-section with phi values of:\n> {phi_values}")

        # (X): Verify that the array of angles is at least 1D:
        verified_phi_values = np.atleast_1d(phi_values)

        # (X): Obtain coefficients:
        coefficient_c_0 = 0.
        coefficient_c_1 = 0.

        # (X): Compute the dfferential cross-section:
        differential_cross_section = coefficient_c_0 + coefficient_c_1 * np.cos(verified_phi_values)

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