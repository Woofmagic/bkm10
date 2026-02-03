"""
Basic validation function for the `DifferentialCrossSection` library.
"""

from bkm10_lib.inputs import BKM10Inputs
from bkm10_lib.cff_inputs import CFFInputs

def validate_configuration(
        configuration_dictionary: dict,
        verbose: bool):
    """
    ## Description:
    Validate user's dict of initialization parameters.
    
    :param dict configuration_dictionary:
        Dictionary of configuration parameters/inputs required
        to evaluate d^{4}\sigma in the BKM10 formalism.

    :param bool verbose: 
        Do you want to see all output of this function evaluation?

    ## Notes:
        1. 2026/02/03
            - Removed `lepton_helicity` and `target_polarization`.
    """

    required_keys = [
        "kinematics",
        "cff_inputs",
        ]
    
    # (X): ...
    if verbose:

        # (X): ...
        print("> [VERBOSE]: Now iterating over required keys...")
        
    for key in required_keys:
        if key not in configuration_dictionary:
            raise ValueError(f"Missing required key in config: {key}")
    
    kinematic_settings = configuration_dictionary["kinematics"]

    # (X): ...
    if verbose:

        # (X): ...
        print("> [VERBOSE]: Extracted kinematics from configuration dictionary.")

    if not kinematic_settings:
        raise ValueError("> Missing 'kinematics' key.")

    if not isinstance(kinematic_settings, BKM10Inputs):
        raise TypeError("> 'kinematics' key must be a BKM10Inputs instance.")

    cff_settings = configuration_dictionary["cff_inputs"]

    # (X): ...
    if verbose:

        # (X): ...
        print("> [VERBOSE]: Extracted CFF settings from configuration dictionary.")

    if not cff_settings:
        raise ValueError("> Missing 'cff_inputs' key.")

    if not isinstance(cff_settings, CFFInputs):
        raise TypeError("> 'cff_inputs' key must be a CFFInputs instance.")
    
    return configuration_dictionary