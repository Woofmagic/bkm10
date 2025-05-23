"""

"""

def validate_configuration(configuration_dictionary: dict, verbose: bool):
    """
    ## Description:
    Validate user's dict of initialization parameters.
    
    ## Parameters:
    configuration_dictionary (dict):
        a dict of user-requests DNN parameters that 
        will be used to attempt to build a TF network.

    verbose (boolean):
        Do you want to see all output of this function evaluation?

    ## Notes
    
    """
    
    try:
        kinematic_settings = int(configuration_dictionary["kinematics"])
        if not kinematic_settings:
            raise ValueError("> That is not acceptable.")
        
        print(kinematic_settings)
        
    except Exception as error:
        raise ValueError("> Bad configuration dictionary!") from error