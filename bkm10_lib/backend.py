"""
The BKM10 formalism has been needed for machine-learning purposes.
Here, we provide the logic that facilitates usage of the library
in TensorFlow contexts.
"""

# 3rd Party Library | NumPy:
import numpy as _np

# 3rd Party Library | TensorFlow:
import tensorflow as _tf

# (X): Set the default backend "math computation" library to be NumPy:
_backend = "numpy"

def set_backend(backend):
    """
    ## Description:
    We provide an interface for the user to inform the library
    what kind of math computation is required.
    """

    # (X): In order for this to work, we need to make the backend option global:
    global _backend

    # (X): If a provided backend value does not correspond to NumPy or TensorFlow..
    if backend not in ["numpy", "tensorflow"]:

        # (X): ... raise a Value error demanding the *value* be NP or TF:
        raise ValueError("Backend must be 'numpy' or 'tensorflow'")
    
    # (X): Otherwise, we go ahead and set the backend to whatever it is:
    _backend = backend

def get_backend():
    """
    ## Description:
    Now that we have provided the opportunity change the backend setting,
    we need to be able to find what it is currently set to.
    """

    # (X): Return the backend variable:
    return _backend

class MathWrapper:
    """
    ## Description:
    The `MathWrapper` class helps us easily pass between computing
    values with NumPy or TensorFlow.
    """
    def __getattr__(self, name):
        """
        ## Description:
        In order to handle equivalent math operations between NumPy and
        TensorFlow, we need to be able to *map* (bjectively) between the
        two libraries. So, we need this wrapper to handle this check: 
        Whenever a "math attribute" is used, we have to check if we're
        going to use NumPy or TensorFlow based on the backend setting and
        then evaluate it accordingly.

        ## Notes:
        The main reason we have this --- perhaps the *only* reason --- is 
        because raising things to powers in NumPy is done with np.power() but
        it is tf.pow() in TensorFlow... Thanks, Obama!
        """

        # (X): Sly trick expose either NP or TF computing based on backkend setting:
        mod = _np if _backend == "numpy" else _tf

        # (X): The major exception is that np.power() is equivalent to tf.pow()... Annoying!
        if name == "power" and _backend == "tensorflow":
            return _tf.pow
       
        # (X): Return wha
        return getattr(mod, name)

math = MathWrapper()
