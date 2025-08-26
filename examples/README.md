This library is a little confusing. There are some examples in this folder that a user of the library can consult for a demonstration.

[NOTE]: Some of these examples will likely change their file names in the future!

1. `compute_coefficients.py`

[NOTE]: This script has not yet been written... It *will* be one that will enable you to see the numerical value of one of the *many* BKM10 coefficients.

2. `configuration.py`

How to compute the BKM10 cross-section using the "standard method" of "filling out the blanks" in a configuration dictionary.

3. `make_plots.py`

Hopefully, this script is self-de(script)ive. (Get it?) Follow this outline to learn how to make plots of the cross-section trend with $\phi$.

4. `tf_backend.py`

Some users want to us this library with TensorFlow. This script demonstrates how one might go about doing that in TensorFlow's *eager-execution mode only!*

5. `tf_cross_section_prediction.py`

An *incomplete* example of how one might use TensorFlow to predict the output of a cross-section provided the BKM10 inputs of the kinematics (`BKM10Inputs`) and (`CFFInputs`).