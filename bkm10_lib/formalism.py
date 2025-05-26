"""

"""

from bkm10_lib.inputs import BKM10Inputs

from bkm10_lib.constants import _MASS_OF_PROTON_IN_GEV

import numpy as np

class BKMFormalism:

    def __init__(self, inputs: BKM10Inputs, formalism_version: str = "10", verbose: bool = False):

        # (X): Collect the inputs:
        self.kinematics = inputs

        # (X): Obtain the BKM formalism version (either 10 or 02):
        self.fomalism_version = formalism_version

        # (X): Define a verbose parameter:
        self.verbose = verbose

        # (X): Derived Quantity | Epsilon:
        self.epsilon = self._calculate_epsilon()

        # (X): Derived Quantity | y:
        self.lepton_energy_fraction = self._calculate_lepton_energy_fraction_y()

        # (X): Derived Quantity | xi:
        self.skewness_parameter = self._calculate_skewness_parameter()

        # (X): Derived Quantity | t_minimum:
        self.t_minimum = self._calculate_t_minimum()

        # (X): Derived Quantity | t':
        self.t_prime = self._calculate_t_prime()

        # (X): Derived Quantity | K_tilde:
        self.k_tilde = self._calculate_k_tilde()

        # (X): Derived Quantity | K:
        self.kinematic_k = self._calculate_k()

        # (X): Derived Quantity | k dot Delta:
        self.k_dot_delta = self._calculate_k_dot_delta()

    def _calculate_epsilon(self) -> float:
        """
        ## Description
        Calculate epsilon, which is just a ratio of kinematic quantities:
        \epsilon := 2 * m_{p} * x_{B} / Q

        ## Parameters:
        squared_Q_momentum_transfer: (float)
            kinematic momentum transfer to the hadron. 

        x_Bjorken: (float)
            kinematic Bjorken X

        verbose: (bool)
            Debugging console output.
        
        ## Notes:
        None!

        ## Examples:
        None!
        """
        squared_Q_momentum_transfer = self.kinematics.squared_Q_momentum_transfer
        x_Bjorken = self.kinematics.x_Bjorken

        try:

            # (1): Calculate Epsilon right away:
            epsilon = (2. * x_Bjorken * _MASS_OF_PROTON_IN_GEV) / np.sqrt(squared_Q_momentum_transfer)

            # (1.1): If verbose, print the result:
            if self.verbose:
                print(f"> Calculated epsilon to be:\n{epsilon}")

            # (2): Return Epsilon:
            return epsilon
        
        except Exception as ERROR:
            print(f"> Error in computing kinematic epsilon:\n> {ERROR}")
            return 0.0
        
    def _calculate_lepton_energy_fraction_y(self) -> float:
        """
        ## Description
        --------------
        Calculate y, which measures the lepton energy fraction.
        y^{2} := \frac{ \sqrt{Q^{2}} }{ \sqrt{\epsilon^{2}} k }

        Parameters
        --------------
        epsilon: (float)
            derived kinematics

        squared_Q_momentum_transfer: (float)
            Q^{2} momentum transfer to the hadron

        lab_kinematics_k: (float)
            lepton momentum loss

        verbose: (bool)
            Debugging console output.

        Notes
        --------------
        """
        lab_kinematics_k = self.kinematics.lab_kinematics_k
        squared_Q_momentum_transfer = self.kinematics.squared_Q_momentum_transfer
        epsilon = self.epsilon

        try:

            # (1): Calculate the y right away:
            lepton_energy_fraction_y = np.sqrt(squared_Q_momentum_transfer) / (epsilon * lab_kinematics_k)

            # (1.1): If verbose output, then print the result:
            if self.verbose:
                print(f"> Calculated y to be:\n{lepton_energy_fraction_y}")

            # (2): Return the calculation:
            return lepton_energy_fraction_y
        
        except Exception as ERROR:
            print(f"> Error in computing lepton_energy_fraction_y:\n> {ERROR}")
            return 0.

    def _calculate_skewness_parameter(self) -> float:
        """
        Description
        --------------
        Calculate the Skewness Parameter
        x_{i} = x_{B} * (1 + \frac{ t Q^{2} }{ 2 } ) ... FUCK OFF

        Parameters
        --------------
        squared_Q_momentum_transfer: (float)
            kinematic momentum transfer to the hadron

        x_Bjorken: (float)
            kinematic Bjorken X

        verbose: (bool)
            Debugging console output.
        

        Notes
        --------------
        """
        squared_hadronic_momentum_transfer_t = self.kinematics.squared_hadronic_momentum_transfer_t
        squared_Q_momentum_transfer = self.kinematics.squared_Q_momentum_transfer
        x_Bjorken = self.kinematics.x_Bjorken

        try:

            # (1): The Numerator:
            numerator = (1. + (squared_hadronic_momentum_transfer_t / (2. * squared_Q_momentum_transfer)))

            # (2): The Denominator:
            denominator = (2. - x_Bjorken + (x_Bjorken * squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer))

            # (3): Calculate the Skewness Parameter:
            skewness_parameter = x_Bjorken * numerator / denominator

            # (3.1): If verbose, print the output:
            if self.verbose:
                print(f"> Calculated skewness xi to be:\n{skewness_parameter}")

            # (4): Return Xi:
            return skewness_parameter
        
        except Exception as ERROR:
            print(f"> Error in computing skewness xi:\n> {ERROR}")
            return 0.
        
    def _calculate_t_minimum(self) -> float:
        """
        Description
        --------------
        Calculate t_{min}.

        Parameters
        --------------
        epsilon: (float)

        Returns
        --------------
        t_minimum: (float)
            t_minimum

        Notes
        --------------
        """
        epsilon = self.epsilon
        squared_Q_momentum_transfer = self.kinematics.squared_Q_momentum_transfer
        x_Bjorken = self.kinematics.x_Bjorken

        try:

            # (1): Calculate 1 - x_{B}:
            one_minus_xb = 1. - x_Bjorken

            # (2): Calculate the numerator:
            numerator = (2. * one_minus_xb * (1. - np.sqrt(1. + epsilon**2))) + epsilon**2

            # (3): Calculate the denominator:
            denominator = (4. * x_Bjorken * one_minus_xb) + epsilon**2

            # (4): Obtain the t minimum
            t_minimum = -1. * squared_Q_momentum_transfer * numerator / denominator

            # (4.1): If verbose, print the result:
            if self.verbose:
                print(f"> Calculated t_minimum to be:\n{t_minimum}")

            # (5): Print the result:
            return t_minimum

        except Exception as ERROR:
            print(f"> Error calculating t_minimum: \n> {ERROR}")
            return 0.    
    
    def _calculate_t_prime(self) -> float:
        """
        Description
        --------------
        Calculate t prime.

        Parameters
        --------------
        squared_hadronic_momentum_transfer_t: (float)

        squared_hadronic_momentum_transfer_t_minimum: (float)

        verbose: (float)

        Returns
        --------------
        t_prime: (float)

        Notes
        --------------
        """
        squared_hadronic_momentum_transfer_t = self.kinematics.squared_hadronic_momentum_transfer_t
        squared_hadronic_momentum_transfer_t_minimum = self.t_minimum

        try:

            # (1): Obtain the t_prime immediately
            t_prime = squared_hadronic_momentum_transfer_t - squared_hadronic_momentum_transfer_t_minimum

            # (1.1): If verbose, print the result:
            if self.verbose:
                print(f"> Calculated t prime to be:\n{t_prime}")

            # (2): Return t_prime
            return t_prime

        except Exception as ERROR:
            print(f"> Error calculating t_prime:\n> {ERROR}")
            return 0.
        
    def _calculate_k_tilde(self) -> float:
        """
        Description
        --------------

        Parameters
        --------------
        epsilon: (float)

        squared_Q_momentum_transfer: (float)

        x_Bjorken: (float)

        lepton_energy_fraction_y: (float)

        squared_hadronic_momentum_transfer_t: (float)

        squared_hadronic_momentum_transfer_t_minimum: (float)

        verbose: (bool)
            Debugging console output.

        Returns
        --------------
        k_tilde : (float)
            result of the operation
        
        Notes
        --------------
        """
        squared_hadronic_momentum_transfer_t = self.kinematics.squared_hadronic_momentum_transfer_t
        squared_Q_momentum_transfer = self.kinematics.squared_Q_momentum_transfer
        x_Bjorken = self.kinematics.x_Bjorken
        squared_hadronic_momentum_transfer_t_minimum = self.squared_hadronic_momentum_transfer_t_minimum
        epsilon = self.epsilon
        
        try:

            # (1): Calculate recurring quantity t_{min} - t
            tmin_minus_t = squared_hadronic_momentum_transfer_t_minimum - squared_hadronic_momentum_transfer_t

            # (2): Calculate the duplicate quantity 1 - x_{B}
            one_minus_xb = 1. - x_Bjorken

            # (3): Calculate the crazy root quantity:
            second_root_quantity = (one_minus_xb * np.sqrt((1. + epsilon**2))) + ((tmin_minus_t * (epsilon**2 + (4. * one_minus_xb * x_Bjorken))) / (4. * squared_Q_momentum_transfer))
            
            # (6): Calculate K_tilde
            k_tilde = np.sqrt(tmin_minus_t) * np.sqrt(second_root_quantity)

            # (6.1): Print the result of the calculation:
            if self.verbose:
                print(f"> Calculated k_tilde to be:\n{k_tilde}")

            # (7) Return:
            return k_tilde

        except Exception as ERROR:
            print(f"> Error in calculating K_tilde:\n> {ERROR}")
            return 0.
        
    def _calculate_k(self) -> float:
        """
        """
        lepton_energy_fraction_y = self.lepton_energy_fraction
        squared_Q_momentum_transfer = self.kinematics.squared_Q_momentum_transfer
        k_tilde = self.k_tilde
        epsilon = self.epsilon

        try:

            # (1): Calculate the amazing prefactor:
            prefactor = np.sqrt(((1. - lepton_energy_fraction_y + (epsilon**2 * lepton_energy_fraction_y**2 / 4.)) / squared_Q_momentum_transfer))

            # (2): Calculate the remaining part of the term:
            kinematic_k = prefactor * k_tilde

            # (2.1); If verbose, log the output:
            if self.verbose:
                print(f"> Calculated kinematic K to be:\n{kinematic_k}")

            # (3): Return the value:
            return kinematic_k

        except Exception as ERROR:
            print(f"> Error in calculating derived kinematic K:\n> {ERROR}")
            return 0.
        
    def _calculate_k_dot_delta(self) -> float:
        """
        Description
        --------------
        Equation (29) in the BKM Formalism, available
        at this link: https://arxiv.org/pdf/hep-ph/0112108.pdf

        Parameters
        --------------
        kinematic_k: (float)
        
        epsilon: (float)

        squared_Q_momentum_transfer: (float)

        x_Bjorken: (float)

        lepton_energy_fraction_y: (float)

        squared_hadronic_momentum_transfer_t: (float)

        azimuthal_phi: (float)

        verbose: (bool)
            Debugging console output.

        Returns
        --------------
        k_dot_delta_result : (float)
            result of the operation
        
        Notes
        --------------
        (1): k-dot-delta shows up in computing the lepton
            propagators. It is Eq. (29) in the following
            paper: https://arxiv.org/pdf/hep-ph/0112108.pdf
        """
        lepton_energy_fraction_y = self.lepton_energy_fraction
        squared_Q_momentum_transfer = self.kinematics.squared_Q_momentum_transfer
        squared_hadronic_momentum_transfer_t = self.kinematics.squared_hadronic_momentum_transfer_t
        x_Bjorken = self.kinematics.x_Bjorken
        kinematic_k = self.kinematic_k
        epsilon = self.epsilon

        try:
        
            # (1): The prefactor: \frac{Q^{2}}{2 y (1 + \varepsilon^{2})}
            prefactor = squared_Q_momentum_transfer / (2. * lepton_energy_fraction_y * (1. + epsilon**2))

            # (2): Second term in parentheses: Phi-Dependent Term: 2 K np.cos(\phi)
            phi_dependence = 2. * kinematic_k * np.cos(np.pi - convert_degrees_to_radians(azimuthal_phi))
            
            # (3): Prefactor of third term in parentheses: \frac{t}{Q^{2}}
            ratio_delta_to_q_squared = squared_hadronic_momentum_transfer_t / squared_Q_momentum_transfer

            # (4): Second term in the third term's parentheses: x_{B} (2 - y)
            bjorken_scaling = x_Bjorken * (2. - lepton_energy_fraction_y)

            # (5): Third term in the third term's parentheses: \frac{y \varepsilon^{2}}{2}
            ratio_y_epsilon = lepton_energy_fraction_y * epsilon**2 / 2.

            # (6): Adding up all the "correction" pieces to the prefactor, written as (1 + correction)
            correction = phi_dependence - (ratio_delta_to_q_squared * (1. - bjorken_scaling + ratio_y_epsilon)) + (ratio_y_epsilon)

            # (7): Writing it explicitly as "1 + correction"
            in_parentheses = 1. + correction

            # (8): The actual equation:
            k_dot_delta_result = -1. * prefactor * in_parentheses

            # (8.1): If verbose, print the output:
            if verbose:
                print(f"> Calculated k dot delta: {k_dot_delta_result}")

            # (9): Return the number:
            return k_dot_delta_result
        
        except Exception as E:
            print(f"> Error in calculating k.Delta:\n> {E}")
            return 0.