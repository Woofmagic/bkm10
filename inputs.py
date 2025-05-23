"""
"""

# (1): Import the specialized `dataclass` library:
from dataclasses import dataclass

@dataclass
class BKM10Inputs:
    # (X): Q^{2}: photon virtuality:
    squared_Q_momentum_transfer: float

    # (X): x_{B}: Bjorken's x:
    x_Bjorken: float

    # (X): t: hadron momentum transfer: (p - p')^{2}:
    squared_hadronic_momentum_transfer_t: float

    # (X): y: lepton energy fraction:
    lepton_energy_fraction_y: float

    # (X): Derived!
    epsilon: float

    # (X): ...
    lab_kinematics_k: float

    # (X): ...
    lepton_helicity: float

    # (X): ...
    target_polarization: float

    # (X): ...
    azimuthal_phi: float

    # (X): ...
    compton_form_factor_h: complex

    # (X): ...
    compton_form_factor_h_tilde: complex

    # (X): ...
    compton_form_factor_e: complex

    # (X): ...
    compton_form_factor_e_tilde: complex

    # (X): ...
    use_ww: bool = False

    # (X): ...
    verbose: bool = False