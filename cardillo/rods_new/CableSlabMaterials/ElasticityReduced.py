# SPDX-License-Identifier: MIT
# Copyright (c) 2011–2026 Joris J.C. Remmers

from typing import Tuple
from .BaseMaterial import BaseMaterial
import numpy as np


class Elasticity(BaseMaterial):
    """
    Isotropic linear elastic material model.

    This class implements a isotropic linear elastic material model for beam elements
    using Hooke's law. The material behavior is fully characterized by two
    elastic constants: Young's modulus (E) and Poisson's ratio (nu). Transverse normal
    stresses and cross-secional shear stresses are assumed to be zero.

    Attributes
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    C : ndarray
        6x6 elastic stiffness matrix (Hookean matrix).

    Notes
    -----
    The stress and strain vectors follow Voigt notation:
    [σ11, σ22, σ33, σ23, σ13, σ12]
    """

    def __init__(self, E, nu) -> None:
        """
        Initialize the isotropic material model.

        Constructs the 6x6 elastic stiffness matrix based on Young's modulus
        and Poisson's ratio.

        Parameters
        ----------

        E : float
            Young's modulus.
        nu : float.
            Poisson's ratio'

        """

        # Call the BaseMaterial constructor
        super().__init__()

        self.E = E
        self.nu = nu

        # Create the hookean matrix
        self.G = self.E / (2.0 * (1.0 + self.nu))
        self.C = np.zeros((6, 6))

        self.C[0, 0] = self.E
        self.C[4, 4] = self.G
        self.C[5, 5] = self.G

    # ---------------------------------------------------------------------------
    #
    # ---------------------------------------------------------------------------

    def getStress(
        self, kinematics, actions, element, intpointID
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute stress and material tangent matrix.

        Parameters
        ----------
        kinematics : object
            Kinematics object containing strain information. Must have:
            - strain : ndarray
                Total strain vector (for total formulation).

        Returns
        -------
        sigma : ndarray
            Stress vector in Voigt notation [σ11, σ22, σ33, σ23, σ13, σ12].
        H : ndarray
            Material tangent stiffness matrix (6x6).
        """
        sigma = self.C @ kinematics.strain

        return sigma, self.C
