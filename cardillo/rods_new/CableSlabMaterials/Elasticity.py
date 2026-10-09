# SPDX-License-Identifier: MIT
# Copyright (c) 2011–2026 Joris J.C. Remmers

from typing import Tuple
from .BaseMaterial import BaseMaterial
import numpy as np


class Elasticity(BaseMaterial):
    """
    Isotropic linear elastic material model.

    This class implements a standard isotropic linear elastic material model
    using Hooke's law. The material behavior is fully characterized by two
    elastic constants: Young's modulus (E) and Poisson's ratio (nu).

    Attributes
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    H : ndarray
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
        self.H = np.zeros((6, 6))

        fac = 1.0 / (2.0 * self.nu * self.nu + self.nu - 1.0)

        self.H[0, 0] = fac * self.E * (self.nu - 1.0)
        self.H[0, 1] = -1.0 * fac * self.E * self.nu
        self.H[0, 2] = self.H[0, 1]
        self.H[1, 0] = self.H[0, 1]
        self.H[1, 1] = self.H[0, 0]
        self.H[1, 2] = self.H[0, 1]
        self.H[2, 0] = self.H[0, 1]
        self.H[2, 1] = self.H[0, 1]
        self.H[2, 2] = self.H[0, 0]
        self.H[3, 3] = self.E / (2.0 + 2.0 * self.nu)
        self.H[4, 4] = self.H[3, 3]
        self.H[5, 5] = self.H[3, 3]

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
        sigma = self.H @ kinematics.strain

        return sigma, self.H
