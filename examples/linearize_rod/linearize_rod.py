import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from cardillo import System
from cardillo.actuators import PDcontroller
from cardillo.discrete import RigidBody, Frame, Meshed, Box
from cardillo.constraints import Prismatic, RigidConnection
from cardillo.forces import Force
from cardillo.math import A_IB_basic, cross3, smoothstep2
from cardillo.solver import BackwardEuler, Newton
from cardillo.rods import (
    CircularCrossSection,
    RectangularCrossSection,
    CrossSectionInertias,
    Simo1986,
)
from cardillo.rods.cosseratRod import make_CosseratRod


def cantilever(Rod, nel):
    # create cardillo system
    system = System()

    # geometry of the rod
    length = 2  # [m]
    # cross section properties
    width = 0.1  # [m]
    height = width  # [m]
    density = 8.0e3  # [kg / m^3]
    cross_section = RectangularCrossSection(width, height)
    cross_section_inertias = CrossSectionInertias(density, cross_section)
    A = cross_section.area  # [m^2]
    Ip, Iy, Iz = np.diagonal(cross_section.second_moment)  # [m^4]

    # material properties
    E = 260.0e9  # [N / m^2]
    G = 100.0e9  # [N / m^2]

    # rod stiffness properties
    shear_corr = 5 / 6
    Ei = np.array([E * A, shear_corr * G * A, shear_corr * G * A])
    Fi = np.array([G * Ip, E * Iy, E * Iz])
    material_model = Simo1986(Ei, Fi)

    # semicircle
    if False:
        r_OP = lambda xi: np.array(
            [0.15 + dh / 2 * np.sin(xi), 0.0, dh / 2 - dh / 2 * np.cos(xi)]
        )
        L = np.pi * dh / 2
        xi1 = np.pi
    elif False:
        r_OP = lambda xi: np.array([0.15 + dh * np.cos(xi), 0.0, dh * np.sin(xi)])
        L = np.pi * dh / 2
        xi1 = np.pi / 2

    # make configurations
    # q0_rod = Rod.serret_frenet_configuration(nel, r_OP, None, None, xi1)
    Q_rod, u0_rod = Rod.straight_initial_configuration(nel, length)
    q0_rod = Q_rod

    rod = Rod(
        cross_section,
        material_model,
        nel,
        Q=Q_rod,
        q0=q0_rod,
        u0=u0_rod,
        cross_section_inertias=cross_section_inertias,
        name="Cosserat_rod",
    )

    system.add(rod)

    left_constraint = RigidConnection(rod, system.origin, xi1=0.0)
    system.add(left_constraint)
    right_constraint = RigidConnection(rod, system.origin, xi1=1.0)
    system.add(right_constraint)

    system.assemble()

    ######################
    # compute eigenmodes #
    ######################
    for c in [None, "NullSpace", "ComplianceProjection"]:
        omegas, modes_dq, sol_modes = system.eigenmodes(
            system.t0,
            system.q0,
            system.la_g0,
            system.la_gamma0,
            system.la_c0,
            constraints=c,
        )

        print(omegas)
        print(len(omegas))

    # vtk-export
    rod._export_dict["level"] = "NodalVolume"
    dir_name = Path(__file__).parent
    system.export(dir_name, f"vtk_modes", sol_modes, fps=25)


if __name__ == "__main__":
    nel = 10
    pDeg = 2
    Rod = make_CosseratRod(
        interpolation="Quaternion",
        mixed=True,
        polynomial_degree=pDeg,  # constraints=[1, 2, 3]
    )

    cantilever(Rod, nel)
