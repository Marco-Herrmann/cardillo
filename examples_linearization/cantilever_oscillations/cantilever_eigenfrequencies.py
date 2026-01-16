from math import pi
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import scipy

from cardillo import System
from cardillo.rods import (
    RectangularCrossSection,
    CrossSectionInertias,
    Simo1986,
)
from cardillo.rods.cosseratRod import make_CosseratRod

""" Eigenfrequencies of shear-deformable cantilever: 
This example shows for pinned-pinned boundary conditions how many 
of the first 10 eigenfrequencies are given within a 1 percentage 
deviation from the analytical solutions.

Analytical solutions for the shear-bending deflections are from
"On the whole spectrum of Timoshenko beams. Part I: a theoretical 
revisitation", Cazzani, A., Stockino, F. and Turco, E., 2016."""


def cantilever(
    Rod,  # TODO: add type hint
    *,
    nelements: int = 5,
    polynomial_degree: int = 3,
    reduced_integration: bool = True,
):
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

    # first 10 longitudinal modes: pinned - pinned aka clamped - clamped
    omega_long = (np.arange(10) + 1) * np.pi * np.sqrt(E / density) / (length)

    # first 10 torsional modes: pinned - pinned aka clamped - clamped
    omega_tors = (np.arange(10) + 1) * np.pi * np.sqrt(G / density) / (length)

    # first 10 bending modes from Cazzani 2016: pinned - pinned
    omega_bend = np.array(
        [
            404.3540829,
            1597.560957,
            3524.348082,
            6104.920320,
            9247.993743,
            12861.93645,
            16862.12383,
            21174.58318,
            25736.94981,
            30497.85749,
        ]
    )

    # construct system
    system = System()

    # compute straight initial configuration of cantilever
    q0 = Rod.straight_configuration(
        nelements, length, polynomial_degree=polynomial_degree
    )

    # construct cantilever
    cantilever = Rod(
        cross_section,
        material_model,
        nelements,
        Q=q0,
        q0=q0,
        polynomial_degree=polynomial_degree,
        reduced_integration=reduced_integration,
        cross_section_inertias=cross_section_inertias,
    )

    # assemble the system
    system.add(cantilever)

    system.assemble()

    # initial position and velocity coordinates
    q0 = system.q0
    u0 = system.u0

    # define free degrees of freedom
    # pinned - pinned
    elDOF_u_constrained = np.concatenate(
        [
            cantilever.elDOF_r_u[0, cantilever.nodalDOF_element_r_u[0]],
            cantilever.elDOF_r_u[-1, cantilever.nodalDOF_element_r_u[-1]],
        ]
    )
    elDOF_u_free = np.setdiff1d(cantilever.elDOF_u, elDOF_u_constrained)

    # compute linearized system equations
    M = system.M(0, system.q0).todense()
    M = M[elDOF_u_free[:, None], elDOF_u_free]
    h_q = system.h_q(0, q0, u0).todense()
    B = system.q_dot_u(0, q0).todense()
    K = -h_q @ B
    K = K[elDOF_u_free[:, None], elDOF_u_free]

    # determine eigenfrequencies
    eigenValues, eigenVectors = scipy.linalg.eig(K, b=M)
    eigenValues = np.real(np.sqrt(eigenValues))

    # search for indices for longitudinal, torsional and bending frequencies
    idx_long = []
    idx_tors = []
    idx_bend = []

    rel_err = 0.1  # [in percent %]
    upper_bound = 1 + 0.5 * rel_err / 100
    lower_bound = 1 - 0.5 * rel_err / 100

    for i in range(10):
        idx_long.append(
            np.where(
                np.logical_and(
                    eigenValues >= omega_long[i] * lower_bound,
                    eigenValues <= omega_long[i] * upper_bound,
                )
            )
        )
        idx_tors.append(
            np.where(
                np.logical_and(
                    eigenValues >= omega_tors[i] * lower_bound,
                    eigenValues <= omega_tors[i] * upper_bound,
                )
            )
        )
        idx_bend.append(
            np.where(
                np.logical_and(
                    eigenValues >= omega_bend[i] * lower_bound,
                    eigenValues <= omega_bend[i] * upper_bound,
                )
            )
        )

    print(f"eigenfrequencies with relative error <= {rel_err}%")
    print("-------------------")
    print("longitudinal modes:")
    for i in range(10):
        print(
            f"{i+1}: mode_ana: {omega_long[i]},  mode_num: {eigenValues[idx_long[i]]}"
        )
    print("-------------------")
    print("torsional modes:")
    for i in range(10):
        print(
            f"{i+1}: mode_ana: {omega_tors[i]},  mode_num: {eigenValues[idx_tors[i]]}"
        )
    print("-------------------")
    print("bending modes:")
    for i in range(10):
        print(
            f"{i+1}: mode_ana: {omega_bend[i]},  mode_num: {eigenValues[idx_bend[i]]}"
        )
    print("-------------------")


if __name__ == "__main__":
    cantilever(
        Rod=make_CosseratRod(interpolation="Quaternion", mixed=False),
        nelements=3,
        polynomial_degree=3,
        reduced_integration=True,
    )
