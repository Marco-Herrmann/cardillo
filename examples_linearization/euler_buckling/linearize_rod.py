import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from cardillo import System
from cardillo.actuators import PDcontroller
from cardillo.discrete import RigidBody, Frame, Meshed, Box
from cardillo.constraints import Prismatic, RigidConnection, Revolute
from cardillo.constraints._base import ProjectedPositionOrientationBase
from cardillo.forces import Force
from cardillo.math import A_IB_basic, cross3, smoothstep2, Exp_SO3_quat, e3
from cardillo.solver import BackwardEuler, Newton, Eigenmodes
from cardillo.rods import (
    CircularCrossSection,
    RectangularCrossSection,
    CrossSectionInertias,
    Simo1986,
)
from cardillo.rods.cosseratRod import make_CosseratRod


def cantilever(Rod, nel, case=1):
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

    # constraint at the left end
    if case in [1, 3, 4]:
        constraints = [RigidConnection(rod, system.origin, xi1=0)]
    elif case == 2:
        constraints = [Revolute(rod, system.origin, axis=1, xi1=0)]
    else:
        raise NotImplementedError

    # constraint at the right end (only the relevant direction for planar consideration)
    if case != 1:
        if case in [2, 3]:
            c_axis = []
        elif case in [4]:
            c_axis = [(0, 2)]

        constraints.append(
            ProjectedPositionOrientationBase(
                rod,
                system.origin,
                [2],
                c_axis,
                xi1=1.0,
            )
        )

    system.add(*constraints)

    # loading
    beta = {1: 2.0, 2: 1.0, 3: 0.699156, 4: 0.5}
    load_factor = 1.01
    F = (
        lambda t: np.array([-np.pi**2 / (beta[case] * length) ** 2 * Fi[1], 0.0, 0.0])
        * t
        * load_factor
    )
    force = Force(F, rod, 1.0)
    system.add(force)

    system.assemble()

    ################
    # solve static #
    ################
    n_steps = 100
    solver_static = Newton(system, n_steps)
    sol = solver_static.solve()

    ######################
    # compute eigenmodes #
    ######################
    solver = Eigenmodes(system, sol)

    nom = 1
    omega_s = np.zeros((n_steps + 1, nom))
    for i in range(n_steps + 1):
        print(f"step: {i: >3}, load factor: {load_factor * i / n_steps:3f}")

        omegas, modes_dq, sol_modes = solver.solve(i)
        # print(f"     {omegas[0]:.5f}")
        omega_s[i] = omegas[:nom]
    # print(omegas)
    # print(len(omegas))
    # omegas, modes_dq, sol_modes = solver.solve(-1)

    fig, ax = plt.subplots(1, 1, squeeze=False)
    [
        ax[0, 0].plot(np.linspace(0, load_factor, n_steps + 1), omega_s[:, i], "-")
        for i in range(nom)
    ]
    ax[0, 0].grid()
    plt.show()

    # vtk-export
    rod._export_dict["level"] = "NodalVolume"
    dir_name = Path(__file__).parent
    system.export(dir_name, f"vtk", sol, fps=25)
    system.export(dir_name, f"vtk_modes", sol_modes, fps=25)


if __name__ == "__main__":
    nel = 8
    pDeg = 3
    rod = "T"
    # rod = "EB"
    # rod = "IEB"

    if rod == "T":
        constraints = [1, 3, 5]
    elif rod == "EB":
        constraints = [1, 2, 3, 5]
    elif rod == "IEB":
        constraints = [0, 1, 2, 3, 5]
    Rod = make_CosseratRod(
        interpolation="Quaternion",
        mixed=True,
        polynomial_degree=pDeg,
        constraints=constraints,
    )

    cantilever(Rod, nel, 3)
