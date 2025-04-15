import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from cardillo import System
from cardillo.actuators import PDcontroller
from cardillo.discrete import RigidBody, Frame, Meshed, Box
from cardillo.constraints import Prismatic, RigidConnection
from cardillo.constraints._base import ProjectedPositionOrientationBase
from cardillo.forces import Force
from cardillo.math import A_IB_basic, cross3, smoothstep2, Exp_SO3_quat, e3
from cardillo.solver import BackwardEuler, Newton
from cardillo.rods import (
    CircularCrossSection,
    RectangularCrossSection,
    CrossSectionInertias,
    Simo1986,
)
from cardillo.rods.cosseratRod import make_CosseratRod


def consistent_constraints(sys, rod, constraints):
    rod_impressed = rod.idx_impressed
    # constraint left side (xi=0)
    g_pos0 = [0, 1, 2]
    g_rot0 = [(1, 2), (2, 0), (0, 1)]

    # constraint at the right side (xi=1) without singularity
    # TODO: these 6 if statements are only valid for rigid-rigid!
    g_pos1 = []
    if 0 in rod_impressed:
        g_pos1.append(0)
    if (1 in rod_impressed) or (5 in rod_impressed):
        g_pos1.append(1)
    if (2 in rod_impressed) or (4 in rod_impressed):
        g_pos1.append(2)

    g_rot1 = []
    if 3 in rod_impressed:
        g_rot1.append((1, 2))
    if 4 in rod_impressed:
        g_rot1.append((2, 0))
    if 5 in rod_impressed:
        g_rot1.append((0, 1))

    g_pos = [g_pos0, g_pos1]
    g_rot = [g_rot0, g_rot1]
    cs = []
    for i, (g_p, g_r, des) in enumerate(zip(g_pos, g_rot, constraints)):
        if des == "free":
            g_r = []
            g_p = []
        elif des == "rigid":
            pass
        elif des == "rot_z":
            if (0, 1) in g_r:
                g_r.remove((0, 1))
        else:
            raise RuntimeError(des, i)

        if len(g_p) > 0 or len(g_r) > 0:
            cs.append(
                ProjectedPositionOrientationBase(rod, sys.origin, g_p, g_r, xi1=i)
            )

    return cs


def cantilever(Rod, nel, constraints=["free", "free"]):
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

    P = np.array([1, 2, 3, 4.1])
    # A_IB = Exp_SO3_quat(P, normalize=True)
    # A_IB = Exp_SO3_quat(2 * np.random.rand(4) - 1, normalize=True)
    A_IB = np.eye(3)
    # make configurations
    # q0_rod = Rod.serret_frenet_configuration(nel, r_OP, None, None, xi1)
    Q_rod, u0_rod = Rod.straight_initial_configuration(nel, length, A_IB0=A_IB)
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

    c = consistent_constraints(system, rod, constraints)
    system.add(*c)

    system.assemble()

    ######################
    # compute eigenmodes #
    ######################
    for c in [None, "NullSpace", "ComplianceProjection"]:
        omegas, modes_dq, sol_modes = system.eigenmodes(
            system.t0,
            system.q0,
            constraints=c,
        )

        print(omegas)
        print(len(omegas))

    # vtk-export
    rod._export_dict["level"] = "NodalVolume"
    dir_name = Path(__file__).parent
    system.export(dir_name, f"vtk_modes", sol_modes, fps=25)


if __name__ == "__main__":
    nel = 1
    pDeg = 1
    Rod = make_CosseratRod(
        interpolation="Quaternion",
        mixed=True,
        polynomial_degree=pDeg,
        constraints=[0, 1, 2, 3, 4, 5],
    )

    constraints = ["free", "free"]
    # constraints = ["rigid", "rigid"]
    # constraints = ["rot_z", "rot_z"]
    cantilever(Rod, nel, constraints)
