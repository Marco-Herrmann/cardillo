import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from cardillo import System
from cardillo.actuators import PDcontroller
from cardillo.discrete import RigidBody, Frame, Meshed, Box
from cardillo.constraints import Prismatic, RigidConnection
from cardillo.constraints._base import ProjectedPositionOrientationBase
from cardillo.forces import Force
from cardillo.math import (
    A_IB_basic,
    cross3,
    smoothstep2,
    Exp_SO3_quat,
    e3,
    norm,
    skew2ax,
)
from cardillo.solver import BackwardEuler, Newton, SolverOptions
from cardillo.rods import (
    CircularCrossSection,
    RectangularCrossSection,
    CrossSectionInertias,
    Simo1986,
)
from cardillo.rods.cosseratRod import make_CosseratRod

PARAMS = {
    "L": 2,  # [m]
    "density": 8.0e3,  # [kg / m^3]
    "cross_section": RectangularCrossSection(0.1, 0.1),  # [m, m]
    "E": 260.0e9,  # [N / m^2]
    "G": 100.0e9,  # [N / m^2]
    "shear_corr": 5 / 6,
}


def create_clamped(sys, rod, xi):
    g_pos = [0, 1, 2]
    g_rot = [(1, 2), (2, 0), (0, 1)]
    return ProjectedPositionOrientationBase(rod, sys.origin, g_pos, g_rot, xi1=xi)


def create_supported(sys, rod, xi):
    g_pos = [0, 1, 2]
    g_rot = []
    return ProjectedPositionOrientationBase(rod, sys.origin, g_pos, g_rot, xi1=xi)


def create_guided(sys, rod, xi):
    g_pos = []
    g_rot = [(1, 2), (2, 0), (0, 1)]
    return ProjectedPositionOrientationBase(rod, sys.origin, g_pos, g_rot, xi1=xi)


def make_constraints(sys, rod, constraints):
    c = []
    # TODO: change this for "IEB"
    for i, cs in enumerate(constraints[:2]):
        if cs == "clamped":
            c.append(create_clamped(sys, rod, i))
        elif cs == "supported":
            c.append(create_supported(sys, rod, i))
        elif cs == "guided":
            c.append(create_guided(sys, rod, i))

    return c


def cantilever(
    Rod, nel, constraints, export_vtk=False, configuration="bent45", name=None
):
    name = name if name else "_".join(constraints + [configuration])
    print(f"{name.replace("_", " ")} nel:{nel:>5}")
    # create cardillo system
    system = System()

    length = PARAMS["L"]
    density = PARAMS["density"]
    cross_section = PARAMS["cross_section"]
    E = PARAMS["E"]
    G = PARAMS["G"]
    shear_corr = PARAMS["shear_corr"]

    A = cross_section.area
    Ip, Iy, Iz = np.diag(cross_section.second_moment)
    cross_section_inertias = CrossSectionInertias(density, cross_section)

    Ei = np.array([E * A, shear_corr * G * A, shear_corr * G * A])
    Fi = np.array([G * Ip, E * Iy, E * Iz])
    material_model = Simo1986(Ei, Fi)

    match configuration:
        case "straight":
            Q_rod = Rod.straight_configuration(nel, length)

        case "bent_45":
            R = length
            r_OC = lambda alpha: R * np.array([np.cos(alpha), np.sin(alpha), 0.0])
            A_IB = lambda alpha: A_IB_basic(alpha + np.pi / 2).z
            xi1 = np.pi / 4

            Q_rod = Rod.pose_configuration(nel, r_OC, A_IB, xi1=xi1)

        case "circular":
            R = length / (2 * np.pi)
            r_OC = lambda alpha: R * np.array([np.cos(alpha), np.sin(alpha), 0.0])
            A_IB = lambda alpha: A_IB_basic(alpha + np.pi / 2).z
            xi1 = 2 * np.pi

            Q_rod = Rod.pose_configuration(nel, r_OC, A_IB, xi1=xi1)

        case "helicoidal":
            R = length
            h = R / 5 * R
            n = 4
            arg_xi = 2 * n * np.pi
            r_OC = lambda xi: np.array(
                [R * np.sin(arg_xi * xi), -R * np.cos(arg_xi * xi), h * xi]
            )
            r_OC_dxi1 = lambda xi: np.array(
                [R * arg_xi * np.cos(arg_xi * xi), R * arg_xi * np.sin(arg_xi * xi), h]
            )
            r_OC_dxi2 = lambda xi: np.array(
                [
                    -R * arg_xi**2 * np.sin(arg_xi * xi),
                    R * arg_xi**2 * np.cos(arg_xi * xi),
                    0.0,
                ]
            )

            Q_rod = Rod.serret_frenet_configuration(
                nel, r_OC, r_OC_dxi1, r_OC_dxi2, xi1=1.0
            )

        case "3D":
            sqrt2_inv = 1 / np.sqrt(2)
            x = lambda xi: xi - xi**3 / 3 + xi**4 / 4 - 2 * xi**5 / 15
            y = lambda xi: sqrt2_inv * (xi**2 - xi**3 / 3 - xi**4 / 12 + xi**5 / 5)
            z = lambda xi: sqrt2_inv * (xi**3 / 3 - xi**4 / 4 + 2 * xi**5 / 15)
            r_OC = lambda xi: np.array([x(xi), y(xi), z(xi)])

            x_dxi1 = lambda xi: 1 - xi**2 + xi**3 - 2 * xi**4 / 3
            y_dxi1 = lambda xi: sqrt2_inv * (2 * xi - xi**2 - xi**3 / 3 + xi**4)
            z_dxi1 = lambda xi: sqrt2_inv * (xi**2 - xi**3 + 2 * xi**4 / 3)
            r_OC_dxi1 = lambda xi: np.array([x_dxi1(xi), y_dxi1(xi), z_dxi1(xi)])

            x_dxi2 = lambda xi: -2 * xi + 3 * xi**2 - 8 * xi**3 / 3
            y_dxi2 = lambda xi: sqrt2_inv * (2 - 2 * xi - xi**2 + 4 * xi**3)
            z_dxi2 = lambda xi: sqrt2_inv * (2 * xi - 3 * xi**2 + 8 * xi**3 / 3)
            r_OC_dxi2 = lambda xi: np.array([x_dxi2(xi), y_dxi2(xi), z_dxi2(xi)])

            alpha = 0.0
            alpha = np.pi / 4
            Q_rod = Rod.serret_frenet_configuration(
                nel, r_OC, r_OC_dxi1, r_OC_dxi2, xi1=1.0, alpha=alpha
            )

    rod = Rod(
        cross_section,
        material_model,
        nel,
        Q=Q_rod,
        cross_section_inertias=cross_section_inertias,
        name="Beam",
    )
    system.add(rod)
    system.add(*make_constraints(system, rod, constraints))

    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))

    ######################
    # compute eigenmodes #
    ######################
    n_steps = 1
    sol = Newton(system, n_steps, verbose=False).solve()
    omegas, modes_dq, sol_modes = system.new_eigenmodes(sol, n_steps)

    if export_vtk:
        rod._export_dict["level"] = "NodalVolume"
        dir_name = Path(__file__).parent
        system.export(dir_name, f"vtk_modes/{name}", sol_modes, fps=25)

    return omegas


def test_all():

    # TODO: add IEB
    for rod in ["T", "EB"]:
        # define rod for numerical solution
        Rod = make_CosseratRod(
            interpolation="Quaternion",
            mixed=True,
            polynomial_degree=3,
            constraints=None if rod == "T" else [1, 2],
        )

        cs = ["clamped", "free", "supported", "guided"]
        configurations = ["straight", "bent_45", "circular", "helicoidal", "3D"]

        for config in configurations:
            for cl in cs:
                for cr in cs:
                    omegas = cantilever(Rod, 8, [cl, cr, rod], False, config)


if __name__ == "__main__":
    test_all()
    exit()

    rod = "EB"
    # rod = "T"

    left = "clamped"
    right = "free"
    left = "free"
    # right = "clamped"

    configuration = "straight"
    # configuration = "bent_45"
    # configuration = "circular"
    # configuration = "helicoidal"
    # configuration = "3D"

    # define rod for numerical solution
    pDeg = 2
    Rod = make_CosseratRod(
        interpolation="Quaternion",
        # interpolation="SE3",
        mixed=True,
        polynomial_degree=pDeg,
        constraints=None if rod == "T" else [1, 2],
    )

    cantilever(Rod, 2, [left, right, rod], True, configuration, name="running")
    exit()

    # analyze convergence by increasing nel
    N0 = 1
    N = 1

    n_compare = 50
    nels = [2**i for i in range(N0, N0 + N)]
    nnodes = np.zeros(N, dtype=int)
    omegas = np.empty((N, n_compare), dtype=float)
    omegas[:] = np.nan
    errors = np.empty((N, n_compare), dtype=float)
    errors[:] = np.nan
    sum_error = np.empty(N, dtype=float)
    sum_error[:] = np.nan
    for i, nel in enumerate(nels):
        nel = 40
        omegas_i = cantilever(Rod, nel, [left, right, rod], False, configuration)
        nnodes[i] = pDeg * nel + 1
        n = np.min([n_compare, len(omegas_i)])
        omegas[i, :n] = omegas_i[:n]

    errors = np.abs(omegas[:-1] - omegas[-1]) / omegas[-1]
    sum_error = np.sum(errors, 1)  # / rod.nnodes

    fig, ax = plt.subplots(1, 1)
    ax.loglog(nels, omegas)
    ax.set_prop_cycle(None)
    ax.set_xlabel("nel")
    ax.set_ylabel("omegas")
    ax.grid()

    nels = nels[:-1]
    fig, ax = plt.subplots(1, 1)
    ax.loglog(nels, errors)
    ax.plot(nels, np.array(nels) ** (-1.0) * 50, "--")
    ax.plot(nels, np.array(nels) ** (-2.0) * 50, "--")
    ax.plot(nels, np.array(nels) ** (-3.0) * 50, "--")
    ax.plot(nels, np.array(nels) ** (-4.0) * 50, "--")
    ax.plot(nels, np.array(nels) ** (-5.0) * 50, "--")
    ax.plot(nels, np.array(nels) ** (-6.0) * 50, "--")
    ax.set_xlabel("nel")
    ax.set_ylabel("e omega i")
    ax.grid()

    fig, ax = plt.subplots(1, 1)
    ax.loglog(nels, sum_error)
    ax.plot(nels, np.array(nels) ** (-1.0) * 50, "--")
    ax.plot(nels, np.array(nels) ** (-2.0) * 50, "--")
    ax.plot(nels, np.array(nels) ** (-3.0) * 50, "--")
    ax.plot(nels, np.array(nels) ** (-4.0) * 50, "--")
    ax.plot(nels, np.array(nels) ** (-5.0) * 50, "--")
    ax.plot(nels, np.array(nels) ** (-6.0) * 50, "--")
    ax.set_xlabel("nel")
    ax.set_ylabel("sum (e omega i) / n")
    ax.grid()

    plt.show()
