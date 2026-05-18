import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from cardillo import System
from cardillo.constraints._base import ProjectedPositionOrientationBase
from cardillo.math import A_IB_basic
from cardillo.solver import SolverOptions, load_solution, Eigenmodes
from cardillo.rods import (
    CircularCrossSection,
    RectangularCrossSection,
    CrossSectionInertias,
    Simo1986,
)
from cardillo.rods.cosseratRod import make_CosseratRod

##########################
# make Sakman parameters #
##########################
length = 1.0
rho = 8.0e3
r = 0.01

# from here nothing to change
cross_section = CircularCrossSection(r)
A = cross_section.area
I = cross_section.second_moment[1, 1]
E = rho * A / I
nu = 0.3
G = E / (2 * (1 + nu))

# estimate with circular beam fixed-fixed
om_ff_c = 59.16
R = 1 / (2 * np.pi)
mass = length * A * rho
theta_steiner = mass * R**2
theta_torus = mass / 8 * (4 * R**2 + 5 * r**2)
G = om_ff_c**2 * (theta_steiner + theta_torus) / (2 * I)
ma_mo = Simo1986(np.array([E * A, G * A, G * A]), np.array([2 * G * I, E * I, E * I]))

PARAMS_SAKMAN = {
    "length": length,
    "material_model": ma_mo,
    "cross_section_inertias": CrossSectionInertias(rho, cross_section),
}

####################
# other parameters #
####################
cross_section = RectangularCrossSection(0.1, 0.1)
rho = 8.0e3
E = 260.0e9
G = 100.0e9
shear_corr = 5 / 6

A = cross_section.area
Ip, Iy, Iz = np.diag(cross_section.second_moment)
cross_section_inertias = CrossSectionInertias(rho, cross_section)

Ei = np.array([E * A, shear_corr * G * A, shear_corr * G * A])
Fi = np.array([G * Ip, E * Iy, E * Iz])
material_model = Simo1986(Ei, Fi)

PARAMS = {
    "L": 2,
    "material_model": material_model,
    "cross_section_inertias": cross_section_inertias,
}


##########################
# define constraint rods #
##########################
constraint_idx = {
    "T": None,
    "EB": [1, 2],
    "IEB": [0, 1, 2],
}


def create_clamped(sys, rod, xi):
    g_pos = [0, 1, 2]
    g_rot = [(1, 2), (2, 0), (0, 1)]
    return ProjectedPositionOrientationBase(rod, sys.origin, g_pos, g_rot, xi1=xi)


def create_supported(sys, rod, xi):
    g_pos = [0, 1, 2]
    g_rot = []
    return ProjectedPositionOrientationBase(rod, sys.origin, g_pos, g_rot, xi1=xi)


def create_simply_supported(sys, rod, xi):
    g_pos = [0, 1, 2]
    g_rot = [(1, 2)]
    A_IJ0 = rod.A_IB(0.0, rod.q0[rod.elDOF_P(xi)], xi)
    return ProjectedPositionOrientationBase(
        rod, sys.origin, g_pos, g_rot, xi1=xi, A_IJ0=A_IJ0
    )


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
        elif cs == "SS":
            c.append(create_simply_supported(sys, rod, i))
        elif cs == "guided":
            c.append(create_guided(sys, rod, i))

    return c


def cantilever(
    Rod,
    nel,
    constraints,
    export_vtk=False,
    configuration="bent45",
    params=PARAMS,
    lump_mass=False,
    save_solution=False,
    name=None,
    save_folder="None",
):
    name = name if name else "_".join(constraints + [configuration])
    print(f"{name.replace("_", " "):<45} nel:{nel:>5}")
    # create cardillo system
    system = System()

    length = params["length"]
    material_model = params["material_model"]
    cross_section_inertias = params["cross_section_inertias"]

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
    solver = Eigenmodes(system, system.sol0)
    omegas, modes_dq, sol_modes = solver.solve(-1)

    if save_solution:
        dir_name = Path(__file__).parent
        save_path = Path(dir_name, "solutions", save_folder)
        save_path.mkdir(parents=True, exist_ok=True)
        sol_modes.save(Path(save_path, f"{name}.pkl"))

    if export_vtk:
        rod._export_dict["level"] = "NodalVolume"
        dir_name = Path(__file__).parent
        system.export(dir_name, f"vtk_modes/{name}", sol_modes, fps=25)

    return omegas


def test_all(p=2, nel=8, save_solution=False, save_folder="all", export_vtk=False):
    for rod in ["T", "EB", "IEB"]:
        # define rod for numerical solution
        Rod = make_CosseratRod(
            interpolation="Quaternion",
            mixed=True,
            polynomial_degree=p,
            constraints=constraint_idx[rod],
        )

        cs = ["clamped", "free", "supported", "SS", "guided"]
        configurations = ["straight", "bent_45", "circular", "helicoidal", "3D"]

        for config in configurations:
            for cl in cs:
                for cr in cs:
                    constraints = [cl, cr, rod]
                    name = "_".join(constraints + [config])
                    omegas = cantilever(
                        Rod,
                        nel,
                        constraints,
                        export_vtk,
                        config,
                        lump_mass=False,
                        params=PARAMS_SAKMAN,
                        save_solution=save_solution,
                        name=name,
                        save_folder=save_folder,
                    )


def make_reference(nel=170, export_vtk=False):
    test_all(
        p=3, nel=nel, save_solution=True, save_folder="reference", export_vtk=export_vtk
    )


def simulate_Sakman(
    p=2, nel=8, save_solution=False, save_folder="sakman", export_vtk=False
):
    rod = "IEB"
    # define rod for numerical solution
    Rod = make_CosseratRod(
        interpolation="Quaternion",
        mixed=True,
        polynomial_degree=p,
        constraints=constraint_idx[rod],
    )

    constraints = [
        ["clamped", "clamped"],
        ["SS", "SS"],
        ["clamped", "SS"],
        ["clamped", "free"],
        ["SS", "free"],
    ]
    configurations = [
        "straight",
        "circular",
        # "helicoidal", # cannot be compared! There is no nothing like n, h, R given
        "3D",
    ]

    for i, (left, right) in enumerate(constraints):
        for configuration in configurations:
            constraints = [left, right, rod]
            name = "_".join(constraints + [configuration])
            omegas = cantilever(
                Rod,
                nel,
                constraints,
                export_vtk,
                configuration,
                lump_mass=False,
                params=PARAMS_SAKMAN,
                save_solution=save_solution,
                name=name,
                save_folder=save_folder,
            )


def compare_with_Sakman():
    """https://link.springer.com/content/pdf/10.1007/s42417-024-01318-y.pdf"""
    rod = "IEB"
    constraints = [
        ["clamped", "clamped"],
        ["SS", "SS"],
        ["clamped", "SS"],
        ["clamped", "free"],
        ["SS", "free"],
    ]
    configurations = [
        "straight",
        "circular",
        # "helicoidal", # cannot be compared! There is no nothing like n, h, R given
        "3D",
    ]

    np.set_printoptions(precision=4, linewidth=300)
    for i, (left, right) in enumerate(constraints):
        for configuration in configurations:
            # load solution
            ref_name = f"{left}_{right}_{rod}_{configuration}.pkl"
            ref_path = Path(Path(__file__).parent, f"solutions/sakman/", ref_name)
            sol_ref = load_solution(ref_path)

            # compute scale
            rod_ = sol_ref.system.contributions[1]
            EI = rod_.material_model.C_m[1, 1]
            Arho0 = rod_.cross_section_inertias.A_rho0
            L = np.sum(rod_.qw_dyn * rod_.J_dyn)
            scl = np.sqrt(Arho0 * L**4 / EI)

            print(scl)

            # extract relevant omegas
            omegas_ref = sol_ref.omegas[0]
            nRB = np.count_nonzero(omegas_ref == 0)
            omegas = omegas_ref[nRB : nRB + 22]
            print(f"Table {i+1}: {configuration}")
            print(omegas * scl)

            if (
                left == "supported"
                and right == "supported"
                and configuration == "straight"
            ):
                pi_vals = np.arange(1, 10) ** 2 * np.pi**2
                print(pi_vals)


if __name__ == "__main__":
    # make_reference(nel=16, export_vtk=True)
    # simulate_Sakman(save_solution=True)
    compare_with_Sakman()
    exit()
    # exit()
    # test_all(export_vtk=True)
    # exit()

    # rod = "IEB"
    # rod = "EB"
    rod = "T"

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
    pDeg = 3
    Rod = make_CosseratRod(
        interpolation="Quaternion",
        # interpolation="SE3",
        mixed=True,
        polynomial_degree=pDeg,
        constraints=constraint_idx[rod],
    )

    cantilever(Rod, 5, [left, right, rod], True, configuration, name="running")
    # exit()

    # load reference solution
    ref_name = f"{left}_{right}_{rod}_{configuration}.pkl"
    ref_path = Path(Path(__file__).parent, f"solutions/reference/", ref_name)
    sol_ref = load_solution(ref_path)

    omegas_ref = sol_ref.omegas[0]
    nRB = np.count_nonzero(omegas_ref == 0)

    # analyze convergence by increasing nel
    N0 = 3
    N = 5

    n_compare = 50
    n_compare = int(4 * (pDeg * 2**N0) / 2)  # number of non-shear felxible DOFs / 2
    nels = [2**i for i in range(N0, N0 + N)]
    nnodes = np.zeros(N, dtype=int)
    omegas = np.empty((N, n_compare - nRB), dtype=float)
    omegas[:] = np.nan
    errors = np.empty((N, n_compare - nRB), dtype=float)
    errors[:] = np.nan
    sum_error = np.empty(N, dtype=float)
    sum_error[:] = np.nan
    for i, nel in enumerate(nels):
        omegas_i = cantilever(Rod, nel, [left, right, rod], False, configuration)
        nnodes[i] = pDeg * nel + 1
        n = np.min([n_compare, len(omegas_i)])
        omegas[i, :n] = omegas_i[nRB:n]

    errors = np.abs(omegas - omegas_ref[nRB:n_compare]) / omegas_ref[nRB:n_compare]
    sum_error = np.sum(errors, 1)  # / rod.nnodes

    fig, ax = plt.subplots(1, 1)
    ax.loglog(nels, omegas)
    ax.set_prop_cycle(None)
    ax.set_xlabel("$n_{el}$")
    ax.set_xticks(nels, nels)
    ax.set_ylabel("omegas")
    ax.grid()

    # nels = nels[:-1]
    fig, ax = plt.subplots(1, 1)
    ax.loglog(nels, errors)
    ax.set_prop_cycle(None)
    ax.plot(nels, np.array(nels) ** (-1.0) * 50, "--")
    ax.plot(nels, np.array(nels) ** (-2.0) * 50, "--")
    ax.plot(nels, np.array(nels) ** (-3.0) * 50, "--")
    ax.plot(nels, np.array(nels) ** (-4.0) * 50, "--")
    ax.plot(nels, np.array(nels) ** (-5.0) * 50, "--")
    ax.plot(nels, np.array(nels) ** (-6.0) * 50, "--")
    ax.set_xlabel("$n_{el}$")
    ax.set_xticks(nels, nels)
    ax.set_ylabel("e omega i")
    ax.grid()

    fig, ax = plt.subplots(1, 1)
    ax.loglog(nels, sum_error)
    ax.set_prop_cycle(None)
    ax.plot(nels, np.array(nels) ** (-1.0) * 50, "--")
    ax.plot(nels, np.array(nels) ** (-2.0) * 50, "--")
    ax.plot(nels, np.array(nels) ** (-3.0) * 50, "--")
    ax.plot(nels, np.array(nels) ** (-4.0) * 50, "--")
    ax.plot(nels, np.array(nels) ** (-5.0) * 50, "--")
    ax.plot(nels, np.array(nels) ** (-6.0) * 50, "--")
    ax.set_xlabel("$n_{el}$")
    ax.set_xticks(nels, nels)
    ax.set_ylabel("sum (e omega i) / n")
    ax.grid()

    plt.show()
