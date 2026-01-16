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


def consistent_constraints(sys, rod, constraints):
    rod_impressed = rod.idx_impressed
    # assert 0 not in rod_impressed, "No axial deformation allowed!"
    assert 2 not in rod_impressed, "No shear deformation in ez_B allowed!"
    # assert 3 not in rod_impressed, "No torsion deformation allowed!"
    assert 4 not in rod_impressed, "No bending deformation around ey_B allowed!"

    assert rod.nelement >= 2, "there must be at least 2 elements!"

    left = constraints[0]
    right = constraints[1]

    cs = ["free", "guided", "clamped", "supported"]
    assert left in cs, left
    assert right in cs, right

    # always restrict x and z motion
    g_pos0 = [0, 2]
    # restrict additional motions
    if left in ["clamped", "supported"]:
        g_pos0.append(1)

    # constraints on the right
    g_pos1 = [0]
    if right in ["clamped", "supported"]:
        g_pos1.append(1)

    # always restrict rotation around x and y
    g_rot0 = [(1, 2), (2, 0)]
    # restrict additional motion
    if left in ["guided", "clamped"]:
        g_rot0.append((0, 1))

    # constraints on the right
    g_rot1 = []
    if right in ["guided", "clamped"]:
        g_rot1.append((0, 1))

    g_pos = [g_pos0, g_pos1]
    g_rot = [g_rot0, g_rot1]
    cs = []
    for i, (g_p, g_r) in enumerate(zip(g_pos, g_rot)):
        if len(g_p) > 0 or len(g_r) > 0:
            cs.append(
                ProjectedPositionOrientationBase(rod, sys.origin, g_p, g_r, xi1=i)
            )

    return cs


def constrain_uDOFs(rod, constraints):
    # free:         no constraints
    # supported:    constrain only position
    # clamped:      constrain position and orientations
    # guided:       constrain only orientations
    nodes = [[0, 0], [rod.nnodes_r - 1, rod.nnodes_p - 1]]
    c_uDOFs = []
    for c, node in zip(constraints[:2], nodes):
        assert c in ["free", "supported", "clamped", "guided"]

        if c in ["clamped", "supported"]:
            c_uDOFs.extend(rod.nodalDOF_r_u[node[0]])
        if c in ["clamped", "guided"]:
            c_uDOFs.extend(rod.nodalDOF_p_u[node[1]])

    # planarize rod
    for node in range(rod.nnodes_r):
        # remove displacements in e_z^B
        c_uDOFs.append(rod.nodalDOF_r_u[node][2])

    for node in range(rod.nnodes_p):
        # remove torsion
        c_uDOFs.append(rod.nodalDOF_p_u[node][0])
        # remove bending around e_y^B
        c_uDOFs.append(rod.nodalDOF_p_u[node][1])

    assert constraints[2] in ["T", "axial"]
    # only simulate axial modes
    if constraints[2] == "axial":
        # remove displacements in e_y^B
        for node in range(rod.nnodes_r):
            c_uDOFs.append(rod.nodalDOF_r_u[node][1])
        # remove bending around e_z^B
        for node in range(rod.nnodes_p):
            c_uDOFs.append(rod.nodalDOF_p_u[node][2])

    return c_uDOFs


def cantilever(Rod, nel, constraints, export_vtk=False, name=None):
    name = name if name else "_".join(constraints)
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
        name="Beam",
    )
    system.add(rod)

    # c = consistent_constraints(system, rod, constraints)
    # system.add(*c)

    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))

    c_uDOFs = constrain_uDOFs(rod, constraints)

    ######################
    # compute eigenmodes #
    ######################
    # omegas, modes_dq, sol_modes = system.eigenmodes(
    #     system.t0, system.q0, constraints=proj, ccic=False
    # )
    omegas, modes_dq, sol_modes = system.new_eigenmodes(
        system.t0, system.q0, remove_uDOFs=c_uDOFs
    )

    if export_vtk:
        rod._export_dict["level"] = "NodalVolume"
        dir_name = Path(__file__).parent
        system.export(dir_name, f"vtk_modes/{name}", sol_modes, fps=25)

    return omegas


def test_all():
    proj = "Nullspace"
    nel = 20
    pDeg = 2

    rods = ["T", "EB"]
    lefts = ["free", "clamped", "supported", "guided"]
    rights = ["free", "clamped", "supported", "guided"]

    for rod in rods:
        c = [0, 1, 2, 3, 4]
        if rod == "T":
            c.remove(1)

        Rod = make_CosseratRod(
            interpolation="Quaternion",
            mixed=True,
            polynomial_degree=pDeg,
            constraints=c,
        )

        for left in lefts:
            for right in rights:
                cantilever(Rod, nel, [left, right, rod], proj)


def analytical_u_psi(constraints, omega_max, u_psi):
    assert omega_max >= 0.0
    omega = -np.inf
    omegas = []

    if constraints[2] == "axial" and u_psi == "psi":
        return np.array([], dtype=float)

    for c in constraints[:2]:
        assert c in ["free", "clamped", "supported", "guided"]

    # add RB mode and select wave velocity
    frac = np.pi / PARAMS["L"]
    if u_psi == "u":
        frac *= np.sqrt(PARAMS["E"] / PARAMS["density"])
        c_set_RB = ["free", "guided"]
    elif u_psi == "psi":
        frac *= np.sqrt(PARAMS["G"] / PARAMS["density"])
        c_set_RB = ["clamped", "support"]

    if constraints[0] in c_set_RB and constraints[1] in c_set_RB:
        omegas.append(0.0)

    if (constraints[0] in c_set_RB) != (constraints[1] in c_set_RB):
        cc = 1
    else:
        cc = 0

    k = 1
    while omega < omega_max:
        if cc == 0:
            f = k
        else:
            f = (2 * k - 1) / 2

        omega = f * frac
        k += 1
        omegas.append(omega)

    omegas.pop()
    return np.array(omegas)


def analytical_T(constraints, omega_max, dir="y"):
    length = PARAMS["L"]
    density = PARAMS["density"]
    cross_section = PARAMS["cross_section"]
    E = PARAMS["E"]
    G = PARAMS["G"]
    shear_corr = PARAMS["shear_corr"]

    A = cross_section.area
    Ip, Iy, Iz = np.diag(cross_section.second_moment)

    I = Iy if dir == "y" else Iz

    # eq. 2.6
    a = G * shear_corr * A
    b = density * A
    c = E * I
    d = density * I

    # eq. 2.20 & 2.21
    b_hat = lambda omega: density * omega**2 * (1 + E / (G * shear_corr))
    c_hat = (
        lambda omega: density
        * omega**2
        * (density * omega**2 / (G * shear_corr) - A / I)
    )

    # eq. 2.22
    lambda_1_star_square = lambda om: 0.5 * (
        -b_hat(om) + np.sqrt(b_hat(om) ** 2 - 4 * c_hat(om))
    )
    lambda_2_star_square = lambda om: 0.5 * (
        -b_hat(om) - np.sqrt(b_hat(om) ** 2 - 4 * c_hat(om))
    )

    # eq. 2.27
    lambda_1_hat = lambda om: np.sqrt(lambda_1_star_square(om))
    lambda_2 = lambda om: np.sqrt(-lambda_2_star_square(om))

    # eq. 2.29
    alpha_1_hat = lambda om: om**2 * b / a + lambda_1_hat(om) ** 2
    alpha_2 = lambda om: om**2 * b / a - lambda_2(om) ** 2

    # transistion frequency
    # eq. 2.23
    omega_transition = np.sqrt(a / d)

    # case 1: omega < omega_transition
    if left in ["supported"]:
        # eq. 3.9
        A_upper = lambda om: np.array(
            [
                [1, 0, 1, 0],
                [alpha_1_hat(om), 0, alpha_2(om), 0],
            ]
        )

    if right in ["supported"]:
        # eq. 3.9
        arg_h = lambda om: lambda_1_hat(om) * length
        arg_t = lambda om: lambda_2(om) * length
        A_lower = lambda om: np.array(
            [
                [
                    np.cosh(arg_h(om)),
                    np.sinh(arg_h(om)),
                    np.cos(arg_t(om)),
                    np.sin(arg_t(om)),
                ],
                [
                    alpha_1_hat(om) * np.cosh(arg_h(om)),
                    alpha_1_hat(om) * np.sinh(arg_h(om)),
                    alpha_2(om) * np.cos(arg_t(om)),
                    alpha_2(om) * np.sin(arg_t(om)),
                ],
            ]
        )

    A = lambda om: np.hstack([A_upper(om), A_lower(om)])

    return []


if __name__ == "__main__":
    # test_all()

    rod = "EB"
    rod = "T"
    # rod = "axial"
    left = right = "free"
    left = right = "supported"

    # get analytical solution
    path = Path(__file__)
    omegas_T = np.loadtxt(
        Path(
            path.parent,
            "analytical",
            "Cazzani_simple_supported.csv",
        ),
        delimiter=",",
        skiprows=1,
    )

    omegas_u = analytical_u_psi([left, right, rod], omegas_T[-1], "u")
    omegas_psi = analytical_u_psi([left, right, rod], omegas_T[-1], "psi")

    # om_T = analytical_T([left, right])

    if rod == "axial":
        omegas_analytical = omegas_u
    else:
        omegas_analytical = np.sort(np.concatenate([omegas_u, omegas_T]))

    # define rod for numerical solution
    pDeg = 3
    Rod = make_CosseratRod(
        interpolation="Quaternion",
        # interpolation="SE3",
        mixed=True,
        polynomial_degree=pDeg,
    )

    # analyze convergence by increasing nel
    N0 = 3
    N = 6  # bei 9 Eskalation!
    n_analytical = len(omegas_analytical)
    nels = [2**i for i in range(N0, N0 + N)]
    nnodes = np.zeros(N, dtype=int)
    omegas = np.empty((N, n_analytical), dtype=float)
    omegas[:] = np.nan
    errors = np.empty((N, n_analytical), dtype=float)
    errors[:] = np.nan
    sum_error = np.empty(N, dtype=float)
    sum_error[:] = np.nan
    for i, nel in enumerate(nels):
        omegas_i = cantilever(Rod, nel, [left, right, rod], False)
        nnodes[i] = pDeg * nel + 1
        n = np.min([n_analytical, len(omegas_i)])
        omegas[i, :n] = omegas_i[:n]

        errors[i, :n] = (
            np.abs((omegas_i[:n] - omegas_analytical[:n])) / omegas_analytical[:n]
        )
        sum_error[i] = np.sum(errors[i, : nnodes[0]]) / nnodes[0]

    cantilever(Rod, 8, [left, right, rod], True, name="running")

    fig, ax = plt.subplots(1, 1)
    ax.loglog(nels, omegas)
    ax.set_prop_cycle(None)
    ax.loglog(nels, len(nels) * [omegas_analytical], "x")
    ax.set_xlabel("nel")
    ax.set_ylabel("omegas")
    ax.grid()

    fig, ax = plt.subplots(1, 1)
    ax.plot(omegas_analytical)
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("omegas")
    ax.grid()

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
