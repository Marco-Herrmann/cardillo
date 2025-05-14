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

    R = length
    r_OP = lambda alpha: R * np.array([np.cos(alpha), np.sin(alpha), 0.0])
    A_IB = lambda alpha: A_IB_basic(alpha + np.pi / 2).z

    Q_rod, u0_rod = Rod.straight_initial_configuration(nel, length)
    Q_rod = Rod.pose_configuration(nel, r_OP, A_IB, xi1=1.0)  # np.pi/2)
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

    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))

    c_uDOFs = constrain_uDOFs(rod, constraints)

    ######################
    # compute eigenmodes #
    ######################
    # omegas, modes_dq, sol_modes = system.eigenmodes(
    #     system.t0, system.q0, constraints=None, ccic=False
    # )
    omegas, modes_dq, sol_modes = system.new_eigenmodes(
        system.t0, system.q0, remove_uDOFs=c_uDOFs
    )

    if export_vtk:
        rod._export_dict["level"] = "NodalVolume"
        dir_name = Path(__file__).parent
        system.export(dir_name, f"vtk_modes/{name}", sol_modes, fps=25)

    return omegas


if __name__ == "__main__":
    rod = "EB"
    rod = "T"

    left = "clamped"
    right = "free"
    # left = "free"
    # right = "clamped"

    # define rod for numerical solution
    pDeg = 2
    Rod = make_CosseratRod(
        interpolation="Quaternion",
        # interpolation="SE3",
        mixed=True,
        polynomial_degree=pDeg,
    )

    # analyze convergence by increasing nel
    N0 = 3
    N = 6

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
        omegas_i = cantilever(Rod, nel, [left, right, rod], False)
        nnodes[i] = pDeg * nel + 1
        n = np.min([n_compare, len(omegas_i)])
        omegas[i, :n] = omegas_i[:n]

    errors = np.abs(omegas[:-1] - omegas[-1]) / omegas[-1]
    sum_error = np.sum(errors, 1) #/ rod.nnodes

    cantilever(Rod, 8, [left, right, rod], True, name="running")

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
