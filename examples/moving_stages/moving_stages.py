import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import trimesh
from scipy.interpolate import interp1d

from cardillo import System
from cardillo.actuators import PDcontroller
from cardillo.discrete import RigidBody, Frame, Meshed, Box
from cardillo.constraints import Prismatic, RigidConnection
from cardillo.forces import Force
from cardillo.math import A_IB_basic, cross3, smoothstep2
from cardillo.solver import BackwardEuler, Newton
from cardillo.rods import CircularCrossSection, CrossSectionInertias, Simo1986
from cardillo.rods.cosseratRod import make_CosseratRod


if __name__ == "__main__":
    # create cardillo system
    system = System()

    # geometry
    dh = 10.0

    # density of blocks
    rho_block = 10.0 / 100**2

    ###########
    # stage x #
    ###########
    dim_x = np.array([30.0, 30.0, dh / 2])
    r_OP0_x = np.array([0.0, 0.0, 0.0])
    A_IB0_x = np.eye(3, dtype=float)
    q0_x = RigidBody.pose2q(r_OP0_x, A_IB0_x)
    u0_x = np.zeros(6, dtype=float)
    stage_x = Box(RigidBody)(dim_x, rho_block, q0=q0_x, u0=u0_x, name="stage_x")

    ###########
    # stage y #
    ###########
    dim_y = np.array([10.0, 30.0, dh / 2])
    r_OP0_y = np.array([10.0, 0.0, dh])
    A_IB0_y = np.eye(3, dtype=float)
    q0_y = RigidBody.pose2q(r_OP0_y, A_IB0_y)
    u0_y = np.zeros(6, dtype=float)
    stage_y = Box(RigidBody)(dim_y, rho_block*3, q0=q0_y, u0=u0_y, name="stage_y")

    ####################
    # disturbing force #
    ####################
    f_d = lambda t: np.array([np.min([1, t]), np.min([2, t]), 0.0])
    force = Force(f_d, stage_y)

    ################
    # adding cable #
    ################
    r_cable = 0.2
    rho_cable = 50.0 / 100**3
    cross_section = CircularCrossSection(r_cable)
    A = cross_section.area
    I = cross_section.second_moment
    E = 210_000 / 100**2
    mu = 0.5
    G = E / (2 * (1 + mu))
    material_model = Simo1986(
        np.array([E * A, G * A, G * A]),
        np.array([G * I[0, 0], E * I[1, 1], E * I[2, 2]]),
    )
    cross_section_inertias = CrossSectionInertias(rho_cable, cross_section)
    nel = 10
    pDeg = 2
    Rod = make_CosseratRod(
        interpolation="Quaternion",
        mixed=True,
        polynomial_degree=pDeg,
    )

    # semicircle
    if True:
        r_OP = lambda xi: np.array(
            [15.0 + dh / 2 * np.sin(xi), 0.0, dh / 2 - dh / 2 * np.cos(xi)]
        )
        L = np.pi * dh / 2
        xi1 = np.pi
    else:
        r_OP = lambda xi: np.array([0.15 + dh * np.cos(xi), 0.0, dh * np.sin(xi)])
        L = np.pi * dh / 2
        xi1 = np.pi / 2

    # make configurations
    q0_rod = Rod.serret_frenet_configuration(nel, r_OP, None, None, xi1)
    Q_rod, u0_rod = Rod.straight_initial_configuration(nel, L)

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

    # constraint
    cable_origin = RigidConnection(system.origin, rod, xi2=0.0)
    cable_stage_x = RigidConnection(stage_x, rod, xi2=0.0)
    cable_stage_y = RigidConnection(stage_y, rod, xi2=1.0)

    ##############################
    # move to workspace position #
    ##############################
    # time grid for planned trajectory
    t0 = 0.0
    t1 = 1.0
    dt = 1e-3

    tN = 0.5  # duration

    # movement in x
    DX = 0
    dx = {-1: -5.0, 0: 0.0, 1: 5.0}[DX]

    # movement in y
    DY = 0
    dy = {-1: -2.5, 0: 0.0, 1: 2.5}[DY]

    # steps
    desired_x = lambda t: dx * smoothstep2(t, x_min=t0, x_max=tN)
    desired_y = lambda t: dy * smoothstep2(t, x_min=t0, x_max=tN)

    # create trajectory, frame and connection
    r_OP_x = lambda t: np.array([desired_x(t), 0.0, 0.0])
    r_OP_y = lambda t: np.array([desired_x(t), desired_y(t), 0.0])

    frame_x = Frame(r_OP_x, name="frame_x")
    frame_y = Frame(r_OP_y, name="frame_y")

    constraint_x = RigidConnection(stage_x, frame_x)
    constraint_y = RigidConnection(stage_y, frame_y)

    # add components to system
    system.add(stage_x, frame_x, constraint_x)
    system.add(stage_y, frame_y, constraint_y)
    # system.add(rod, cable_origin, cable_stage_y)
    system.add(rod, cable_stage_x, cable_stage_y)

    # assemble the system
    system.assemble()

    ##############
    # simulation #
    ##############
    # solver = BackwardEuler(system, t1, dt)
    solver = Newton(system, n_load_steps=30)
    sol = solver.solve()

    # vtk-export
    rod._export_dict["level"] = "NodalVolume"
    dir_name = Path(__file__).parent
    system.export(dir_name, f"vtk", sol, fps=25)

    ###############################
    # constraining stage movement #
    ###############################
    prismatic_x = Prismatic(system.origin, stage_x, 0)
    prismatic_y = Prismatic(stage_x, stage_y, 1)
    
    prismatic_x.name="prismatic_x"
    prismatic_y.name="prismatic_y"

    system.remove(constraint_x, constraint_y)
    system.remove(frame_x, frame_y)
    system.add(prismatic_x, prismatic_y)

    # system.remove(rod, cable_origin, cable_stage_y)

    system.set_new_initial_state(sol.q[-1], sol.u[-1], t0)

    ######################
    # compute eigenmodes #
    ######################
    omegas, modes_dq, sol_modes = system.eigenmodes(
        system.t0, system.q0
    )

    print(omegas)
    print(len(omegas))

    # vtk-export
    dir_name = Path(__file__).parent
    system.export(dir_name, f"vtk_dx{DX}_dy{DY}", sol_modes, fps=25)

    ###################
    # post-processing #
    ###################
    t = sol.t
    q = sol.q
    u = sol.u

    nplot = 3
    fig, ax = plt.subplots(2, nplot)

    for i in range(nplot):
        ax[0, i].plot(t, q[:, stage_x.qDOF[i]])
        ax[0, i].plot(t, q[:, stage_y.qDOF[i]], "--")
        ax[0, i].grid()

        ax[1, i].plot(t, u[:, stage_x.uDOF[i]])
        ax[1, i].plot(t, u[:, stage_y.uDOF[i]], "--")
        ax[1, i].grid()

    plt.show()
