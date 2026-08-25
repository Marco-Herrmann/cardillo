import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation

from cardillo import System
from cardillo.constraints import RigidConnection
from cardillo.discrete import RigidBody, Cylinder
from cardillo.forces import Force
from cardillo.math import e3
from cardillo.rods import animate_beam
from cardillo.rods_new import (
    CircularCrossSection,
    CrossSectionInertias,
    Simo1986,
    make_CosseratRod,
)
from cardillo.solver import (
    Newton,
    BackwardEuler,
    SolverOptions,
    ScipyDAE,
    DualStormerVerlet,
    Eigenmodes,
)

if __name__ == "__main__":
    # nturns = 3  # number of coils
    # nturns = 10  # number of coils
    nturns = 20  # number of coils Harsch2021

    t1 = 20  #

    #########
    # gravity
    #########
    gravity = 9.81

    #######################
    # spring modeled as rod
    #######################
    system = System()
    Rod = make_CosseratRod()

    polynomial_degree = 2
    elements_per_turn = 12
    nelements = int(elements_per_turn * nturns)

    ############
    # Harsch2021
    ############
    rho = 7850  # [kg / m^3]
    G = 81.5e9
    E = 206.0e9
    print(f"G: {G}; E: {E}")

    # 1mm cross sectional diameter
    wire_diameter = 1e-3
    wire_radius = wire_diameter / 2

    # helix parameter
    coil_diameter = 32.0e-3
    coil_radius = coil_diameter / 2
    pitch_unloaded = wire_diameter
    c = pitch_unloaded / (coil_radius * 2 * np.pi)

    # rod cross-section
    cross_section = CircularCrossSection(wire_radius)
    cross_section_inertias = CrossSectionInertias(rho, cross_section)
    A = cross_section.area(0.0)
    I_ii = np.diag(cross_section.second_moment(0.0))

    A_rho0 = rho * A
    Ei = np.array([E * A, G * A, G * A])
    Fi = np.array([G * I_ii[0], E * I_ii[1], E * I_ii[2]])
    material_model = Simo1986(Ei, Fi)
    print(f"Ei: {Ei}")
    print(f"Fi: {Fi}")

    # gravity load
    f_g_rod_statics = (
        lambda t, xi: -(2 * t if t <= 0.5 else 1.0) * A_rho0 * gravity * e3
    )
    f_g_rod = lambda t, xi: -A_rho0 * gravity * e3

    # helix and derivatives
    def r(xi, phi0=0):
        alpha = 2 * np.pi * nturns * xi
        return coil_radius * np.array(
            [np.sin(alpha + phi0), -np.cos(alpha + phi0), c * alpha]
        )

    def dr(xi, phi0=0):
        alpha = 2 * np.pi * nturns * xi
        return (
            coil_radius
            * 2
            * np.pi
            * nturns
            * np.array([np.cos(alpha + phi0), np.sin(alpha + phi0), c])
        )

    def ddr(xi, phi0=0):
        alpha = 2 * np.pi * nturns * xi
        return (
            coil_radius
            * (2 * np.pi * nturns) ** 2
            * np.array([-np.sin(alpha + phi0), np.cos(alpha + phi0), 0])
        )

    # definition of the parametric curve
    curve = lambda xi: r(xi, phi0=np.pi)
    dcurve = lambda xi: dr(xi, phi0=np.pi)
    ddcurve = lambda xi: ddr(xi, phi0=np.pi)

    q0_rod = Rod.serret_frenet_configuration(
        nelements,
        curve,
        dcurve,
        ddcurve,
        xi1=1,
        r_OP0=np.zeros(3, dtype=float),
        A_IB0=np.eye(3, dtype=float),
    )

    rod = Rod(
        cross_section,
        material_model,
        nelements,
        Q=q0_rod,
        distributed_load=[f_g_rod_statics, None],
        cross_section_inertias=cross_section_inertias,
    )
    system.add(rod)

    ##############
    # pendulum bob
    ##############
    R = 25e-3  # radius of the main cylinder
    h = 34e-3  # height of the main cylinder
    density = 7850  # [kg / m^3]; steel
    r_OS0 = np.array([0, 0, -h / 2 - wire_radius])
    p0 = np.array([1, 0, 0, 0], dtype=float)
    q0_bob = np.concatenate((r_OS0, p0))
    bob = Cylinder(RigidBody)(radius=R, height=h, density=rho, q0=q0_bob)

    f_g_bob_statics = lambda t: -(2 * t if t <= 0.5 else 1.0) * bob.mass * gravity * e3
    f_g_bob = lambda t: -bob.mass * gravity * e3
    gravity_bob_statics = Force(f_g_bob_statics, bob, name="bob_grav_stat")
    gravity_bob = Force(f_g_bob, bob, name="bob_grav")
    system.add(bob, gravity_bob_statics)

    pulling_factor = 0.3
    # pulling_factor = 0.1
    f_pulling = (
        lambda t: -(2 * (t - 0.5) if t >= 0.5 else 0.0)
        * bob.mass
        * gravity
        * e3
        * pulling_factor
    )
    pulling_force = Force(f_pulling, bob, name="puliing")

    joint1 = RigidConnection(system.origin, rod, xi2=1, name="ground-rod")
    joint2 = RigidConnection(bob, rod, xi2=0, name="bob-rod")

    #####################
    # assemble the system
    #####################
    # system.add(rod, joint1, force_rod)
    system.add(
        joint1,
        joint2,
        pulling_force,
    )
    system.assemble()

    #####################
    # solve static system
    #####################
    n_load_steps = 10
    sol = Newton(
        system,
        n_load_steps=n_load_steps,
        # t1 = 0.5,
        t1=1.0,
    ).solve()
    q = sol.q
    q_stat = q
    nt = len(q)
    t = sol.t[:nt]

    i_stat0 = np.argwhere(sol.t == 0.5)[0, 0]
    print(bob.r_OP(t[i_stat0], q[i_stat0, bob.qDOF]))
    print(bob.r_OP(t[-1], q[-1, bob.qDOF]))

    ################
    # blender export
    ################
    dir_name = Path(__file__).parent
    system.export_blender(dir_name, "blend_stat", sol, create_blend=True)

    ##############
    # Eigenmodes #
    ##############
    solver_eig = Eigenmodes(system, sol)
    omegas = np.zeros((n_load_steps + 1, 10))
    omegas_cheap = np.zeros((n_load_steps + 1, 10))
    # for i in range(n_load_steps + 1):
    #     print(f"step {i}/{n_load_steps}")
    #     sol_eig = solver_eig.solve(i, n_eig=11, compute_dense=False)
    #     omegas[i] = sol_eig.omegas[:10]

    #     # print(f"solve cheap")
    #     # _, _, sol_eig_cheap = solver_eig.solve_cheap(i)
    #     # omegas_cheap[i] = sol_eig_cheap.omegas[0, :10]

    #     # print(f"export")
    #     # system.export_blender(dir_name, f"blend_eig{i}", sol_eig, create_blend=True)

    # fig, ax = plt.subplots(1, 1)
    # ax.plot(t, omegas)
    # ax.plot(t, omegas_cheap, "--")
    # plt.show()

    # compute with reduced number of DOFs
    sol_stat0 = solver_eig.solve(i_stat0, n_eig=11, compute_dense=True)

    system.set_new_initial_state(q0=sol.q[-1], u0=sol.u[-1])

    rod.set_parameter(
        distributed_load=[f_g_rod, None],
    )

    system.remove(gravity_bob_statics, pulling_force)
    system.add(gravity_bob)
    system.assemble()

    solver = ScipyDAE(
        system,
        t1=t1,
        # t1 = 5.0e-3,
        dt=1.0e-3,
        method="Radau",
        atol=1e-3,
        rtol=1e-3,
        stages=3,
    )
    # solver = DualStormerVerlet(
    #     system,
    #     t1=t1,
    #     dt=1.0e-3,
    #     options=SolverOptions(
    #         fixed_point_atol=1e-3,
    #         fixed_point_rtol=1e-3,
    #     ),
    # )

    sol = solver.solve()
    q = sol.q
    q_dyn = q
    nt = len(q)
    t = sol.t[:nt]

    ################################
    # plot characteristic quantities
    ################################
    r_OS = np.array([bob.r_OP(ti, qi[bob.qDOF]) for (ti, qi) in zip(sol.t, sol.q)])

    ordering = "zyx"
    angles = np.array(
        [
            Rotation.from_matrix(bob.A_IB(ti, qi[bob.qDOF])).as_euler(ordering)
            for (ti, qi) in zip(sol.t, sol.q)
        ]
    )

    ###############
    # visualization
    ###############
    fig, ax = plt.subplots(2, 1)

    ax[0].plot(t, r_OS[:, 0], label="x")
    ax[0].plot(t, r_OS[:, 1], label="y")
    ax[0].plot(t, r_OS[:, 2], label="z")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(t, np.rad2deg(angles[:, 0]), label="alpha")
    ax[1].plot(t, np.rad2deg(angles[:, 1]), label="beta")
    ax[1].plot(t, np.rad2deg(angles[:, 2]), label="gamma")
    ax[1].legend()
    ax[1].grid()

    ###########
    # animation
    ###########
    _ = animate_beam(t, q, [rod], 0.05, scale_di=0.01, show=False)

    plt.show()

    ################
    # blender export
    ################
    system.export_blender(dir_name, "blend_dyn", sol, create_blend=True)
