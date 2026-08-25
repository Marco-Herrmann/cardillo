import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import warnings

from cardillo import System
from cardillo.constraints import RigidConnection
from cardillo.forces import Force, B_Force, Moment, B_Moment
from cardillo.math import e1, e2, e3, A_IB_basic
from cardillo.rods import RectangularCrossSection, animate_beam, Simo1986
from cardillo.rods_new import (
    Simo1986 as Simo1986_new,
    RectangularCrossSection as RectangularCrossSection_new,
    make_CosseratRod as make_CosseratRod_new,
)
from cardillo.rods.cosseratRod import make_CosseratRod
from cardillo.solver import Newton, SolverOptions
from cardillo.utility.sensor import Sensor


def bent_45(
    Rod,
    constitutive_law,
    *,
    nelements: int = 10,
    slenderness: float = 1e1,
    tolType: str = "",
    #
    n_load_steps: int = 20,
    #
    VTK_export: bool = False,
    name: str = "simulation",
    show_plots: bool = False,
    save_tip_displacement: bool = False,
    save_stresses: bool = False,
    new_interface: bool = False,
):
    # handle name
    plot_name = name.replace("_", " ")
    save_name = f'{name.replace(" ", "_")}_nel{nelements}'
    print(f"Slenderness: {slenderness:1.0e}, Rod: {plot_name}, nel: {nelements}")

    # geometry
    R = 100

    # create function of circle
    r_OP_circle = lambda alpha: R * np.array([np.sin(alpha), np.cos(alpha), 0])
    A_IB_circle = lambda alpha: A_IB_basic(-alpha).z

    # define angle
    angle = 45 * np.pi / 180
    r_OP0 = lambda xi: r_OP_circle(xi * angle)
    A_IB0 = lambda xi: A_IB_circle(xi * angle)

    # cross section
    w = R / slenderness
    if new_interface:
        cross_section = RectangularCrossSection_new(w, w)
        A = cross_section.area(0.0)
        I1, I2, I3 = np.diag(cross_section.second_moment(0.0))
    else:
        cross_section = RectangularCrossSection(w, w)
        A = cross_section.area
        I1, I2, I3 = np.diag(cross_section.second_moment)

    # material model
    E = 1e7
    G = E / 2
    Ei = np.array([E * A, G * A, G * A])
    Fi = np.array([G * I1, E * I2, E * I3])
    material_model = constitutive_law(Ei, Fi)

    # initialize system
    system = System()

    # create rod
    q0 = Rod.pose_configuration(nelements, r_OP0, A_IB0)
    # q0 = Rod.straight_configuration(nelements, R, r_OP0=r_OP0(0.0), A_IB0=A_IB0(0.0))
    rod = Rod(cross_section, material_model, nelements, Q=q0)
    system.add(rod)

    # connect to origin
    clamping = RigidConnection(rod, system.origin, xi1=0)
    system.add(clamping)

    # tip load
    Fz_dict = {1e1: 6e6, 1e2: 6e2, 1e3: 6e-2, 1e4: 6e-6}
    tip_force = Fz_dict[slenderness]
    F = lambda t: tip_force * t**2 * e3
    # F = lambda t: tip_force * t * e1
    force = B_Force(F, rod, xi=1)
    force = Force(F, rod, xi=1)
    system.add(force)

    # sensor at tip
    sensor = Sensor(rod, xi=1, name="Tip")
    system.add(sensor)

    # assemble system
    system.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))

    ############
    # simulation
    ############
    atols_dict_MX = {1e1: 1e-2, 1e2: 1e-6, 1e3: 1e-10, 1e4: 1e-13}
    atols_dict_DB = {1e1: 1e-2, 1e2: 1e-6, 1e3: 1e-8, 1e4: 1e-10}
    if tolType == "MX":
        # Domenico MX
        atols_dict = atols_dict_MX
    elif tolType == "DB":
        # Domenico DB
        atols_dict = atols_dict_DB
    elif tolType == "":
        warnings.warn("No tolType was specified!")
        atols_dict = atols_dict_MX
    else:
        raise NotImplementedError
    solver = Newton(
        system,
        n_load_steps=n_load_steps,
        options=SolverOptions(newton_atol=atols_dict[slenderness]),  # rtol=0
    )
    sol = solver.solve()  # solve static equilibrium equations
    dir_name = Path(sys.argv[0]).parent
    system.export_blender(dir_name, f"blend_stat", sol, create_blend=True)

    # read solution
    t = sol.t
    q = sol.q
    la_c = sol.la_c
    la_g = sol.la_g

    fig, ax = plt.subplots(6)
    for i in range(6):
        ax[i].plot(t, la_g[:, i])
    # plt.show()

    ##############
    # Eigenmodes #
    ##############
    from cardillo.solver import Eigenmodes, FrequencyResponseFunction

    np.set_printoptions(precision=2, linewidth=500)
    solver_eig = Eigenmodes(system, sol)
    solver_frf = FrequencyResponseFunction(system, sol)
    omegas = np.zeros((n_load_steps + 1, 10))
    omegas_DAE = np.zeros((n_load_steps + 1, 10))
    omegas_proj = np.zeros((n_load_steps + 1, 10))
    omegas_cheap = np.zeros((n_load_steps + 1, 10))

    iom = 1j * np.logspace(-2, 4, 51)
    frfs = np.zeros((n_load_steps + 1, len(iom), 6, 3), dtype=complex)

    for i in range(n_load_steps + 1):
        print(f"step {i}/{n_load_steps}")
        sol_eig = solver_eig.solve(i)
        omegas[i] = sol_eig.omegas[0, :10]

        sol_DAE = solver_eig.solve(
            i, bilateral_constraints=dict(Method="DAE", alpha=1.0, gamma=1.0)
        )
        omegas_DAE[i] = sol_DAE.omegas[0, :10]

        sol_proj = solver_eig.solve(i, bilateral_constraints=dict(Method="Proj"))
        omegas_proj[i] = sol_proj.omegas[0, :10]

        # print(f"solve cheap")
        # _, _, sol_eig_cheap = solver_eig.solve_cheap(i)
        # omegas_cheap[i] = sol_eig_cheap.omegas[0, :10]

        # # if i == 0 or i == n_load_steps:
        # #     dir_name = Path(sys.argv[0]).parent
        # #     system.export_blender(dir_name, f"blend_eig{i}", sol_eig, create_blend=True)

        if i == 0 or i == n_load_steps:
            frfs[i] = solver_frf.solve(i, iom)

    fig, ax = plt.subplots(6, 3)
    for i in range(6):
        for j in range(3):
            # for l in range(n_load_steps + 1):
            l = 0
            ax[i, j].loglog(iom.imag, np.abs(frfs[l, :, i, j]))

            ax[i, j].grid()
    plt.show()

    fig, ax = plt.subplots(1, 1)
    ax.plot(t, omegas, ".-", label="My method")
    ax.plot(t, omegas_cheap, "x--", label="Cheap method")
    ax.legend()
    plt.show()

    exit()

    #######################
    # test KN and Wla_g_q #
    #######################
    # for i in range(len(sol.t)):
    #     ti = sol.t[i]
    #     qi = sol.q[i]
    #     la_gi = sol.la_g[i]
    #     KN = [A.toarray() for A in system.KN_g(ti, qi, la_gi)]
    #     Wla_g_q = system.Wla_g_q(ti, qi, la_gi).toarray()
    #     B = system.q_dot_u(ti, qi).toarray()

    #     # this should not have N contributions
    #     print(f"norm(N): {np.linalg.norm(KN[1])}")

    #     # K is on LHS, but Wla_g on RHS
    #     K_proj = - (Wla_g_q @ B + B.T @ Wla_g_q.T) / 2

    #     # TODO: I don't get why it is not symmetric?
    #     # # check for symmetry
    #     # print(f"K_proj - K_proj.T: {np.linalg.norm(K_proj - K_proj.T)}")

    #     # compute difference
    #     diff = K_proj - KN[0]
    #     print(f"norm(diff): {np.linalg.norm(diff)}")

    solver_eig = Eigenmodes(system, sol)
    for i in range(0, 1):  # len(sol.t)):
        _, _, sol_eig = solver_eig.solve(i)
        omegas = sol_eig.omegas[0]

        _, _, sol_eig_cheap = solver_eig.solve_cheap(i)
        omegas_cheap = sol_eig_cheap.omegas[0]

        diff = omegas - omegas_cheap
        print(np.linalg.norm(diff))

    #################
    # post-processing
    #################
    # vtk-export
    dir_name = Path(sys.argv[0]).parent
    if VTK_export:
        system.export(dir_name, f"vtk/slen_{slenderness:1.0e}/{save_name}", sol)

    system.export_blender(dir_name, f"blender", sol, create_blend=True)

    ##########################
    # matplotlib visualization
    ##########################
    # construct animation of rods
    if show_plots:
        scale = R / 1.5
        fig_animate, ax, anim = animate_beam(
            t,
            q,
            [rod],
            scale=scale,
            scale_di=0.05 * R,
            show=False,
            n_frames=rod.nelement + 1,
            repeat=True,
        )
        # plot animation
        ax.azim = 30 + 180
        ax.elev = 25

        # move axes around
        ax.set_xlim3d(left=-0.5 * scale, right=1.5 * scale)
        ax.set_ylim3d(bottom=0.5 * scale, top=2.5 * scale)
        ax.set_zlim3d(bottom=-0.5 * scale, top=1.5 * scale)

    # tip displacement over load steps
    fig, ax = plt.subplots(1, 1)
    if len(t) == n_load_steps + 1:
        qDOF_tip = rod.local_qDOF_P(1)
        r_OP0_tip = rod.r_OP(0, q0[qDOF_tip], 1)
        delta_tip_header = "load, delta_x, delta_y, delta_z"
        delta_tip = np.zeros((4, n_load_steps + 1), dtype=float)
        for i in range(n_load_steps + 1):
            delta_tip[0, i] = t[i] * tip_force
            qe = q[i][rod.qDOF][qDOF_tip]
            delta_tip[1:, i] = rod.r_OP(t[i], qe, 1) - r_OP0_tip

        fig.suptitle(f"Tip displacement {name}")
        ax.plot(delta_tip[0], delta_tip[1], "r-", marker="x")
        ax.plot(delta_tip[0], delta_tip[2], "g-", marker="o")
        ax.plot(delta_tip[0], delta_tip[3], "b-", marker="s")
        ax.grid()
        ax.set_xlabel("$F_z$")
        ax.set_ylabel("Tip displacement")

        if save_tip_displacement:
            path_tip = Path(
                dir_name, "csv", f"slen_{slenderness:1.0e}", "tip_displacement"
            )
            path_tip.mkdir(parents=True, exist_ok=True)
            np.savetxt(
                path_tip / f"{save_name}.csv",
                delta_tip.T,
                delimiter=", ",
                header=delta_tip_header,
                comments="",
            )

    # stresses along the rod
    nxi_ges_min = 201
    nxi_el = max(11, int(np.ceil((nxi_ges_min + rod.nelement - 1) / rod.nelement)))
    stresses_header = "xi, nx, ny, nz, mx, my, mz"
    if new_interface:
        xis, B_n, B_m = rod.eval_stresses(
            t[-1], q[-1], la_c[-1], la_g[-1], n_per_element=nxi_el
        )
        stresses = np.hstack([xis[:, None], B_n, B_m]).T
    else:
        stresses = np.zeros((7, nxi_el * rod.nelement), dtype=float)
        for el in range(rod.nelement):
            xi_el = np.linspace(*rod.element_interval(el), nxi_el)
            for i in range(nxi_el):
                idx = el * nxi_el + i
                stresses[0, idx] = xi_el[i]
                B_n, B_m = rod.eval_stresses(
                    t[-1], q[-1], la_c[-1], la_g[-1], xi_el[i], el
                )
                stresses[1:, idx] = *B_n, *B_m

    fig2, ax2 = plt.subplots(2, 1)
    fig2.suptitle(f"Stresses {name}")
    for i in range(2):
        ax2[i].plot(stresses[0], stresses[3 * i + 1], "r")
        ax2[i].plot(stresses[0], stresses[3 * i + 2], "g")
        ax2[i].plot(stresses[0], stresses[3 * i + 3], "b")
        ax2[i].grid()

    ax2[0].set_ylabel(r"$_B n$")
    ax2[1].set_ylabel(r"$_B m$")
    ax2[1].set_xlabel(r"$\xi$")

    if save_stresses:
        path_stresses = Path(dir_name, "csv", f"slen_{slenderness:1.0e}", "stresses")
        path_stresses.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            path_stresses / f"{save_name}.csv",
            stresses.T,
            delimiter=", ",
            header=stresses_header,
            comments="",
        )

    if show_plots:
        plt.show()


if __name__ == "__main__":
    formulation = "old"
    formulation = "new"

    if formulation == "old":
        Rod = make_CosseratRod(
            interpolation="Quaternion",
            mixed=False,
            polynomial_degree=2,
            reduced_integration=True,
        )
    elif formulation == "new":
        Rod = make_CosseratRod_new(
            polynomial_degree=2,
            # idx_constraints=[0, 1, 2, 5],
            # idx_displacement_based=[0, 1, 2, 3, 4, 5],
            # continuity=1,
            # quadrature_int=6,
        )

    bent_45(
        Rod,
        Simo1986 if formulation == "old" else Simo1986_new,
        nelements=25,
        slenderness=1e1,
        tolType="MX",
        n_load_steps=20,
        show_plots=True,
        name="bent 45",
        new_interface=not (formulation == "old"),
    )
