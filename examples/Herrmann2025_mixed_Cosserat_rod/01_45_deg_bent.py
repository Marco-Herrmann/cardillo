import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import warnings

from cardillo import System
from cardillo.constraints import RigidConnection
from cardillo.forces import Force
from cardillo.math import e3, A_IB_basic
from cardillo.rods import RectangularCrossSection, animate_beam, Simo1986
from cardillo.rods.cosseratRod import make_CosseratRod
from cardillo.rods.boostedCosseratRod import make_BoostedCosseratRod
from cardillo.solver import Newton, SolverOptions


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
    new_eval: bool = False,
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
    # q0 = Rod.straight_configuration(nelements, R * np.pi / 4)
    rod = Rod(cross_section, material_model, nelements, Q=q0)
    system.add(rod)

    # connect to origin
    clamping = RigidConnection(rod, system.origin, xi1=0)
    system.add(clamping)

    # tip load
    Fz_dict = {1e1: 6e6, 1e2: 6e2, 1e3: 6e-2, 1e4: 6e-6}
    tip_force = Fz_dict[slenderness]
    F = lambda t: tip_force * t * e3
    force = Force(F, rod, xi=1)
    system.add(force)

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

    # read solution
    t = sol.t
    q_step = sol.q
    la_c_step = sol.la_c
    la_g_step = sol.la_g

    #################
    # post-processing
    #################
    # vtk-export
    dir_name = Path(sys.argv[0]).parent
    if VTK_export:
        system.export(dir_name, f"vtk/slen_{slenderness:1.0e}/{save_name}", sol)

    ##########################
    # matplotlib visualization
    ##########################
    # construct animation of rods
    if show_plots:
        scale = R / 1.5
        fig_animate, ax, anim = animate_beam(
            t,
            q_step,
            [rod],
            scale=scale,
            scale_di=0.05 * R,
            show=False,
            n_frames=rod.nelement + 1,
            repeat=False,
        )
        # plot animation
        # ax.azim = 30 + 180
        # ax.elev = 25

        # move axes around
        ax.set_xlim3d(left=-0.5 * scale, right=1.5 * scale)
        ax.set_ylim3d(bottom=0.5 * scale, top=2.5 * scale)
        ax.set_zlim3d(bottom=-0.5 * scale, top=1.5 * scale)

    # tip displacement over load steps
    fig, ax = plt.subplots(1, 1)
    if len(t) == n_load_steps + 1:
        qDOF_tip = rod.elDOF_P(1)
        r_OP0_tip = rod.r_OP(0, q0[qDOF_tip], 1)
        delta_tip_header = "load, delta_x, delta_y, delta_z"
        delta_tip = np.zeros((4, n_load_steps + 1), dtype=float)
        for i in range(n_load_steps + 1):
            delta_tip[0, i] = t[i] * tip_force
            qe = q_step[i][rod.qDOF][qDOF_tip]
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
    if new_eval:
        # TODO: can we not slice like sol[-1]?
        xis, B_n, B_m = rod.eval_stresses(
            t[-1], q_step[-1], la_c_step[-1], la_g_step[-1], n_per_element=nxi_el
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
                    t[-1], q_step[-1], la_c_step[-1], la_g_step[-1], xi_el[i], el
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

    #######################
    # degree of asymmetry #
    #######################
    from scipy.sparse.linalg import norm as sparse_norm
    from scipy.sparse import coo_array

    for iStep in range(n_load_steps + 1):
        t_step = sol.t[iStep]
        q_step, la_g_step, la_c_step, la_N_step = (
            sol.q[iStep],
            sol.la_g[iStep],
            sol.la_c[iStep],
            sol.la_N[iStep],
        )
        x_step = np.concatenate([q_step, la_g_step, la_c_step, la_N_step])
        # update W_g, W_c, ...
        f_step = solver.fun(x_step, t_step)
        # compute jacobian
        J_step = solver.jac(x_step, t_step)

        # method a: take whole iteration matrix at static equilibrium
        frob_Ka = sparse_norm(J_step, "fro")
        frob_Ka_a = sparse_norm(0.5 * (J_step - J_step.T), "fro")
        da_a = frob_Ka_a / frob_Ka

        # method b: eliminate constraints
        nx = solver.x.shape[1]
        qDOF_c = np.arange(0, solver.split_x[0])
        la_gDOF_c = np.arange(solver.split_x[0], solver.split_x[1])
        la_cDOF_c = np.arange(solver.split_x[1], solver.split_x[2])
        la_NDOF_c = np.arange(solver.split_x[2], nx)
        cDOF = np.concatenate([qDOF_c[7:], la_cDOF_c, la_NDOF_c])

        uDOF_r = np.arange(0, solver.split_f[0])
        la_gDOF_r = np.arange(solver.split_f[0], solver.split_f[1])
        la_cDOF_r = np.arange(solver.split_f[1], solver.split_f[2])
        la_SDOF_r = np.arange(solver.split_f[2], solver.split_f[3])
        la_NDOF_r = np.arange(solver.split_f[3], nx)
        rDOF = np.concatenate([uDOF_r[6:], la_cDOF_r, la_SDOF_r[1:], la_NDOF_r])

        J_elim = J_step[rDOF[:, None], cDOF]
        frob_Kb = sparse_norm(J_elim, "fro")
        frob_Kb_a = sparse_norm(0.5 * (J_elim - J_elim.T), "fro")
        db_a = frob_Kb_a / frob_Kb

        # method c: only use h_q @ B
        c_split = np.cumsum(np.array([rod.nu - 6, rod._nla_c, rod._nla_g], dtype=int))
        B = system.q_dot_u(t_step, q_step)[7:, 6:]

        hs_q = []
        if hasattr(rod, "h_q"):
            hs_q.append(rod.h_q(t_step, q_step[rod.qDOF], np.zeros(rod.nu)))
        if hasattr(rod, "Wla_c_q"):
            hs_q.append(rod.Wla_c_q(t_step, q_step[rod.qDOF], la_c_step[rod.la_cDOF]))
        if hasattr(rod, "Wla_g_q"):
            hs_q.append(rod.Wla_g_q(t_step, q_step[rod.qDOF], la_g_step[rod.la_gDOF]))

        K = sum(hs_q).tocoo()[6:, 7:] @ B

        # compose projected jacobian
        J_c = coo_array((c_split[-1], c_split[-1]))
        J_c[: c_split[0], : c_split[0]] = K

        if rod._nla_c > 0:
            J_c[: c_split[0], c_split[0] : c_split[1]] = rod.W_c(
                t_step, q_step[rod.qDOF]
            ).tocoo()[6:]
            J_c[c_split[0] : c_split[1], : c_split[0]] = (
                rod.c_q(
                    t_step, q_step[rod.qDOF], np.zeros(rod.nu), la_c_step[rod.la_cDOF]
                ).tocoo()[:, 7:]
                @ B
            )
            J_c[c_split[0] : c_split[1], c_split[0] : c_split[1]] = rod.c_la_c()

        if rod._nla_g > 0:
            print("Not tested yet!")
            J_c[: c_split[1], c_split[1] : c_split[2]] = rod.W_g(
                t_step, q_step[rod.qDOF]
            ).tocoo()[6:]
            J_c[c_split[1] : c_split[2], : c_split[1]] = (
                rod.g_q(t_step, q_step[rod.qDOF]).tocoo()[:, 7:] @ B
            )

        frob_Kc = sparse_norm(J_c, "fro")
        frob_Kc_a = sparse_norm(0.5 * (J_c - J_c.T), "fro")
        dc_a = frob_Kc_a / frob_Kc

        print(f"step {iStep:>2}, da_a: {da_a:.5f}, db_a: {db_a:.5f}, db_c: {dc_a:.5f}")

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
    Rod = make_BoostedCosseratRod(
        polynomial_degree=2,
        # idx_displacement_based=[0, 1, 2, 3, 4, 5],
    )

    bent_45(
        Rod,
        Simo1986,
        nelements=10,
        slenderness=1e1,
        tolType="MX",
        n_load_steps=20,
        show_plots=True,
        name="bent 45",
        new_eval=True,
    )
