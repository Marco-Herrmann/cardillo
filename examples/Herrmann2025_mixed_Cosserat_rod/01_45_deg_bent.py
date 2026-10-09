import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import warnings

from cardillo import System
from cardillo.constraints import RigidConnection
from cardillo.forces import Force, B_Force
from cardillo.math import e1, e2, e3, A_IB_basic
from cardillo.rods import RectangularCrossSection, animate_beam
from cardillo.rods_new import (
    Simo1986,
    RectangularCrossSection,
    make_CosseratRod,
    Elasticity,
    Rectangle_Quadrature,
)
from cardillo.solver import Newton, SolverOptions
from cardillo.utility.sensor import Sensor


def bent_45(
    Rod,
    *,
    nelements: int = 10,
    slenderness: float = 1e1,
    tolType: str = "",
    #
    n_load_steps: int = 20,
    #
    VTK_export: bool = False,
    Blender_export: bool = False,
    name: str = "simulation",
    show_plots: bool = False,
    save_tip_displacement: bool = False,
    save_stresses: bool = False,
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
    A = cross_section.area(0.0)
    I1, I2, I3 = np.diag(cross_section.second_moment(0.0))

    # material model
    E = 1e7
    G = E / 2
    Ei = np.array([E * A, G * A, G * A])
    Fi = np.array([G * I1, E * I2, E * I3])
    material_model1 = Simo1986(Ei, Fi)

    cross_section_quadrature = Rectangle_Quadrature(5, 10, w, w)
    material_model2 = Elasticity(E, 0.0, cross_section_quadrature)
    material_model2 = Elasticity(E, 0.3, cross_section_quadrature)

    # initialize system
    system = System()

    # create rod
    q0 = Rod.pose_configuration(nelements, r_OP0, A_IB0)
    # q0 = Rod.straight_configuration(nelements, R, r_OP0=r_OP0(0.0), A_IB0=A_IB0(0.0))
    rod1 = Rod(cross_section, material_model1, nelements, Q=q0, name="rod1_Simo")
    rod2 = Rod(cross_section, material_model2, nelements, Q=q0, name="rod2_Elasticity")
    system.add(rod1, rod2)

    # connect to origin
    clamping1 = RigidConnection(rod1, system.origin, xi1=0, name="rigid_connection_1")
    clamping2 = RigidConnection(rod2, system.origin, xi1=0, name="rigid_connection_2")
    system.add(clamping1, clamping2)

    # tip load
    Fz_dict = {1e1: 6e6, 1e2: 6e2, 1e3: 6e-2, 1e4: 6e-6}
    tip_force = Fz_dict[slenderness]
    F = lambda t: tip_force * t * e3
    force1 = Force(F, rod1, xi=1, name="force_1")
    force2 = Force(F, rod2, xi=1, name="force_2")
    system.add(force1, force2)

    # sensor at tip
    sensor1 = Sensor(rod1, xi=1, name="Tip1")
    sensor2 = Sensor(rod2, xi=1, name="Tip2")
    system.add(sensor1, sensor2)

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

    #################
    # post-processing
    #################
    # read solution
    t = sol.t
    q = sol.q
    la_c = sol.la_c
    la_g = sol.la_g

    # export
    dir_name = Path(sys.argv[0]).parent
    if VTK_export:
        warnings.warn("VTK export not implemented for the new rod!")
        system.export(dir_name, f"vtk/slen_{slenderness:1.0e}/{save_name}", sol)
    if Blender_export:
        system.export_blender(dir_name, f"bent45/blend_stat", sol, create_blend=True)

    ##########################
    # matplotlib visualization
    ##########################
    # construct animation of rods
    if show_plots:
        scale = R / 1.5
        fig_animate, ax, anim = animate_beam(
            t,
            q,
            [rod1, rod2],
            scale=scale,
            scale_di=0.05 * R,
            show=False,
            n_frames=rod1.nelement + 1,
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
        for j, rod in enumerate([rod1, rod2]):
            qDOF_tip = rod.local_qDOF_P(1)
            r_OP0_tip = rod.r_OP(0, q0[qDOF_tip], 1)
            delta_tip_header = "load, delta_x, delta_y, delta_z"
            delta_tip = np.zeros((4, n_load_steps + 1), dtype=float)
            for i in range(n_load_steps + 1):
                delta_tip[0, i] = t[i] * tip_force
                qe = q[i][rod.qDOF][qDOF_tip]
                delta_tip[1:, i] = rod.r_OP(t[i], qe, 1) - r_OP0_tip

            fig.suptitle(f"Tip displacement {name}")
            ax.plot(delta_tip[0], delta_tip[1], f"r-{'-' if j==1 else ''}", marker="x")
            ax.plot(delta_tip[0], delta_tip[2], f"g-{'-' if j==1 else ''}", marker="o")
            ax.plot(delta_tip[0], delta_tip[3], f"b-{'-' if j==1 else ''}", marker="s")
            ax.grid()
            ax.set_xlabel("$F_z$")
            ax.set_ylabel("Tip displacement")

            if save_tip_displacement:
                path_tip = Path(
                    dir_name,
                    "csv",
                    f"slen_{slenderness:1.0e}",
                    f"tip_displacement_{'SIMO' if j==0 else 'ELASTICITY'}",
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
    nxi_el = max(11, int(np.ceil((nxi_ges_min + rod1.nelement - 1) / rod1.nelement)))
    stresses_header = "xi, nx, ny, nz, mx, my, mz, nxE, nyE, nzE, mxE, myE, MzE"
    xis, B_n, B_m = rod1.eval_stresses(
        t[-1], q[-1], la_c[-1], la_g[-1], n_per_element=nxi_el
    )
    _, B_nE, B_mE = rod2.eval_stresses(
        t[-1], q[-1], la_c[-1], la_g[-1], n_per_element=nxi_el
    )
    stresses = np.hstack([xis[:, None], B_n, B_m, B_nE, B_mE]).T

    fig2, ax2 = plt.subplots(2, 1)
    fig2.suptitle(f"Stresses {name}")
    for i in range(2):
        ax2[i].plot(stresses[0], stresses[3 * i + 1], "r", label="SIMO")
        ax2[i].plot(stresses[0], stresses[3 * i + 2], "g")
        ax2[i].plot(stresses[0], stresses[3 * i + 3], "b")
        ax2[i].plot(stresses[0], stresses[3 * i + 1 + 6], "--r", label="3D elasticity")
        ax2[i].plot(stresses[0], stresses[3 * i + 2 + 6], "--g")
        ax2[i].plot(stresses[0], stresses[3 * i + 3 + 6], "--b")
        ax2[i].grid()

    ax2[0].set_ylabel(r"$_B n$")
    ax2[1].set_ylabel(r"$_B m$")
    ax2[1].set_xlabel(r"$\xi$")
    ax2[0].legend()

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
    Rod = make_CosseratRod(
        polynomial_degree=2,
        idx_displacement_based=np.arange(6),
    )

    bent_45(
        Rod,
        nelements=25,
        slenderness=1e1,
        tolType="MX",
        n_load_steps=20,
        show_plots=True,
        name="bent 45",
    )
