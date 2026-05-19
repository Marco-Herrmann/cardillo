import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import trimesh
import pandas as pd

from cardillo import System
from cardillo.constraints import Cylindrical, Prismatic, RigidConnection, Revolute
from cardillo.discrete import Frame, RigidBody, Cylinder, Meshed
from cardillo.forces import B_Moment, Moment, Force
from cardillo.math import e1, e2, e3, A_IB_basic, Spurrier, ax2skew
from cardillo.rods import *
from cardillo.rods.cosseratRod import make_CosseratRod
from cardillo.rods.force_line_distributed import Force_line_distributed


from cardillo.solver import (
    Newton,
    Riks,
    SolverOptions,
    BackwardEuler,
    Rattle,
    ScipyIVP,
    ScipyDAE,
    DualStormerVerlet,
)

from scipy.integrate import cumulative_trapezoid


def rotating_shaft(
    Rod,
    constitutive_law,
    *,
    nelement: int = 10,
    #
    t_final: float = 10,
    compute_parameters: bool = False,
    dt: float = 0.1,
    #
    show_plots: bool = True,
    VTK_export: bool = False,
    name: str = "rod",
):
    # handle name
    plot_name = name.replace("_", " ")
    save_name = name.replace(" ", "_")

    # global properties
    L = 6.0  # [m] total lenghth of the system
    density = 7_800  # [kg / m^3]
    E = 210 * 1e9  # [N / m^2]
    nu = 0.3  # Poisson's ratio
    G = E / (2 * (1 + nu))  # [N / m^2]

    r_I = 0.045  # [m] inner radius of the rod
    r_O = 0.050  # [m] outer radius of the rod

    r_d = 0.24  # [m] radius of the disc
    t_d = 0.05  # [m] width of the disc
    d = 0.05  # [m] z-distance from the centerline to the center of mass of the disc

    # mass of the disc and moment of inertia
    if compute_parameters:
        m_d = np.pi * r_d**2 * t_d * density
        theta_d_x = 1 / 2 * m_d * r_d**2
        theta_d_y = theta_d_z = 1 / 4 * m_d * r_d**2 + 1 / 12 * m_d * t_d**2

    else:
        m_d = 70.573
        theta_d_x = 2.0325 * 1e-3
        theta_d_y = theta_d_z = 1.0163 * 1e-3

    # mass moment of inertia matrix with respect to the center of mass C of the disc in the body-fixed frame
    B_Theta_C = np.diag(np.array([theta_d_x, theta_d_y, theta_d_z]))  # [kg m^2]

    # annular circular cross section as user-defined cross section
    area = np.pi * (r_O**2 - r_I**2)  # [m^2]
    first_moment = np.array([0, 0, 0])  # [m^3]
    second_moment = np.diag([2, 1, 1]) / 4 * np.pi * (r_O**4 - r_I**4)  # [m^4]

    cross_section = UserDefinedCrossSection(area, first_moment, second_moment)
    # cross_section = AnnularCrossSection(radius_o, area, first_moment, second_moment)
    A = cross_section.area  # [m^2]
    Ip, Iy, Iz = np.diag(cross_section.second_moment)

    cross_section_inertias = CrossSectionInertias(density, cross_section)
    A_rho0 = cross_section_inertias.A_rho0

    # material properties rods
    if compute_parameters:
        Ei = np.array([E * A, G * A, G * A])
        Fi = np.array([G * Ip, E * Iy, E * Iz])

    else:
        Ei = np.array([313.4, 60.5, 60.5]) * 1e6  # [N]
        Fi = np.array([272.7, 354.5, 354.5]) * 1e3  # [N]

    material_model = constitutive_law(Ei, Fi)

    # initialize system
    system = System()

    #################################
    # Excitation via actuated frame on the left end
    #################################
    A1 = 0.8  # [m] first amplitude of the excitation
    A2 = 1.2  # [m] second amplitude of the excitation

    T1 = 0.5  # [s] first time period
    T2 = 1.0  # [s] second time period
    T3 = 1.25  # [s] third time period

    omega = 60.0  # [rad/s] frequency of the excitation (close to the first natural frequency of the system at 56.7 rad/s)

    om_step1 = np.pi / T1
    om_step2 = np.pi / (T3 - T2)

    def Omega(t):
        if t <= T1:
            Omega = A1 * omega * (1 - np.cos(t * om_step1)) / 2
        elif t <= T2:
            Omega = A1 * omega
        elif t <= T3:
            Omega = (
                A1 * omega + (A2 - A1) * omega * (1 - np.cos((t - T2) * om_step2)) / 2
            )
        else:
            # case t > T3
            Omega = A2 * omega
        return Omega

    # Analytical phi(t)
    def phi(t, phi0=0.0):
        _phi_T0 = phi0
        _phi_T1 = _phi_T0 + A1 * omega * T1 / 2
        _phi_T2 = _phi_T1 + A1 * omega * (T2 - T1)
        _phi_T3 = _phi_T2 + (A1 * omega * (T3 - T2) + (A2 - A1) * omega * (T3 - T2) / 2)

        if t <= T1:
            _phi = _phi_T0 + A1 * omega * (t - np.sin(t * om_step1) / om_step1) / 2

        elif t <= T2:
            _phi = _phi_T1 + A1 * omega * (t - T1)

        elif t <= T3:
            _phi = _phi_T2 + (
                A1 * omega * (t - T2)
                + (A2 - A1)
                * omega
                * (t - T2 - np.sin((t - T2) * om_step2) / om_step2)
                / 2
            )

        else:
            _phi = _phi_T3 + A2 * omega * (t - T3)

        return _phi

    ######################################
    # important positions and velocities #
    ######################################
    # R: beginn of the rod (driven)
    # M: middle of the rod (disc)
    # T: end of the rod (undriven)
    r_OR0 = np.zeros(3, dtype=float)
    A_IR0 = np.eye(3, dtype=float)

    r_OM0 = np.array([L / 2, 0.0, 0.0])
    A_IM0 = np.eye(3, dtype=float)

    # r_OT0 = np.array([L, 0.0, 0.0])
    # A_IT0 = np.eye(3, dtype=float)

    v_P0 = np.zeros(3, dtype=float)
    B_omega_IB0 = Omega(0.0) * e1

    ########
    # Disc #
    ########
    # D: disc's center of mass and cosys
    B_r_DM = np.array([0.0, 0.0, -d])
    r_OD0 = r_OM0 - A_IM0 @ B_r_DM

    q0 = RigidBody.pose2q(r_OD0, A_IM0)  # initial position of the disc
    u0 = np.hstack([v_P0, B_omega_IB0])  # initial velocity of the disc

    # create mesh of a solid cylinder (disc with radius_d and height=width)
    disc_mesh = trimesh.creation.cylinder(radius=r_d, height=t_d)

    disc = Meshed(RigidBody)(
        mesh_obj=disc_mesh,
        B_r_CP=B_r_DM,  # vector from the center of mass to the centerline
        A_BM=A_IB_basic(np.deg2rad(-90)).y,
        mass=m_d,
        B_Theta_C=B_Theta_C,
        q0=q0,
        u0=u0,
        name="disc",
    )

    # gravity load of the disc
    F_g_disc = -m_d * 9.81 * e3
    force_disc = Force(F_g_disc, disc, name="gravity_disc")
    system.add(disc, force_disc)

    ##########
    # rod(s) #
    ##########
    assert nelement % 2 == 0, "Number of elements must be even for disc in the middle!"
    q0, u0 = Rod.straight_initial_configuration(
        nelement, L, r_OR0, A_IR0, v_P0, B_omega_IB0
    )

    rod = Rod(
        cross_section,
        material_model,
        nelement,
        Q=q0,
        q0=q0,
        u0=u0,
        cross_section_inertias=cross_section_inertias,
        name="rod",
    )

    # add force line distributed gravity load
    F_g = lambda t, xi: -9.81 * A_rho0 * e3
    # F_g = lambda t, xi: -9.81 * e3
    force_line = Force_line_distributed(F_g, rod)

    # connect disc to rod
    connection_cylinder = RigidConnection(rod, disc, xi1=0.5, r_OJ0=r_OM0)
    system.add(rod, force_line, connection_cylinder)

    ##############################
    # add cylindrical constraint #
    ##############################
    cylindrical = Cylindrical(rod, system.origin, axis=0, xi1=1)
    system.add(cylindrical)

    #################################
    # solver for static equilibrium #
    #################################
    clamping_static = RigidConnection(rod, system.origin, xi1=0)
    system.add(clamping_static)

    # assemble system
    system.assemble(SolverOptions(compute_consistent_initial_conditions=False))
    sol = Newton(system, 1).solve()

    system.set_new_initial_state(sol.q[-1], sol.u[-1])
    system.remove(clamping_static)

    # creating rotation around x-axis of the actuated rod with the time dependent angle excitation phi(t)
    phi0 = 0.0  # initial rotation angle of the actuated rod
    A_IR = lambda t: A_IB_basic(phi(t, phi0)).x
    A_IR_t = lambda t: A_IB_basic(phi(t, phi0)).dx * Omega(t)
    # connect actuated rod to the rotating frame for displacement controlled case
    rotating_frame = Frame(
        r_OP=r_OR0,
        A_IB=A_IR,
        A_IB_t=A_IR_t,
        name="RotatingFrameActuation",
    )
    clamping = RigidConnection(rod, rotating_frame, xi1=0)
    system.add(rotating_frame, clamping)

    # assemble system
    system.assemble()

    # Solver parameters
    solver = ScipyDAE(system, t_final, dt, method="Radau")
    solver = DualStormerVerlet(system, t_final, dt)

    #################
    # Simulation
    #################
    sol = solver.solve()
    q = sol.q
    nt = len(q)
    t = sol.t[:nt]
    u = sol.u

    #################
    # post-processing
    #################

    # VTK export
    dir_name = Path(__file__).parent
    if VTK_export:
        # trick rod to export a circular cross section
        rod.cross_section = CircularCrossSection(r_O)

        # trick system to export disc with base vectors
        B2 = RigidBody(m_d, B_Theta_C, name="Disc_axes")
        B2.qDOF = disc.qDOF
        B2.uDOF = disc.uDOF

        system.add(B2)
        system.export(dir_name, f"vtk/", sol)

    path = Path(dir_name, "csv")
    path.mkdir(parents=True, exist_ok=True)

    ##################
    # Exciation: angular velocity and rotation angle of the actuated rod
    ##################

    # Evaluate and plot
    t_vals = np.linspace(0, t_final, 1_000)
    phi0 = 0.0
    phi_vals = np.array([phi(t, phi0) for t in t_vals])
    omega_vals = np.array([Omega(t) for t in t_vals])

    phi_numerical = cumulative_trapezoid(omega_vals, t_vals, initial=0)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(t_vals, omega_vals, label="Omega(t)", color="steelblue")
    plt.title("Angular Velocity")
    plt.xlabel("Time [s]")
    plt.ylabel("Omega [rad/s]")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(t_vals, phi_vals, label="Analytical phi(t)", color="darkorange")
    plt.plot(t_vals, phi_numerical, "--", label="Numerical phi(t)", color="green")
    plt.title("Comparison of Analytical vs Numerical phi(t)")
    plt.xlabel("Time [s]")
    plt.ylabel("phi [rad]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # save the figure
    plt.savefig(f"{dir_name}/{save_name}_excitation.png", dpi=300, bbox_inches="tight")

    # Create a DataFrame
    df = pd.DataFrame(
        {
            "time [s]": t_vals,
            "Omega(t) [rad/s]": omega_vals,
            "phi_analytical(t) [rad]": phi_vals,
            "phi_numerical(t) [rad]": phi_numerical,
        }
    )
    csv_path = path / f"excitation_data_{save_name}.csv"

    # Export to CSV
    df.to_csv(csv_path, index=False)
    print(f"Excitation data saved to {csv_path}")

    ##############################################
    # plot for displacement in prescribed system #
    ##############################################
    r_OM = np.array(
        [disc.r_OP(ti, qi, B_r_CP=B_r_DM) for (ti, qi) in zip(t, q[:, disc.qDOF])]
    )
    ex_D = np.array([disc.A_IB(ti, qi) @ e1 for (ti, qi) in zip(t, q[:, disc.qDOF])])
    R_r_OM = np.array(
        [
            A_IR(ti).T @ disc.r_OP(ti, qi, B_r_CP=B_r_DM)
            for (ti, qi) in zip(t, q[:, disc.qDOF])
        ]
    )
    fig, ax = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    ax[1].scatter(R_r_OM[:, 1], R_r_OM[:, 2], c=t, cmap="plasma", s=10)
    ax[1].grid()

    ax[0].scatter(r_OM[:, 1], r_OM[:, 2], c=t, cmap="plasma", s=10)
    ax[0].grid()

    fig, ax = plt.subplots(1, 1)
    ax.plot(t, ex_D[:, 0], label="ex_D @ ex_I")
    ax.plot(t, ex_D[:, 1], label="ex_D @ ey_I")
    ax.plot(t, ex_D[:, 2], "--", label="ex_D @ ez_I")
    ax.set_title("ex Disc")
    ax.grid()
    ax.legend()
    ax.set_xlabel("t")

    #################
    # phase plot
    #################
    u_x = r_OM[:, 0] - r_OM0[0] * np.ones_like(
        r_OM[:, 0]
    )  # x-displacement of the center point of the disc
    u_y = r_OM[:, 1] - r_OM0[1] * np.ones_like(
        r_OM[:, 1]
    )  # y-displacement of the center point of the disc
    u_z = r_OM[:, 2] - r_OM0[2] * np.ones_like(
        r_OM[:, 2]
    )  # z-displacement of the center point of the disc

    v_M = np.array(
        [
            disc.v_P(ti, qi, ui, B_r_CP=B_r_DM)
            for (ti, qi, ui) in zip(t, q[:, disc.qDOF], u[:, disc.uDOF])
        ]
    )
    v_x = v_M[:, 0]  # x-velocity of the center point of the disc
    v_y = v_M[:, 1]  # y-velocity of the center point of the disc
    v_z = v_M[:, 2]  # z-velocity of the center point of the disc

    # Create figure and subplots
    fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    # colormap = 'viridis'
    colormap = "plasma"

    # --- Subplot 1: u_y vs v_y ---
    sc1 = axs[0].scatter(u_y, v_y, c=t, cmap=colormap, s=10)
    axs[0].set_xlabel("Displacement u_y (m)")
    axs[0].set_ylabel("Velocity v_y (m/s)")
    axs[0].set_title("Phase Plot: Y-direction")
    axs[0].grid(True)

    # --- Subplot 2: u_z vs v_z ---
    sc2 = axs[1].scatter(u_z, v_z, c=t, cmap=colormap, s=10)
    axs[1].set_xlabel("Displacement u_z (m)")
    axs[1].set_ylabel("Velocity v_z (m/s)")
    axs[1].set_title("Phase Plot: Z-direction")
    axs[1].grid(True)

    # Add one shared colorbar to the right of both plots
    fig.subplots_adjust(right=0.85)  # Make room for the colorbar
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(sc1, cax=cbar_ax)
    cbar.set_label("Time (s)")

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # leave space for colorbar

    # Save the figure
    fig.savefig(f"{dir_name}/{save_name}_phase_plot.png", dpi=300, bbox_inches="tight")

    # Create a DataFrame
    df = pd.DataFrame(
        {
            "time [s]": t,
            "u_y [m]": u_y,
            "v_y [m/s]": v_y,
            "u_z [m]": u_z,
            "v_z [m/s]": v_z,
        }
    )
    csv_path = path / f"phase_plot_data_{save_name}.csv"
    # Export to CSV
    df.to_csv(csv_path, index=False)
    print(f"Phase plot data saved to {csv_path}")

    ##########################
    # matplotlib visualization of the beams
    ##########################
    # construct animation of beam
    fig1, ax1, anim1 = animate_beam(
        t,
        q,
        [rod],
        scale=L,
        scale_di=0.1,
        show=False,
        n_frames=rod.nelement + 1,
        repeat=True,
    )

    if show_plots:
        plt.show()

    anim1.save(f"{dir_name/save_name}.mp4", writer="ffmpeg", fps=30)


if __name__ == "__main__":
    # rotating_shaft(
    #     Rod=make_CosseratRod(
    #         interpolation="Quaternion", mixed=True, constraints=[0, 1, 2]
    #     ),
    #     constitutive_law=Simo1986,
    #     nelement=20,
    #     t_final=2.5,
    #     dt=0.1 * 1e-2,
    #     show_plots=False,
    #     VTK_export=True,
    #     name="IEB",
    # )
    rotating_shaft(
        Rod=make_CosseratRod(interpolation="Quaternion", mixed=True),
        constitutive_law=Simo1986,
        nelement=20,
        t_final=2.5,
        dt=0.1 * 1e-3,
        VTK_export=True,
        name="Cosserat",
    )
