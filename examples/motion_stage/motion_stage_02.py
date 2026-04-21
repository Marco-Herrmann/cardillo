import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from cardillo import System
from cardillo.rods.boostedCosseratRod import make_BoostedCosseratRod
from cardillo.rods import (
    RectangularCrossSection,
    CrossSectionInertias,
    Simo1986,
    CircularCrossSection,
)
from cardillo.discrete import RigidBody, Frame
from cardillo.constraints import RigidConnection, Prismatic
from cardillo.forces import Force
from cardillo.force_laws import KelvinVoigtElement as SpringDamper
from cardillo.interactions import TwoPointInteraction
from cardillo.rods import animate_beam

from cardillo.contacts import Sphere2Plane

from cardillo.math import A_IB_basic, e1, e2

from cardillo.solver import (
    Newton,
    Moreau,
    DualStormerVerlet,
    BackwardEuler,
    SolverOptions,
    Solution,
)

# sources:
# https://ioplus.nl/en/posts/solving-dynamic-challenges-in-next-generation-cable-slabs
# https://past.isma-isaac.be/downloads/isma2014/papers/isma2014_0509.pdf
# https://www.youtube.com/watch?v=sgF-0xj2sD8
# https://pure.tue.nl/admin/files/317778356/1722107-ThesisProjectReport.pdf


def motion_stage(l_x0, l_y0):
    assert -35.0 <= l_x0 <= 35.0, "Bounds for l_x0 (long stroke) are +- 35.0"
    assert -15.0 <= l_y0 <= 15.0, "Bounds for l_y0 (short stroke) are +- 20.0"

    # geometry
    length = 1.5
    height = 0.3
    h = 0.02
    w = 0.1

    # material
    E = 1.0 * 1e4
    G = 0.5 * 1e4
    density = 1.0

    # actuation
    actuation_amplitude = 0.2
    actuation_frequency = 2.0  # [1/s = Hz]

    # discretization and model
    system = System()
    nelement = 20
    Cable = make_BoostedCosseratRod(
        # idx_constraints=[0, 1, 2],
    )

    t1 = 1.0
    dt = 0.001

    # Rigid bodies for table, long stroke and short stroke
    table = RigidBody(
        mass=1.0,
        B_Theta_C=np.diag([1.0, 1.0, 1.0]),
        q0=RigidBody.pose2q(np.array([0.0, 0.0, -5.0]), np.eye(3)),
        name="Table",
    )

    long_stroke = RigidBody(
        mass=1.0,
        B_Theta_C=np.diag([1.0, 1.0, 1.0]),
        q0=RigidBody.pose2q(np.array([0.0, 0.0, 32.5]), np.eye(3)),
        name="long_stroke",
    )

    short_stroke = RigidBody(
        mass=1.0,
        B_Theta_C=np.diag([1.0, 1.0, 1.0]),
        q0=RigidBody.pose2q(np.array([0.0, 0.0, 39.0]), np.eye(3)),
        name="short_stroke",
    )

    # cross section
    cross_section = RectangularCrossSection(width=w, height=h)
    A = cross_section.area
    Ix, Iy, Iz = np.diag(cross_section.second_moment)

    # TODO: get this into Simo
    Ei = np.array([E * A, G * A, G * A])
    Fi = np.array([G * Ix, E * Iy, E * Iz])
    material_model = Simo1986(Ei, Fi)

    # cross section inertias
    cross_section_inertias = CrossSectionInertias(density, cross_section)

    b = lambda t, xi: np.array([0.0, 0.0, -9.81 * A * density])

    # reference configuration
    Q = Cable.straight_configuration(nelement, length)

    # initial configuration
    l_u = np.pi * height / 2
    l_s = (length - l_u) / 2
    print(
        f"length on ground: {l_s}, length in semi-circle: {l_u},total length: {length}, height: {height}"
    )

    def r_OP0(xi):
        if xi * length <= l_s:
            return np.array([xi * length, 0.0, 0.0])
        elif xi * length <= l_s + l_u:
            alpha = (xi * length - l_s) / l_u
            return np.array(
                [
                    l_s + height / 2 * np.sin(np.pi * alpha),
                    0.0,
                    height / 2 * (1 - np.cos(np.pi * alpha)),
                ]
            )
        elif xi * length <= l_s + l_u + l_s:
            return np.array([(1 - xi) * length, 0.0, height])
        else:
            raise NotImplementedError

    def A_IB0(xi):
        if xi * length <= l_s:
            alpha = 0.0
        elif xi * length <= l_s + l_u:
            alpha = -(xi * length - l_s) / l_u * np.pi
        elif xi * length <= l_s + l_u + l_s:
            alpha = -np.pi
        return A_IB_basic(alpha).y

    q0 = Cable.pose_configuration(nelement, r_OP0, A_IB0)

    cable = Cable(
        cross_section=cross_section,
        material_model=material_model,
        nelement=nelement,
        cross_section_inertias=cross_section_inertias,
        Q=Q,
        q0=q0,
        distributed_load=[b, None],
        name="cable",
    )

    # clamp to ground
    constraint_lower = RigidConnection(
        cable,
        system.origin,
        xi1=0.0,
        r_OJ0=system.origin.r_OP(0.0),
        name="constraint0",
    )

    # prescribe stage movement
    r_OP_stat = lambda t: np.array([l_x0 * t, 0.0, height])
    omega = 2 * np.pi * actuation_frequency
    r_OP_dyn = lambda t: r_OP_stat(1) + actuation_amplitude * np.sin(omega * t) * e1
    contr_stat_dyn = [[], []]
    r_OP_stat_dyn = [
        r_OP_stat,
        r_OP_dyn,
    ]
    A_IB_stat_dyn = [
        A_IB_basic(-np.pi).y,
        A_IB_basic(-np.pi).y,
    ]

    for i, (c, r_OP, A_IB) in enumerate(
        zip(contr_stat_dyn, r_OP_stat_dyn, A_IB_stat_dyn)
    ):
        stat_dyn = "static" if i == 0 else "dynamic"
        frame_upper = Frame(
            r_OP=r_OP,
            A_IB=A_IB,
            name=f"frame_upper_{stat_dyn}",
        )
        constraint_upper = RigidConnection(
            cable,
            frame_upper,
            xi1=1.0,
            r_OJ0=frame_upper.r_OP(0.0),
            name=f"constraint1_{stat_dyn}",
        )
        c.extend([frame_upper, constraint_upper])

    # contacts
    for node in range(1, cable.nnodes - 1):
        # connect to origin
        contact_lower = Sphere2Plane(
            system.origin,
            cable,
            mu=0.0,
            r=0.0,
            xi=node / (cable.nnodes - 1),
            name=f"contact_node{node:0>2d}_lower",
        )
        # system.add(contact_lower)

        # connect to stage (static and dynamic)
        for i, c in enumerate(contr_stat_dyn):
            stat_dyn = "static" if i == 0 else "dynamic"
            contact_upper = Sphere2Plane(
                c[0],
                cable,
                mu=0.0,
                r=0.0,
                xi=node / (cable.nnodes - 1),
                name=f"contact_node{node:0>2d}_upper_{stat_dyn}",
            )
            c.append(contact_upper)

    # constrain rigid bodies
    table_constraint = RigidConnection(system.origin, table, name="origin-table")
    frame_long_stat = Frame(
        lambda t: np.array([l_x0 * t, 0.0, 0.0]), name="frame_long_stat"
    )
    frame_short_stat = Frame(
        lambda t: np.array([l_x0 * t, l_y0 * t, 0.0]), name="frame_long_stat"
    )
    long_stroke_constraint_stat = RigidConnection(
        long_stroke, frame_long_stat, name="long_stroke_stat"
    )
    short_stroke_constraint_stat = RigidConnection(
        short_stroke, frame_short_stat, name="short_stroke_stat"
    )
    long_stroke_constraint_dyn = Prismatic(
        table, long_stroke, axis=0, name="table-long_stroke_dyn"
    )
    short_stroke_constraint_dyn = Prismatic(
        long_stroke, short_stroke, axis=1, name="long-stroke-short_stroke_dyn"
    )

    force_long = (
        lambda t: e1
        * (long_stroke.mass + short_stroke.mass)
        * 1e2
        * np.cos(omega * t)
        * 0.0
    )
    forcing_long = [
        Force(force_long, long_stroke, name="forcing_long+"),
        Force(lambda t: -force_long(t), table, name="forcing_long-"),
    ]
    force_short = lambda t: e2 * short_stroke.mass * 1e3 * np.cos(2 * omega * t) * 0.0
    forcing_short = [
        Force(force_short, short_stroke, name="forcing_short+"),
        Force(lambda t: -force_short(t), long_stroke, name="forcing_short-"),
    ]

    # assemble system
    assemble_options = SolverOptions(compute_consistent_initial_conditions=False)
    system.add(table, long_stroke, short_stroke, table_constraint)
    system.add(
        long_stroke_constraint_stat,
        short_stroke_constraint_stat,
        frame_long_stat,
        frame_short_stat,
    )
    # system.add(cable)
    # system.add(constraint_lower)
    # system.add(*contr_stat_dyn[0])
    system.assemble(options=assemble_options)

    # static solver
    solver_stat = Newton(system, 10)
    sol_stat = solver_stat.solve()

    # # visualize static result
    # animate_beam(sol_stat.t, sol_stat.q, [cable], scale=1.0, scale_di=0.1)

    # prepare for dynamic simulation
    system.set_new_initial_state(
        sol_stat.q[-1], sol_stat.u[-1], t0=0.0, options=assemble_options
    )
    system.remove(
        long_stroke_constraint_stat,
        short_stroke_constraint_stat,
        frame_long_stat,
        frame_short_stat,
    )
    system.add(long_stroke_constraint_dyn, short_stroke_constraint_dyn)
    system.add(*forcing_long, *forcing_short)
    # system.remove(*contr_stat_dyn[0])
    # system.add(*contr_stat_dyn[1])
    system.assemble(options=assemble_options)

    # dynamic solver
    solver_dyn = BackwardEuler(system, t1=t1, dt=dt)
    # solver_dyn = DualStormerVerlet(system, t1=t1, dt=dt)
    # solver_dyn = Moreau(system, t1=t1, dt=dt)
    sol_dyn = solver_dyn.solve()

    # visualize dynamic result
    # animate_beam(sol_dyn.t, sol_dyn.q, [cable], scale=1.0, scale_di=0.1)

    x_long = np.array(
        [sol_dyn.q[i, long_stroke.qDOF[0]] for i in range(len(sol_dyn.t))]
    )
    y_short = np.array(
        [sol_dyn.q[i, short_stroke.qDOF[1]] for i in range(len(sol_dyn.t))]
    )

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(sol_dyn.t, x_long)
    ax[1, 0].plot(sol_dyn.t, y_short)
    ax[0, 1].plot(sol_dyn.t, [force_long(ti)[0] for ti in sol_dyn.t])
    ax[1, 1].plot(sol_dyn.t, [force_short(ti)[1] for ti in sol_dyn.t])
    plt.show()

    # vtk-export
    dir_name = Path(__file__).parent
    system.export(dir_name, "vtk_02", sol_dyn, fps=25)

    # make nice visuals with multiple individual cables
    if False:
        ny = 10
        nz = 2
        r_cable = h / nz / 2
        cross_section_vis = CircularCrossSection(radius=r_cable)
        r_OP_vis = lambda t, q, xi, B_r_CP: cable.r_OP(
            t, q[cable.qDOF[cable.local_qDOF_P(xi)]], xi=xi, B_r_CP=B_r_CP
        )
        A_IB_vis = lambda t, q, xi: cable.A_IB(
            t, q[cable.qDOF[cable.local_qDOF_P(xi)]], xi=xi
        )
        system_mult = System()
        q0s_mult = np.zeros((ny, nz, len(sol_dyn.t), cable.nq))
        cables = np.zeros((ny, nz), dtype=object)
        for iy in range(ny):
            for iz in range(nz):
                B_r_CP = (
                    np.array([0.0, iy - ny / 2 + 1 / 2, iz - nz / 2 + 1 / 2])
                    * r_cable
                    * 2
                )
                for it in range(len(sol_dyn.t)):
                    q0 = Cable.pose_configuration(
                        nelement,
                        lambda xi: r_OP_vis(sol_dyn.t[it], sol_dyn.q[it], xi, B_r_CP),
                        lambda xi: A_IB_vis(sol_dyn.t[it], sol_dyn.q[it], xi),
                    )
                    q0s_mult[iy, iz, it, :] = q0
                cables[iy, iz] = Cable(
                    cross_section_vis,
                    material_model,
                    nelement,
                    Q=q0s_mult[iy, iz, 0, :],
                    name=f"cable_{iy:0>2d}_{iz:0>2d}",
                )
                system_mult.add(cables[iy, iz])

        system_mult.assemble(options=assemble_options)
        q_mult = np.zeros((len(sol_dyn.t), system_mult.nq))
        for it in range(len(sol_dyn.t)):
            for iy in range(ny):
                for iz in range(nz):
                    q_mult[it, cables[iy, iz].qDOF] = q0s_mult[iy, iz, it, :]

        sol_mult = Solution(
            system_mult,
            t=sol_dyn.t,
            q=q_mult,
        )
        system_mult.export(dir_name, "vtk_02_multiple", sol_mult, fps=25)


if __name__ == "__main__":
    # -35 <= l_x <= 35, -15 <= l_y <= 15
    motion_stage(-30.0, -15.0)
