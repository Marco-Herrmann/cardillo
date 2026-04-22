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


def r_OP0_fun(xi, length, l_s, l_u, height):
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
    else:
        return np.array([l_s - (xi * length - l_u - l_s), 0.0, height])


def A_IB0_fun(xi, length, l_s, l_u, height):
    if xi * length <= l_s:
        alpha = 0.0
    elif xi * length <= l_s + l_u:
        alpha = -(xi * length - l_s) / l_u * np.pi
    else:
        alpha = -np.pi
    return A_IB_basic(alpha).y


def motion_stage(l_x0, l_y0):
    assert -35.0 <= l_x0 <= 35.0, "Bounds for l_x0 (long stroke) are +- 35.0"
    assert -15.0 <= l_y0 <= 15.0, "Bounds for l_y0 (short stroke) are +- 20.0"

    # cable dimensions geometry
    r_cable = 0.375 / 2

    # cable materialz
    E = 1.0 * 1e7
    G = 0.5 * 1e7
    density = 1.0

    # geometry
    r_OP1_lower = np.array([-42.0, 47.0, 24.3])
    r_OP1_upper = np.array([-11.45, 47.0, 35.8])
    r_OP2a_lower = np.array([42.0, -47.0, 24.3])
    r_OP2a_upper = np.array([14.55, -47.0, 35.8])
    r_OP2b_lower = np.array([17.05, -33.5, 30.3])
    r_OP2b_upper = np.array([17.05, 12.0, 37.7])
    length1 = 96.9
    length2a = 100.0
    length2b = 78.1

    # actuation
    actuation_amplitude_long = 1e2
    actuation_amplitude_short = 0.0
    actuation_frequency_long = 2.0  # [1/s = Hz]
    actuation_frequency_short = 2.0  # [1/s = Hz]

    # masses of stages
    mass_long = 1.0
    mass_short = 1.0

    # discretization and model
    system = System()
    nelement1 = 20
    nelement2a = 20
    nelement2b = 20
    Cable = make_BoostedCosseratRod()

    t1 = 1.0
    dt = 0.001

    # Rigid bodies for table, long stroke and short stroke
    # TODO: spring-damper on table <--> system.origin do excite some dynamics
    table = RigidBody(
        mass=1.0,
        B_Theta_C=np.diag([1.0, 1.0, 1.0]),
        q0=RigidBody.pose2q(np.array([0.0, 0.0, -5.0]), np.eye(3)),
        name="Table",
    )

    # TODO: B_Theta_C
    long_stroke = RigidBody(
        mass=mass_long,
        B_Theta_C=np.diag([1.0, 1.0, 1.0]) * mass_long,
        q0=RigidBody.pose2q(np.array([0.0, 0.0, 32.5]), np.eye(3)),
        name="long_stroke",
    )

    # TODO: B_Theta_C
    short_stroke = RigidBody(
        mass=mass_short,
        B_Theta_C=np.diag([1.0, 1.0, 1.0]) * mass_short,
        q0=RigidBody.pose2q(np.array([0.0, 0.0, 39.0]), np.eye(3)),
        name="short_stroke",
    )

    # cross section
    cross_section = RectangularCrossSection(width=8 * r_cable, height=2 * r_cable)
    A = cross_section.area
    Ix, Iy, Iz = np.diag(cross_section.second_moment)

    # TODO: get this into Simo
    Ei = np.array([E * A, G * A, G * A])
    Fi = np.array([G * Ix, E * Iy, E * Iz])
    material_model = Simo1986(Ei, Fi)

    # cross section inertias
    cross_section_inertias = CrossSectionInertias(density, cross_section)

    b = lambda t, xi: np.array([0.0, 0.0, -9.81 * A * density])

    # frames for contact
    # TODO: restrict plane where contact can happen
    frame_contact_lower = Frame(r_OP1_lower, name="frame_contact_lower")
    frame_contact_upper = Frame(
        r_OP1_upper, A_IB=A_IB_basic(np.pi).y, name="frame_contact_upper"
    )
    frame_contact_lower_2b = Frame(r_OP2b_lower, name="frame_contact_lower_2b")
    system.add(frame_contact_lower, frame_contact_upper, frame_contact_lower_2b)

    ########################
    # Cable 1: Long stroke #
    ########################
    # reference configuration
    Q1 = Cable.straight_configuration(nelement1, length1)

    # initial configuration
    height1 = (r_OP1_upper - r_OP1_lower)[2]
    l_u1 = np.pi * height1 / 2
    # TODO: is there an elegant way to get l_s?
    l_s1 = (length1 - l_u1) / 2
    d_x1 = (r_OP1_upper - r_OP1_lower)[0]
    l_s1 = (length1 - l_u1 + d_x1) / 2
    print(
        f"length on ground: {l_s1}, length in semi-circle: {l_u1},total length: {length1}, height: {height1}"
    )
    q01 = Cable.pose_configuration(
        nelement1,
        lambda xi: r_OP0_fun(xi, length1, l_s1, l_u1, height1),
        lambda xi: A_IB0_fun(xi, length1, l_s1, l_u1, height1),
        r_OP0=r_OP1_lower,
    )

    cable1 = Cable(
        cross_section=cross_section,
        material_model=material_model,
        nelement=nelement1,
        cross_section_inertias=cross_section_inertias,
        Q=Q1,
        q0=q01,
        distributed_load=[b, None],
        name="cable1",
    )

    # clamp cable
    constraint1_lower = RigidConnection(
        cable1,
        system.origin,
        xi1=0.0,
        r_OJ0=r_OP1_lower,
        name="constraint1_lower",
    )
    constraint1_upper = RigidConnection(
        cable1,
        long_stroke,
        xi1=1.0,
        r_OJ0=r_OP1_upper,
        name="constraint1_upper",
    )

    # contacts
    for node in range(1, cable1.nnodes - 1):
        contact_lower = Sphere2Plane(
            frame_contact_lower,
            cable1,
            mu=0.0,
            r=0.0,
            xi=node / (cable1.nnodes - 1),
            name=f"contact_cable1_node{node:0>2d}_lower",
        )
        contact_upper = Sphere2Plane(
            frame_contact_upper,
            cable1,
            mu=0.0,
            r=0.0,
            xi=node / (cable1.nnodes - 1),
            name=f"contact_cable1_node{node:0>2d}_upper",
        )
        system.add(contact_lower, contact_upper)

    ###########################
    # Cable 2: Long stroke    #
    # a: table to long stroke #
    # b: long to short stroke #
    ###########################
    # reference configuration
    Q2a = Cable.straight_configuration(nelement2a, length2a)

    # initial configuration
    height2a = (r_OP2a_upper - r_OP2a_lower)[2]
    l_u2a = np.pi * height2a / 2
    l_s2a = np.abs((r_OP2a_upper - r_OP2a_lower)[0])
    d_x2a = (r_OP2a_upper - r_OP2a_lower)[0]
    l_s2a = (length2a - l_u2a - d_x2a) / 2
    print(
        f"length on ground: {l_s2a}, length in semi-circle: {l_u2a},total length: {length2a}, height: {height2a}"
    )
    q02a = Cable.pose_configuration(
        nelement2a,
        lambda xi: r_OP0_fun(xi, length2a, l_s2a, l_u2a, height2a),
        lambda xi: A_IB0_fun(xi, length2a, l_s2a, l_u2a, height2a),
        r_OP0=r_OP2a_lower,
        A_IB0=A_IB_basic(np.pi).z,
    )

    cable2a = Cable(
        cross_section=cross_section,
        material_model=material_model,
        nelement=nelement2a,
        cross_section_inertias=cross_section_inertias,
        Q=Q2a,
        q0=q02a,
        distributed_load=[b, None],
        name="cable2a",
    )

    # clamp cable
    constraint2a_lower = RigidConnection(
        cable2a,
        system.origin,
        xi1=0.0,
        r_OJ0=r_OP2a_lower,
        name="constraint2a_lower",
    )
    constraint2a_upper = RigidConnection(
        cable2a,
        long_stroke,
        xi1=1.0,
        r_OJ0=r_OP2a_upper,
        name="constraint2a_upper",
    )

    # contacts
    for node in range(1, cable2a.nnodes - 1):
        contact_lower = Sphere2Plane(
            frame_contact_lower,
            cable2a,
            mu=0.0,
            r=0.0,
            xi=node / (cable2a.nnodes - 1),
            name=f"contact_cable2a_node{node:0>2d}_lower",
        )
        contact_upper = Sphere2Plane(
            frame_contact_upper,
            cable2a,
            mu=0.0,
            r=0.0,
            xi=node / (cable2a.nnodes - 1),
            name=f"contact_cable2a_node{node:0>2d}_upper",
        )
        system.add(contact_lower, contact_upper)

    # reference configuration
    Q2b = Cable.straight_configuration(nelement2b, length2b)

    # initial configuration
    height2b = (r_OP2b_upper - r_OP2b_lower)[2]
    l_u2b = np.pi * height2b / 2
    l_s2b = (length2b - l_u2b) / 2
    d_x2b = (r_OP2b_upper - r_OP2b_lower)[1]
    l_s2b = (length2b - l_u2b + d_x2b) / 2
    print(
        f"length on ground: {l_s2b}, length in semi-circle: {l_u2b},total length: {length2b}, height: {height2b}"
    )
    q02b = Cable.pose_configuration(
        nelement2b,
        lambda xi: r_OP0_fun(xi, length2b, l_s2b, l_u2b, height2b),
        lambda xi: A_IB0_fun(xi, length2b, l_s2b, l_u2b, height2b),
        r_OP0=r_OP2b_lower,
        A_IB0=A_IB_basic(np.pi / 2).z,
    )

    cable2b = Cable(
        cross_section=cross_section,
        material_model=material_model,
        nelement=nelement2b,
        cross_section_inertias=cross_section_inertias,
        Q=Q2b,
        q0=q02b,
        distributed_load=[b, None],
        name="cable2b",
    )

    # clamp cable
    constraint2b_lower = RigidConnection(
        cable2b,
        long_stroke,
        xi1=0.0,
        r_OJ0=r_OP2b_lower,
        name="constraint2b_lower",
    )
    constraint2b_upper = RigidConnection(
        cable2b,
        short_stroke,
        xi1=1.0,
        r_OJ0=r_OP2b_upper,
        name="constraint2b_upper",
    )

    # contacts
    for node in range(1, cable2b.nnodes - 1):
        contact_lower = Sphere2Plane(
            frame_contact_lower_2b,
            cable2b,
            mu=0.0,
            r=0.0,
            xi=node / (cable2b.nnodes - 1),
            name=f"contact_cable2b_node{node:0>2d}_lower",
        )
        # no upper contact
        system.add(contact_lower)

    # constrain rigid bodies
    table_constraint = RigidConnection(system.origin, table, name="origin-table")
    frame_long_stat = Frame(
        lambda t: np.array([l_x0 * t, 0.0, 0.0]), name="frame_long_stat"
    )
    frame_short_stat = Frame(
        lambda t: np.array([l_x0 * t, l_y0 * t, 0.0]), name="frame_short_stat"
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

    force_long_ampl = (
        e1 * (long_stroke.mass + short_stroke.mass) * actuation_amplitude_long
    )
    force_long = lambda t: force_long_ampl * np.cos(
        2 * np.pi * actuation_frequency_long * t
    )
    forcing_long = [
        Force(force_long, long_stroke, name="forcing_long+"),
        Force(lambda t: -force_long(t), table, name="forcing_long-"),
    ]
    force_short_ampl = e2 * short_stroke.mass * actuation_amplitude_short
    force_short = lambda t: force_short_ampl * np.cos(
        2 * np.pi * actuation_frequency_short * t
    )
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
    system.add(cable1, constraint1_lower, constraint1_upper)
    system.add(cable2a, constraint2a_lower, constraint2a_upper)
    system.add(cable2b, constraint2b_lower, constraint2b_upper)
    system.assemble(options=assemble_options)

    # static solver
    solver_stat = Newton(system, 10)
    sol_stat = solver_stat.solve()

    # visualize static result
    animate_beam(
        sol_stat.t,
        sol_stat.q,
        [cable1, cable2a, cable2b],
        scale=length2a,
        scale_di=length2a / 10,
    )

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
    ax[0, 1].plot(sol_dyn.t, x_long)
    ax[0, 0].plot(sol_dyn.t, [force_long(ti)[0] for ti in sol_dyn.t])
    ax[1, 1].plot(sol_dyn.t, y_short)
    ax[1, 0].plot(sol_dyn.t, [force_short(ti)[1] for ti in sol_dyn.t])
    plt.show()

    # vtk-export
    dir_name = Path(__file__).parent
    system.export(dir_name, "vtk_02", sol_dyn, fps=25)

    # make nice visuals with multiple individual cables
    if False:
        ny = 10
        nz = 2
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
    # motion_stage(0.0, 0.0)
    motion_stage(-30.0, -15.0)
