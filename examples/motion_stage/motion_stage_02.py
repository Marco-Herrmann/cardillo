import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from cardillo import System
from cardillo.actuators.constraint import ActuatedConstraint
from cardillo.rods_new import (
    RectangularCrossSection,
    CrossSectionInertias,
    Simo1986,
    CircularCrossSection,
)
from cardillo.rods_new2 import make_CosseratRod
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
    Eigenmodes,
    FrequencyResponseFunction,
)

# sources:
# https://ioplus.nl/en/posts/solving-dynamic-challenges-in-next-generation-cable-slabs
# https://past.isma-isaac.be/downloads/isma2014/papers/isma2014_0509.pdf
# https://www.youtube.com/watch?v=sgF-0xj2sD8
# https://pure.tue.nl/admin/files/317778356/1722107-ThesisProjectReport.pdf

eye3 = np.eye(3, dtype=float)


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
    n_stat = 10

    # cable dimensions geometry
    r_cable = 0.375 / 2

    # TODO: tune these parameters
    # cable material
    E = 5.0 * 1e5  # 5 MPa
    G = 2.5 * 1e5
    density = 3.5 * 1e-3  # 3_500 kg/m^3

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
    actuation_amplitude_long = 5.0
    actuation_amplitude_short = 2.0
    actuation_frequency_long = 2.0  # [1/s = Hz]
    actuation_frequency_short = 2.0  # [1/s = Hz]

    # masses of stages
    mass_long = 50.0
    mass_short = 25.0

    # discretization and model
    system = System()
    nelement1 = 20
    nelement2a = 20
    nelement2b = 20
    Cable = make_CosseratRod(
        idx_constraints=[3],
    )

    t1 = 1.0
    dt = 0.001

    # Rigid bodies for table, long stroke and short stroke
    # TODO: spring-damper on table <--> system.origin do excite some dynamics
    r_OP0_table = np.array([0.0, 0.0, -5.0])
    A_IB0_table = eye3
    table = RigidBody(
        mass=1.0,
        B_Theta_C=np.diag([1.0, 1.0, 1.0]),
        q0=RigidBody.pose2q(r_OP0_table, A_IB0_table),
        name="Table",
    )

    # TODO: B_Theta_C
    r_OP0_long_stroke = np.array([0.0, 0.0, 32.5])
    A_IB0_long_stroke = eye3
    long_stroke = RigidBody(
        mass=mass_long,
        B_Theta_C=np.diag([1.0, 1.0, 1.0]) * mass_long,
        q0=RigidBody.pose2q(r_OP0_long_stroke, A_IB0_long_stroke),
        name="long_stroke",
    )

    # TODO: B_Theta_C
    r_OP0_short_stroke = np.array([0.0, 0.0, 39.0])
    A_IB0_short_stroke = eye3
    short_stroke = RigidBody(
        mass=mass_short,
        B_Theta_C=np.diag([1.0, 1.0, 1.0]) * mass_short,
        q0=RigidBody.pose2q(r_OP0_short_stroke, A_IB0_short_stroke),
        name="short_stroke",
    )

    # constrain rigid bodies
    table_constraint = RigidConnection(system.origin, table, name="origin-table")
    long_stroke_constraint = Prismatic(
        table, long_stroke, axis=0, name="table-long_stroke_dyn"
    )
    short_stroke_constraint = Prismatic(
        long_stroke, short_stroke, axis=1, name="long-stroke-short_stroke_dyn"
    )

    actuation_long_stat = lambda t: l_x0 * t
    actuation_long_dyn = lambda t: actuation_amplitude_long * np.sin(
        2 * np.pi * actuation_frequency_long * t
    )

    actuation_short_stat = lambda t: l_y0 * t
    actuation_short_dyn = lambda t: actuation_amplitude_short * np.sin(
        2 * np.pi * actuation_frequency_short * t
    )

    ####
    # actuation static to go to 9 positions in the workspace
    ####
    #         ^ y
    #         |
    #   2-----3-----4
    #   |           |
    # - 1-----0 - - 5 - - > x
    #               |
    #   8-----7-----6

    lx_min = 35.0
    ly_min = 15.0

    def actuation_long_stat(t):
        if t < 1 / 9:
            return -t * 9 * lx_min
        elif t < 2 / 9:
            return -lx_min
        elif t < 4 / 9:
            return (t - 3 / 9) * 9 * lx_min
        elif t < 6 / 9:
            return lx_min
        elif t < 8 / 9:
            return -(t - 7 / 9) * 9 * lx_min
        else:
            return -lx_min

    def actuation_short_stat(t):
        if t < 1 / 9:
            return 0
        elif t < 2 / 9:
            return (t - 1 / 9) * 9 * ly_min
        elif t < 4 / 9:
            return ly_min
        elif t < 6 / 9:
            return -(t - 5 / 9) * 9 * ly_min
        else:
            return -ly_min

    n_stat = 90

    ts_test = np.linspace(0, 1, 91)
    als = [actuation_long_stat(t) for t in ts_test]
    ass = [actuation_short_stat(t) for t in ts_test]
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(ts_test, als)
    ax[0, 1].plot(ts_test, ass)
    ax[1, 0].plot(als, ass)
    plt.show()

    actuation_long_stroke = ActuatedConstraint(
        long_stroke_constraint,
        actuation_long_stat,
    )
    actuation_short_stroke = ActuatedConstraint(
        short_stroke_constraint,
        actuation_short_stat,
    )

    # cross section
    cross_section = RectangularCrossSection(width=8 * r_cable, height=2 * r_cable)
    A = cross_section.area(0.0)
    Ix, Iy, Iz = np.diag(cross_section.second_moment(0.0))

    A = 4 * np.pi * r_cable**2
    # cable can slide along each other (without steiner)
    Iy_f = Iz_f = np.pi * r_cable**4

    # cable cannot slide along each other (with steiner)
    Iy_s = np.pi * r_cable**4
    Iz_s = 21 * np.pi * r_cable**4

    eta_steiner = 0.75
    Iy = (1 - eta_steiner) * Iy_f + eta_steiner * Iy_s
    Iz = (1 - eta_steiner) * Iz_f + eta_steiner * Iz_s
    Ix = Iy + Iz

    Ei = np.array([E * A, G * A, G * A])
    Fi = np.array([G * Ix, E * Iy, E * Iz])
    material_model = Simo1986(Ei, Fi)

    # cross section inertias
    cross_section_inertias = CrossSectionInertias(density, cross_section)
    cross_section_inertias = CrossSectionInertias(
        A_rho0=A * density, B_I_rho0=np.diag([Ix, Iy, Iz]) * density
    )

    b = lambda t, xi: np.array([0.0, 0.0, -9.81 * A * density])

    # relative contact point of planes w.r.t. center of mass
    B_r_CP1_lower = A_IB0_table.T @ (r_OP1_lower - r_OP0_table)
    B_r_CP1_upper = A_IB0_long_stroke.T @ (r_OP1_upper - r_OP0_long_stroke)
    B_r_CP2a_lower = A_IB0_table.T @ (r_OP2a_lower - r_OP0_table)
    B_r_CP2a_upper = A_IB0_long_stroke.T @ (r_OP2a_upper - r_OP0_long_stroke)
    B_r_CP2b_lower = A_IB0_long_stroke.T @ (r_OP2b_lower - r_OP0_long_stroke)

    properties = [
        # Cable 1: Long stroke
        dict(
            name="cable1",
            nelement=nelement1,
            length=length1,
            r_OP_lower=r_OP1_lower,
            r_OP_upper=r_OP1_upper,
            body_lower=table,
            body_upper=long_stroke,
            B_r_CPlane_lower=B_r_CP1_lower,
            A_BP_lower=eye3,
            B_r_CPlane_upper=B_r_CP1_upper,
            A_BP_upper=A_IB_basic(np.pi).y,
        ),
        # Cable 2a: short stroke cable: Table to long stroke
        dict(
            name="cable2a",
            nelement=nelement2a,
            length=length2a,
            r_OP_lower=r_OP2a_lower,
            r_OP_upper=r_OP2a_upper,
            body_lower=table,
            body_upper=long_stroke,
            B_r_CPlane_lower=B_r_CP2a_lower,
            A_BP_lower=eye3,
            B_r_CPlane_upper=B_r_CP2a_upper,
            A_BP_upper=A_IB_basic(np.pi).y,
        ),
        # Cable 2b: short stroke cable: long stroke to short stroke
        dict(
            name="cable2b",
            nelement=nelement2b,
            length=length2b,
            r_OP_lower=r_OP2b_lower,
            r_OP_upper=r_OP2b_upper,
            body_lower=long_stroke,
            body_upper=short_stroke,
            B_r_CPlane_lower=B_r_CP2b_lower,
            A_BP_lower=eye3,
            B_r_CPlane_upper=None,
            A_BP_upper=None,
        ),
    ]
    # properties = []

    ######################
    # Cables in the loop #
    ######################
    cables = []
    # TODO: add planarizer for cable 2b
    for cable_properties in properties:
        nelement = cable_properties["nelement"]
        length = cable_properties["length"]
        r_OP_lower = cable_properties["r_OP_lower"]
        r_OP_upper = cable_properties["r_OP_upper"]

        # reference configuration
        Q = Cable.straight_configuration(nelement, length)

        # initial configuration
        d_x, d_y, d_z = r_OP_upper - r_OP_lower
        l_u = np.pi * d_z / 2
        l_s = (
            length - l_u + np.sqrt(d_x**2 + d_y**2)
        ) / 2  # TODO: check +-, maybe use flag
        print(
            f"length on ground: {l_s}, length in semi-circle: {l_u}, total length: {length}, height: {d_z}"
        )
        q0 = Cable.pose_configuration(
            nelement,
            lambda xi: r_OP0_fun(xi, length, l_s, l_u, d_z),
            lambda xi: A_IB0_fun(xi, length, l_s, l_u, d_z),
            r_OP0=r_OP_lower,
            A_IB0=A_IB_basic(np.atan2(d_y, d_x)).z,
        )
        cable = Cable(
            cross_section=cross_section,
            material_model=material_model,
            nelement=nelement,
            cross_section_inertias=cross_section_inertias,
            Q=Q,
            q0=q0,
            distributed_load=[b, None],
            name=cable_properties["name"],
        )

        # clamp cable
        constraint_lower = RigidConnection(
            cable,
            cable_properties["body_lower"],
            xi1=0.0,
            r_OJ0=r_OP_lower,
            name=f"constraint_lower_{cable.name}",
        )
        constraint_upper = RigidConnection(
            cable,
            cable_properties["body_upper"],
            xi1=1.0,
            r_OJ0=r_OP_upper,
            name=f"constraint_upper_{cable.name}",
        )
        cables.append(cable)
        system.add(cable, constraint_lower, constraint_upper)

        # contacts
        for node in range(1, cable.nnodes - 1):
            contact_lower = Sphere2Plane(
                cable_properties["body_lower"],
                cable,
                mu=1.0,
                radius=0.0,
                B_r_CP1=cable_properties["B_r_CPlane_lower"],
                A_B1P=cable_properties["A_BP_lower"],
                xi2=node / (cable.nnodes - 1),
                name=f"contact_lower_{cable.name}_{node:0>2d}",
            )
            system.add(contact_lower)
            if cable_properties["B_r_CPlane_upper"] is not None:
                contact_upper = Sphere2Plane(
                    cable_properties["body_upper"],
                    cable,
                    mu=1.0,
                    radius=0.0,
                    B_r_CP1=cable_properties["B_r_CPlane_upper"],
                    A_B1P=cable_properties["A_BP_upper"],
                    xi2=node / (cable.nnodes - 1),
                    name=f"contact_upper_{cable.name}_{node:0>2d}",
                )
                system.add(contact_upper)

    # assemble system
    assemble_options = SolverOptions(compute_consistent_initial_conditions=False)
    assemble_options_ccic = SolverOptions(compute_consistent_initial_conditions=True)
    newton_options = SolverOptions(newton_atol=1e-8, newton_max_iter=50)
    system.add(table, long_stroke, short_stroke)
    system.add(
        table_constraint,
        long_stroke_constraint,
        short_stroke_constraint,
        actuation_long_stroke,
        actuation_short_stroke,
    )
    system.assemble(options=assemble_options)

    # static solver
    solver_stat = Newton(system, n_stat, options=newton_options)
    solver_stat = Newton(
        system, int(2 * n_stat / 10 + 1), t1=0.2, options=newton_options
    )
    # solver_stat = Newton(system, 1, t1=0.0, options=newton_options)
    sol_stat = solver_stat.solve()

    # blender export
    dir_name = Path(__file__).parent
    sol_stat.t *= 10
    system.export_blender(dir_name, "blender_02_stat", sol_stat, create_blend=True)

    # visualize static result
    if len(cables) > 0:
        animate_beam(
            sol_stat.t,
            sol_stat.q,
            cables,
            scale=length2a,
            scale_di=length2a / 10,
        )

    # prepare for eigenmodes and dynamic simulation
    system.set_new_initial_state(
        sol_stat.q[-1], sol_stat.u[-1], t0=0.0, options=assemble_options
    )

    # remove constraint but add constraint force as external one
    actuation_long_stroke.update_actuation(
        active=False, inactive_force=sol_stat.la_g[-1, actuation_long_stroke.la_gDOF]
    )
    actuation_short_stroke.update_actuation(
        active=False, inactive_force=sol_stat.la_g[-1, actuation_short_stroke.la_gDOF]
    )

    # assemble
    system.assemble(options=assemble_options_ccic)

    # solve to make sure that the system is in static equilibrium
    # TODO: is this necessary?
    solver_stat2 = Newton(system, 1, t1=1.0)
    sol_stat2 = solver_stat2.solve()

    # eigenmodes
    solver_eig = Eigenmodes(system, sol_stat2)
    sol_eig = solver_eig.solve(-1)
    dir_name = Path(__file__).parent
    system.export_blender(dir_name, "blender_02_eig", sol_eig, create_blend=True)

    print(sol_eig.omegas[:10])

    # frequency response function
    iom = 1j * np.logspace(-3, 1, 201)
    solver_FRF = FrequencyResponseFunction(system, sol_stat2)
    sol_FRF = solver_FRF.solve(-1, iom)

    # Inpu-output of interest
    outDOF_long = long_stroke.outDOF[0]
    outDOF_short = short_stroke.outDOF[1]

    inDOF_long = actuation_long_stroke.inDOF[0]
    inDOF_short = actuation_short_stroke.inDOF[0]

    frf_long_long = sol_FRF[:, outDOF_long, inDOF_long]
    frf_short_short = sol_FRF[:, outDOF_short, inDOF_short]

    # FRF-plots
    fig, ax = plt.subplots(2, 2, sharex=True)
    ax[0, 0].loglog(iom.imag, np.abs(frf_long_long))
    ax[1, 0].plot(iom.imag, np.degrees(np.angle(frf_long_long)))
    ax[0, 1].loglog(iom.imag, np.abs(frf_short_short))
    ax[1, 1].plot(iom.imag, np.degrees(np.angle(frf_short_short)))

    ax[0, 0].grid()
    ax[1, 0].grid()
    ax[0, 1].grid()
    ax[1, 1].grid()

    # expectation from free rigid body
    ax[0, 0].loglog(iom.imag, 1 / (iom.imag**2 * (mass_long + mass_short)), "--")
    ax[1, 0].plot(iom.imag, np.ones_like(iom.imag) * 180, "--")
    ax[0, 1].loglog(iom.imag, 1 / (iom.imag**2 * mass_short), "--")
    ax[1, 1].plot(iom.imag, np.ones_like(iom.imag) * 180, "--")
    plt.show()

    exit()

    # dynamic actuation
    actuation_long_stroke.update_actuation(tau=actuation_long_dyn)
    actuation_short_stroke.update_actuation(tau=actuation_short_dyn)

    # assemble again
    system.assemble(options=assemble_options)

    # dynamic solver
    # solver_dyn = BackwardEuler(system, t1=t1, dt=dt)
    solver_dyn = DualStormerVerlet(system, t1=t1, dt=dt)
    # solver_dyn = Moreau(system, t1=t1, dt=dt)
    # solver_dyn = Newton(system, t1=t1, n_load_steps=int((t1 - system.t0) / dt * 1e3))
    sol_dyn = solver_dyn.solve()

    # visualize dynamic result
    # animate_beam(sol_dyn.t, sol_dyn.q, [cable], scale=1.0, scale_di=0.1)

    x_long = (
        np.array([sol_dyn.q[i, long_stroke.qDOF[0]] for i in range(len(sol_dyn.t))])
        - l_x0
    )
    y_short = (
        np.array([sol_dyn.q[i, short_stroke.qDOF[1]] for i in range(len(sol_dyn.t))])
        - l_y0
    )

    if not hasattr(sol_dyn, "P_g"):
        sol_dyn.P_g = sol_dyn.la_g

    fig, ax = plt.subplots(3, 2)
    ax[0, 0].plot(sol_dyn.t, [actuation_long_stroke.tau(ti) for ti in sol_dyn.t])
    ax[0, 1].plot(sol_dyn.t, [actuation_short_stroke.tau(ti) for ti in sol_dyn.t])
    ax[1, 0].plot(sol_dyn.t, x_long)
    ax[1, 1].plot(sol_dyn.t, y_short)
    ax[2, 0].plot(
        sol_dyn.t, [la_g[actuation_long_stroke.la_gDOF] for la_g in sol_dyn.P_g]
    )
    ax[2, 1].plot(
        sol_dyn.t, [la_g[actuation_short_stroke.la_gDOF] for la_g in sol_dyn.P_g]
    )
    ax[0, 0].set_title("Long stroke")
    ax[0, 1].set_title("Short stroke")
    ax[0, 0].set_ylabel("Actuation")
    ax[1, 0].set_ylabel("Position")
    ax[2, 0].set_ylabel("Force")
    plt.show()

    # vtk-export
    dir_name = Path(__file__).parent
    system.export_blender(dir_name, "blender_02", sol_dyn, create_blend=True)

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
