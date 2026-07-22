import numpy as np
from pathlib import Path
import pytest

from cardillo import System
from cardillo.discrete import RigidBody, Box, Sphere, Frame, Tetrahedron
from cardillo.forces import Force
from cardillo.force_laws import KelvinVoigtElement as SpringDamper
from cardillo.interactions import TwoPointInteraction
from cardillo.contacts import Sphere2Plane, Sphere2Sphere, Sphere2PlaneOld
from cardillo.solver import Moreau, BackwardEuler, SolverOptions
from cardillo.math import A_IB_basic, Exp_SO3
from cardillo.math.approx_fprime import approx_fprime


def run(solver=Moreau, VTK_export=False):
    ############################################################################
    #                   system setup
    ############################################################################

    ###################
    # solver parameters
    ###################
    t_span = (0.0, 2)
    t0, t1 = t_span
    dt = 1.0e-3

    ############
    # parameters
    ############
    radius = 0.05  # radius of ball
    mass = 1  # mass ball
    density = mass / (4 / 3 * np.pi * radius**3)  # density of ball
    g = np.array([0, 0, -10])  # gravitational acceleration
    e_N = 0.0  # restitution coefficient in normal direction
    e_F = 0.0  # restitution coefficient in tangent direction
    mu = 0.3  # frictional coefficient

    # initialize system
    system = System()
    # floor
    omega = 2 * np.pi * 0.5
    amplitude = radius
    # r_OP=lambda t: amplitude * np.array([np.sin(omega * t), 0.0, 0.0])
    # r_OP=lambda t: amplitude * np.array([0.0, np.sin(omega * t), 0.0])
    r_OP = lambda t: amplitude * np.array([0.0, 0.0, np.sin(omega * t)])
    # r_OP = lambda t: amplitude * np.array([0.0, 0.0, 0.0])

    angle = np.deg2rad(20)
    # A_IB = A_IB_basic(np.deg2rad(10)).x @ A_IB_basic(np.deg2rad(10)).y
    # A_IB=lambda t: A_IB_basic(angle * np.sin(omega * t)).x
    # A_IB=lambda t: A_IB_basic(angle * np.sin(omega * t)).y
    # A_IB = lambda t: A_IB_basic(angle * np.sin(omega * t)).z
    A_IB = (
        lambda t: A_IB_basic(angle * np.sin(omega * t)).y
        @ A_IB_basic(angle * np.sin(omega * t)).z
    )

    floor = Box(Frame)(
        dimensions=[4.5, 4.5, 0.0001],
        r_OP=r_OP,
        A_IB=A_IB,
        name="floor",
    )
    system.add(floor)  # (only for visualization purposes)

    # initial conditions ball
    initial_gap = 0.01 * radius + radius
    r_OC0 = np.array([0, 0, radius + initial_gap])
    q0 = RigidBody.pose2q(r_OC0, np.eye(3))
    u0 = np.zeros(6)

    # ball as sphere
    ball = Sphere(RigidBody)(
        radius=radius,
        density=density,
        subdivisions=3,
        q0=q0,
        u0=u0,
        name="ball",
    )

    system.add(ball)

    # gravity of ball
    system.add(Force(ball.mass * g, ball, name="gravity_" + ball.name))

    # contact between ball and plane
    contact = Sphere2PlaneOld(
        floor,
        ball,
        mu=mu,
        r=radius,
        e_N=e_N,
        e_F=e_F,
        name="floor2" + ball.name,
    )
    contact = Sphere2Plane(
        floor,
        ball,
        mu=mu * 0.0,
        radius=radius,
        e_N=e_N,
        e_F=e_F,
        name="floor2" + ball.name,
    )
    system.add(contact)

    # add tetrahedron
    edge = 0.1
    density = 7700
    mu = 0.3
    r_OC0_tetra = np.array([10 * edge, 0, edge])
    q0_tetra = RigidBody.pose2q(r_OC0_tetra, np.eye(3))
    u0_tetra = np.zeros(6)

    tetrahedron = Tetrahedron(RigidBody)(
        edge=edge, density=density, q0=q0_tetra, u0=u0_tetra, name="tetrahedron"
    )

    system.add(tetrahedron)

    # gravity of ball
    system.add(
        Force(tetrahedron.mass * g, tetrahedron, name="gravity_" + tetrahedron.name)
    )

    for i, vertex in enumerate(tetrahedron.B_visual_mesh.vertices):
        contacti = Sphere2PlaneOld(
            floor,
            tetrahedron,
            mu=mu,
            r=0,
            e_N=e_N,
            e_F=e_F,
            B_r_CP=vertex,
            name=f"floor2{tetrahedron.name}_{i}",
        )
        contacti = Sphere2Plane(
            floor,
            tetrahedron,
            mu=mu * 0.0,
            radius=0,
            e_N=e_N,
            e_F=e_F,
            B_r_CP2=vertex,
            name=f"floor2{tetrahedron.name}_{i}",
        )
        system.add(contacti)

    # assemble system
    system.assemble()

    ############
    # simulation
    ############
    solver = solver(
        system,
        t1,
        dt,
        options=SolverOptions(prox_scaling=0.4, continue_with_unconverged=False),
    )  # create solver
    sol = solver.solve()  # simulate system

    # vtk-export
    if VTK_export:
        dir_name = Path(__file__).parent
        system.export(dir_name, "vtk", sol)


@pytest.mark.filterwarnings("ignore: 'approx_fprime' is used")
def test_implementation():
    m = 1.0
    B_Theta_C = np.diag([1.0, 1.0, 1.0])
    q01 = np.random.rand(7) * 5
    u01 = np.random.rand(6) * 3
    q02 = np.random.rand(7) * 5
    u02 = np.random.rand(6) * 3

    body1 = RigidBody(m, B_Theta_C, q01, u01, name="Body1")
    body2 = RigidBody(m, B_Theta_C, q02, u02, name="Body2")

    B_r_CP1 = np.random.rand(3)
    B_r_CP2 = np.random.rand(3)
    A_B1P = Exp_SO3(np.random.rand(3))

    B1_t1 = A_B1P[:, 0]
    B1_t2 = A_B1P[:, 1]
    B1_n = A_B1P[:, 2]

    mu = 0.0  # TODO: friction
    radius = np.random.rand()

    contact = Sphere2Plane(body1, body2, mu, radius, B_r_CP1, B_r_CP2, A_B1P)

    # assembly
    t0 = np.random.rand()
    body1.qDOF = np.arange(0, 7)
    body2.qDOF = np.arange(7, 14)
    body1.uDOF = np.arange(0, 6)
    body2.uDOF = np.arange(6, 12)
    contact.assembler_callback()
    q0 = np.array([*q01, *q02])
    u0 = np.array([*u01, *u02])
    q0_dot = np.array([*body1.q_dot(t0, q01, u01), *body2.q_dot(t0, q02, u02)])

    u0_dot = np.random.rand(12)
    la_N0 = np.random.rand(1)

    # compute contact kinematics analytically
    n = body1.A_IB(t0, q01) @ B1_n
    r_OP1 = body1.r_OP(t0, q01, B_r_CP=B_r_CP1)
    r_OP2 = body2.r_OP(t0, q02, B_r_CP=B_r_CP2)

    g_N_ana = n @ (r_OP2 - r_OP1) - radius

    ####################
    # normal direction #
    ####################
    # g_N
    g_N = contact.g_N(t0, q0)[0]
    assert np.isclose(g_N, g_N_ana), f"g_N: {g_N} != {g_N_ana}"

    # g_N_q
    g_N_q = contact.g_N_q(t0, q0)
    g_N_q_num = approx_fprime(q0, lambda q_: contact.g_N(t0, q_))
    assert np.all(
        np.isclose(g_N_q, g_N_q_num, rtol=1e-5)
    ), f"g_N_q: {g_N_q} != {g_N_q_num}"

    # g_N_dot
    g_N_dot = contact.g_N_dot(t0, q0, u0)
    g_N_dot_num = g_N_q_num @ q0_dot
    assert np.isclose(g_N_dot, g_N_dot_num), f"g_N_dot: {g_N_dot} != {g_N_dot_num}"

    # # g_N_dot_q
    g_N_dot_q = contact.g_N_dot_q(t0, q0, u0)
    g_N_dot_q_num = approx_fprime(q0, lambda q_: contact.g_N_dot(t0, q_, u0))
    assert np.all(
        np.isclose(g_N_dot_q, g_N_dot_q_num, rtol=1e-5)
    ), f"g_N_dot_q: {g_N_dot_q} != {g_N_dot_q_num}"

    # g_N_dot_u
    g_N_dot_u = contact.g_N_dot_u(t0, q0)
    g_N_dot_u_num = approx_fprime(u0, lambda u_: contact.g_N_dot(t0, q0, u_))
    assert np.all(
        np.isclose(g_N_dot_u, g_N_dot_u_num, rtol=1e-5)
    ), f"g_N_dot_u: {g_N_dot_u} != {g_N_dot_u_num}"

    # W_N
    W_N = contact.W_N(t0, q0)
    assert np.all(
        np.isclose(W_N, g_N_dot_u.T, rtol=1e-5)
    ), f"W_N: {W_N} != {g_N_dot_u.T}"

    # g_N_ddot
    g_N_ddot = contact.g_N_ddot(t0, q0, u0, u0_dot)
    g_N_ddot_num = g_N_dot_q_num @ q0_dot + g_N_dot_u @ u0_dot
    assert np.isclose(g_N_ddot, g_N_ddot_num), f"g_N_ddot: {g_N_ddot} != {g_N_ddot_num}"

    # Wla_N_q
    Wla_N_q = contact.Wla_N_q(t0, q0, la_N0)
    Wla_N_q_num = approx_fprime(q0, lambda q_: contact.W_N(t0, q_) @ la_N0)
    assert np.all(
        np.isclose(Wla_N_q, Wla_N_q_num, rtol=1e-5)
    ), f"Wla_N_q: {Wla_N_q} != {Wla_N_q_num}"

    # KN_N
    K_num = -Wla_N_q @ np.block(
        [
            [body1.q_dot_u(t0, q01), np.zeros((7, 6))],
            [np.zeros((7, 6)), body2.q_dot_u(t0, q02)],
        ]
    )
    K_num = (K_num + K_num.T) / 2
    N_num = np.zeros_like(K_num)
    K, N = contact.KN_N(t0, q0, la_N0)

    assert np.all(np.isclose(K, K.T)), f"K-symmetry: {K} != {K.T}"
    assert np.all(np.isclose(K, K_num)), f"K-num: {K} != {K_num}"
    assert np.all(np.isclose(N, N_num)), f"N: {N} != {N_num}"

    ########################
    # tangential direction #
    ########################
    # TODO


# def test_with_Moreau():
#     run(Moreau)

# def test_with_BackwardEuler():
#     run(BackwardEuler)

if __name__ == "__main__":
    for i in range(1_000):
        test_implementation()
    exit()
    run(Moreau)
    run(BackwardEuler)
