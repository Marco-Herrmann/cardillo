import numpy as np
import pytest
from scipy.linalg import null_space

from cardillo import System
from cardillo.discrete import RigidBody
from cardillo.constraints import (
    Spherical,
    RigidConnection,
    Revolute,
    Prismatic,
    Cylindrical,
    Planarizer,
    FixedDistance,
)
from cardillo.math import ax2skew
from cardillo.math.approx_fprime import approx_fprime


def _random_rigid_body(rng):
    q0 = rng.random(7)
    q0[3:] /= np.linalg.norm(q0[3:])
    return RigidBody(1.0, np.eye(3), q0=q0, name=f"RB_{rng.random()}")


@pytest.mark.filterwarnings("ignore: 'approx_fprime' is used")
def test_rigid_body_translational():
    rng = np.random.default_rng(0)
    q0 = rng.random(7)
    q0[3:] /= np.linalg.norm(q0[3:])
    rb = RigidBody(1.0, np.eye(3), q0=q0)

    t0 = 0.0
    F = rng.random(3)
    B_r_CP = rng.random(3)

    B0 = rb.q_dot_u(t0, q0)
    J_P = rb.J_P(t0, q0, B_r_CP=B_r_CP)
    J2_P = rb.J2_P(t0, q0, B_r_CP=B_r_CP)

    V = lambda q: rb.r_OP(t0, q, B_r_CP=B_r_CP) @ F
    dV_dq = approx_fprime(q0, V, eps=1e-5)
    d2V_dq2 = approx_fprime(q0, lambda q: approx_fprime(q, V, eps=1e-5), eps=1e-5)
    assert np.all(np.isclose(d2V_dq2, d2V_dq2.T, atol=1e-6)), "d2V/dq2 is not symmetric"

    assert np.all(
        np.isclose(dV_dq @ B0, J_P.T @ F, atol=1e-4)
    ), "dV/dq @ B != J_P.T @ F"

    d2V_pulled = B0.T @ d2V_dq2 @ B0
    d2V_ana = np.einsum("ijk,i->jk", J2_P, F)
    assert np.all(
        np.isclose(d2V_pulled, d2V_ana, atol=1e-4)
    ), f"B.T @ d2V/dq2 @ B != einsum(J2_P, F)\n{d2V_pulled}\n{d2V_ana}"


@pytest.mark.filterwarnings("ignore: 'approx_fprime' is used")
def test_rigid_body_rotational():
    rng = np.random.default_rng(0)
    q0 = rng.random(7)
    q0[3:] /= np.linalg.norm(q0[3:])
    rb = RigidBody(1.0, np.eye(3), q0=q0)

    t0 = 0.0
    F = rng.random(3)
    B_r_CP = rng.random(3)
    B0 = rb.q_dot_u(t0, q0)

    A_IB = rb.A_IB(t0, q0)
    B_J_R = rb.B_J_R(t0, q0)
    B_J2_R = rb.B_J2_R(t0, q0)

    B_F = A_IB.T @ F
    B_M = np.cross(B_r_CP, B_F)

    d2W2 = (
        np.einsum("ijk,i->jk", B_J2_R, B_M)
        + B_J_R.T @ ax2skew(B_r_CP) @ ax2skew(B_F) @ B_J_R
    )

    V = lambda q: rb.r_OP(t0, q, B_r_CP=B_r_CP) @ F
    d2V_dq2 = approx_fprime(q0, lambda q: approx_fprime(q, V, eps=1e-5), eps=1e-5)
    d2V_pulled = B0.T @ d2V_dq2 @ B0

    assert np.all(
        np.isclose(d2W2, d2V_pulled, atol=1e-4)
    ), f"B_J_R/B_J2_R construction != B.T @ d2V/dq2 @ B\n{d2W2}\n{d2V_pulled}"


def _setup(make_constraint, seed):
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed + 1)
    rb1 = _random_rigid_body(rng1)
    rb2 = _random_rigid_body(rng2)
    constraint = make_constraint(rb1, rb2)

    system = System()
    system.add(rb1, rb2, constraint)
    system.assemble()

    t0 = 0.0
    q0 = system.q0
    B0 = system.q_dot_u(t0, q0)
    return rng1, rng2, constraint, t0, q0, B0


# (name, constraint factory, seed)
KN_g_cases = [
    # PositionOrientationBase.KN_g
    ("RigidConnection", lambda rb1, rb2: RigidConnection(rb1, rb2), 0),
    ("Spherical", lambda rb1, rb2: Spherical(rb1, rb2, r_OJ0=np.zeros(3)), 0),
    ("Revolute axis=0", lambda rb1, rb2: Revolute(rb1, rb2, axis=0), 0),
    ("Revolute axis=1", lambda rb1, rb2: Revolute(rb1, rb2, axis=1), 1),
    ("Revolute axis=2", lambda rb1, rb2: Revolute(rb1, rb2, axis=2), 2),
    # ProjectedPositionOrientationBase.KN_g
    ("Prismatic axis=0", lambda rb1, rb2: Prismatic(rb1, rb2, axis=0), 0),
    ("Prismatic axis=1", lambda rb1, rb2: Prismatic(rb1, rb2, axis=1), 1),
    ("Prismatic axis=2", lambda rb1, rb2: Prismatic(rb1, rb2, axis=2), 2),
    ("Cylindrical axis=0", lambda rb1, rb2: Cylindrical(rb1, rb2, axis=0), 0),
    ("Cylindrical axis=1", lambda rb1, rb2: Cylindrical(rb1, rb2, axis=1), 1),
    ("Cylindrical axis=2", lambda rb1, rb2: Cylindrical(rb1, rb2, axis=2), 2),
    ("Planarizer axis=0", lambda rb1, rb2: Planarizer(rb1, rb2, axis=0), 0),
    ("Planarizer axis=1", lambda rb1, rb2: Planarizer(rb1, rb2, axis=1), 1),
    ("Planarizer axis=2", lambda rb1, rb2: Planarizer(rb1, rb2, axis=2), 2),
    # standalone implementation (not PositionOrientationBase-derived)
    ("FixedDistance", lambda rb1, rb2: FixedDistance(rb1, rb2), 0),
]


@pytest.mark.filterwarnings("ignore: 'approx_fprime' is used")
@pytest.mark.parametrize("name, make_constraint, seed", KN_g_cases)
def test_KN_g(name, make_constraint, seed):
    rng1, rng2, constraint, t0, q0, B0 = _setup(make_constraint, seed)

    # g_dot_u = W_g.T = g_q @ B, directly from the chain rule
    # g_dot = g_q @ q_dot = g_q @ B @ u -- no la_g involved here at all
    g_q = np.atleast_2d(constraint.g_q(t0, q0))
    assert np.all(
        np.isclose(g_q @ B0, constraint.W_g(t0, q0).T, atol=1e-6)
    ), f"{name}: g_q @ B != W_g.T"

    la_g = rng1.random(constraint.nla_g)
    # np.atleast_1d: FixedDistance.g returns a bare scalar (nla_g=1), unlike
    # the array-valued g of the PositionOrientationBase-derived joints
    V = lambda q: np.atleast_1d(constraint.g(t0, q)) @ la_g

    dV_dq = approx_fprime(q0, V, eps=1e-5)
    d2V_dq2 = approx_fprime(q0, lambda q: approx_fprime(q, V, eps=1e-5), eps=1e-5)
    assert np.all(
        np.isclose(d2V_dq2, d2V_dq2.T, atol=1e-6)
    ), f"{name}: d2V/dq2 is not symmetric"

    lhs1 = B0.T @ dV_dq
    rhs1 = constraint.W_g(t0, q0) @ la_g
    assert np.all(
        np.isclose(lhs1, rhs1, atol=1e-4)
    ), f"{name}: B.T @ dV/dq != W_g @ la_g\n{lhs1}\n{rhs1}"

    # K + N must equal -B.T @ d2V/dq2 @ B
    K, N = constraint.KN_g(t0, q0, la_g)
    lhs2 = B0.T @ d2V_dq2 @ B0
    assert np.all(
        np.isclose(lhs2, -(K + N), atol=1e-4)
    ), f"{name}: B.T @ d2V/dq2 @ B != -(K + N)\n{lhs2}\n{-(K + N)}"


KN_l_cases = [
    ("Prismatic axis=0", lambda rb1, rb2: Prismatic(rb1, rb2, axis=0), 0),
    ("Prismatic axis=1", lambda rb1, rb2: Prismatic(rb1, rb2, axis=1), 1),
    ("Prismatic axis=2", lambda rb1, rb2: Prismatic(rb1, rb2, axis=2), 2),
    ("Revolute axis=0", lambda rb1, rb2: Revolute(rb1, rb2, axis=0), 0),
    ("Revolute axis=1", lambda rb1, rb2: Revolute(rb1, rb2, axis=1), 1),
    ("Revolute axis=2", lambda rb1, rb2: Revolute(rb1, rb2, axis=2), 2),
]


@pytest.mark.filterwarnings("ignore: 'approx_fprime' is used")
@pytest.mark.parametrize("name, make_constraint, seed", KN_l_cases)
def test_KN_l(name, make_constraint, seed):
    rng1, rng2, constraint, t0, q0, B0 = _setup(make_constraint, seed)

    # l_dot_u = W_l = l_q @ B, directly from the chain rule; l is scalar, so
    # both sides are already flat (nu,) vectors -- no transpose needed
    l_q = constraint.l_q(t0, q0).flatten()
    assert np.all(
        np.isclose(l_q @ B0, constraint.W_l(t0, q0).flatten(), atol=1e-6)
    ), f"{name}: l_q @ B != W_l"

    la_l = rng1.random()
    V = lambda q: constraint.l(t0, q) * la_l

    dV_dq = approx_fprime(q0, V, eps=1e-5)
    d2V_dq2 = approx_fprime(q0, lambda q: approx_fprime(q, V, eps=1e-5), eps=1e-5)

    assert np.all(
        np.isclose(d2V_dq2, d2V_dq2.T, atol=1e-6)
    ), f"{name}: d2V/dq2 is not symmetric"

    lhs1 = B0.T @ dV_dq
    rhs1 = constraint.W_l(t0, q0) * la_l
    assert np.all(
        np.isclose(lhs1, rhs1.flatten(), atol=1e-4)
    ), f"{name}: B.T @ dV/dq != W_l * la_l\n{lhs1}\n{rhs1}"

    K, N = constraint.KN_l(t0, q0, la_l)
    lhs2 = B0.T @ d2V_dq2 @ B0

    if not np.all(np.isclose(lhs2, -(K + N), atol=1e-4)):
        print(f"{name}: Unprojected! (K+N)_num != (K+N)")

    # check only with admissible gen. virtual displacements:  delta_s = T_adm @ delta_z, where W_g.T @ T_adm = 0
    T_adm = null_space(constraint.W_g(t0, q0).T)

    lhs3 = T_adm.T @ lhs2 @ T_adm
    rhs3 = T_adm.T @ (K + N) @ T_adm

    assert np.all(
        np.isclose(lhs3, -rhs3, atol=1e-4)
    ), f"{name}: Projected one! (K+N)_num != (K+N)"
    print(np.linalg.norm(rhs3))


if __name__ == "__main__":
    test_rigid_body_translational()
    test_rigid_body_rotational()
    for case in KN_g_cases:
        test_KN_g(*case)
    for case in KN_l_cases:
        test_KN_l(*case)
    print("all checks passed")
