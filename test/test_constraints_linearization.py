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
from cardillo.math.rotations import Exp_SO3, Spurrier


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
    rng = np.random.default_rng(seed)
    rb1 = _random_rigid_body(rng)
    rb2 = _random_rigid_body(rng)
    constraint = make_constraint(rb1, rb2)

    system = System()
    system.add(rb1, rb2, constraint)
    system.assemble()

    t0 = 0.0
    q0 = system.q0
    B0 = system.q_dot_u(t0, q0)
    return rng, constraint, t0, q0, B0


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


def _check_KN_g(name, constraint, t0, q0, B0, la_g):
    """K + N must equal -B.T @ Hess_q(g @ la_g) @ B, checked both via the
    first variation (g_q @ B == W_g.T) and the second (KN_g itself)."""
    # g_dot_u = W_g.T = g_q @ B, directly from the chain rule
    # g_dot = g_q @ q_dot = g_q @ B @ u -- no la_g involved here at all
    g_q = np.atleast_2d(constraint.g_q(t0, q0))
    assert np.all(
        np.isclose(g_q @ B0, constraint.W_g(t0, q0).T, atol=1e-6)
    ), f"{name}: g_q @ B != W_g.T"

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

    K, N = constraint.KN_g(t0, q0, la_g)
    lhs2 = B0.T @ d2V_dq2 @ B0
    assert np.all(
        np.isclose(lhs2, -(K + N), atol=1e-4)
    ), f"{name}: B.T @ d2V/dq2 @ B != -(K + N)\n{lhs2}\n{-(K + N)}"


@pytest.mark.filterwarnings("ignore: 'approx_fprime' is used")
@pytest.mark.parametrize("name, make_constraint, seed", KN_g_cases)
def test_KN_g(name, make_constraint, seed):
    rng, constraint, t0, q0, B0 = _setup(make_constraint, seed)
    la_g = rng.random(constraint.nla_g)
    _check_KN_g(name, constraint, t0, q0, B0, la_g)


KN_l_cases = [
    ("Prismatic axis=0", lambda rb1, rb2: Prismatic(rb1, rb2, axis=0), 0),
    ("Prismatic axis=1", lambda rb1, rb2: Prismatic(rb1, rb2, axis=1), 1),
    ("Prismatic axis=2", lambda rb1, rb2: Prismatic(rb1, rb2, axis=2), 2),
    ("Revolute axis=0", lambda rb1, rb2: Revolute(rb1, rb2, axis=0), 0),
    ("Revolute axis=1", lambda rb1, rb2: Revolute(rb1, rb2, axis=1), 1),
    ("Revolute axis=2", lambda rb1, rb2: Revolute(rb1, rb2, axis=2), 2),
]


def _check_KN_l(name, constraint, t0, q0, B0, la_l):
    """K + N must equal -B.T @ Hess_q(l * la_l) @ B, checked both via the
    first variation (l_q @ B == W_l) and the second (KN_l itself) -- the
    latter both unprojected and restricted to admissible generalized
    virtual displacements delta_s = T_adm @ delta_z, W_g.T @ T_adm == 0."""
    # l_dot_u = W_l = l_q @ B, directly from the chain rule; l is scalar, so
    # both sides are already flat (nu,) vectors -- no transpose needed
    l_q = constraint.l_q(t0, q0).flatten()
    assert np.all(
        np.isclose(l_q @ B0, constraint.W_l(t0, q0).flatten(), atol=1e-6)
    ), f"{name}: l_q @ B != W_l"

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

    assert np.all(
        np.isclose(lhs2, -(K + N), atol=1e-4)
    ), f"{name}: B.T @ d2V/dq2 @ B != -(K + N)\n{lhs2}\n{-(K + N)}"

    T_adm = null_space(constraint.W_g(t0, q0).T)
    lhs3 = T_adm.T @ lhs2 @ T_adm
    rhs3 = T_adm.T @ (K + N) @ T_adm
    assert np.all(
        np.isclose(lhs3, -rhs3, atol=1e-4)
    ), f"{name}: Projected one! (K+N)_num != (K+N)"


@pytest.mark.filterwarnings("ignore: 'approx_fprime' is used")
@pytest.mark.parametrize("name, make_constraint, seed", KN_l_cases)
def test_KN_l(name, make_constraint, seed):
    rng, constraint, t0, q0, B0 = _setup(make_constraint, seed)
    la_l = rng.random()
    _check_KN_l(name, constraint, t0, q0, B0, la_l)


def _revolute_q_with_angle(rb1, rb2, rev, angle, rng):
    """Build q = (q1, q2) with body 1's pose fully random and body 2's pose
    chosen so that the joint's constraint g == 0 holds exactly, with the
    relative joint angle set to `angle`.

    This matters because the q0 that `System.assemble()` itself produces
    always has angle == 0: its reference frames (r_OJ0, A_IJ0) are picked
    from the bodies' pose at assembly time, which forces l(t0, q0) == 0 by
    construction (see examples/revolute_kn_l_debug/quaternion_only.py).
    That makes q0 a poor stand-in for "a generic point on the constraint
    manifold" -- every other joint angle is just as valid a configuration
    with g == 0, and KN_l has to be correct there too. This reuses the
    joint's *own* fixed attachment offsets (recovered from the original
    assembly-time poses of rb1, rb2) to place body 2 at an arbitrary,
    explicitly chosen angle instead of the trivial one.
    """
    t0 = 0.0

    # recover the joint's fixed, assembly-time attachment offsets
    q1_0, q2_0 = rb1.q0, rb2.q0
    A_IB1_0, A_IB2_0 = rb1.A_IB(t0, q1_0), rb2.A_IB(t0, q2_0)
    r_OP1_0, r_OP2_0 = rb1.r_OP(t0, q1_0), rb2.r_OP(t0, q2_0)

    B1_r_P1J0 = A_IB1_0.T @ (rev.r_OJ0 - r_OP1_0)
    B2_r_P2J0 = A_IB2_0.T @ (rev.r_OJ0 - r_OP2_0)
    A_K1J0 = A_IB1_0.T @ rev.A_IJ0
    A_K2J0 = A_IB2_0.T @ rev.A_IJ0

    # body 1: fully random pose
    r_OP1 = rng.random(3)
    p1 = rng.random(4)
    p1 /= np.linalg.norm(p1)
    q1 = np.concatenate([r_OP1, p1])

    A_IB1 = rb1.A_IB(t0, q1)
    r_OJ1 = r_OP1 + A_IB1 @ B1_r_P1J0
    A_IJ1 = A_IB1 @ A_K1J0

    # body 2: chosen so the joint point coincides (position constraint) and
    # the shared axis is aligned (rotation constraint), with the relative
    # rotation about that axis equal to `angle` -- the joint's own free DOF
    e_c1 = A_IJ1[:, rev.axis]
    A_IJ2 = Exp_SO3(angle * e_c1) @ A_IJ1
    A_IB2 = A_IJ2 @ A_K2J0.T
    r_OP2 = r_OJ1 - A_IB2 @ B2_r_P2J0
    p2 = Spurrier(A_IB2)
    q2 = np.concatenate([r_OP2, p2])

    return np.concatenate([q1, q2])


Revolute_random_angle_cases = [
    ("Revolute axis=0", 0, 0),
    ("Revolute axis=1", 1, 1),
    ("Revolute axis=2", 2, 2),
]


def _setup_revolute_random_angle(axis, seed):
    """Same shape as `_setup`, but for an explicitly constructed, generic
    point on the constraint manifold (random body 1 pose, random nonzero
    joint angle) rather than the assembler's always-angle-0 q0."""
    rng = np.random.default_rng(seed)
    rb1 = _random_rigid_body(rng)
    rb2 = _random_rigid_body(rng)
    rev = Revolute(rb1, rb2, axis=axis)

    system = System()
    system.add(rb1, rb2, rev)
    system.assemble()

    t0 = 0.0
    angle = rng.uniform(-np.pi, np.pi)
    q0 = _revolute_q_with_angle(rb1, rb2, rev, angle, rng)
    B0 = system.q_dot_u(t0, q0)

    # sanity: the constructed configuration really is on the manifold, at
    # the intended angle
    assert np.all(
        np.isclose(rev.g(t0, q0), 0.0, atol=1e-10)
    ), "constructed q0 does not satisfy g == 0"
    # rev.l is a stateful quadrant-unwrapping angle (see Revolute.l): a
    # single fresh evaluation only reports the geometric angle up to a
    # multiple of 2*pi (its hardcoded initial `previous_quadrant = 1`
    # bookkeeping doesn't know our `angle` was negative), so compare
    # modulo 2*pi rather than exactly.
    angle_diff = rev.l(t0, q0) - angle
    assert np.isclose(
        angle_diff, 2 * np.pi * np.round(angle_diff / (2 * np.pi)), atol=1e-10
    ), "constructed q0 does not have the intended joint angle (mod 2*pi)"

    return rng, rev, t0, q0, B0


@pytest.mark.filterwarnings("ignore: 'approx_fprime' is used")
@pytest.mark.parametrize("name, axis, seed", Revolute_random_angle_cases)
def test_KN_l_revolute_random_angle(name, axis, seed):
    rng, rev, t0, q0, B0 = _setup_revolute_random_angle(axis, seed)
    la_l = rng.random()
    _check_KN_l(name, rev, t0, q0, B0, la_l)


@pytest.mark.filterwarnings("ignore: 'approx_fprime' is used")
@pytest.mark.parametrize("name, axis, seed", Revolute_random_angle_cases)
def test_KN_g_revolute_random_angle(name, axis, seed):
    rng, rev, t0, q0, B0 = _setup_revolute_random_angle(axis, seed)
    la_g = rng.random(rev.nla_g)
    _check_KN_g(name, rev, t0, q0, B0, la_g)


def _prismatic_q_with_displacement(rb1, rb2, pris, displacement, rng):
    """Build q = (q1, q2) with body 1's pose fully random and body 2's pose
    chosen so that the joint's constraint g == 0 holds exactly, with the
    relative displacement along the free axis set to `displacement`.

    Same idea as `_revolute_q_with_angle`: `System.assemble()`'s own q0
    always has displacement == 0 (its reference frames are picked from the
    bodies' pose at assembly time), which is just as special a case here
    as angle == 0 was for the revolute joint. A Prismatic constrains ALL
    three relative orientations (A_IJ2 must equal A_IJ1 exactly -- there is
    no rotational freedom at all) and the two displacement axes orthogonal
    to `axis`; only the displacement along `axis` is free.
    """
    t0 = 0.0

    q1_0, q2_0 = rb1.q0, rb2.q0
    A_IB1_0, A_IB2_0 = rb1.A_IB(t0, q1_0), rb2.A_IB(t0, q2_0)
    r_OP1_0, r_OP2_0 = rb1.r_OP(t0, q1_0), rb2.r_OP(t0, q2_0)

    B1_r_P1J0 = A_IB1_0.T @ (pris.r_OJ0 - r_OP1_0)
    B2_r_P2J0 = A_IB2_0.T @ (pris.r_OJ0 - r_OP2_0)
    A_K1J0 = A_IB1_0.T @ pris.A_IJ0
    A_K2J0 = A_IB2_0.T @ pris.A_IJ0

    # body 1: fully random pose
    r_OP1 = rng.random(3)
    p1 = rng.random(4)
    p1 /= np.linalg.norm(p1)
    q1 = np.concatenate([r_OP1, p1])

    A_IB1 = rb1.A_IB(t0, q1)
    r_OJ1 = r_OP1 + A_IB1 @ B1_r_P1J0
    A_IJ1 = A_IB1 @ A_K1J0

    # body 2: no rotational freedom (A_IJ2 == A_IJ1 exactly), and the joint
    # point displaced from J1 by exactly `displacement` along the free axis
    A_IJ2 = A_IJ1
    A_IB2 = A_IJ2 @ A_K2J0.T
    r_OJ2 = r_OJ1 + displacement * A_IJ1[:, pris.axis]
    r_OP2 = r_OJ2 - A_IB2 @ B2_r_P2J0
    p2 = Spurrier(A_IB2)
    q2 = np.concatenate([r_OP2, p2])

    return np.concatenate([q1, q2])


Prismatic_random_displacement_cases = [
    ("Prismatic axis=0", 0, 0),
    ("Prismatic axis=1", 1, 1),
    ("Prismatic axis=2", 2, 2),
]


def _setup_prismatic_random_displacement(axis, seed):
    """Same shape as `_setup`, but for an explicitly constructed, generic
    point on the constraint manifold (random body 1 pose, random nonzero
    displacement along the free axis) rather than the assembler's
    always-displacement-0 q0."""
    rng = np.random.default_rng(seed)
    rb1 = _random_rigid_body(rng)
    rb2 = _random_rigid_body(rng)
    pris = Prismatic(rb1, rb2, axis=axis)

    system = System()
    system.add(rb1, rb2, pris)
    system.assemble()

    t0 = 0.0
    displacement = rng.uniform(-2.0, 2.0)
    q0 = _prismatic_q_with_displacement(rb1, rb2, pris, displacement, rng)
    B0 = system.q_dot_u(t0, q0)

    # sanity: the constructed configuration really is on the manifold, at
    # the intended displacement
    assert np.all(
        np.isclose(pris.g(t0, q0), 0.0, atol=1e-10)
    ), "constructed q0 does not satisfy g == 0"
    assert np.isclose(
        pris.l(t0, q0), displacement, atol=1e-10
    ), "constructed q0 does not have the intended displacement"

    return rng, pris, t0, q0, B0


@pytest.mark.filterwarnings("ignore: 'approx_fprime' is used")
@pytest.mark.parametrize("name, axis, seed", Prismatic_random_displacement_cases)
def test_KN_l_prismatic_random_displacement(name, axis, seed):
    rng, pris, t0, q0, B0 = _setup_prismatic_random_displacement(axis, seed)
    la_l = rng.random()
    _check_KN_l(name, pris, t0, q0, B0, la_l)


@pytest.mark.filterwarnings("ignore: 'approx_fprime' is used")
@pytest.mark.parametrize("name, axis, seed", Prismatic_random_displacement_cases)
def test_KN_g_prismatic_random_displacement(name, axis, seed):
    rng, pris, t0, q0, B0 = _setup_prismatic_random_displacement(axis, seed)
    la_g = rng.random(pris.nla_g)
    _check_KN_g(name, pris, t0, q0, B0, la_g)


if __name__ == "__main__":
    test_rigid_body_translational()
    test_rigid_body_rotational()
    for case in KN_g_cases:
        test_KN_g(*case)
    for case in KN_l_cases:
        test_KN_l(*case)
    for case in Revolute_random_angle_cases:
        test_KN_l_revolute_random_angle(*case)
        test_KN_g_revolute_random_angle(*case)
    for case in Prismatic_random_displacement_cases:
        test_KN_l_prismatic_random_displacement(*case)
        test_KN_g_prismatic_random_displacement(*case)
    print("all checks passed")
