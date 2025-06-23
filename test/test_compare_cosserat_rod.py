import pytest
from timeit import timeit
from itertools import product


from cardillo.rods import (
    CrossSectionInertias,
    RectangularCrossSection,
    Simo1986,
    Harsch2021,
    animate_beam,
)

from cardillo.rods.cosseratRod import make_CosseratRod
from cardillo.rods.boostedCosseratRod import make_BoostedCosseratRod

from cardillo.constraints import RigidConnection
from cardillo.solver import Newton, SolverOptions
from cardillo.forces import Force, B_Moment

from cardillo.math import e2, e3

from cardillo.visualization import Export

from cardillo import System

import numpy as np

import matplotlib.pyplot as plt
from pathlib import Path

pDeg = 2
c = None
RI = True
Rod_slow = make_CosseratRod(
    interpolation="Quaternion",
    mixed=True,
    constraints=c,
    polynomial_degree=pDeg,
    reduced_integration=RI,
)
Rod_boosted = make_BoostedCosseratRod(polynomial_degree=pDeg, constraints=c)

# geometry of the rod
length = 2 * np.pi

# cross section properties for visualization purposes
slenderness = 1.0e2
width = length / slenderness
cross_section = RectangularCrossSection(width, 2 * width)

# material properties
Ei = np.array([5, 1, 1])
Fi = np.array([0.5, 2, 2])
material_model = Simo1986(Ei, Fi)

cross_section_inertias = CrossSectionInertias(1.0, cross_section)

nelement = 15

# create rods
Rods = [Rod_slow, Rod_boosted]
# the order in q and u is identical for both implementations!
Q_ref, u_ref = Rod_slow.straight_initial_configuration(nelement, length)
Q_test = Q_ref + np.random.rand(len(Q_ref)) * length
u_test = u_ref + np.random.rand(len(u_ref)) * length

rods = []
us = []
for i, Rod in enumerate(Rods):
    rod = Rod(
        cross_section,
        material_model,
        nelement,
        Q=Q_ref,
        q0=Q_test,
        u0=u_test,
        cross_section_inertias=cross_section_inertias,
        name=f"Rod {i}",
    )

    rods.append(rod)
    us.append(np.random.rand(rod.nu))

sys = System()
sys.add(*rods)
sys.assemble(options=SolverOptions(compute_consistent_initial_conditions=False))


def args_dict_random(random):
    qRand = np.random.rand(int(sys.nq / 2))
    uRand = np.random.rand(int(sys.nu / 2))
    u_dotRand = np.random.rand(int(sys.nu / 2))
    la_cRand = np.random.rand(int(sys.nla_c / 2))

    t = 0.0 + random * np.random.rand()
    q_ = sys.q0 + random * np.concatenate([qRand, qRand])
    u_ = sys.u0 + random * np.concatenate([uRand, uRand])
    u_dot = sys.u_dot0 + random * np.concatenate([u_dotRand, u_dotRand])
    la_c = sys.la_c0 + random * np.concatenate([la_cRand, la_cRand])

    zeros3 = np.zeros(3, dtype=float)

    def args_dict(rod, step_callback=False, xi=0.0, B_r_CP=zeros3):
        if step_callback:
            q, u = sys.step_callback(t, q_, u_)
        else:
            q, u = q_, u_

        qRod = q[rod.qDOF]
        qeRod = qRod[rod.elDOF_P(xi)]

        uRod = u[rod.uDOF]
        ueRod = uRod[rod.elDOF_P_u(xi)]

        u_dotRod = u_dot[rod.uDOF]
        ue_dotRod = u_dotRod[rod.elDOF_P_u(xi)]

        if hasattr(rod, "la_cDOF"):
            la_cRod = la_c[rod.la_cDOF]
        else:
            la_cRod = None

        return {
            "t": t,
            "xi": xi,
            "q": qRod,
            "qe": qeRod,
            "u": uRod,
            "ue": ueRod,
            "ue_dot": ue_dotRod,
            "la_c": la_cRod,
            "B_r_CP": B_r_CP,
        }

    return args_dict


# Testfunctions to check identical implementations
def test_nodes():
    r0 = rods[0].nodes(sys.q0)
    r1 = rods[1].nodes(sys.q0)
    nr = np.linalg.norm(r0 - r1)
    assert np.isclose(nr, 0.0, atol=1e-5), nr


def test_nodalValues_global():
    r0, ex0, ey0, ez0 = rods[0].nodalFrames(sys.q0)
    r1, ex1, ey1, ez1 = rods[1].nodalFrames(sys.q0)

    nr = np.linalg.norm(r0 - r1)
    nex = np.linalg.norm(ex0 - ex1)
    ney = np.linalg.norm(ey0 - ey1)
    nez = np.linalg.norm(ez0 - ez1)
    assert np.isclose(nr, 0.0, atol=1e-5), nr
    assert np.isclose(nex, 0.0, atol=1e-5), nex
    assert np.isclose(ney, 0.0, atol=1e-5), ney
    assert np.isclose(nez, 0.0, atol=1e-5), nez


def test_nodalValues_elementwise():
    r0, ex0, ey0, ez0 = rods[0].nodalFrames(sys.q0, True)
    r1, ex1, ey1, ez1 = rods[1].nodalFrames(sys.q0, True)

    nr = np.linalg.norm(r0 - r1)
    nex = np.linalg.norm(ex0 - ex1)
    ney = np.linalg.norm(ey0 - ey1)
    nez = np.linalg.norm(ez0 - ez1)
    assert np.isclose(nr, 0.0, atol=1e-5), nr
    assert np.isclose(nex, 0.0, atol=1e-5), nex
    assert np.isclose(ney, 0.0, atol=1e-5), ney
    assert np.isclose(nez, 0.0, atol=1e-5), nez


# function name - arguments - sparse
# grouped for testing the speedup with same arguments
funcs_args_grouped = [
    [
        ["M", ["t", "q"], True],
    ],
    [
        ["q_dot", ["t", "q", "u"], False],
        ["q_dot_q", ["t", "q", "u"], True],
        ["q_dot_u", ["t", "q"], True],
    ],
    [
        ["h", ["t", "q", "u"], False],
        ["h_u", ["t", "q", "u"], True],
    ],
    [
        ["g_S", ["t", "q"], False],
        ["g_S_q", ["t", "q"], True],
    ],
    [
        ["c_la_c", [], True],
        ["la_c", ["t", "q", "u"], False],
        ["W_c", ["t", "q"], True],
        ["c", ["t", "q", "u", "la_c"], False],
        ["c_q", ["t", "q", "u", "la_c"], True],
        # TODO: add W_g, g, g_q also here
    ],
]

funcs_args = [v for group in funcs_args_grouped for v in group]


# calls for interaction
xis = [0.0, 1.0, int(nelement / 2) / nelement, np.random.rand()]
elDOFs = [[rods[0].elDOF_P(xi), rods[0].elDOF_P_u(xi)] for xi in xis]

B_r_CPs = [
    np.zeros(3, dtype=float),
    np.array([0, 0, 0]),
    np.random.rand(3),
    np.array([0.0, 1.0, 2.0]),
]

funcs_interaction = [
    ["r_OP", ["t", "qe", "xi", "B_r_CP"]],
    ["r_OP_q", ["t", "qe", "xi", "B_r_CP"]],
    ["J_P", ["t", "qe", "xi", "B_r_CP"]],
    ["J_P_q", ["t", "qe", "xi", "B_r_CP"]],
    #
    ["v_P", ["t", "qe", "ue", "xi", "B_r_CP"]],
    ["v_P_q", ["t", "qe", "ue", "xi", "B_r_CP"]],
    #
    ["a_P", ["t", "qe", "ue", "ue_dot", "xi", "B_r_CP"]],
    ["a_P_q", ["t", "qe", "ue", "ue_dot", "xi", "B_r_CP"]],
    ["a_P_u", ["t", "qe", "ue", "ue_dot", "xi", "B_r_CP"]],
    #
    #
    ["A_IB", ["t", "qe", "xi"]],
    ["A_IB_q", ["t", "qe", "xi"]],
    ["B_J_R", ["t", "qe", "xi"]],
    ["B_J_R_q", ["t", "qe", "xi"]],
    #
    ["B_Omega", ["t", "qe", "ue", "xi"]],
    ["B_Omega_q", ["t", "qe", "ue", "xi"]],
    #
    ["B_Psi", ["t", "qe", "ue", "ue_dot", "xi"]],
    ["B_Psi_q", ["t", "qe", "ue", "ue_dot", "xi"]],
    ["B_Psi_u", ["t", "qe", "ue", "ue_dot", "xi"]],
]

cases_interaction = np.array(
    list(product(funcs_interaction, xis, B_r_CPs)), dtype=object
)


@pytest.mark.parametrize("name, args, sparse", funcs_args)
def test_cardillo_core(name, args, sparse):
    args_dict = args_dict_random(True)
    res = []
    for rod in rods:
        assert hasattr(rod, name)
        fct = getattr(rod, name)
        ddict = args_dict(rod)
        a = [ddict[ai] for ai in args]
        res.append(fct(*a))

    if sparse:
        res = [r.tocsc().toarray() for r in res]

    n = np.linalg.norm(res[0] - res[1])
    assert np.isclose(n, 0.0, atol=1e-5), n


@pytest.mark.parametrize("xi", xis)
def test_elDOFs(xi):
    print(xi)
    res = []
    res_u = []
    for rod in rods:
        res.append(rod.elDOF_P(xi))
        res_u.append(rod.elDOF_P_u(xi))

    n = np.linalg.norm(res[0] - res[1])
    n_u = np.linalg.norm(res_u[0] - res_u[1])
    assert n == 0, n
    assert n_u == 0, n_u


@pytest.mark.parametrize("name_args, xi, B_r_CP", cases_interaction)
def test_cardillo_interaction(name_args, xi, B_r_CP):
    name = name_args[0]
    args = name_args[1]
    args_dict = args_dict_random(True)
    res = []
    for rod in rods:
        assert hasattr(rod, name)
        fct = getattr(rod, name)
        ddict = args_dict(rod, xi=xi, B_r_CP=B_r_CP)
        a = [ddict[ai] for ai in args]
        res.append(fct(*a))

    n = np.linalg.norm(res[0] - res[1])
    assert np.isclose(n, 0.0, atol=1e-5), n


def performance_cardillo_core(group):
    name = ", ".join([p[0] for p in group])
    print(f"{name}")

    # test time result
    def wrapper(rod):
        args_ = args_dict_random(1)(rod, 0)
        for name, args, sparse in group:
            fct = getattr(rod, name)
            a = [args_[ai] for ai in args]
            fct(*a)

    num = 500
    tmts = []
    for i, rod in enumerate(rods):
        globals().update(locals())
        tmt = timeit("wrapper(rod)", globals=globals(), number=num)
        print(f"    {rod.name}: {tmt:.5e}")
        tmts.append(tmt)

    print(f"        Speedup: {(1 - tmts[1] / tmts[0]) * 100:.2f}%")


def performance_compare():
    num = 5_000
    # current
    globals().update(locals())
    tmtc = timeit("rods[0]._M_coo()", globals=globals(), number=num)
    print(f"    Current: {tmtc:.5e}")

    # option 1
    tmt1 = timeit("rods[1].M1()", globals=globals(), number=num)
    print(f"    Option1: {tmt1:.5e}")

    # option 2
    tmt2 = timeit("rods[1].M2()", globals=globals(), number=num)
    print(f"    Option1: {tmt2:.5e}")

    print(f"1 : {(1 - tmt1 / tmtc) * 100:.2f}")
    print(f"2 : {(1 - tmt2 / tmtc) * 100:.2f}")


if __name__ == "__main__":
    # test_nodes()
    # test_nodalValues_global()
    # test_nodalValues_elementwise()
    # compliance and constraints

    # performance_compare()

    for case in cases_interaction:
        test_cardillo_interaction(*case)

    for group in funcs_args_grouped:
        performance_cardillo_core(group)
