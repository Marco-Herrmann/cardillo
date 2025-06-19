import pytest
from timeit import timeit


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

q0, u0 = sys.step_callback(0.0, sys.q0, sys.u0)  # normalize quaternions
t = 0.0
sys.q0 = q0
sys.u0 = u0


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


######################
# cardillo functions #
######################

properties = []

properties.extend(["q_dot", "q_dot_q", "q_dot_u"])

properties.extend(["g"])
properties.extend(["gamma"])

properties.extend(["c", "c_q", "c_u"])


# function name - arguments - sparse
# grouped for testing the speedup with same arguments
funcs_args_grouped = [
    [
        ["M", ["t", "q"], True],
    ],
    [
        ["q_dot", ["t", "q", "u"], False],
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

args_dict = lambda rod: {
    "t": 0.0,
    "q": q0[rod.qDOF],
    "u": u0[rod.uDOF],
    "la_c": np.zeros(rod.nla_c),
}
args_dict_random = lambda rod: {
    "t": np.random.rand(),
    "q": q0[rod.qDOF] + np.random.rand(rod.nq),
    "u": u0[rod.uDOF] + np.random.rand(rod.nu),
    "la_c": np.zeros(rod.nla_c) + np.random.rand(rod.nla_c),
}


@pytest.mark.parametrize("name, args, sparse", funcs_args)
def test_cardillo_core(name, args, sparse):
    res = []
    for rod in rods:
        assert hasattr(rod, name)
        fct = getattr(rod, name)
        a = [args_dict(rod)[ai] for ai in args]
        res.append(fct(*a))

    if sparse:
        res = [r.tocsc().toarray() for r in res]

    n = np.linalg.norm(res[0] - res[1])
    assert np.isclose(n, 0.0, atol=1e-5), n


def performance_cardillo_core(group):
    name = ", ".join([p[0] for p in group])
    print(f"{name}")

    # test time result
    def wrapper(rod):
        args_ = args_dict_random(rod)
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


if __name__ == "__main__":
    # test_nodes()
    # test_nodalValues_global()
    # test_nodalValues_elementwise()
    # compliance and constraints

    for group in funcs_args_grouped:
        performance_cardillo_core(group)
