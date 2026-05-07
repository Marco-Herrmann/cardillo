import numpy as np
from pathlib import Path

from cardillo import System
from cardillo.math import e1, e2, e3
from cardillo.math.rotations import A_IB_basic
from cardillo.rods.cosseratRod import make_CosseratRod
from cardillo.rods import CircularCrossSection, CrossSectionInertias, Simo1986
from cardillo.rods.force_line_distributed import Force_line_distributed

from cardillo.forces import Force
from cardillo.discrete import Frame, RigidBody, Meshed, Box, Cylinder
from cardillo.constraints import RigidConnection, Revolute

from cardillo.contacts import Sphere2Sphere

from cardillo.solver import Newton, SolverOptions, DualStormerVerlet


system = System()


Rod = make_CosseratRod(
    interpolation="Quaternion",
    mixed=True,
    constraints=[0, 1, 2],
    polynomial_degree=2,
    reduced_integration=True,
)

E = 210_000_000.0
mu = 0.5
G = E / (2 * (1 + mu))
rho_rope = 7_800.0
r_rope = 0.025

L = 10.0
nelement = 25
# because of the angle, we can start at the right and elevated
r_OP0 = np.array([2.0 - 0.45, 0.0, 2.0])
A_IB0 = A_IB_basic(3 * np.pi / 4).y

cross_section = CircularCrossSection(r_rope)
cross_section_intertias = CrossSectionInertias(rho_rope, cross_section)

I = cross_section.second_moment[1, 1]
Ei = np.ones(3, dtype=float)
Fi = np.array([G * I, E * I, E * I])
material_model = Simo1986(Ei, Fi)

q0 = Rod.straight_configuration(nelement, L, r_OP0, A_IB0)
rope = Rod(
    cross_section,
    material_model,
    nelement,
    Q=q0,
    q0=q0,
    cross_section_inertias=cross_section_intertias,
    name="Rope",
)

rope_gravity = [
    Force_line_distributed(
        lambda t, xi: -cross_section.area * 9.81 * e3 * np.max([2 * t - 1, 0.0]), rope
    ),
    Force_line_distributed(lambda t, xi: -cross_section.area * 9.81 * e3, rope),
]
rope_tension = [
    Force(lambda t: -100 * 9.81 * (e3 + e1) * t, rope, xi=1, name=f"tenioning_static"),
    Force(lambda t: -100 * 9.81 * (e3 + e1), rope, xi=1, name=f"tenioning_dynamic"),
]


# TODO: think of a nice way for the guidance!
v_max = 2.0

# time values to change
t_acc = 3.0
t_drive = 4.0
t_break = 1.4

# compute coefficients
a_acc = np.pi / t_acc
a_break = np.pi / t_break


def r_OF(t):
    if t <= t_acc:
        return r_OP0 + np.array([v_max / 2 * (t - np.sin(a_acc * t) / a_acc), 0.0, 0.0])
    elif t <= t_acc + t_drive:
        return r_OF(t_acc) + np.array([v_max * (t - t_acc), 0.0, 0.0])
    elif t <= t_acc + t_drive + t_break:
        tau = t_acc + t_drive
        return r_OF(tau) + np.array(
            [
                v_max / 2 * (t - tau + np.sin(a_break * (t - tau)) / a_break),
                0.0,
                0.0,
            ]
        )
    else:
        return r_OF(t_acc + t_drive + t_break)


guidance = Frame(r_OF)
rope_guidance = [
    RigidConnection(rope, system.origin, xi1=0),
    Revolute(rope, guidance, axis=1, xi1=0),
]

import matplotlib.pyplot as plt

fig, ax = plt.subplots(3, 1)
ts = np.linspace(0, 25.0, 201)
ax[0].plot(ts, [guidance.r_OP__(ti)[0] for ti in ts])
ax[1].plot(ts, [guidance.r_OP_t__(ti)[0] for ti in ts])
ax[2].plot(ts, [guidance.r_OP_tt__(ti)[0] for ti in ts])
[axi.grid() for axi in ax]

plt.show()
# exit()


# create roba
rho_roba = 2_000
r_rol = 0.45 / 2
d_rol = 0.6


roba_angle = -np.pi / 4
A_IB_wippe = A_IB_basic(roba_angle).y
A_IB0_rol = A_IB_basic(np.pi / 2).x

B_CP_rol = np.array(
    [
        [-d_rol / 2, 0.0, 0.0],
        [d_rol / 2, 0.0, 0.0],
    ]
)

B_CP_wippe = np.array(
    [
        [-d_rol, 0.0, 0.0],
        [d_rol, 0.0, 0.0],
    ]
)

dim_main = [2 * d_rol, 0.2, 0.2]
dim_w2 = [d_rol, 0.15, 0.15]
r_OPm = np.zeros(3, dtype=float)
q0_main = RigidBody.pose2q(r_OPm, A_IB_wippe)
wippe_main = Box(RigidBody)(dim_main, density=rho_roba, q0=q0_main, name="wippe_main")
c_main = Revolute(system.origin, wippe_main, axis=1, r_OJ0=r_OPm, name="joint_main")


wippen_constraints = [wippe_main, c_main]
rols = []

# go down hierarchy
for i, B_CP_wi in enumerate(B_CP_wippe):
    r_OPw = r_OPm + wippe_main.A_IB(0.0, q0_main) @ B_CP_wi
    q0_wippe = RigidBody.pose2q(r_OPw, A_IB_wippe)
    wippe = Box(RigidBody)(dim_w2, density=rho_roba, q0=q0_wippe, name=f"wippe2_{i}")
    c_wippe = Revolute(wippe_main, wippe, axis=1, r_OJ0=r_OPw, name=f"joint_wippe_{i}")
    wippen_constraints.extend([wippe, c_wippe])

    for j, B_CP_rj in enumerate(B_CP_rol):
        r_OPr = r_OPw + wippe.A_IB(0.0, q0_wippe) @ B_CP_rj
        q0_rol = RigidBody.pose2q(r_OPr, A_IB0_rol)
        rol = Cylinder(RigidBody)(
            r_rol, height=0.1, density=rho_roba, q0=q0_rol, name=f"rol_{i}_{j}"
        )
        c_rol = Revolute(wippe, rol, axis=1, r_OJ0=r_OPr, name=f"joint_rol_{i}_{j}")
        wippen_constraints.append(c_rol)
        rols.append(rol)

# conctact
contacts = []
mu = 0.3
# for i, rol in enumerate(rols[:1]):
for i, rol in enumerate(rols):
    for j, xi_contact in enumerate(np.linspace(0, 1, int(L / (2 * r_rope)))):
        contacts.append(
            Sphere2Sphere(
                rol, rope, r_rol, r_rope, mu=mu, xi2=xi_contact, name=f"contact_{i}_{j}"
            )
        )


system.add(rope, rope_gravity[0], rope_guidance[0], rope_tension[0])
system.assemble()

sol_static = Newton(system, 5).solve()

dir_name = Path(__file__).parent
system.export(dir_name, "vtk_static", sol_static)

system.set_new_initial_state(sol_static.q[-1], sol_static.u[-1], 0.0)


system.remove(rope_gravity[0], rope_guidance[0], rope_tension[0])
system.add(guidance, rope_gravity[1], rope_guidance[1], rope_tension[1])


# add rols, wippen, constraints and contacts
system.add(*rols, *wippen_constraints, *contacts)

# prepare for solve
system.assemble()

t1 = 5.0
dt = 1e-2
sol_dynamic = DualStormerVerlet(system, t1, dt).solve()
system.export(dir_name, "sol_dynamic", sol_dynamic)
