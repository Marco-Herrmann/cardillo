import numpy as np
from pathlib import Path


from cardillo import System
from cardillo.math import A_IB_basic
from cardillo.discrete import Frame
from cardillo.constraints import RigidConnection
from cardillo.rods_new import (
    CircularCrossSection,
    Simo1986,
    make_CosseratRod,
    CrossSectionInertias,
)
from cardillo.solver import Newton, Solution, SolverOptions, Eigenmodes


def pose_velocity(
    nu_node,
    nnodes,
    v_P,
    B_Omega,
    xi1=1.0,
):
    xis = np.linspace(0, xi1, nnodes)

    # nodal positions and unit quaternions
    vO = np.zeros((nnodes, nu_node))
    for i, xii in enumerate(xis):
        vO[i, :3] = v_P(xii)
        vO[i, 3:] = B_Omega(xii)

    return vO.reshape(-1)


r = 0.01
L = 1.0

nelement = 128

cross_section = CircularCrossSection(r)
cross_section_inertias = CrossSectionInertias(1.0, cross_section)

# material properties
E = 1e4
mu = 0.3
G = E / (2 * (1 + mu))
A = cross_section.area(0.0)
Ip, Iy, Iz = np.diag(cross_section.second_moment(0.0))
Ei = np.array([E * A, G * A, G * A])
Fi = np.array([G * Ip, E * Iy, E * Iz])

material_model = Simo1986(Ei, Fi)

Rod = make_CosseratRod()
q0 = Rod.straight_configuration(nelement=nelement, L=L)
rod = Rod(
    cross_section,
    material_model,
    nelement,
    cross_section_inertias=cross_section_inertias,
    Q=q0,
    name="Rod",
)


u1_x = -0.6
u1_y = -0.1
u1_z = u1_x

A_IL = lambda t: A_IB_basic(-np.pi / 6 * t).z
r_OR = lambda t: L * np.array(
    [(1 + u1_x) - u1_x * np.cos(t * np.pi / 2), t * u1_y, t * u1_z]
)
A_IR = lambda t: A_IB_basic(5.5 * np.pi * t).x

frame_left = Frame(name="frame_left", A_IB=A_IL)
frame_right = Frame(name="frame_right", r_OP=r_OR, A_IB=A_IR)

connection_left = RigidConnection(rod, frame_left, xi1=0, name="connection_left")
connection_right = RigidConnection(rod, frame_right, xi1=1, name="connection_right")

system = System()
system.add(rod, frame_left, frame_right, connection_left, connection_right)
system.assemble()

# solve
solver = Newton(system, n_load_steps=200)
sol = solver.solve()

solver_eig = Eigenmodes(system, sol)
sol_eig = solver_eig.solve(-1)

dir_name = Path(__file__).parent
print("Export static solution (single)")
system.export_blender(dir_name, "cable_single", sol, create_blend=True)
print("Export eigenmodes (single)")
system.export_blender(dir_name, "cable_single_eig", sol_eig, create_blend=True)

# make nice visuals with 3 individual cables
ny = 10
nz = 2
r_cable = 0.005
h2 = np.sqrt(3) / 3
B_r_CPs = r_cable * np.array(
    [
        [0.0, 0.0, 2 * h2],
        [0.0, 1.0, -h2],
        [0.0, -1.0, -h2],
    ]
)
cross_section_vis = CircularCrossSection(radius=r_cable, export_resolution=36)
r_OP_vis = lambda t, q, xi, B_r_CP: rod.r_OP(
    t, q[rod.qDOF[rod.local_qDOF_P(xi)]], xi=xi, B_r_CP=B_r_CP
)
A_IB_vis = lambda t, q, xi: rod.A_IB(t, q[rod.qDOF[rod.local_qDOF_P(xi)]], xi=xi)

Dr_P_vis = lambda t, q, Dz, xi, B_r_CP: rod.v_P(
    t,
    q[rod.qDOF[rod.local_qDOF_P(xi)]],
    Dz[rod.qDOF[rod.local_uDOF_P(xi)]],
    xi=xi,
    B_r_CP=B_r_CP,
)
B_Dphi_vis = lambda t, q, Dz, xi: rod.B_Omega(
    t, q[rod.qDOF[rod.local_qDOF_P(xi)]], Dz[rod.qDOF[rod.local_uDOF_P(xi)]], xi=xi
)

system_mult = System()
system_mult.add(frame_left, frame_right)
q0s_mult = np.zeros((3, len(sol.t), rod.nq))
Dzs_mult = np.zeros((3, 10, rod.nu))
cables = np.zeros(3, dtype=object)
for i, B_r_CP in enumerate(B_r_CPs):
    for it in range(len(sol.t)):
        q0 = Rod.pose_configuration(
            nelement,
            lambda xi: r_OP_vis(sol.t[it], sol.q[it], xi, B_r_CP),
            lambda xi: A_IB_vis(sol.t[it], sol.q[it], xi),
        )
        q0s_mult[i, it] = q0
    cables[i] = Rod(
        cross_section_vis,
        material_model,
        nelement,
        Q=q0s_mult[i, 0],
        name=f"cable_{i}",
    )
    system_mult.add(cables[i])

    for iM in range(10):
        Dz_single = sol_eig.Delta_z[:, iM]
        Dzs_mult[i, iM] = pose_velocity(
            6,
            rod.nnodes,
            lambda xi: Dr_P_vis(sol.t[-1], sol.q[-1], Dz_single, xi, B_r_CP),
            lambda xi: B_Dphi_vis(sol.t[-1], sol.q[-1], Dz_single, xi),
        )

assemble_options = SolverOptions(compute_consistent_initial_conditions=False)
system_mult.assemble(options=assemble_options)
q_mult = np.zeros((len(sol.t), system_mult.nq))
for it in range(len(sol.t)):
    for i in range(3):
        q_mult[it, cables[i].qDOF] = q0s_mult[i, it]

Dz_mult = np.zeros((system_mult.nu, 10))
for iM in range(10):
    for i in range(3):
        Dz_mult[cables[i].uDOF, iM] = Dzs_mult[i, iM]


sol_mult = Solution(
    system_mult,
    t=sol.t,
    q=q_mult,
)
sol_mult_eig = Solution(
    system_mult,
    sol_eig.t,
    sol_mult.q[-1],
    omegas=sol_eig.omegas[:10],
    Delta_z=Dz_mult,
)
system_mult.export_blender(dir_name, "cable_mult", sol_mult, create_blend=True)
system_mult.export_blender(dir_name, "cable_mult_eig", sol_mult_eig, create_blend=True)
