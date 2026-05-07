import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from pprint import pprint
import scipy
import scipy.linalg

from cardillo import System
from cardillo.discrete import Box, Sphere, RigidBody, Axis
from cardillo.force_laws import KelvinVoigtElement as SpringDamper
from cardillo.forces import Force
from cardillo.interactions import TwoPointInteraction
from cardillo.math import Exp_SO3_quat, e3, norm
from cardillo.solver import Eigenmodes, Newton


def solve_quadratic(a, b, c):
    D = b**2 - 4 * a * c
    if D < 0:
        raise ValueError("no real roots")
    else:
        sqrt_D = np.sqrt(D)
        return (-b + sqrt_D) / (2 * a), (-b - sqrt_D) / (2 * a)


def inertia_axes_perp_analytic(I, n):
    I = np.asarray(I, dtype=float)
    n = np.asarray(n, dtype=float)
    n /= np.linalg.norm(n)

    # --- 1. wähle u ⟂ n (robust) ---
    # n nicht parallel zu e_z?
    if abs(n[2]) < 0.9:
        u = np.cross(n, [0, 0, 1])
    else:
        u = np.cross(n, [0, 1, 0])
    u /= np.linalg.norm(u)

    # --- 2. zweiter Basisvektor ---
    v = np.cross(n, u)

    # --- 3. 2×2-Trägheitstensor ---
    A = u @ I @ u
    B = u @ I @ v
    C = v @ I @ v

    # --- 4. analytische Eigenwerte ---
    tr = A + C
    disc = np.sqrt((A - C) ** 2 + 4 * B**2)

    I1 = 0.5 * (tr + disc)  # Maximum
    I2 = 0.5 * (tr - disc)  # Minimum

    # --- 5. Richtungen ---
    if abs(B) < 1e-12:
        if A > C:
            e1 = u
            e2 = v
        else:
            e1 = v
            e2 = -u
    else:
        theta = 0.5 * np.arctan2(2 * B, A - C)
        e1 = np.cos(theta) * u + np.sin(theta) * v
        e2 = -np.sin(theta) * u + np.cos(theta) * v

    # return None, I1, I2
    return e1, e2, I1, I2


def get_omega_swing(m, theta, k, g, l0, r):
    k_xx = k * m * g / (k * l0 + m * g)
    k_xalpha = -k_xx * r
    k_alphaalpha = m * g * r * (k * l0 + k * r + m * g) / (k * l0 + m * g)

    a = m * theta
    b = -theta * k_xx - m * k_alphaalpha
    c = k_xx * k_alphaalpha - k_xalpha**2
    omega_squared1, omega_squared2 = solve_quadratic(a, b, c)
    return np.sqrt(omega_squared1), np.sqrt(omega_squared2)


if __name__ == "__main__":
    ###################
    # system parameters
    ###################
    block_dim = np.array([5.0, 3.0, 1.0])  # size of the block
    density = 0.1
    mass = density * np.prod(block_dim)

    l0 = 3.0  # rest length of the spring
    k = 300  # spring stiffness
    d = 2  # damping constant
    d = 0

    g = 9.81

    # initial conditions
    A_IB = np.eye(3, dtype=float)
    # A_IB = Exp_SO3_quat(np.array([1.0, 2.0, 3.0, 4.0]), normalize=True)
    A_IB = Exp_SO3_quat(2 * np.random.rand(4) - 1, normalize=True)

    hook_position = "corner"
    # hook_position = "edge"
    # hook_position = "surface"
    # hook_position = "center"
    if hook_position == "corner":
        B_r_CP = block_dim / 2
        n_correct = 0
    elif hook_position == "edge":
        B_r_CP = block_dim / 2
        B_r_CP[0] = 0
        n_correct = 2
    elif hook_position == "surface":
        B_r_CP = block_dim / 2
        B_r_CP[0] = 0
        B_r_CP[1] = 0
        n_correct = 4
    elif hook_position == "center":
        B_r_CP = block_dim / 2
        B_r_CP[0] = 0
        B_r_CP[1] = 0
        B_r_CP[2] = 0
        n_correct = 4

    r_CP = A_IB @ B_r_CP
    r = norm(B_r_CP)
    if r == 0.0:
        e_CP = e3
    else:
        e_CP = r_CP / r
    l_c = r + mass * g / k + l0
    r_OC = -l_c * e_CP

    # initialize rigid body
    q0 = RigidBody.pose2q(r_OC, A_IB)
    block = Box(RigidBody)(
        dimensions=block_dim,
        density=density,
        q0=q0,
        name="block",
    )

    #################
    # assemble system
    #################
    # initialize system
    system = System()

    # spring-damper interactions
    spring_damper = SpringDamper(
        TwoPointInteraction(
            block,
            system.origin,
            B_r_CP1=B_r_CP,
        ),
        k,
        d,
        l_ref=l0,
        compliance_form=False,
        # compliance_form=True,
        name=f"spring_damper",
    )

    gravity = Force(lambda t: -mass * g * e_CP, block, name="gravity")

    system.add(block, spring_damper, gravity)
    system.assemble()

    print(
        f"static equilibrium: {norm(system.h(system.t0, system.q0, system.u0) + system.W_c(system.t0, system.q0) @ system.la_c0)}"
    )

    # compute eigenmodes
    solver = Eigenmodes(system, system.sol0)
    omegas, modes_dq, sol = solver.solve(-1)

    # theoretical values
    # only (easily) possible for hook_position in ["center", "surface"]
    # partially for "edge", but not for "corner", see n_correct
    B_e_max, B_e_min, theta_max, theta_min = inertia_axes_perp_analytic(
        block.B_Theta_C, A_IB.T @ e_CP
    )
    A_BR = np.vstack([A_IB.T @ e_CP, B_e_max, B_e_min]).T
    omega_swing3, omega_swing1 = get_omega_swing(block.mass, theta_max, k, g, l0, r)
    omega_swing4, omega_swing2 = get_omega_swing(block.mass, theta_min, k, g, l0, r)

    print(f"Theoretical values with  {n_correct} correct swing omegas:")
    print(f"    omega swing1: {omega_swing1}")
    print(f"    omega swing2: {omega_swing2}")
    print(f"    omega swing3: {omega_swing3}")
    print(f"    omega swing4: {omega_swing4}")
    print(f"  omega_vertical: {np.sqrt(k / block.mass)}")
    print(f" Computed values: {omegas}")

    #####
    # do the test with the first order system
    #####
    M = system.M(system.t0, system.q0).toarray()
    h_q = system.h_q(system.t0, system.q0, system.u0).toarray()
    B = system.q_dot_u(system.t0, system.q0).toarray()
    # minus at both sides helps somehow to get more precise eigenvalues
    eigs = scipy.linalg.eigvals(-h_q @ B, -M)
    print(
        f"1st order system: {np.real(np.emath.sqrt(np.sort(-eigs)))} (only for compliance_form=False)"
    )

    # vtk-export
    dir_name = Path(__file__).parent
    block_fake = Axis(RigidBody)(mass=1.0, B_Theta_C=np.eye(3), name="block_fake")
    block_fake.qDOF = block.qDOF
    block_fake.uDOF = block.uDOF
    system.add(block_fake)
    system.export(dir_name, f"vtk", sol, fps=25)
