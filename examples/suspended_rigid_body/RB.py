import matplotlib.pyplot as plt
import numpy as np

from cardillo import System
from cardillo.discrete import Box, Sphere, RigidBody, Axis
from cardillo.force_laws import KelvinVoigtElement as SpringDamper
from cardillo.forces import Force
from cardillo.interactions import TwoPointInteraction
from cardillo.math import Exp_SO3_quat, e3, norm
from cardillo.solver import Newton

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

    # make r_CP upwards
    from cardillo.math.SmallestRotation import SmallestRotation

    A_SR = SmallestRotation(e_CP, i=3).A_RJ

    e_CP = A_SR.T @ e_CP
    r_OC = A_SR.T @ r_OC

    # initialize rigid body
    q0 = RigidBody.pose2q(r_OC, A_SR.T @ A_IB)
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

    assert system.nla_c == 0

    h_q = system.h_q(system.t0, system.q0, system.u0).toarray()
    g_S_q = system.g_S_q(system.t0, system.q0).toarray()

    K = np.vstack([h_q, g_S_q])
    K_s = 0.5 * (K + K.T)
    K_a = 0.5 * (K - K.T)
    dK_a = np.linalg.norm(K_a, "fro") / np.linalg.norm(K, "fro")

    B = system.q_dot_u(system.t0, system.q0).toarray()

    KB = K @ B

    print(f"projection error: {np.linalg.norm(KB[-1])}")

    KB = KB[:-1]
    KB_s = 0.5 * (KB + KB.T)
    KB_a = 0.5 * (KB - KB.T)
    dKB_a = np.linalg.norm(KB_a, "fro") / np.linalg.norm(KB, "fro")

    print(f" dK_a: {dK_a:.5f}")
    print(f"dKB_a: {dKB_a:.5f}")

    from cardillo.solver.solution import Solution

    sol = Solution(
        system, np.array([system.t0]), np.array([system.q0]), np.zeros((1, system.nu))
    )

    # vtk-export
    from pathlib import Path

    dir_name = Path(__file__).parent
    block_fake = Axis(RigidBody)(
        mass=1.0, B_Theta_C=np.eye(3), name="block_fake", origin_size=0.3
    )
    block_fake = RigidBody(mass=1.0, B_Theta_C=np.eye(3), name="block_fake")
    block_fake.qDOF = block.qDOF
    block_fake.uDOF = block.uDOF
    system.add(block_fake)
    system.export(dir_name, f"vtk", sol, fps=25)
