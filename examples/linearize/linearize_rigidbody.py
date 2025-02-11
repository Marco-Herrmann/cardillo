import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from pprint import pprint

from cardillo import System
from cardillo.discrete import Box, PointMass, Frame, RigidBody
from cardillo.force_laws import KelvinVoigtElement as SpringDamper
from cardillo.interactions import TwoPointInteraction
from cardillo.math import Exp_SO3_quat, ax2skew, ax2skew_squared
from cardillo.solver import ScipyIVP, Solution

if __name__ == "__main__":
    ###################
    # system parameters
    ###################
    m = 1.0  # mass
    block_dim = np.array([5.0, 3.0, 2.0])  # size of the block
    B_Theta_C = m / 12 * np.diag(np.diag(-ax2skew_squared(block_dim)))

    l0 = 3.0  # rest length of the spring
    k = 100  # spring stiffness
    d = 2  # damping constant
    d = 0

    # initial conditions
    r_OC = np.array([0, 0, 0], dtype=float)
    A_IB = np.eye(3, dtype=float)
    A_IB = Exp_SO3_quat(np.array([1.0, 2.0, 3.0, 4.0]), normalize=True)

    # initialize rigid body
    q0 = RigidBody.pose2q(r_OC, A_IB)
    block = RigidBody(m, B_Theta_C, q0=q0, u0=np.zeros(6, dtype=float), name="block")
    block = Box(RigidBody)(
        dimensions=block_dim,
        mass=m,
        B_Theta_C=B_Theta_C,
        q0=q0,
        u0=np.zeros(6, dtype=float),
        name="block",
    )

    # compute offsets
    B_r_CP0 = 0.5 * np.array([block_dim[0], -block_dim[1], block_dim[2]], dtype=float)
    B_r_CP1 = 0.5 * np.array([block_dim[0], block_dim[1], block_dim[2]], dtype=float)
    B_r_CP2 = 0.5 * np.array([-block_dim[0], block_dim[1], block_dim[2]], dtype=float)
    B_r_CP3 = 0.5 * np.array([-block_dim[0], -block_dim[1], block_dim[2]], dtype=float)

    B_r_CPis = [B_r_CP0, B_r_CP1, B_r_CP2, B_r_CP3]
    r_OQis = [r_OC + A_IB @ (B_r_CPi + np.array([0, 0, l0])) for B_r_CPi in B_r_CPis]

    #################
    # assemble system
    #################

    # initialize system
    system = System()

    # spring-damper interactions
    spring_dampers = [
        SpringDamper(
            TwoPointInteraction(
                block,
                system.origin,
                B_r_CP1=B_r_CP0,
                B_r_CP2=r_OQ0,
            ),
            k,
            d,
            l_ref=l0,
            compliance_form=False,
            name=f"spring_damper{i}",
        )
        for i, (B_r_CP0, r_OQ0) in enumerate(zip(B_r_CPis, r_OQis))
    ]

    system.add(block, *spring_dampers)

    system.assemble()

    # compute matrices
    t0 = 0.0
    q0 = system.q0
    u0 = system.u0

    M = system.M(t0, q0).toarray()
    M_inv = np.linalg.inv(M)

    h = system.h(t0, q0, u0)
    h_q = system.h_q(t0, q0, u0).toarray()
    h_u = system.h_u(t0, q0, u0).toarray()

    q_dot = system.q_dot(t0, q0, u0)
    B = system.q_dot_u(t0, q0).toarray()

    A = np.block(
        [[np.linalg.solve(M, h_u), np.linalg.solve(M, h_q)], [B, np.zeros([7, 7])]]
    )

    if False:
        print(f"M: \n{M}")
        print(f"h: \n{h}")
        print(f"h_q: \n{h_q}")
        print(f"h_u: \n{h_u}")
        print(f"q_dot: \n{q_dot}")
        print(f"B: \n{B}")
        print(f"A: \n{A}")

    print("Theoretical values: ")
    print(f"  omega_vertical: {np.sqrt(4 * k / m)}")
    print(f" omega long axis: {np.sqrt(k * block_dim[1]**2 / B_Theta_C[0, 0])}")
    print(f"omega short axis: {np.sqrt(k * block_dim[0]**2 / B_Theta_C[1, 1])}")

    # compute eigenvalues
    lambdas, vs = np.linalg.eig(A)
    print("Computed values: ")
    pprint([*lambdas])

    v_u = vs[:6]
    v_q = vs[6:]

    for i in range(len(lambdas)):
        vq = v_q[:, i]
        vu = v_u[:, i]
        print(f"Eigenvalue {i}: {np.real(lambdas[i]):.5f} + {np.imag(lambdas[i]):.5f}j")
        print(f"    Re(v_u): {[str('{s: .5f}').format(s=s) for s in np.real(vu)]}")
        print(f"    Im(v_u): {[str('{s: .5f}').format(s=s) for s in np.imag(vu)]}")
        print(f"    Re(v_q): {[str('{s: .5f}').format(s=s) for s in np.real(vq)]}")
        print(f"    Im(v_q): {[str('{s: .5f}').format(s=s) for s in np.imag(vq)]}")

        # export
        if True:
            T = 2
            nt = 51
            t = np.linspace(0, T, nt)
            # avoid quaternion to be 0
            scl = 0.9

            q = np.zeros([nt, 7])
            u = np.zeros([nt, 6])

            delta_q_form = vq / np.linalg.norm(vq)

            for j, tj in enumerate(t):
                mtpj = scl * np.exp(tj / T * 2 * np.pi * 1j)
                q[j] = q0 + np.real(mtpj * delta_q_form)
                u[j] = u0 + np.real(mtpj * v_u[:, i])

            sol = Solution(system, t, q, u)

            # vtk-export
            dir_name = Path(__file__).parent
            system.export(dir_name, f"vtk_{i}", sol, fps=25)
