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
    block_dim = np.array([5.0, 3.0, 2.0])  # size of the block

    l0 = 3.0  # rest length of the spring
    k = 300  # spring stiffness
    d = 2  # damping constant
    d = 0

    # initial conditions
    r_OC = np.array([0, 0, 0], dtype=float)
    A_IB = np.eye(3, dtype=float)
    A_IB = Exp_SO3_quat(np.array([1.0, 2.0, 3.0, 4.0]), normalize=True)

    # initialize rigid body
    q0 = RigidBody.pose2q(r_OC, A_IB)
    block = Box(RigidBody)(
        dimensions=block_dim,
        density=0.1,
        q0=q0,
        name="block",
    )

    # get offsets of upper vertices
    B_r_CPis = block.B_r_CQi_T[:, block.B_r_CQi_T[2, :] > 0].T
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
        [[np.zeros([7, 7]), B], [np.linalg.solve(M, h_q), np.linalg.solve(M, h_u)]]
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
    print(f"  omega_vertical: {np.sqrt(4 * k / block.mass)}")
    print(f" omega long axis: {np.sqrt(k * block_dim[1]**2 / block.B_Theta_C[0, 0])}")
    print(f"omega short axis: {np.sqrt(k * block_dim[0]**2 / block.B_Theta_C[1, 1])}")

    # compute eigenvalues
    lambdas, vs = np.linalg.eig(A)
    print("Computed values: ")

    v_q = vs[:7]
    v_u = vs[7:]

    for i in range(len(lambdas)):
        vq = v_q[:, i]
        vu = v_u[:, i]
        print(f"Eigenvalue {i}: {np.real(lambdas[i]):.5f} + {np.imag(lambdas[i]):.5f}j")
        print(f"    Re(v_q): {[str('{s: .5f}').format(s=s) for s in np.real(vq)]}")
        print(f"    Im(v_q): {[str('{s: .5f}').format(s=s) for s in np.imag(vq)]}")
        print(f"    Re(v_u): {[str('{s: .5f}').format(s=s) for s in np.real(vu)]}")
        print(f"    Im(v_u): {[str('{s: .5f}').format(s=s) for s in np.imag(vu)]}")

        # export nonlinear
        if False:
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

    # export linearized
    if True:
        T = 1
        nt = 1
        t = np.linspace(0, T, nt)
        q = np.array([q0] * nt)
        u = np.array([u0] * nt)

        v_q_Re = np.real(v_q)
        v_q_Im = np.imag(v_q)
        v_q_abs = np.abs(v_q)
        modes_dq = np.zeros_like(v_q.T, dtype=float)
        for i in range(len(lambdas)):
            v_qi = v_q_Re[:, i]
            v_qi = v_q_abs[:, i]
            # modes_dq[i] = v_qi
            modes_dq[i] = v_qi / np.linalg.norm(v_qi)
        omegas = np.array([np.imag(lambdas)])
        modes_dq = np.array([modes_dq.T])

        # compose solution object with omegas and modes
        sol = Solution(system, t, q, u, omegas=omegas, modes_dq=modes_dq)

        # vtk-export
        dir_name = Path(__file__).parent
        system.export(dir_name, f"vtk", sol, fps=25)

        # to visualize this export:
        # - load files in paraview as usual
        # - add filter "Warp By Vector" (Filters -> Common -> Warp By Vector)
        # - select desired mode in WarpByVector -> Properties -> Vectors
        # - Time Manager (View -> Time Inspector to show window)
        #       - untik time sources
        #       - increase number of frames
        #       - Animations -> WrapByVector -> Scale Factor -> klick on "+"
        #       - edit this animation: Interpolation -> Sinusoid (Phase, Frequency, Offset as default)
        #       - set Value to desired amplitude (Value of Time 1 is not used)
        # - activate repeat and play animation
        # - show other modes by changing the vector in WarpByVector -> Properties -> Vectors
