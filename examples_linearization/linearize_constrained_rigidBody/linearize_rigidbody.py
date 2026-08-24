import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from pprint import pprint
import scipy
import scipy.linalg

from cardillo import System
from cardillo.discrete import Box, RigidBody
from cardillo.force_laws import KelvinVoigtElement as SpringDamper
from cardillo.forces import Force
from cardillo.interactions import TwoPointInteraction
from cardillo.constraints import RigidConnection, FixedDistance, Revolute, Cylindrical
from cardillo.math import Exp_SO3_quat, ax2skew, cross3, ei, norm
from cardillo.solver import Eigenmodes, Newton, FrequencyResponseFunction

if __name__ == "__main__":
    ###################
    # system parameters
    ###################
    block_dim = np.array([5.0, 3.0, 2.0])  # size of the block
    n_blocks = 5  # number of blocks
    axis = 2  # axis of rotation

    # initialize system
    system = System()

    l0 = 3.0  # rest length of the spring
    k = 300  # spring stiffness
    d = 2  # damping constant
    # d = 0

    # initial conditions
    r_OC0 = np.array([0, 0, 0], dtype=float)
    A_IB = np.eye(3, dtype=float)
    # A_IB = Exp_SO3_quat(np.array([1.0, 2.0, 3.0, 4.0]), normalize=True)
    # A_IB = Exp_SO3_quat(2 * np.random.rand(4) - 1, normalize=True)

    # initialize rigid bodies
    blocks = []
    for i in range(n_blocks):
        r_OC = r_OC0 + i * block_dim  # np.array([i*block_dim[0], 0.0, 0.0])
        r_OC[axis] = 0.0
        q0i = RigidBody.pose2q(r_OC, A_IB)
        blocki = Box(RigidBody)(
            dimensions=block_dim,
            density=0.1,
            q0=q0i,
            name=f"block{i}",
        )
        blocks.append(blocki)

    # add origin to the beginning to have an easy handling of the constriants
    blocks.insert(0, system.origin)

    # initialize constraints
    constraints = []
    for i in range(n_blocks):
        # r_OJ0i = np.array([-block_dim[0]/2 + i * block_dim[0], 0.0, 0.0])
        r_OJ0i = -block_dim / 2 + i * block_dim
        r_OJ0i[axis] = 0.0
        constrainti = Revolute(
            blocks[i], blocks[i + 1], axis=axis, r_OJ0=r_OJ0i, name=f"constraint{i}"
        )
        constraints.append(constrainti)

    preloaded = False
    preloaded = True
    if preloaded:
        e_diag = block_dim.copy()
        # e_diag[axis] = 0.0
        e_diag /= norm(e_diag)

        f0 = 100.0

        f = lambda t: f0 * e_diag
        force = Force(f, blocks[-1], B_r_CP=block_dim / 2, name="force")
        system.add(force)

    #################
    # assemble system
    #################
    system.add(*blocks[1:], *constraints)
    system.assemble()

    if preloaded:
        static_solver = Newton(system)
        sol0 = static_solver.solve()
        # print(sol0.la_g)
    else:
        sol0 = system.sol0

    #######################
    # test KN and Wla_g_q #
    #######################
    for i in range(len(sol0.t)):
        ti = sol0.t[i]
        qi = sol0.q[i]
        la_gi = sol0.la_g[i]
        KN = [A.toarray() for A in system.KN_g(ti, qi, la_gi)]
        Wla_g_q = system.Wla_g_q(ti, qi, la_gi).toarray()
        B = system.q_dot_u(ti, qi).toarray()

        # this should not have N contributions
        print(f"norm(N): {np.linalg.norm(KN[1])}")

        # K is on LHS, but Wla_g on RHS
        K_proj = -Wla_g_q @ B
        K_proj_sym = (K_proj + K_proj.T) / 2

        # TODO: I don't get why it is not symmetric?
        # check for symmetry
        print(f"K_proj - K_proj.T: {np.linalg.norm(K_proj - K_proj.T)}")

        # compute difference
        diff_sym = K_proj_sym - KN[0]
        print(f"norm(diff_sym): {np.linalg.norm(diff_sym)}")
        diff = K_proj - KN[0]
        print(f"norm(diff): {np.linalg.norm(diff)}")

    solver = Eigenmodes(system, system.sol0)
    sol_modes = solver.solve(-1)

    print(sol_modes.omegas)

    iom = 1j * np.logspace(-0.5, 2, 500)
    solver_frf = FrequencyResponseFunction(system, system.sol0)
    frfs = solver_frf.solve(-1, iom)

    fig, ax = plt.subplots(3, 2, sharex=True)
    for i in range(6 * (n_blocks)):
        for j in range(2):
            i_fig = i % 6
            if i_fig == 0:
                i_fig = 0
            elif i_fig == 1:
                i_fig = 1
            elif i_fig == 5:
                i_fig = 2
            else:
                continue

            ax[i_fig, j].loglog(iom.imag, np.abs(frfs[:, i, j]), label=f"Body {i//6}")
            ax[i_fig, j].grid()

    ax[0, 0].legend()
    ax[0, 0].set_title("X-excitation")
    ax[0, 1].set_title("Y-excitation")
    ax[0, 0].set_ylabel("X-Amplitude")
    ax[1, 0].set_ylabel("Y-Amplitude")
    ax[2, 0].set_ylabel("Alpha-Amplitude")
    plt.show()

    # vtk-export
    dir_name = Path(__file__).parent
    # system.export(dir_name, f"vtk", sol, fps=25)
    system.export_blender(dir_name, f"blender", sol_modes, create_blend=True)
