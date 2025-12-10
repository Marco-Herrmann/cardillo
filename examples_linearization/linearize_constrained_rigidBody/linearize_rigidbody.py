import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from pprint import pprint
import scipy
import scipy.linalg

from cardillo import System
from cardillo.discrete import Box, RigidBody
from cardillo.force_laws import KelvinVoigtElement as SpringDamper
from cardillo.interactions import TwoPointInteraction
from cardillo.constraints import RigidConnection, FixedDistance, Revolute, Cylindrical
from cardillo.math import Exp_SO3_quat, ax2skew, cross3, ei
from cardillo.solver import Solution


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

    # connection = RigidConnection(block, system.origin)
    # connection = FixedDistance(block, system.origin, B2_r_P2J2=block_dim / 2)
    # connection1 = Revolute(system.origin, blocks[1], axis=2, r_OJ0=np.array([-block_dim[0]/2, 0, 0]))
    # connection2 = Revolute(blocks[1], blocks[2], axis=2, r_OJ0=np.array([block_dim[0]/2, 0, 0]))
    # connection3 = Revolute(blocks[2], blocks[3], axis=2, r_OJ0=np.array([3*block_dim[0]/2, 0, 0]))
    # connection = Cylindrical(block, system.origin, axis=2, r_OJ0=block_dim/2)

    #################
    # assemble system
    #################
    system.add(*blocks[1:], *constraints)
    system.assemble()

    for proj in [None, "NullSpace", "ComplianceProjection"]:
        omegas, modes_dq, sol = system.eigenmodes(
            system.t0,
            system.q0,
            system.la_g0,
            system.la_gamma0,
            system.la_c0,
            constraints=proj,
        )

        print(omegas)

        # vtk-export
        dir_name = Path(__file__).parent
        system.export(dir_name, f"vtk{'_'+proj if proj else ''}", sol, fps=25)

