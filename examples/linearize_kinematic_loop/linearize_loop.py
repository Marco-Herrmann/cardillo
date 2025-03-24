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
from cardillo.constraints._base import ProjectedPositionOrientationBase
from cardillo.math import Exp_SO3_quat, ax2skew, cross3, ei, A_IB_basic
from cardillo.solver import Solution


if __name__ == "__main__":
    ###################
    # system parameters
    ###################
    block_dim = np.array([5.0, 3.0, 2.0])  # size of the block
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

    l, w, h = block_dim
    Bi_r_OBi = np.array([0.0, -l / 2 - w / 2, 0.0])
    Bi_r_BiJ = np.array([l / 2, w / 2, 0.0])

    # initialize rigid bodies
    n_blocks = 4
    blocks = []
    for i in range(n_blocks):
        A_IBi = A_IB_basic(2 * i * np.pi / 4).z
        r_OCi = r_OC0 + A_IBi @ Bi_r_OBi
        q0i = RigidBody.pose2q(r_OCi, A_IBi)
        blocki = Box(RigidBody)(
            dimensions=block_dim,
            density=0.1,
            q0=q0i,
            name=f"block{i}",
        )
        blocks.append(blocki)

    # add origin to the beginning to have an easy handling of the constriants
    # blocks.insert(0, system.origin)

    # initialize constraints
    constraints = []
    for i in range(n_blocks - 1):
        r_OJ0i = r_OC0 + blocks[i].r_OP(0, blocks[i].q0, B_r_CP=Bi_r_BiJ)
        # r_OJ0i = np.array([0.0, 0.0, 0.0])
        constrainti = Revolute(
            blocks[i], blocks[i + 1], axis=axis, r_OJ0=r_OJ0i, name=f"constriant{i}"
        )
        constraints.append(constrainti)

    # constrain first rigid body to ground
    connection = RigidConnection(blocks[0], system.origin)

    # closing condition
    r_OJ03 = np.array([-l / 2, -l / 2, 0.0])
    closing = ProjectedPositionOrientationBase(
        blocks[0],
        blocks[-1],
        constrained_axes_translation=[0, 1],
        projection_pairs_rotation=[],
        r_OJ0=r_OJ03,
    )

    # connection = FixedDistance(block, system.origin, B2_r_P2J2=block_dim / 2)
    # connection1 = Revolute(system.origin, blocks[1], axis=2, r_OJ0=np.array([-block_dim[0]/2, 0, 0]))
    # connection2 = Revolute(blocks[1], blocks[2], axis=2, r_OJ0=np.array([block_dim[0]/2, 0, 0]))
    # connection3 = Revolute(blocks[2], blocks[3], axis=2, r_OJ0=np.array([3*block_dim[0]/2, 0, 0]))
    # connection = Cylindrical(block, system.origin, axis=2, r_OJ0=block_dim/2)

    #################
    # assemble system
    #################
    system.add(*blocks, *constraints)
    # system.add(connection)
    system.add(closing)
    system.assemble()

    omegas, modes_dq, sol = system.eigenmodes(
        system.t0, system.q0, system.la_g0, system.la_gamma0, system.la_c0
    )

    print(omegas)

    # vtk-export
    dir_name = Path(__file__).parent
    system.export(dir_name, f"vtk", sol, fps=25)

    # Following https://public.kitware.com/pipermail/paraview/2017-October/041077.html to visualize this export:
    # - load files in paraview as usual
    # ---> multiple objects?
    #       - yes:  - select block1 and block2
    #               - add filter "Group Datasets" (Filters -> Common -> Group Datasets)
    #               - change coloring, if needed
    #               -> continue
    #       - no:   -> continue
    # - add filter "Warp By Vector" (Filters -> Common -> Warp By Vector) (to GroupDatasets-object or single object)
    # - select desired mode in WarpByVector -> Properties -> Vectors
    # - Time Manager (View -> Time Inspector to show window)
    #       - untik time sources
    #       - increase number of frames
    #       - Animations -> WrapByVector -> Scale Factor -> klick on "+"
    #       - edit this animation: Interpolation -> Sinusoid (Phase, Frequency, Offset as default)
    #       - set Value to desired amplitude (Value of Time 1 is not used)
    # - activate repeat and play animation
    # - show other modes by changing the vector in WarpByVector -> Properties -> Vectors

    # Adding object to GroupDataset:
    # - right click on the GroupDataset (or on one of the indicating objects)
    # - Change Input -> select other objects (use Crtl/Shift)
