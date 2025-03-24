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


def scipy_eig(*args, **kwargs):
    eig = scipy.linalg.eig(*args, **kwargs)
    if eig[-1].dtype == complex:
        return eig
    elif eig[-1].dtype == float:
        eig = list(eig)
        assert np.isclose(np.linalg.norm(np.imag(eig[0])), 0.0)
        eig[0] = eig[0].astype(float)
        return tuple(eig)


if __name__ == "__main__":
    ###################
    # system parameters
    ###################
    block_dim = np.array([5.0, 3.0, 2.0])  # size of the block

    l0 = 3.0  # rest length of the spring
    k = 300  # spring stiffness
    d = 2  # damping constant
    # d = 0

    # initial conditions
    r_OC = np.array([0, 0, 0], dtype=float)
    A_IB = np.eye(3, dtype=float)
    # A_IB = Exp_SO3_quat(np.array([1.0, 2.0, 3.0, 4.0]), normalize=True)
    # A_IB = Exp_SO3_quat(2 * np.random.rand(4) - 1, normalize=True)

    r_OC2 = np.array([5, 0, 0])

    # initialize rigid body
    q01 = RigidBody.pose2q(r_OC, A_IB)
    block1 = Box(RigidBody)(
        dimensions=block_dim,
        density=0.1,
        q0=q01,
        name="block1",
    )
    q02 = RigidBody.pose2q(r_OC2, A_IB)
    block2 = Box(RigidBody)(
        dimensions=block_dim,
        density=0.1,
        q0=q02,
        name="block2",
    )

    # get offsets of upper vertices and compute positions of suspension
    B_r_CPis = block1.B_r_CQi_T[:, block1.B_r_CQi_T[2, :] > 0].T
    r_OQis = [r_OC + A_IB @ (B_r_CPi + l0 * ei(2)) for B_r_CPi in B_r_CPis]

    #################
    # assemble system
    #################

    # initialize system
    system = System()

    # spring-damper interactions
    # connection = RigidConnection(block, system.origin)
    # connection = FixedDistance(block, system.origin, B2_r_P2J2=block_dim / 2)
    connection1 = Revolute(system.origin, block1, axis=2, r_OJ0=-block_dim / 2)
    connection2 = Revolute(block1, block2, axis=2, r_OJ0=block_dim / 2)
    # connection = Cylindrical(block, system.origin, axis=2, r_OJ0=block_dim/2)
    system.add(block1, block2, connection1, connection2)
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
