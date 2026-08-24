import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from pprint import pprint
import scipy
import scipy.linalg

from cardillo import System
from cardillo.discrete import Box, RigidBody
from cardillo.forces import Force
from cardillo.force_laws import KelvinVoigtElement as SpringDamper
from cardillo.interactions import TwoPointInteraction
from cardillo.math import Exp_SO3_quat, e3
from cardillo.solver import Newton, Eigenmodes, FrequencyResponseFunction

if __name__ == "__main__":
    ###################
    # system parameters
    ###################
    block_dim = np.array([5.0, 3.0, 2.0])  # size of the block

    l0 = np.sqrt(3.0)  # rest length of the spring
    k = 0.5  # spring stiffness
    d = 2  # damping constant
    d = 0.1
    # d = 10

    # initial conditions
    r_OC = np.array([0, 0, 0], dtype=float)
    A_IB = np.eye(3, dtype=float)
    # A_IB = Exp_SO3_quat(np.array([1.0, 2.0, 3.0, 4.0]), normalize=True)
    # A_IB = Exp_SO3_quat(2 * np.random.rand(4) - 1, normalize=True)

    # initialize rigid body
    q0 = RigidBody.pose2q(r_OC, A_IB)
    block = Box(RigidBody)(
        dimensions=block_dim,
        density=0.01,
        # density=1,
        q0=q0,
        name="block",
    )

    # get offsets of upper vertices and compute positions of suspension
    B_r_CPis = block.B_r_CQi_T.T
    r_OQis = []
    for i, B_r_CPi in enumerate(B_r_CPis):
        e_PiQi = np.random.rand(3)
        # e_PiQi = np.array([1.0, 1.0, 1.0])

        r_CPi = A_IB @ B_r_CPi

        if r_CPi[0] < 0:
            e_PiQi[0] *= -1

        if r_CPi[1] < 0:
            e_PiQi[1] *= -1

        if r_CPi[2] < 0:
            e_PiQi[2] *= 0.5

        r_OQis.append(r_OC + r_CPi + l0 * e_PiQi / np.linalg.norm(e_PiQi))

    #################
    # assemble system
    #################

    # initialize system
    system = System()

    # spring-damper interactions
    spring_dampers = [
        SpringDamper(
            TwoPointInteraction(
                block, system.origin, B_r_CP1=B_r_CPi, B_r_CP2=r_OQi, name=f"TPI{i}"
            ),
            k,
            d,
            l_ref=l0,
            compliance_form=False,
            name=f"spring_damper{i}",
        )
        for i, (B_r_CPi, r_OQi) in enumerate(zip(B_r_CPis, r_OQis))
    ]

    # gravity
    force = Force(
        lambda t: -block.mass * np.array([0, 0, 9.81]) * t,
        block,
        name="gravity",
    )

    # forcing for linearization
    forcing = Force(
        np.zeros(3),
        block,
        B_r_CP=np.array([block_dim[0] / 2, 0.0, 0.0]),
        name="forcing",
    )

    system.add(block, *spring_dampers, force, forcing)
    system.assemble()

    # solve static equilibrium
    solver_stat = Newton(system, 10)
    sol_stat = solver_stat.solve()

    # compute eigenmodes
    solver_eig = Eigenmodes(system, sol_stat)
    sol_eig = solver_eig.solve(-1)

    # prepare a second system, where there is no gravity, but the system is in the deflected configuration
    system_B = System()
    r_OC_B = block.r_OP(None, sol_stat.q[-1])
    A_IB_B = block.A_IB(None, sol_stat.q[-1])
    q0_B = RigidBody.pose2q(r_OC_B, A_IB_B)
    block_B = Box(RigidBody)(
        dimensions=block_dim,
        density=0.01,
        # density=1,
        q0=q0_B,
        name="block",
    )

    # spring-damper interactions
    spring_dampers_B = [
        SpringDamper(
            TwoPointInteraction(
                block_B, system_B.origin, B_r_CP1=B_r_CPi, B_r_CP2=r_OQi, name=f"TPI{i}"
            ),
            k,
            d,
            l_ref=np.linalg.norm(r_OC_B + A_IB_B @ B_r_CPi - r_OQi),
            compliance_form=False,
            name=f"spring_damper{i}",
        )
        for i, (B_r_CPi, r_OQi) in enumerate(zip(B_r_CPis, r_OQis))
    ]

    # forcing for linearization
    forcing_B = Force(
        np.zeros(3),
        block_B,
        B_r_CP=np.array([block_dim[0] / 2, 0.0, 0.0]),
        name="forcing",
    )

    system_B.add(block_B, *spring_dampers_B, forcing_B)
    system_B.assemble()

    # solve static equilibrium
    solver_stat_B = Newton(system_B, 10)
    sol_stat_B = solver_stat_B.solve()

    # compute eigenmodes
    solver_eig_B = Eigenmodes(system_B, sol_stat_B)
    sol_eig_B = solver_eig_B.solve(-1)

    ###################
    # post-processing #
    ###################
    # blender export
    dir_name = Path(__file__).parent
    system.export_blender(
        dir_name, f"blender_stat", sol_stat, create_blend=True, verbose=False
    )
    system.export_blender(
        dir_name, f"blender_eig", sol_eig, create_blend=True, verbose=False
    )
    system_B.export_blender(
        dir_name, f"blender_stat_B", sol_stat_B, create_blend=True, verbose=False
    )
    system_B.export_blender(
        dir_name, f"blender_eig_B", sol_eig_B, create_blend=True, verbose=False
    )

    print(f"\n\n\nOmegas  : {sol_eig.omegas}")
    print(f"\n\n\nOmegas_B: {sol_eig_B.omegas}")
    exit()

    iom = 1j * np.logspace(0, 4, 1000)
    solver_frf = FrequencyResponseFunction(system, sol_stat)
    frfs = solver_frf.solve(-1, s_val=iom)[:, :, 3:]

    print(frfs.shape)

    fig, ax = plt.subplots(2, 3, sharex=True)
    for i in range(3):
        ax[0, i].loglog(iom.imag, np.abs(frfs[:, 0, i]))
        ax[1, i].semilogx(iom.imag, np.angle(frfs[:, 0, i]))

    plt.show()
