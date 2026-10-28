import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from cardillo import System
from cardillo.discrete import Box, RigidBody
from cardillo.forces import Force
from cardillo.interactions import TwoPointInteraction
from cardillo.solver import MoreauTheta
from cardillo.force_laws_new import KelvinVoigtElement as SpringDamperNew
from cardillo.force_laws import KelvinVoigtElement as SpringDamperOld


def make_system(
    use_force_laws_new=True,
    compliance_form=True,
    nsprings=6,
    k=0.5,
    d=0.1,
    seed=0,
):
    """Rigid block suspended by spring-dampers at randomly chosen vertices.

    Parameters
    ----------
    use_force_laws_new : bool
        True: new scalar force laws (la_e/la_d), False: old ones (la_c).
    compliance_form : bool
        Passed to the spring-dampers.
    nsprings : int
        Number of spring-dampers (at most 8, one per vertex).
    k, d : float
        Stiffness and damping constant of the spring-dampers.
    seed : int
        Seed of the random geometry.

    Returns
    -------
    system : System
        Assembled system.
    block : RigidBody
        The suspended block.
    """
    block_dim = np.array([5.0, 3.0, 2.0])  # size of the block
    density = 0.01
    l0 = np.sqrt(3.0)  # rest length of the springs

    # random but reproducible geometry
    rng = np.random.default_rng(seed=seed)

    # initial conditions
    r_OC = np.zeros(3, dtype=float)
    A_IB = np.eye(3, dtype=float)

    # initialize rigid body
    q0 = RigidBody.pose2q(r_OC, A_IB)
    block = Box(RigidBody)(
        dimensions=block_dim,
        density=density,
        q0=q0,
        name="block",
    )

    # pick random vertices and compute positions of suspension points, such
    # that the springs point outwards and are undeformed in the initial
    # configuration
    B_r_CPis = block.B_r_CQi_T.T[rng.choice(8, size=nsprings, replace=False)]
    r_OQis = []
    for B_r_CPi in B_r_CPis:
        e_PiQi = rng.random(3)

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
    system = System()

    # spring-damper interactions
    SpringDamper = SpringDamperNew if use_force_laws_new else SpringDamperOld
    spring_dampers = [
        SpringDamper(
            TwoPointInteraction(
                block, system.origin, B_r_CP1=B_r_CPi, B_r_CP2=r_OQi, name=f"TPI{i}"
            ),
            k,
            d,
            l_ref=l0,
            compliance_form=compliance_form,
            name=f"spring_damper{i}",
        )
        for i, (B_r_CPi, r_OQi) in enumerate(zip(B_r_CPis, r_OQis))
    ]

    # gravity
    gravity = Force(
        -block.mass * np.array([0, 0, 9.81]),
        block,
        name="gravity",
    )

    system.add(block, *spring_dampers, gravity)
    system.assemble()

    return system, block


def spring_forces(sol):
    """Total spring-damper forces: la_e + la_d (new) or la_c (old)."""
    if sol.la_e is not None and sol.la_e.shape[1] > 0:
        la = sol.la_e.copy()
        if sol.la_d is not None and sol.la_d.shape[1] > 0:
            la += sol.la_d
        return la
    return sol.la_c


if __name__ == "__main__":
    # simulation parameters
    t1 = 20.0
    dt = 1e-2
    theta = 0.5

    system, block = make_system(use_force_laws_new=True)
    print(f"nla_c: {system.nla_c}, nla_e: {system.nla_e}, nla_d: {system.nla_d}")

    ##################
    # forward dynamics
    ##################
    sol = MoreauTheta(system, t1, dt, theta=theta).solve()
    t, q = sol.t, sol.q

    ###################
    # post-processing #
    ###################
    r_OC = np.array([block.r_OP(ti, qi[block.qDOF]) for ti, qi in zip(t, q)])
    la_springs = spring_forces(sol)

    fig, ax = plt.subplots(2, 1, sharex=True)
    for i, coord in enumerate("xyz"):
        ax[0].plot(t, r_OC[:, i], label=f"${coord}_C$")
    ax[0].set_ylabel("position of center of mass")
    ax[0].legend()
    ax[0].grid()

    for i in range(la_springs.shape[1]):
        ax[1].plot(t, la_springs[:, i], label=f"spring_damper{i}")
    ax[1].legend()
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("spring-damper force")
    ax[1].grid()

    plt.tight_layout()
    plt.show()

    # blender export
    # dir_name = Path(__file__).parent
    # system.export_blender(dir_name, "blender", sol, create_blend=True)
