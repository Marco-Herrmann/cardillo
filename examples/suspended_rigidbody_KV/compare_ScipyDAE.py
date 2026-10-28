import matplotlib.pyplot as plt
import numpy as np

from cardillo.solver import MoreauTheta, ScipyDAE

from suspended_rigidbody_KV import make_system, spring_forces

# Reference: old force laws (la_c, damping inside c) with ScipyDAE (Radau IIA).
# Compared: new force laws (la_e, la_d) with MoreauTheta for several step sizes.

if __name__ == "__main__":
    t1 = 20.0
    d = 0.1
    theta = 0.5
    dts = [4e-2, 2e-2, 1e-2, 5e-3]

    ###########
    # reference
    ###########
    system_ref, block_ref = make_system(use_force_laws_new=False, d=d)
    sol_ref = ScipyDAE(system_ref, t1, dts[-1], rtol=1e-8, atol=1e-10).solve()
    t_ref = sol_ref.t
    r_ref = np.array(
        [block_ref.r_OP(t, q[block_ref.qDOF]) for t, q in zip(t_ref, sol_ref.q)]
    )
    la_ref = spring_forces(sol_ref)

    ##########################
    # MoreauTheta, new laws
    ##########################
    results = []
    for dt in dts:
        system, block = make_system(use_force_laws_new=True, d=d)
        sol = MoreauTheta(system, t1, dt, theta=theta).solve()
        r = np.array([block.r_OP(t, q[block.qDOF]) for t, q in zip(sol.t, sol.q)])
        la = spring_forces(sol)

        # compare on the common time grid of the reference
        r_i = np.column_stack([np.interp(t_ref, sol.t, r[:, j]) for j in range(3)])
        la_i = np.column_stack(
            [np.interp(t_ref, sol.t, la[:, j]) for j in range(la.shape[1])]
        )
        err_r = np.max(np.linalg.norm(r_i - r_ref, axis=1))
        err_la = np.max(np.abs(la_i - la_ref))
        results.append((dt, sol, r, la, err_r, err_la))
        print(
            f"dt = {dt:.1e}: max |r_OC - r_OC_ref| = {err_r:.3e}, max |la - la_ref| = {err_la:.3e}"
        )

    # observed order of convergence
    for (dt0, *_, e0, _), (dt1, *_, e1, _) in zip(results[:-1], results[1:]):
        print(
            f"order r_OC ({dt0:.1e} -> {dt1:.1e}): {np.log(e0 / e1) / np.log(dt0 / dt1):.2f}"
        )

    ###################
    # post-processing #
    ###################
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 9))
    for j, coord in enumerate("xyz"):
        ax[0].plot(
            t_ref,
            r_ref[:, j],
            "k-",
            lw=2.5,
            alpha=0.4,
            label=f"${coord}_C$ ScipyDAE (old)" if j == 0 else None,
        )
    dt, sol, r, la, *_ = results[-1]
    for j, coord in enumerate("xyz"):
        ax[0].plot(
            sol.t, r[:, j], "--", label=f"${coord}_C$ MoreauTheta (new), dt={dt:.0e}"
        )
    ax[0].set_ylabel("position of center of mass")
    ax[0].legend(fontsize=7)
    ax[0].grid()

    for j in range(la_ref.shape[1]):
        ax[1].plot(t_ref, la_ref[:, j], "k-", lw=2.5, alpha=0.4)
        ax[1].plot(sol.t, la[:, j], "--")
    ax[1].set_ylabel("spring-damper force\n(gray: ScipyDAE)")
    ax[1].grid()

    for dt, sol, r, la, *_ in results:
        r_i = np.column_stack([np.interp(t_ref, sol.t, r[:, j]) for j in range(3)])
        ax[2].semilogy(
            t_ref, np.linalg.norm(r_i - r_ref, axis=1) + 1e-16, label=f"dt={dt:.0e}"
        )
    ax[2].set_xlabel("t")
    ax[2].set_ylabel("|r_OC - r_OC_ref|")
    ax[2].legend()
    ax[2].grid()

    plt.tight_layout()
    plt.show()
