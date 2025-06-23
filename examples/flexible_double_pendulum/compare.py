import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from cardillo.solver import load_solution


###############
# load solution
###############
path = Path(__file__)
sols = [
    load_solution(Path(path.parent, "sol_boosted.pkl")),
    load_solution(Path(path.parent, "sol_default.pkl")),
]


# plot results
fig, ax = plt.subplots(2, 1)
ax = np.array([ax])


styles = ["-", "--"]
names = ["boosted", "default"]

for i, sol in enumerate(sols):

    contributions = sol.system.contributions
    rods = [c for c in contributions if "Rod" in c.name]

    # get position of the end
    qDOF_end = rods[1].qDOF[rods[1].elDOF_P(1.0)]
    # qDOF_end = rods[1].qDOF[rods[1].elDOF[-1]]
    r_OE = np.array(
        [rods[1].r_OP(ti, qi[qDOF_end], 1.0) for (ti, qi) in zip(sol.t, sol.q)]
    )
    qDOF_middle = rods[0].qDOF[rods[0].elDOF[-1]]
    r_OM = np.array(
        [rods[0].r_OP(ti, qi[qDOF_middle], 1.0) for (ti, qi) in zip(sol.t, sol.q)]
    )

    ax[0, 0].plot(sol.t, r_OE[:, 0], f"{styles[i]}r", label=names[i])
    ax[0, 0].plot(sol.t, r_OE[:, 1], f"{styles[i]}g", label=names[i])
    ax[0, 0].plot(sol.t, r_OE[:, 2], f"{styles[i]}b", label=names[i])
    ax[0, 0].grid(True)
    ax[0, 0].legend()

    ax[0, 1].plot(sol.t, r_OM[:, 0], f"{styles[i]}r", label=names[i])
    ax[0, 1].plot(sol.t, r_OM[:, 1], f"{styles[i]}g", label=names[i])
    ax[0, 1].plot(sol.t, r_OM[:, 2], f"{styles[i]}b", label=names[i])
    ax[0, 1].grid(True)
    ax[0, 1].legend()

plt.show()
