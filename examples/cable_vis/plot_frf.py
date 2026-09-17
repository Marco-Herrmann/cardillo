import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

csv_dir = Path(__file__).parent / "csv"
csv_paths = sorted(
    csv_dir.glob("FRF_zz_angle_multiplier_*.csv"),
    key=lambda p: int(re.search(r"_(\d+)\.csv$", p.name).group(1)),
)

colors = ["#4477AA", "#228833", "#EE6677"]  # paul_bright_blue, _green, _red

fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={"hspace": 0.05})
for i, csv_path in enumerate(csv_paths):
    angle_multiplier = int(re.search(r"_(\d+)\.csv$", csv_path.name).group(1))

    with open(csv_path) as f:
        meta_line = f.readline()
    meta = dict(
        re.findall(r"(\w+)=([\d.eE+-]+)", meta_line)
    )

    omega, amplitude, angle_deg = np.loadtxt(
        csv_path, delimiter=",", skiprows=2, unpack=True
    )

    color = colors[i % len(colors)]
    label = f"angle_multiplier={angle_multiplier}"
    ax[0].loglog(omega, amplitude, color=color, label=label)
    ax[1].semilogx(omega, angle_deg, color=color, label=label)

    if i==0:
        m = float(meta["mass_RB"])
        k = float(meta["stiffness_RB"])
        d = float(meta["damping_RB"])

        # fig.suptitle(f"m={m:.3g}, k={k:.3g}, d={d:.3g}")
    else:
        this_m = float(meta["mass_RB"])
        this_k = float(meta["stiffness_RB"])
        this_d = float(meta["damping_RB"])

        assert this_m == m, f"m don't match for (0-{i})"
        assert this_k == k, f"k don't match for (0-{i})"
        assert this_d == d, f"d don't match for (0-{i})"


# discrete (analytic) FRF of the isolated m-d-k harmonic oscillator, for
# comparison against the actual (cable-coupled) response above
H = 1.0 / (k - m * omega**2 + 1j * d * omega)
ax[0].loglog(
    omega, np.abs(H), ":", color="k", label=f"{label} (harmonic osc.)"
)
ax[1].semilogx(omega, np.angle(H, deg=True), ":", color="k")

ax[0].grid(which="both")
ax[1].grid(which="both")
# ax[1].legend()

for a in ax:
    a.tick_params(
        axis="both",
        which="both",
        length=0,
        labelbottom=False,
        labelleft=False,
    )

fig_path = Path(__file__).parent / "csv" / "FRF_zz.svg"
background_alpha = 0.5  # 0 = fully transparent, 1 = fully opaque
fig.savefig(
    fig_path, dpi=300, facecolor=(1, 1, 1, background_alpha), bbox_inches="tight"
)
print(f"Saved figure to {fig_path}")

plt.show()
