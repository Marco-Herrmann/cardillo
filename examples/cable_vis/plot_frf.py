import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FRF_STRING = "zz"
# FRF_STRING = "DF"

csv_dir = Path(__file__).parent / "csv"
csv_paths = sorted(
    csv_dir.glob(f"FRF_{FRF_STRING}_angle_multiplier_*.csv"),
    key=lambda p: int(re.search(r"_(\d+)\.csv$", p.name).group(1)),
)

colors = ["#4477AA", "#228833", "#EE6677"]  # paul_bright_blue, _green, _red
variant_labels = ["Design A", "Design B", "Design C"]
linewidth = 2.5

titles = {
    "zz": "Forcing right → Displacement right",
    "DF": "Displacement right → Reaction force left",
}

fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={"hspace": 0.05})
fig_width, fig_height = fig.get_size_inches()
fig_amp, ax_amp = plt.subplots(1, 1, figsize=(fig_width, fig_height / 2))
for i, csv_path in enumerate(csv_paths):
    angle_multiplier = int(re.search(r"_(\d+)\.csv$", csv_path.name).group(1))

    with open(csv_path) as f:
        meta_line = f.readline()
    meta = dict(re.findall(r"(\w+)=([\d.eE+-]+)", meta_line))

    omega, amplitude, angle_deg = np.loadtxt(
        csv_path, delimiter=",", skiprows=2, unpack=True
    )

    if i == 0:
        m = float(meta["mass_RB"])
        k = float(meta["stiffness_RB"])
        d = float(meta["damping_RB"])

        omega0 = np.sqrt(k / m)

        # fig.suptitle(f"m={m:.3g}, k={k:.3g}, d={d:.3g}")
    else:
        this_m = float(meta["mass_RB"])
        this_k = float(meta["stiffness_RB"])
        this_d = float(meta["damping_RB"])

        assert this_m == m, f"m don't match for (0-{i})"
        assert this_k == k, f"k don't match for (0-{i})"
        assert this_d == d, f"d don't match for (0-{i})"

    # plot normalized
    omega_norm = omega / omega0
    color = colors[i % len(colors)]
    label = variant_labels[i % len(variant_labels)]
    ax[0].loglog(omega_norm, amplitude, color=color, label=label, linewidth=linewidth)
    ax[1].semilogx(omega_norm, angle_deg, color=color, label=label, linewidth=linewidth)
    ax_amp.loglog(omega_norm, amplitude, color=color, label=label, linewidth=linewidth)


if FRF_STRING == "zz":
    # discrete (analytic) FRF of the isolated m-d-k harmonic oscillator, for
    # comparison against the actual (cable-coupled) response above
    H = 1.0 / (k - m * omega**2 + 1j * d * omega)
    ax[0].loglog(
        omega_norm, np.abs(H), ":", color="k", label="No cable", linewidth=linewidth
    )
    ax[1].semilogx(
        omega_norm, np.angle(H, deg=True), ":", color="k", linewidth=linewidth
    )
    ax_amp.loglog(
        omega_norm, np.abs(H), ":", color="k", label="No cable", linewidth=linewidth
    )

ax[0].legend(loc="upper right")
amp_legend_loc = "upper right" if FRF_STRING == "zz" else "lower left"
ax_amp.legend(loc=amp_legend_loc)

title = titles.get(FRF_STRING)
if title is not None:
    ax_amp.set_title(title)

ax[1].set_yticks([-180, -90, 0, 90, 180])
ax[1].set_xticks([0.5, 1.0, 5.0])
ax[1].set_xticklabels(["0.5", "1.0", "5.0"])
ax[1].set_xlabel(r"Frequency $\dfrac{\omega}{\omega_0}$")
ax_amp.set_xlabel(r"Frequency $\dfrac{\omega}{\omega_0}$")

ax[0].grid(which="both")
ax[1].grid(which="both")
ax_amp.grid(which="major")
# ax[1].legend()

for a in ax:
    a.tick_params(
        axis="both",
        which="both",
        length=0,
        labelbottom=False,
        labelleft=False,
    )

ax[0].tick_params(axis="y", which="major", labelleft=True)
ax[1].tick_params(axis="both", which="major", labelleft=True, labelbottom=True)

ax_amp.tick_params(
    axis="both",
    which="both",
    length=0,
    labelbottom=False,
    labelleft=False,
)
ax_amp.tick_params(axis="both", which="major", labelleft=True, labelbottom=True)
ax_amp.set_xticks([0.5, 1.0, 5.0])
ax_amp.set_xticklabels(["0.5", "1.0", "5.0"])

fig_path = Path(__file__).parent / "csv" / f"FRF_{FRF_STRING}"
fig_amp_path = Path(__file__).parent / "csv" / f"FRF_{FRF_STRING}_amplitude"
background_alpha = 0.5  # 0 = fully transparent, 1 = fully opaque

for a in ax:
    a.set_facecolor((1, 1, 1, background_alpha))
ax_amp.set_facecolor((1, 1, 1, background_alpha))

for ending in [".png", ".svg"]:
    fig.savefig(
        fig_path.with_suffix(ending),
        dpi=300,
        facecolor=(1, 1, 1, background_alpha),
        bbox_inches="tight",
    )
    fig_amp.savefig(
        fig_amp_path.with_suffix(ending),
        dpi=300,
        facecolor=(1, 1, 1, background_alpha),
        bbox_inches="tight",
    )
print(f"Saved figure to {fig_path}")
print(f"Saved figure to {fig_amp_path}")

plt.show()
