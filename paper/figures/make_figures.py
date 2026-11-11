"""Draws fig_casimir.png: Casimir values on exterior powers and on spinors."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
here = os.path.dirname(os.path.abspath(__file__))
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), dpi=200)
for ax, m in zip(axes, (7, 15)):
    ks = list(range(m + 1))
    vals = [k * (m - k) for k in ks]
    ax.bar(ks, vals, color="#4a7bb7", width=0.6, label=r"$\Lambda^k\mathbb{R}^{%d}$: $k(%d-k)$" % (m, m))
    ax.axhline(m * (m - 1) / 8, color="#c0392b", lw=1.6, label=r"spinors: $%d\cdot%d/8$" % (m, m - 1))
    ax.set_xlabel(r"$k$")
    ax.set_ylabel("Casimir")
    ax.set_title(r"$\mathfrak{so}(%d)$" % m)
    ax.set_xticks(ks if m == 7 else ks[::2])
    ax.legend(fontsize=8, loc="upper right" if m == 7 else "lower center", framealpha=0.9)
    ax.set_ylim(0, max(vals) * 1.35)
fig.tight_layout()
fig.savefig(os.path.join(here, "fig_casimir.png"))
