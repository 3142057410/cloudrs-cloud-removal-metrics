"""
replot_prodval.py —— 用已有的产品级验证结果重画 fig9 / fig10（高分辨率版）

为什么单独有这个脚本：`lai_product_validation.py` 里的图按 200 dpi 存，fig10 印到
0.86\textwidth（5.9 in）时只有 163 dpi，达不到 Elsevier 的 300 dpi 要求。
本脚本只读 CSV、不重算，几秒钟出图。

同时把 fig9 的口径与表 13 对齐：散点画的是**云掩膜内经元**的 LAI（LAI_ret_cld），
图例里的括号是同一口径的 RMSE（rmse_cld），因此与表 13 的 RMSE 列逐位相同。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TAB = Path("results/tables_boa")
FIG = Path("results/figures")
DPI = 400

per = pd.read_csv(TAB / "product_validation_per_patch.csv")
ms = pd.read_csv(TAB / "product_validation_method.csv", index_col=0)
ms = ms.loc[[m for m in ["zero_fill", "mean_fill", "nearest_fill", "blur_fill",
                         "inpaint_telea", "knn_spatial", "hist_match",
                         "unet_opt", "dsen2cr_opt"] if m in ms.index]]

plt.rcParams.update({"font.size": 9.5, "axes.grid": True, "grid.alpha": .25,
                     "axes.axisbelow": True, "savefig.dpi": DPI})

# ---------------- Fig 9 ----------------
fig, ax = plt.subplots(figsize=(5.0, 4.5))
cmap = plt.get_cmap("tab10")
for i, m in enumerate(ms.index):
    g = per[per.method == m]
    ax.scatter(g.lai_ref, g.LAI_ret_cld, s=7, alpha=.45,
               color=cmap(i % 10), linewidths=0,
               label=f"{m} ({ms.rmse_cld[m]:.2f})")
lim = [0, max(per.lai_ref.max(), per.LAI_ret_cld.max()) * 1.05]
ax.plot(lim, lim, "k--", lw=.9)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("MCD15A3H LAI (reference)")
ax.set_ylabel("LAI retrieved on cloud-masked pixels")
ax.legend(fontsize=6.4, ncol=2, loc="upper left", frameon=False,
          handletextpad=.2, columnspacing=.9, borderaxespad=.3)
fig.tight_layout()
fig.savefig(FIG / "fig9_lai_product_scatter.png")
plt.close(fig)
print("fig9", (FIG / "fig9_lai_product_scatter.png").stat().st_size)

# ---------------- Fig 10 ----------------
bins = [0, .2, .4, .6, .8, 1.01]
lab = ["0–20", "20–40", "40–60", "60–80", "80–100"]
per["cf_bin"] = pd.cut(per.cloud_fraction, bins, labels=lab, right=False)
fig, ax = plt.subplots(figsize=(5.4, 3.9))
for m in ["zero_fill", "mean_fill", "hist_match", "unet_opt", "dsen2cr_opt"]:
    g = per[per.method == m].groupby("cf_bin", observed=True)
    y, sd = g.bias_all.mean(), g.bias_all.std()
    ax.errorbar(range(len(y)), y, yerr=sd, marker="o", ms=3.4, lw=1.1,
                capsize=2.2, elinewidth=.9, label=m)
ax.axhline(0, color="k", lw=.9, ls=":")
ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab)
ax.set_xlabel("Cloud fraction of patch (%)")
ax.set_ylabel("LAI bias vs MCD15A3H")
ax.legend(fontsize=7, ncol=2, frameon=False, loc="lower left",
          handletextpad=.4, columnspacing=1.0)
ax.set_title("Product-level bias grows with cloud cover", fontsize=9.5)
fig.tight_layout()
fig.savefig(FIG / "fig10_bias_vs_cloud.png")
plt.close(fig)
print("fig10", (FIG / "fig10_bias_vs_cloud.png").stat().st_size)
