"""
ndvi_corr.py —— 第三下游任务（NDVI 分类）的指标相关性分析

之前这段是临时命令算的，加方法后无法复现，故固化成脚本。

产出：
  results/tables/corr_ndvi_task.csv   两种口径 × 6 指标 × 3 个目标

用法：
  python src/ndvi_corr.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

METRICS = ["PSNR", "SSIM", "SAM", "ERGAS", "MAE", "RMSE"]
TARGETS = ["dLAI_mean", "dCab_mean", "dclass"]
HIGH_THR = 25.0


def main():
    cfg = yaml.safe_load(open("configs/p1.yaml", encoding="utf-8"))
    tab = Path(cfg["output"]["tables"])

    img = pd.read_csv(tab / "per_patch_metrics.csv")
    dn = pd.read_csv(tab / "per_patch_downstream.csv")[["method", "patch", "dLAI_mean", "dCab_mean"]]
    nd = pd.read_csv(tab / "ndvi_task.csv")
    d = img.merge(dn, on=["method", "patch"]).merge(nd, on=["method", "patch"])
    print(f"样本 {len(d)} 行 | 方法 {d.method.nunique()} | patch {d.patch.nunique()}")

    g = d.groupby("method")[METRICS + TARGETS].mean()
    hi = g[g.PSNR >= HIGH_THR]
    print(f"合格方法（PSNR≥{HIGH_THR:.0f}）{len(hi)} 个")

    rows = []
    for scope, gg in (("全量方法层", g), ("合格方法层", hi)):
        for m in METRICS:
            for t in TARGETS:
                rows.append({"scope": scope, "n_methods": len(gg), "metric": m,
                             "target": t,
                             "rho": round(float(spearmanr(gg[m], gg[t]).correlation), 3)})
    res = pd.DataFrame(rows)
    res.to_csv(tab / "corr_ndvi_task.csv", index=False)

    print()
    print(res.pivot_table(index=["scope", "metric"], columns="target", values="rho")
          .round(3).to_string())

    print("\n影像块层（n=%d）：" % len(d))
    for m in METRICS:
        print(f"  {m:5s}  vs Δclass {spearmanr(d[m], d.dclass).correlation:+.3f}"
              f"   vs ΔLAI {spearmanr(d[m], d.dLAI_mean).correlation:+.3f}")
    print(f"\n-> {tab/'corr_ndvi_task.csv'}")


if __name__ == "__main__":
    main()
