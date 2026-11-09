"""
robust_drop_weak.py —— 剔除弱基线后的稳健性检查

背景：主表里有一个明显偏弱的生成式基线（扩散，PSNR ≈ 18 dB）。它的存在会被审稿人
当作"稻草人"，进而怀疑"PSNR 在全部方法上相关很高"这个结论是被它抬起来的。

这个脚本做两件事：
  1) 去掉最弱的方法后重算三层相关，看结论是否变化 —— 若不变，说明结论不依赖弱基线
  2) 对"高质量子集里 SAM 优于 PSNR"做配对 bootstrap，给出差值的置信区间
     —— 只报"0.905 vs 0.810"不够，必须给出这两个数之差本身的可信范围

输出：
  results/tables/robust_drop_weak.csv
  results/tables/pairwise_metric_diff.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

METRICS = ["PSNR", "SSIM", "SAM", "ERGAS", "MAE", "RMSE"]
HIGH_THR = 25.0


def boot_method_rho(metric_vals, target_vals, n_boot=2000, seed=0):
    """对方法层（点少）做 bootstrap，返回 rho 与其分布，用于配对比较。"""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(metric_vals, float), np.asarray(target_vals, float)
    n = len(x)
    out = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if np.std(x[i]) < 1e-12 or np.std(y[i]) < 1e-12:
            continue
        r = spearmanr(x[i], y[i]).correlation
        if np.isfinite(r):
            out.append(r)
    return np.array(out)


def main():
    cfg = yaml.safe_load(open("configs/p1.yaml", encoding="utf-8"))
    tab = Path(cfg["output"]["tables"])
    img = pd.read_csv(tab / "per_patch_metrics.csv")
    dn = pd.read_csv(tab / "per_patch_downstream.csv")
    df = img.merge(dn, on=["method", "patch"], how="inner")

    weak = df.groupby("method")["PSNR"].mean().idxmin()
    all_means = df.groupby("method")["PSNR"].mean()
    very_weak = all_means[all_means < 20.0].index.tolist()   # PSNR<20 的一律算弱基线
    print("=" * 72)
    print(f"最弱方法：{weak}（PSNR {all_means.min():.2f} dB）")
    print(f"PSNR<20 dB 的方法：{very_weak}")
    print("=" * 72)

    rows = []
    sets = [("全部方法", df),
            ("剔除最弱一个", df[df.method != weak]),
            ("剔除全部弱基线(PSNR<20)", df[~df.method.isin(very_weak)])]
    for tag, d in sets:
        g = d.groupby("method")[METRICS + ["dLAI_mean"]].mean()
        hi = g[g.PSNR >= HIGH_THR]
        for metric in METRICS:
            rows.append({
                "set": tag, "metric": metric,
                "n_methods": len(g),
                "rho_method": round(float(spearmanr(g[metric], g.dLAI_mean).correlation), 4),
                "rho_sample": round(float(spearmanr(d[metric], d.dLAI_mean).correlation), 4),
                "rho_highq": round(float(spearmanr(hi[metric], hi.dLAI_mean).correlation), 4)
                if len(hi) >= 4 else np.nan,
                "n_highq": len(hi),
            })
    res = pd.DataFrame(rows)
    res.to_csv(tab / "robust_drop_weak.csv", index=False)
    print("\n【剔除弱基线前后对比】")
    piv = res.pivot_table(index="metric", columns="set",
                          values=["rho_method", "rho_sample", "rho_highq"])
    print(piv.round(3).to_string())

    # 高质量子集里各指标与最优指标之间的配对 bootstrap
    g = df.groupby("method")[METRICS + ["dLAI_mean"]].mean()
    hi = g[g.PSNR >= HIGH_THR]
    print(f"\n高质量子集 n={len(hi)}，配对 bootstrap 比较（2000 次）")
    dists = {m: boot_method_rho(hi[m], hi.dLAI_mean) for m in METRICS}
    boots = []
    base = "SAM"
    for m in METRICS:
        if m == base:
            continue
        a, b = dists[base], dists[m]
        k = min(len(a), len(b))
        diff = np.abs(a[:k]) - np.abs(b[:k])          # |rho_SAM| - |rho_other|
        lo, hi95 = np.percentile(diff, [2.5, 97.5])
        boots.append({"pair": f"|SAM| - |{m}|", "delta_median": round(float(np.median(diff)), 4),
                      "ci_low": round(float(lo), 4), "ci_high": round(float(hi95), 4),
                      "p_gt0": round(float((diff > 0).mean()), 3)})
    bt = pd.DataFrame(boots)
    bt.to_csv(tab / "pairwise_metric_diff.csv", index=False)
    print(bt.to_string(index=False))
    print("\n判读：p_gt0 > 0.95 才可以说'SAM 明显优于该指标'；")
    print("      若置信区间跨 0，只能说'方向一致但证据不足'，不能写成确定结论。")
    print(f"\n-> {tab/'robust_drop_weak.csv'}")


if __name__ == "__main__":
    main()
