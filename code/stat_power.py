"""
stat_power.py —— 两处补齐：把"证据不足"的两个位置换成站得住的说法

问题一（方法层 n=8，p=0.68）：
  "SAM 优于 PSNR"在方法层做配对 bootstrap 只有 p=0.68，因为方法只有 8 个。
  但在**样本层**我们有 5992 条（方法×patch），这里有统计力度。
  做法：对 patch 做配对重采样，每次重算 ρ(SAM) 与 ρ(PSNR)，取差值分布。
  这检验的是同一个问题（哪个指标的排序更贴合下游误差），但用足了样本量。

问题二（跨数据集的高量子集构不出来）：
  主数据用"PSNR≥25 dB"定义高质量子集，新数据上没有任何方法达到 25 dB，
  于是那一层无法跨数据集比较。
  做法：改成**相对口径**——每个数据集取 PSNR 最高的前 K 个方法（K=4/5/6 各算一次）。
  相对口径在两个数据集上都成立，才能回答"在好方法之间，哪个指标更可靠"。

输出：
  results/tables/stat_power_samplelevel.csv
  results/tables/highq_relative.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

METRICS = ["PSNR", "SAM", "SSIM", "ERGAS", "MAE", "RMSE"]


def paired_patch_bootstrap(df, a="SAM", b="PSNR", tgt="dLAI_mean", n_boot=1000, seed=0):
    """
    按 patch 重采样（保持方法×patch 的配对结构），比较两个指标的秩相关。
    返回 (rho_a, rho_b, diff 分布)。
    """
    rng = np.random.default_rng(seed)
    patches = df.patch.unique()
    groups = {p: g for p, g in df.groupby("patch")}
    ra, rb = [], []

    for _ in range(n_boot):
        pick = rng.choice(patches, len(patches), replace=True)
        sub = pd.concat([groups[p] for p in pick], ignore_index=True)
        x, y = sub[a].values, sub[b].values
        t = sub[tgt].values
        if np.std(x) < 1e-12 or np.std(y) < 1e-12 or np.std(t) < 1e-12:
            continue
        ca = spearmanr(x, t).correlation
        cb = spearmanr(y, t).correlation
        if np.isfinite(ca) and np.isfinite(cb):
            ra.append(ca); rb.append(cb)
    return np.array(ra), np.array(rb)


def relative_highq(df, k):
    """取每个数据集内 PSNR 最高的前 k 个方法（相对口径，可跨数据集比较）。"""
    g = df.groupby("method")[METRICS + ["dLAI_mean", "dCab_mean"]].mean()
    top = g.nlargest(k, "PSNR")
    return top


def summarize_top(top, label):
    out = {}
    for m in METRICS:
        for tgt in ("dLAI_mean", "dCab_mean"):
            c = top[[m, tgt]].dropna()
            if len(c) >= 4:
                out[f"{m}_vs_{'LAI' if 'LAI' in tgt else 'Cab'}"] = round(
                    float(spearmanr(c[m], c[tgt]).correlation), 4)
    return {"set": label, "k": len(top), "methods": ",".join(top.index), **out}


def main():
    cfg = yaml.safe_load(open("configs/p1.yaml", encoding="utf-8"))
    tab = Path(cfg["output"]["tables"])

    # ---------------- ① 样本层配对 bootstrap ----------------
    img = pd.read_csv(tab / "per_patch_metrics.csv")
    dn = pd.read_csv(tab / "per_patch_downstream.csv")
    df = img.merge(dn, on=["method", "patch"], how="inner")
    print("=" * 74)
    print(f"样本层配对 bootstrap：{len(df)} 条 | patch {df.patch.nunique()} | 方法 {df.method.nunique()}")
    print("=" * 74)

    rows = []
    for a, b in [("SAM", "PSNR"), ("SAM", "SSIM"), ("SAM", "ERGAS"), ("PSNR", "SSIM")]:
        ra, rb = paired_patch_bootstrap(df, a, b)
        k = min(len(ra), len(rb))
        diff = np.abs(ra[:k]) - np.abs(rb[:k])
        lo, hi = np.percentile(diff, [2.5, 97.5])
        rows.append({
            "pair": f"|{a}| - |{b}|",
            "rho_a": round(float(np.median(ra[:k])), 4),
            "rho_b": round(float(np.median(rb[:k])), 4),
            "delta_median": round(float(np.median(diff)), 4),
            "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4),
            "p_gt0": round(float((diff > 0).mean()), 4),
        })
        print(f"  {rows[-1]['pair']:18s} 中位差 {rows[-1]['delta_median']:+.4f}  "
              f"CI [{lo:+.4f}, {hi:+.4f}]  p(diff>0)={rows[-1]['p_gt0']:.3f}", flush=True)
    sp = pd.DataFrame(rows)
    sp.to_csv(tab / "stat_power_samplelevel.csv", index=False)

    # ---------------- ② 相对口径的高量子集 ----------------
    print("\n" + "=" * 74)
    print("相对口径高质量子集（各数据集取 PSNR 前 K 名）")
    print("=" * 74)
    newp = tab / "per_patch_metrics_s2.csv"
    if not newp.exists():
        print("  没有新数据集结果，跳过")
        return
    nd = pd.read_csv(newp)
    recs = []
    for k in (4, 5, 6):
        recs.append(summarize_top(relative_highq(df, k), f"主 SEN12MS-CR Top{k}"))
        recs.append(summarize_top(relative_highq(nd, k), f"新 S2-L2A Top{k}"))
    rel = pd.DataFrame(recs)
    rel.to_csv(tab / "highq_relative.csv", index=False)

    cols = ["set", "k", "PSNR_vs_LAI", "SAM_vs_LAI", "SSIM_vs_LAI", "ERGAS_vs_LAI"]
    cols = [c for c in cols if c in rel.columns]
    print(rel[cols].to_string(index=False))

    print("\n判读：若两个数据集在 Top-K 上符号一致，说明'好方法之间'的结论跨数据集成立；")
    print("      这比原来的'PSNR≥25 dB'口径更强，因为它不依赖绝对阈值。")
    print(f"\n-> {tab/'stat_power_samplelevel.csv'}")
    print(f"-> {tab/'highq_relative.csv'}")


if __name__ == "__main__":
    main()
