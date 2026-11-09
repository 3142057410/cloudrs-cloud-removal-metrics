"""
subset_methods.py —— 方法数从 14 减到 10 会怎样？

背景：用户想把方法数减到 10 个以降低工作量。形式上 JSTARS 要求 ≥6 个方法即可，
但"减哪几个"会改变结论，必须先算清楚。

关键约束（从数据里推出来的）：
  PSNR ≥ 25 dB 的"合格方法"恰好有 8 个，而"合格方法之间"这一层是全文核心分析。
  所以减到 10 个时，这 8 个必须全部保留，只剩 2 个名额给弱方法。

输出：results/tables/subset_comparison.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

TAB = Path("results/tables")
METRICS = ["PSNR", "SSIM", "SAM", "ERGAS", "MAE", "RMSE"]
THR = 25.0


def load():
    img = pd.read_csv(TAB / "per_patch_metrics.csv")
    dn = pd.read_csv(TAB / "per_patch_downstream.csv")
    return img.merge(dn, on=["method", "patch"], how="inner")


def summarize(df, methods, label):
    d = df[df.method.isin(methods)]
    g = d.groupby("method")[METRICS + ["dLAI_mean"]].mean()
    hi = g[g.PSNR >= THR]
    out = {"subset": label, "n_methods": len(g), "n_highq": len(hi)}
    for m in METRICS:
        out[f"method_{m}"] = round(float(spearmanr(g[m], g.dLAI_mean).correlation), 3) if len(g) >= 4 else np.nan
        out[f"sample_{m}"] = round(float(spearmanr(d[m], d.dLAI_mean).correlation), 3)
    # 核心判断：合格方法里 SAM 是否高于 PSNR、SSIM 是否最弱
    if len(hi) >= 4:
        r = {m: abs(float(spearmanr(hi[m], hi.dLAI_mean).correlation)) for m in METRICS}
        out["hiq_SAM"] = round(r["SAM"], 3)
        out["hiq_PSNR"] = round(r["PSNR"], 3)
        out["hiq_SSIM"] = round(r["SSIM"], 3)
        out["SAM_gt_PSNR"] = r["SAM"] > r["PSNR"]
        out["SSIM_worst"] = r["SSIM"] <= min(v for k, v in r.items() if k != "SSIM")
    return out


def main():
    df = load()
    means = df.groupby("method")["PSNR"].mean().sort_values(ascending=False)
    print("=" * 78)
    print("各方法 PSNR 均值（降序）—— PSNR≥25 的即为“合格方法”")
    print("=" * 78)
    for m, v in means.items():
        print(f"  {m:16s} {v:6.2f} dB {'  ← 合格' if v >= THR else ''}")

    hiq = [m for m in means.index if means[m] >= THR]
    weak = [m for m in means.index if means[m] < THR]
    print(f"\n合格方法 {len(hiq)} 个：{hiq}")
    print(f"弱方法   {len(weak)} 个：{weak}")

    n_all, n_hi = len(means.index), len(hiq)
    cands = {
        f"全部 {n_all} 个（当前）": list(means.index),
        f"{n_all - 2} 个（保留全部弱方法，去 nearest/hist）": [
            m for m in means.index if m not in ("nearest_fill", "hist_match")],
        f"{n_hi + 2} 个：合格 {n_hi} + zero_fill + diffusion（保三族）": hiq + ["zero_fill", "diffusion"],
        f"{n_hi + 2} 个：合格 {n_hi} + zero_fill + hist_match（去生成族）": hiq + ["zero_fill", "hist_match"],
        f"{n_hi + 2} 个：合格 {n_hi} + hist_match + diffusion": hiq + ["hist_match", "diffusion"],
        f"{n_hi} 个：仅合格方法（无弱基线）": hiq,
    }

    rows = [summarize(df, v, k) for k, v in cands.items()]
    res = pd.DataFrame(rows)
    res.to_csv(TAB / "subset_comparison.csv", index=False)

    cols = ["subset", "n_methods", "n_highq", "method_PSNR", "method_SAM", "method_SSIM",
            "sample_PSNR", "sample_SAM", "sample_SSIM", "hiq_SAM", "hiq_PSNR", "hiq_SSIM",
            "SAM_gt_PSNR", "SSIM_worst"]
    print("\n" + "=" * 78)
    print("不同方法子集下的结论是否保持")
    print("=" * 78)
    print(res[cols].to_string(index=False))

    print("\n判读：")
    print("  · n_highq 若 < 8，'合格方法之间'这一层就变了，全文核心分析需要重写")
    print("  · SAM_gt_PSNR 为 False 时，'好方法之间 SAM 反超'这一结论不成立")
    print("  · SSIM_worst 应恒为 True —— 这是全文唯一决定性结论，最该保住")
    print(f"\n-> {TAB/'subset_comparison.csv'}")


if __name__ == "__main__":
    main()
