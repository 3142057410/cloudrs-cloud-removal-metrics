"""
eval_s2_crossregion.py —— 在新数据集（Sentinel-2 L2A，自建）上复算"指标 vs 下游误差"

目的：检验主结论（指标判别力依赖任务与候选池）是否只在 SEN12MS-CR 上成立。

两个数据集的差别必须说清，否则会被质疑"不可比"：
  · 主实验：SEN12MS-CR，L1C **TOA** 反射率，10m，13 波段齐全
  · 本实验：S2 L2A **地表**反射率，20m 为主（B2/B3/B4/B8 为 10m），
    只下 10 个可用波段（B1/B9/B10 填空），云掩膜来自官方 SCL 分类
因此这里**只比"相关性方向与相对大小"，不比绝对数值**。

方法集合只保留不依赖 SAR 的 9 个（新数据没下 S1，sar 通道为全零占位，
依赖 SAR 的方法会静默退化，必须排除，否则结果是假的）。

用法：
    python src/eval_s2_crossregion.py --processed data/s2_processed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import METHODS
from dataset import USABLE_IDX
from downstream import DownstreamRetrieval, retrieval_delta
from metrics import evaluate_all

# 不带 SAR 的方法（新数据集无 S1，必须排除 sar_* 与 *_sar）
OPTICAL_ONLY = ["zero_fill", "mean_fill", "nearest_fill", "blur_fill",
                "inpaint_telea", "knn_spatial", "hist_match",
                "unet_opt", "dsen2cr_opt"]
METRICS = ["PSNR", "SSIM", "SAM", "ERGAS", "MAE", "RMSE"]
HIGH_THR = 25.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/s2_processed")
    ap.add_argument("--lut", default="data/processed/prosail_lut.npz")
    ap.add_argument("--out", default="results/tables")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    root = Path(args.processed)
    files = sorted(root.glob("*.npz"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"{root} 里没有 npz，先跑 build_s2_dataset.py")
    ds = DownstreamRetrieval(args.lut)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print(f"跨区域评估：{len(files)} 个 patch × {len(OPTICAL_ONLY)} 个方法（不含 SAR）")
    print("=" * 74)

    rows = []
    for mi, m in enumerate(OPTICAL_ONLY, 1):
        fn = METHODS[m]
        for f in files:
            d = np.load(f)
            opt, tgt, mask = d["optical"], d["target"], d["mask"]
            if mask.sum() < 20:
                continue
            pred = fn(opt, d["sar"], mask)
            # 与主流程保持一致：只在实际可用的 10 个波段上算指标
            # （新数据的 B1/B9/B10 是填空零值，不排除会稀释 SSIM/SAM）
            met = evaluate_all(pred[USABLE_IDX], tgt[USABLE_IDX], mask=mask)
            dd = retrieval_delta(ds, pred, tgt, mask)
            rows.append({"method": m, "patch": f.stem, **met,
                         "dLAI_mean": dd["dLAI_mean"],
                         "dCab_mean": dd.get("dCab_mean", np.nan)})
        print(f"  [{mi}/{len(OPTICAL_ONLY)}] {m} 完成", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_patch_metrics_s2.csv", index=False)
    print(f"\n合计 {len(df)} 行")

    g = df.groupby("method")[METRICS + ["dLAI_mean", "dCab_mean"]].mean()
    print("\n各方法均值：")
    print(g.round(4).to_string())

    # 与主数据集同样的两层口径
    res = []
    for name, gg in [("all", g), ("highq", g[g.PSNR >= HIGH_THR])]:
        if len(gg) < 4:
            continue
        for metric in ("PSNR", "SAM", "SSIM", "ERGAS"):
            for tgt in ("dLAI_mean", "dCab_mean"):
                # 变量名不能再用 d：上面循环里 d 是 np.load 出来的文件对象，
                # 会造成 KeyError: '[PSNR, dLAI_mean] is not a file in the archive'
                c = gg[[metric, tgt]].dropna()
                if len(c) < 4:
                    continue
                res.append({"scope": name, "n_methods": len(c), "metric": metric,
                            "target": tgt,
                            "rho": round(float(spearmanr(c[metric], c[tgt]).correlation), 4)})
    rr = pd.DataFrame(res)
    rr.to_csv(out / "corr_s2_crossregion.csv", index=False)
    print("\n【跨区域相关性】")
    print(rr.pivot_table(index=["scope", "metric"], columns="target", values="rho").round(3).to_string())

    # 与主数据集对照
    main_f = out / "corr_highq_subset.csv"
    if main_f.exists():
        mh = pd.read_csv(main_f).set_index("metric")["spearman_in_highq"]
        # 新数据上没有任何方法达到 PSNR≥25，高量子集为空 → 只能比"全量层"
        print("\n【对照主数据集】全量方法层（高量子集在新数据上为空，见下）")
        for metric in ("PSNR", "SAM", "SSIM", "ERGAS"):
            a = mh.get(metric, np.nan)
            b_row = rr[(rr.scope == "all") & (rr.metric == metric) & (rr.target == "dLAI_mean")]
            b = float(b_row.rho.iloc[0]) if len(b_row) else np.nan
            if not (np.isfinite(a) and np.isfinite(b)):
                verdict = "无数据"
            elif np.sign(a) == np.sign(b):
                verdict = "一致" if abs(abs(a) - abs(b)) < 0.1 else "符号一致、量级有差"
            else:
                verdict = "**符号相反**"
            print(f"  {metric:5s} 主(高效) {a:+.3f}  新(全量) {b:+.3f}   {verdict}")
        n_hi = int(rr[rr.scope == "highq"].n_methods.max()) if (rr.scope == "highq").any() else 0
        print(f"\n  新数据里 PSNR≥{HIGH_THR} 的方法数：{n_hi}"
              f"（最高 PSNR {g.PSNR.max():.2f} dB）")
        if n_hi < 4:
            print("  → 高量子集无法在新数据上构建，该层结论本轮未获跨数据集验证，须写进 limitation")
    print(f"\n-> {out/'corr_s2_crossregion.csv'}")


if __name__ == "__main__":
    main()
