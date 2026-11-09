"""
band_analysis.py —— 机制实验：为什么 PSNR 与 ΔLAI 不一致

两个假设需要被量化：
  H1（PSNR 被谁主导）：全局 PSNR 是 10 个波段 MSE 的融合，但高反射率波段
      （B8/B8A/B11/B12）的绝对误差天然更大，会主导全局 PSNR。
  H2（LAI 对谁敏感）：LAI 反演主要靠红边与红光（B4–B7），对短波红外的敏感度低。
若 H1 的主导波段 ≠ H2 的敏感波段，则"PSNR 高但 ΔLAI 大"就是必然，而不是偶然。

三组量都归一到同一根波段轴上：
  1) mse_share_b  —— 波段 b 对全局 MSE 的贡献占比（谁在决定 PSNR）
  2) lai_sens_b   —— 该波段反射率 +0.01 时 LAI 反演值的变化（谁在决定 ΔLAI）
  3) rho_b        —— 逐波段 PSNR 与 ΔLAI 的秩相关（谁真正"预测"了下游误差）

输出：
  results/tables/per_band_psnr.csv
  results/tables/band_mechanism.csv
  results/figures/fig10_band_mechanism.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import METHODS
from dataset import S2_BAND_NAMES, USABLE_IDX, load_patch
from downstream import DownstreamRetrieval, patch_spectra

BANDS = [S2_BAND_NAMES[i] for i in USABLE_IDX]


def band_mse_share(pred, tgt, mask):
    """每个波段的 MSE 及其在总 MSE 中的占比。"""
    m = mask.astype(bool)
    p, t = pred[USABLE_IDX][:, m], tgt[USABLE_IDX][:, m]
    mse_b = ((p - t) ** 2).mean(axis=1)
    return mse_b, mse_b / (mse_b.sum() + 1e-12)


def lai_sensitivity(ds, n=3000, delta=0.01, seed=0):
    """
    用查找表估计"每个波段对 LAI 反演的敏感度"：
    取 n 条 LUT 光谱，把波段 b 抬 0.01，看重近邻反演出的 LAI 变化多少。
    这不是解析 Jacobian（逆问题不可微），而是局部的有限差分，物理含义一致。
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds.refl), min(n, len(ds.refl)), replace=False)
    base_spec = ds.refl[idx]                       # (n, 10)，已排除大气波段
    base, _ = ds.invert(base_spec)
    sens = np.zeros(len(ds.keep), np.float64)
    for b in range(len(ds.keep)):
        pert = base_spec.copy()
        pert[:, b] += delta
        p, _ = ds.invert(pert)
        sens[b] = np.mean(np.abs(p - base))
    return sens / delta                            # 单位：LAI / 单位反射率


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=None,
                    help="逗号分隔，只算这些方法（缺省为 config 里的全部）")
    ap.add_argument("--append", action="store_true",
                    help="与已有 per_band_psnr.csv 合并（分批跑，避免一次加载太多模型把显存吃满）")
    ap.add_argument("--plot-only", action="store_true",
                    help="不重算，直接读 per_band_psnr.csv 重算机制表并重绘（改图尺寸时用）")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 个 patch（冒烟）")
    args = ap.parse_args()

    cfg = yaml.safe_load(open("configs/p1.yaml", encoding="utf-8"))
    cache = Path(cfg["data"]["cache_dir"])
    tab = Path(cfg["output"]["tables"])
    figdir = Path(cfg["output"]["figures"])
    tab.mkdir(parents=True, exist_ok=True)

    names = json.load(open(cache / "split.json", encoding="utf-8"))["test"]
    if args.limit:
        names = names[: args.limit]
    allm = [m for m in cfg["methods"] if m in METHODS]
    methods = allm if not args.methods else [x.strip() for x in args.methods.split(",") if x.strip()]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise SystemExit(f"未知方法 {unknown}")
    ds = DownstreamRetrieval(cfg["downstream"]["lut_path"])

    part = tab / "per_band_psnr.csv"
    if args.plot_only:
        if not part.exists():
            raise SystemExit(f"缺少 {part}")
        pb = pd.read_csv(part)
        print(f"仅重算机制表：读入 {len(pb)} 行 / {pb.method.nunique()} 个方法")
    else:
        print("=" * 72)
        print(f"分波段机制分析 | {len(methods)} 个方法 × {len(names)} 个 patch × {len(BANDS)} 个波段")
        print("=" * 72)

        rows = []
        for mi, mname in enumerate(methods, 1):
            fn = METHODS[mname]
            for n in names:
                d = load_patch(cache, n)
                m = d["mask"].astype(bool)
                if m.sum() < 20:
                    continue
                pred = fn(d["optical"], d["sar"], d["mask"])
                mse_b, share = band_mse_share(pred, d["target"], d["mask"])
                psnr_b = 10 * np.log10(1.0 / np.maximum(mse_b, 1e-8))
                rec = {"method": mname, "patch": n, "n_px": int(m.sum())}
                for i, b in enumerate(BANDS):
                    rec[f"PSNR_{b}"] = float(psnr_b[i])
                    rec[f"share_{b}"] = float(share[i])
                rows.append(rec)
            print(f"  [{mi}/{len(methods)}] {mname} 完成", flush=True)
            # 分族跑时每算完一个方法就释放，否则连续加载十几个网络会静默退出
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

        pb = pd.DataFrame(rows)
        if args.append and part.exists():
            old = pd.read_csv(part)
            before = (len(old), old.method.nunique())
            pb = pd.concat([old, pb], ignore_index=True).drop_duplicates(
                subset=["method", "patch"], keep="last")
            print(f"  合并：{before[0]} 行/{before[1]} 方法 -> "
                  f"{len(pb)} 行/{pb.method.nunique()} 方法")
        pb.to_csv(part, index=False)

    dn = pd.read_csv(tab / "per_patch_downstream.csv")
    df = pb.merge(dn[["method", "patch", "dLAI_mean"]], on=["method", "patch"], how="inner")
    print(f"\n合并下游 {len(df)} 行 / {df.method.nunique()} 个方法")

    # --- 1) 各波段的 MSE 占比（方法平均）---
    share_cols = [f"share_{b}" for b in BANDS]
    share_mean = df[share_cols].mean()
    share_mean.index = BANDS

    # --- 2) LAI 敏感度 ---
    sens = lai_sensitivity(ds)
    print("\n【波段敏感度】LAI 反演对各波段反射率 +0.01 的响应（LAI / 单位反射率）")
    for b, s in zip(BANDS, sens):
        print(f"  {b:4s}  {s:8.3f}")

    # --- 3) 逐波段 PSNR 与 ΔLAI 的相关 ---
    rho_s, rho_m = [], []
    g = df.groupby("method")[[f"PSNR_{b}" for b in BANDS] + ["dLAI_mean"]].mean()
    for b in BANDS:
        rho_s.append(spearmanr(df[f"PSNR_{b}"], df["dLAI_mean"]).correlation)
        rho_m.append(spearmanr(g[f"PSNR_{b}"], g["dLAI_mean"]).correlation)

    mech = pd.DataFrame({
        "band": BANDS,
        "mse_share": share_mean.values,
        "lai_sensitivity": sens,
        "rho_sample": rho_s,
        "rho_method": rho_m,
    })
    mech["share_x_sens"] = mech.mse_share * mech.lai_sensitivity
    mech.to_csv(tab / "band_mechanism.csv", index=False)

    print("\n【机制表】按 MSE 占比降序")
    print(mech.sort_values("mse_share", ascending=False).round(4).to_string(index=False))

    # 关键判据：MSE 占比最大的三个波段，与敏感度最大的三个波段是否重合
    top_share = mech.nlargest(3, "mse_share").band.tolist()
    top_sens = mech.nlargest(3, "lai_sensitivity").band.tolist()
    inter = set(top_share) & set(top_sens)
    print(f"\n主导 PSNR 的波段（MSE 占比 Top3）：{top_share}")
    print(f"主导 ΔLAI 的波段（敏感度 Top3）  ：{top_sens}")
    print(f"重合 {len(inter)} 个：{sorted(inter) if inter else '无'}")
    r = np.corrcoef(mech.mse_share, mech.lai_sensitivity)[0, 1]
    print(f"MSE 占比 与 敏感度 的相关性 r = {r:+.2f}")
    print("判读：r 越低 / 重合越少，说明'PSNR 在意的地方'与'反演在意的地方'越错位，")
    print("      这就是 PSNR 与 ΔLAI 不一致的物理原因，可直接写进讨论节。")

    # --- 图 ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.01))
    x = np.arange(len(BANDS))
    axes[0].bar(x, mech.mse_share.values, color="#2f6fb3")
    axes[0].set_title("(a) Who drives PSNR\n(MSE contribution per band)", fontsize=9.5)
    axes[0].set_ylabel("MSE share")

    axes[1].bar(x, mech.lai_sensitivity.values, color="#c0392b")
    axes[1].set_title("(b) Who drives ΔLAI\n(LAI sensitivity per band)", fontsize=9.5)
    axes[1].set_ylabel("ΔLAI per unit reflectance")

    colors = ["#2c7a4b" if v < 0 else "#c0392b" for v in mech.rho_sample.values]
    axes[2].bar(x, mech.rho_sample.values, color=colors, alpha=.85)
    axes[2].axhline(0, color="k", lw=.8)
    axes[2].set_title("(c) Which band predicts the error\n(band-wise PSNR vs ΔLAI)", fontsize=9.5)
    axes[2].set_ylabel("Spearman ρ")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(BANDS, rotation=45, fontsize=8)
        ax.grid(axis="y", alpha=.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(figdir / "fig10_band_mechanism.png")
    plt.close(fig)
    print(f"\n图 -> {figdir/'fig10_band_mechanism.png'}")


if __name__ == "__main__":
    main()
