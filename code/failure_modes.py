"""
failure_modes.py —— 把 PSNR 用的 MSE 精确拆成三种失败形态

动机：现在只有定性的失败案例图。审稿人会问"指标失灵时长什么样"，需要定量。

方法（不做阈值分类，做精确分解）：
    对每个方法、每个影像块、每个波段，在云掩膜像素上把预测 p 对真值 t 做最小二乘：
        p ≈ μ_p + a·(t − μ_t)
    则  MSE = E[(p−t)²] = (μ_p−μ_t)² + (1−a)²·Var(t) + Var(e)
    三项互相正交（交叉项因 OLS 残差与 t 正交、且均值残差为 0 而消失），总和**恰好**等于 MSE。
    → 偏移项：整体亮度错了多少（光谱整体偏移）
    → 对比度项：把地物差异压平或放大（过度平滑 / 过增强）
    → 残差项：空间与光谱结构没对上（纹理错位、虚假纹理）

产出：
    results/tables/failure_decomposition.csv   方法 × 三项 RMS 与占比
    results/tables/failure_vs_downstream.csv   各项与三个下游误差的秩相关
    results/figures/fig13_failure_modes.png    左：占比堆叠条；右：各项的判别力对比

用法：
    python src/failure_modes.py                 # 全量 15 方法
    python src/failure_modes.py --limit 40      # 冒烟
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import METHODS
from dataset import USABLE_IDX, load_patch

TAB = Path("results/tables")
FIG = Path("results/figures")


def decompose(pred, tgt, mask):
    """在云掩膜像素上做精确三分解，返回 (bias², contrast², resid²) 的波段均值。"""
    m = mask.astype(bool)
    if m.sum() < 50:
        return None
    p = pred[USABLE_IDX][:, m].astype(np.float64)     # (B, n)
    t = tgt[USABLE_IDX][:, m].astype(np.float64)
    mu_p, mu_t = p.mean(1), t.mean(1)
    td = t - mu_t[:, None]
    var_t = (td ** 2).mean(1)
    cov = (p * td).mean(1)                             # 因 mean(td)=0
    a = np.where(var_t > 1e-12, cov / np.maximum(var_t, 1e-12), 0.0)
    e = p - (mu_p[:, None] + a[:, None] * td)
    bias = (mu_p - mu_t) ** 2
    contrast = (1.0 - a) ** 2 * var_t
    resid = (e ** 2).mean(1)
    return float(bias.mean()), float(contrast.mean()), float(resid.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p1.yaml")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--methods", default=None, help="逗号分隔，缺省全部")
    ap.add_argument("--append", action="store_true", help="与已有结果合并（分设备批次跑）")
    ap.add_argument("--plot-only", action="store_true",
                    help="不重算分解，直接读 failure_decomposition.csv 重绘 fig13（改图尺寸/字号时用）")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    cache = Path(cfg["data"]["cache_dir"])
    names = json.load(open(cache / "split.json", encoding="utf-8"))["test"]
    if args.limit:
        names = names[: args.limit]

    part = TAB / "failure_decomposition.csv"
    if args.plot_only:
        if not part.exists():
            raise SystemExit(f"缺少 {part}，无法只重绘")
        df = pd.read_csv(part)
        print(f"仅重绘：读入 {len(df)} 行 / {df.method.nunique()} 个方法")
    else:
        todo = args.methods.split(",") if args.methods else list(METHODS)
        unknown = [m for m in todo if m not in METHODS]
        if unknown:
            raise SystemExit(f"未知方法 {unknown}")

        print(f"失败模式分解：{len(todo)} 个方法 × {len(names)} 个影像块")
        rows = []
        t0 = time.time()
        for mi, mname in enumerate(todo, 1):
            fn = METHODS[mname]
            for n in names:
                d = load_patch(cache, n)
                pred = fn(d["optical"], d["sar"], d["mask"])
                r = decompose(pred, d["target"], d["mask"])
                if r is None:
                    continue
                rows.append({"method": mname, "patch": n,
                             "bias2": r[0], "contrast2": r[1], "resid2": r[2]})
            print(f"  [{mi}/{len(todo)}] {mname} 完成  "
                  f"{(time.time()-t0)/60:.1f} 分钟", flush=True)

        df = pd.DataFrame(rows)

    if args.append and not args.plot_only and part.exists():
        old = pd.read_csv(part)[["method", "patch", "bias2", "contrast2", "resid2"]]
        df = pd.concat([old, df], ignore_index=True).drop_duplicates(
            subset=["method", "patch"], keep="last")
        print(f"  合并后 {df.method.nunique()} 个方法 / {len(df)} 行")

    tot = df[["bias2", "contrast2", "resid2"]].sum(1)
    for c in ["bias2", "contrast2", "resid2"]:
        df[c.replace("2", "_share")] = df[c] / np.maximum(tot, 1e-12)
    df["mse"] = tot
    df["PSNR"] = 10 * np.log10(1.0 / np.maximum(df.mse, 1e-12))
    df.to_csv(TAB / "failure_decomposition.csv", index=False)

    # ---- 方法级汇总，并与下游误差对齐 ----
    g = df.groupby("method")[["bias2", "contrast2", "resid2",
                              "bias_share", "contrast_share", "resid_share",
                              "mse", "PSNR"]].mean()
    for c in ["bias2", "contrast2", "resid2"]:
        g[c.replace("2", "_rms")] = np.sqrt(g[c])
    dn = pd.read_csv(TAB / "per_patch_downstream.csv").groupby("method")[
        ["dLAI_mean", "dCab_mean"]].mean()
    nd = pd.read_csv(TAB / "ndvi_task.csv").groupby("method")[["dclass"]].mean()
    g = g.join(dn).join(nd).reset_index()
    g.sort_values("PSNR").to_csv(TAB / "failure_summary.csv", index=False)

    # ---- 三项分别与下游误差的相关（方法层）----
    out = []
    for comp, label in [("mse", "总 MSE（即 PSNR）"),
                        ("bias_rms", "偏移项"), ("contrast_rms", "对比度项"),
                        ("resid_rms", "残差项")]:
        rec = {"component": label, "n_methods": len(g)}
        for tgt in ["dLAI_mean", "dCab_mean", "dclass"]:
            rec[tgt] = round(float(spearmanr(g[comp], g[tgt]).correlation), 3)
        out.append(rec)
    cmp_df = pd.DataFrame(out)
    cmp_df.to_csv(TAB / "failure_vs_downstream.csv", index=False)

    print("\n三项与下游误差的秩相关（方法层，n=%d）" % len(g))
    print(cmp_df.to_string(index=False))

    print("\n各方法的三项占比（按 PSNR 升序）")
    print(g.sort_values("PSNR")[["method", "PSNR", "bias_share", "contrast_share",
                                 "resid_share", "dLAI_mean"]].round(3).to_string(index=False))

    # ---- 出图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 与其他图件保持同一套中文字体，否则中文标签会变成方框
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    gg = g.sort_values("PSNR")
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.87),
                             gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    y = np.arange(len(gg))
    left = np.zeros(len(gg))
    for c, lab, col in [("bias_share", "Global bias", "#EF9F27"),
                        ("contrast_share", "Contrast compression", "#378ADD"),
                        ("resid_share", "Structural residual", "#1D9E75")]:
        v = gg[c].to_numpy() * 100
        ax.barh(y, v, left=left, color=col, label=lab, height=0.72)
        left += v
    ax.set_yticks(y)
    ax.set_yticklabels([f"{m}  ({p:.1f} dB)" for m, p in zip(gg.method, gg.PSNR)], fontsize=7.5)
    ax.set_xlabel("Share of global MSE (%)", fontsize=8, labelpad=2)
    ax.set_title("(a) Error composition (sorted by PSNR)", fontsize=9)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.30),
              ncol=3, frameon=False, handlelength=1.2, columnspacing=1.0)
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.25, lw=0.5)

    ax = axes[1]
    labels = cmp_df.component.tolist()
    vals = cmp_df.dLAI_mean.abs().to_numpy()
    cols = ["#888780", "#EF9F27", "#378ADD", "#1D9E75"]
    b = ax.bar(range(len(labels)), vals, color=cols, width=0.62)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(["Total MSE\n(=PSNR)", "Bias", "Contrast", "Residual"],
                          fontsize=7.5)
    ax.set_ylabel("Spearman $|\\rho|$ vs ΔLAI", fontsize=8)
    ax.set_title("(b) Which component predicts retrieval error", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", fontsize=7.5)
    plt.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG / "fig13_failure_modes.png", dpi=650, bbox_inches="tight")
    print(f"\n图 -> {FIG/'fig13_failure_modes.png'}")
    print(f"表 -> {TAB/'failure_decomposition.csv'} / failure_summary.csv / failure_vs_downstream.csv")


if __name__ == "__main__":
    main()
