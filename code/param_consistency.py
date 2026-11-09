"""
param_consistency.py —— 换一个下游变量（Cab），结论还成立吗？

动机：审稿人最容易攻击的一点是"ΔLAI 是你自己用 PROSAIL 查找表定义的自洽代理"。
如果同一批去云结果、同一个反演器、只把下游参数从 LAI 换成叶绿素 Cab（光谱敏感区
明显不同：Cab 主控可见光蓝红吸收，LAI 主控近红外/短波红外散射），
而"哪个指标能选出好方法"的结论仍然一致，那么结论就不再依赖单一参数的选择。

输出：
  results/tables/corr_vs_dCab.csv     各指标的 ρ（对 LAI / 对 Cab，全量与高量子集）
  results/tables/rank_agreement.csv    方法排序在两种参数下的秩一致性
  results/figures/fig11_param_consistency.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

METRICS = ["PSNR", "SSIM", "SAM", "ERGAS", "MAE", "RMSE"]
HIGH_THR = 25.0


def main():
    cfg = yaml.safe_load(open("configs/p1.yaml", encoding="utf-8"))
    tab = Path(cfg["output"]["tables"])
    figdir = Path(cfg["output"]["figures"])

    img = pd.read_csv(tab / "per_patch_metrics.csv")
    dn = pd.read_csv(tab / "per_patch_downstream.csv")
    df = img.merge(dn, on=["method", "patch"], how="inner")
    if "dCab_mean" not in df.columns or df.dCab_mean.isna().all():
        raise SystemExit("下游表里没有 dCab_mean，请先用新版 downstream.py 重跑 run_downstream.py")
    df = df.dropna(subset=METRICS + ["dLAI_mean", "dCab_mean"])

    g = df.groupby("method")[METRICS + ["dLAI_mean", "dCab_mean"]].mean()
    hi = g[g["PSNR"] >= HIGH_THR]
    print("=" * 76)
    print(f"样本 {len(df)} 行 | 方法 {len(g)} | 高质量子集 {len(hi)} 个（PSNR≥{HIGH_THR}）")
    print("=" * 76)

    rows = []
    for m in METRICS:
        r_s_lai = spearmanr(df[m], df.dLAI_mean).correlation
        r_s_cab = spearmanr(df[m], df.dCab_mean).correlation
        r_m_lai = spearmanr(g[m], g.dLAI_mean).correlation
        r_m_cab = spearmanr(g[m], g.dCab_mean).correlation
        r_h_lai = spearmanr(hi[m], hi.dLAI_mean).correlation
        r_h_cab = spearmanr(hi[m], hi.dCab_mean).correlation
        rows.append({
            "metric": m,
            "rho_method_LAI": round(r_m_lai, 4), "rho_method_Cab": round(r_m_cab, 4),
            "rho_sample_LAI": round(r_s_lai, 4), "rho_sample_Cab": round(r_s_cab, 4),
            "rho_highq_LAI": round(r_h_lai, 4), "rho_highq_Cab": round(r_h_cab, 4),
        })
    pc = pd.DataFrame(rows)
    pc["sign_agree_highq"] = np.sign(pc.rho_highq_LAI) == np.sign(pc.rho_highq_Cab)
    pc.to_csv(tab / "corr_vs_dCab.csv", index=False)

    print("\n【各指标对两个下游变量的秩相关】")
    print(pc.to_string(index=False))

    # 方法排序在两种参数下是否一致
    r1 = g["dLAI_mean"].rank(ascending=True)
    r2 = g["dCab_mean"].rank(ascending=True)
    tau = spearmanr(g.dLAI_mean, g.dCab_mean).correlation
    agree = pd.DataFrame({"method": g.index, "rank_dLAI": r1.values, "rank_dCab": r2.values})
    agree["shift"] = (agree.rank_dLAI - agree.rank_dCab).abs()
    agree = agree.sort_values("shift", ascending=False)
    agree.to_csv(tab / "rank_agreement.csv", index=False)

    print(f"\n【方法排序一致性】dLAI 与 dCab 的秩相关 ρ = {tau:.3f}")
    print(agree.to_string(index=False))

    # 高质量子集里，两种参数下各指标的排名
    hq_lai = hi[["dLAI_mean"]].join(hi[METRICS]).apply(lambda s: spearmanr(s, hi.dLAI_mean).correlation)
    hq_cab = hi[["dCab_mean"]].join(hi[METRICS]).apply(lambda s: spearmanr(s, hi.dCab_mean).correlation)
    hq_lai, hq_cab = hq_lai[METRICS], hq_cab[METRICS]
    best_lai = hq_lai.abs().idxmax()
    best_cab = hq_cab.abs().idxmax()
    print(f"\n高质量子集里判别力最强的指标：对 LAI 是 {best_lai}（|ρ|={abs(hq_lai[best_lai]):.3f}），"
          f"对 Cab 是 {best_cab}（|ρ|={abs(hq_cab[best_cab]):.3f}）")
    print("判读：")
    if best_lai == best_cab:
        print(f"  ✓ 两种下游变量下都是 {best_lai} 最强 → '用 SAM 选优'的结论不依赖具体参数，可直接写进论文")
    else:
        print(f"  ⚠ 两种参数下最优指标不同（{best_lai} vs {best_cab}）→ 结论要弱化为"
              "'指标选择依赖下游关注的具体参数'，这本身也是有价值的发现")

    # ---------------- 图 ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.45))
    x = np.arange(len(METRICS))
    w = 0.38
    for ax, cols, title in [
            (axes[0], ("rho_highq_LAI", "rho_highq_Cab"),
             f"High-quality subset (n={len(hi)})"),
            (axes[1], ("rho_method_LAI", "rho_method_Cab"),
             f"All {len(g)} methods")]:
        ax.bar(x - w / 2, pc[cols[0]].abs().values, w, label="vs ΔLAI", color="#2f6fb3")
        ax.bar(x + w / 2, pc[cols[1]].abs().values, w, label="vs ΔCab", color="#c0392b")
        ax.set_xticks(x)
        ax.set_xticklabels(METRICS, fontsize=7.5)
        ax.set_ylabel("|Spearman ρ|")
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(figdir / "fig11_param_consistency.png")
    plt.close(fig)
    print(f"\n图 -> {figdir/'fig11_param_consistency.png'}")


if __name__ == "__main__":
    main()
