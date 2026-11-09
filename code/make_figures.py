"""
make_figures.py —— 生成 P1 论文所需的全部图

产出（results/figures/）：
  fig1_method_scatter.png   方法层：各指标 vs ΔLAI 散点（每指标一格，标出方法名）
  fig2_sample_corr.png      样本层：秩相关及 bootstrap 95% CI 条形图
  fig3_stratified.png       分层：云量四分位下的秩相关热图
  fig4_rank_compare.png     排序对比：PSNR 排名 vs ΔLAI 排名（斜率图）
  fig5_qualitative.png      定性示例：有云/掩膜/真值/三种方法结果
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import METHODS
from dataset import load_patch

plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 600
plt.rcParams["savefig.dpi"] = 650  # tight bbox 会裁掉边距，留余量保证版面有效 ≥600 dpi
plt.rcParams["savefig.pad_inches"] = 0.02
plt.rcParams["font.size"] = 8
plt.rcParams["savefig.bbox"] = "tight"

METRICS = ["PSNR", "SSIM", "SAM", "ERGAS", "MAE", "RMSE"]
TARGET = "dLAI_mean"
# 指标方向：越大越好 / 越小越好，画图时统一成"越靠右上越好"
HIGHER_BETTER = {"PSNR": True, "SSIM": True, "SAM": False,
                 "ERGAS": False, "MAE": False, "RMSE": False}
C_GOOD, C_BAD = "#2c7a4b", "#c0392b"

# 图例里类别的显示顺序（默认按字母序会变成 Mid/Low/High，读着别扭）
CAT_ORDER = {"Low NDVI": 0, "Mid NDVI": 1, "High NDVI": 2,
             "Thin cloud": 0, "Medium cloud": 1, "Thick cloud": 2,
             "Cirrus": 0, "Water cloud": 1}

# 分层表里的类别值是中文，出英文图时统一映射（不改 CSV，免得动到论文里的表）
ZH2EN = {
    "低NDVI": "Low NDVI", "中NDVI": "Mid NDVI", "高NDVI": "High NDVI",
    "薄云": "Thin cloud", "中云": "Medium cloud", "厚云": "Thick cloud",
    "卷云型": "Cirrus", "水云型": "Water cloud",
    "Q1低云": "Q1 (low)", "Q2": "Q2", "Q3": "Q3", "Q4高云": "Q4 (high)",
}


def _place_labels(ax, pts, labels, fontsize=6.0, base=5.5, marker_s=42):
    """把点标签排开：按方位依次尝试偏移，取第一个不与已放标签重叠的位置。

    图缩到印刷宽度（7.16 in）后，原来固定的 (4,3) 偏移会让密集区的标签叠在一起。
    这里用贪心搜索，不引入第三方依赖。

    marker_s: scatter 的 s（点面积，单位 pt²）。必须传，否则占位半径算小了，
              标签仍会压在点上。
    """
    import numpy as _np
    from matplotlib.transforms import Bbox
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    ax_bb = ax.get_window_extent()
    placed = []

    P = _np.asarray(pts, float)
    # 点的显示半径：s 是面积(pt²) -> r(pt)=sqrt(s/pi) -> 再换成像素
    r_px = _np.sqrt(marker_s / _np.pi) * fig.dpi / 72.0 + 2.0
    for (px_, py_) in P:
        px, py = ax.transData.transform((px_, py_))
        placed.append(Bbox.from_bounds(px - r_px, py - r_px, 2 * r_px, 2 * r_px))
    d = _np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    _np.fill_diagonal(d, _np.inf)
    cx, cy = P.mean(0)          # 团簇中心，标签优先朝"背离中心"的方向摆
    angs = _np.arange(0, 360, 20)
    # 先放孤立的点（到最近邻远的），密集区后放，冲突更少
    for k in _np.argsort(-d.min(1)):
        x, y = P[k]
        th0 = _np.degrees(_np.arctan2(y - cy, x - cx))
        outward = angs[_np.argsort(_np.abs((angs - th0 + 180) % 360 - 180))]
        done = False
        for radius in (1.0, 1.8, 2.6, 3.6, 4.8, 6.2, 8.0):
            for ang in outward:
                r = base * radius
                dx, dy = r * _np.cos(_np.deg2rad(ang)), r * _np.sin(_np.deg2rad(ang))
                t = ax.annotate(labels[k], (x, y), fontsize=fontsize,
                                xytext=(dx, dy), textcoords="offset points",
                                ha="left" if dx >= 0 else "right",
                                va="bottom" if dy >= 0 else "top",
                                alpha=.92, annotation_clip=False)
                bb = t.get_window_extent(renderer=rend).expanded(1.04, 1.12)
                inside = (bb.x0 >= ax_bb.x0 - 14 and bb.x1 <= ax_bb.x1 + 14 and
                          bb.y0 >= ax_bb.y0 - 14 and bb.y1 <= ax_bb.y1 + 14)
                if inside and not any(bb.overlaps(p) for p in placed):
                    placed.append(bb)
                    done = True
                    break
                t.remove()
            if done:
                break
        if not done:
            t = ax.annotate(labels[k], (x, y), fontsize=fontsize, xytext=(4, 3),
                            textcoords="offset points", alpha=.92)
            placed.append(t.get_window_extent(renderer=rend))


def fig1_method_scatter(g, out):
    # 15 个方法全标注在 2.2 in 宽的面板里必然压字：只标正文讨论的 4 个代表方法。
    # 每个标注用固定的偏移方向（按它在散点里的位置人工定过），保证不压点、不互相压。
    KEYS = {
        "zero_fill":     ("Zero fill",  0,  7, "center", "bottom"),
        "diffusion":     ("Diffusion",  7, -2, "left",   "center"),
        "hist_match":    ("Hist. match", 7,  2, "left",  "center"),
        "uncrtaints_ft": ("UnCRtainTS", 0, -8, "center", "top"),
    }
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.13))
    for ax, m in zip(axes.ravel(), METRICS):
        ax.scatter(g[m], g[TARGET], s=42, c="#2f6fb3", edgecolor="k", linewidth=.35, zorder=3)
        for k, (lab, dx, dy, ha, va) in KEYS.items():
            if k not in g.index:
                continue
            ax.annotate(lab, (g.loc[k, m], g.loc[k, TARGET]), fontsize=6.5,
                        xytext=(dx, dy), textcoords="offset points",
                        ha=ha, va=va, alpha=.95, annotation_clip=False)
        rho = pd.Series(g[m]).corr(pd.Series(g[TARGET]), method="spearman")
        ax.set_title(f"{m}  (ρ={rho:+.2f})", fontsize=8)
        ax.set_xlabel(m)
        ax.set_ylabel("ΔLAI")
        ax.margins(.28)
        ax.grid(alpha=.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(out / "fig1_method_scatter.png")
    plt.close(fig)


def fig2_sample_corr(sl, out):
    d = sl.sort_values("spearman_vs_dLAI")
    fig, ax = plt.subplots(figsize=(7.16, 3.44))
    y = np.arange(len(d))
    colors = [C_GOOD if v < 0 else C_BAD for v in d.spearman_vs_dLAI]  # 与ΔLAI负相关=好
    ax.barh(y, d.spearman_vs_dLAI, color=colors, alpha=.85)
    for i, (_, r) in enumerate(d.iterrows()):
        ax.plot([r.ci_low, r.ci_high], [i, i], color="k", lw=1.6)
        ax.text(r.spearman_vs_dLAI, i, f" {r.spearman_vs_dLAI:+.3f}",
                va="center", ha="left" if r.spearman_vs_dLAI >= 0 else "right", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(d.metric)
    ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel("Spearman ρ (metric vs ΔLAI, bootstrap 95% CI)")
    ax.grid(axis="x", alpha=.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(out / "fig2_sample_corr.png")
    plt.close(fig)


def fig3_stratified(st, out):
    piv = st.pivot(index="metric", columns="cloud_bin", values="spearman")
    fig, ax = plt.subplots(figsize=(3.5, 2.19))
    v = np.nanmax(np.abs(piv.values))
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([ZH2EN.get(str(c), str(c)) for c in piv.columns], fontsize=7.5)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.values[i, j]
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=8,
                    color="w" if abs(val) > v * .55 else "k")
    fig.colorbar(im, ax=ax, label="Spearman ρ")
    fig.tight_layout()
    fig.savefig(out / "fig3_stratified.png")
    plt.close(fig)


def fig4_rank_compare(g, out):
    """斜率图：PSNR 排名 vs ΔLAI 排名。交叉的线就是"排序不一致"的证据。"""
    psnr_rank = g["PSNR"].rank(ascending=False)
    dlai_rank = g[TARGET].rank(ascending=True)   # ΔLAI 越小越好
    fig, ax = plt.subplots(figsize=(3.5, 2.94))
    for name in g.index:
        r1, r2 = psnr_rank[name], dlai_rank[name]
        color = C_GOOD if abs(r1 - r2) < 1.5 else C_BAD
        ax.plot([0, 1], [r1, r2], "-o", color=color, lw=1.8, ms=5, alpha=.9)
        ax.text(-0.04, r1, f"{name} ", ha="right", va="center", fontsize=8)
        ax.text(1.04, r2, f" {name}", ha="left", va="center", fontsize=8)
    ax.set_xlim(-0.55, 1.55)
    ax.invert_yaxis()
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Ranked by PSNR", "Ranked by ΔLAI"])
    ax.set_ylabel("Rank (1 = best)")
    ax.grid(axis="y", alpha=.2, linestyle=":")
    fig.tight_layout()
    fig.savefig(out / "fig4_rank_compare.png")
    plt.close(fig)


def fig5_qualitative(cfg, out):
    cache = Path(cfg["data"]["cache_dir"])
    split = json.load(open(cache / "split.json", encoding="utf-8"))
    # 选一个云量中等的 patch，展示更有代表性
    meta = pd.DataFrame(json.load(open(cache / "meta.json", encoding="utf-8")))
    cand = meta[meta.name.isin(split["test"])].copy()
    # 优先用春季场景：冬季反射率只有其一半，五个面板共用一条拉伸曲线时会整体发暗，
    # 印刷后看不清方法之间的差别。
    spring = cand[cand.name.str.startswith("spring")]
    if not spring.empty:
        cand = spring
    cand = cand[(cand.cloud_fraction > .15) & (cand.cloud_fraction < .45)]
    if cand.empty:
        cand = meta[meta.name.isin(split["test"])]
    name = cand.sort_values("cloud_fraction").iloc[len(cand) // 2]["name"]
    d = load_patch(cache, name)

    show = ["nearest_fill", "hist_match", "sar_guided"]
    # 冬季场景反射率本就低，按 [0,1] 直接显示几乎全黑。改用**云图自身的 2–98 分位**拉伸，
    # 四个面板共用同一组上下限，方法之间的差异仍然可比，也不会因为各自归一化而失真。
    _o = np.transpose(d["optical"][[3, 2, 1]], (1, 2, 0))
    # 冬季场景反射率很低，按 [0,1] 直接显示几乎全黑。用**晴空像素**的分位数拉伸
    # （云很亮，若把云算进去，地面会被压到全黑）；上下限取标量、三通道共用，
    # 这样不会因逐通道归一化而把白云染成粉色。最后加 gamma 提亮中间调。
    _clear = ~d["mask"].astype(bool)
    # 逐通道分位数（不是标量）：冬季场景蓝光散射强，标量拉伸会把地面整体压成蓝调；
    # 逐通道拉伸能还原地物色相，代价是极亮的云会略偏色。
    _lo = np.percentile(_o[_clear], 2, axis=0)
    _hi = np.percentile(_o[_clear], 98, axis=0)

    def rgb(a):
        x = np.transpose(a[[3, 2, 1]], (1, 2, 0))          # B4,B3,B2
        y = np.clip((x - _lo) / np.maximum(_hi - _lo, 1e-6), 0, 1)
        return y ** 0.8

    # 云图 / 真值 / 三种方法：题注里承诺了 ground truth，之前漏掉，评阅时会被抓
    panels = [("Cloudy L1C", d["optical"]), ("Ground truth", d["target"])]
    panels += [(m, METHODS[m](d["optical"], d["sar"], d["mask"])) for m in show]

    fig, axes = plt.subplots(1, len(panels), figsize=(7.16, 1.72))
    for ax, (t, a) in zip(axes, panels):
        ax.imshow(rgb(a))
        ax.set_title(t, fontsize=7.5)
        ax.axis("off")
    # 用细轮廓标出云区，比半透明色块覆盖更不伤图像
    axes[0].contour(d["mask"].astype(float), levels=[0.5],
                    colors="#e8443a", linewidths=0.45)
    fig.tight_layout()
    fig.savefig(out / "fig5_qualitative.png")
    plt.close(fig)

def fig6_subset(ml, hq, out):
    """
    论文的核心图：把"明显很差的方法"剔除后，各指标的判别力怎么变。
    全量里 zero_fill 这种一眼就能看出差的样本会把所有指标的相关性一起抬高；
    只保留 PSNR≥25 dB 的方法后，PSNR 掉到不显著，SAM 基本不掉。
    """
    full = ml.set_index("metric")["spearman_vs_dLAI"].abs()
    sub = hq.set_index("metric")["spearman_in_highq"].abs()
    pv = hq.set_index("metric")["p_value"]
    order = full.sort_values(ascending=False).index
    n_hi = int(hq["n_methods"].iloc[0])      # 方法数，不是指标数（len(sub) 是 6 个指标）
    n_all = int(ml["n_methods"].iloc[0])

    fig, ax = plt.subplots(figsize=(7.16, 3.72))
    x = np.arange(len(order))
    w = 0.36
    ax.bar(x - w / 2, [full[m] for m in order], w, label=f"All {n_all} methods", color="#9bb7d4")
    ax.bar(x + w / 2, [sub[m] for m in order], w,
           label=f"High-quality only (PSNR≥25 dB, n={n_hi})", color="#2f6fb3")
    for i, m in enumerate(order):
        star = "p<.05" if pv.get(m, 1) < .05 else "n.s."
        ax.text(x[i] + w / 2, sub[m] + .03, star, ha="center", fontsize=8,
                color=C_GOOD if star == "p<.05" else C_BAD)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("|Spearman ρ| (larger is better)")
    ax.set_ylim(0, 1.05)
    ax.axhline(.5, color="gray", ls=":", lw=1)
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=2, frameon=False, handlelength=1.3, columnspacing=1.2)
    ax.grid(axis="y", alpha=.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(out / "fig6_highq_subset.png")
    plt.close(fig)


def fig7_season(ss, out):
    """季节分层：符号一致说明结论不依赖季节；幅度差异说明哪一季更难评。"""
    piv = ss.pivot(index="metric", columns="season", values="spearman")
    fig, ax = plt.subplots(figsize=(7.16, 3.78))
    x = np.arange(len(piv))
    w = 0.38
    seasons = list(piv.columns)
    colors = {"spring": "#4f9a6a", "winter": "#2f6fb3"}
    for i, s in enumerate(seasons):
        off = (i - (len(seasons) - 1) / 2) * w
        ax.bar(x + off, piv[s].values, w, label=f"{s} (n≈{int(ss[ss.season==s].n.iloc[0])})",
               color=colors.get(s, "#888"))
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index)
    ax.set_ylabel("Spearman ρ (vs ΔLAI)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(out / "fig7_season.png")
    plt.close(fig)


def fig8_groups(lc, ct, out, cty=None):
    """地表类型、云厚度、云类型三个分层并排展示。"""
    panels = [(lc, "landcover", "(a) Surface type (NDVI tertiles)"),
              (ct, "cloud_thickness", "(b) Cloud thickness (brightness tertiles)")]
    if cty is not None:
        panels.append((cty, "cloud_type", "(c) Cloud type (B10 cirrus band)"))
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(7.16, 2.45), squeeze=False)
    for ax, (df_, col, title) in zip(axes.ravel(), panels):
        piv = df_.pivot(index="metric", columns=col, values="spearman").loc[METRICS]
        piv = piv[sorted(piv.columns, key=lambda c: CAT_ORDER.get(ZH2EN.get(str(c), str(c)), 9))]
        x = np.arange(len(piv))
        w = 0.8 / piv.shape[1]
        for i, c in enumerate(piv.columns):
            ax.bar(x + (i - (piv.shape[1] - 1) / 2) * w, piv[c].values, w,
                   label=ZH2EN.get(str(c), str(c)))
        ax.axhline(0, color="k", lw=.8)
        ax.set_xticks(x)
        ax.set_xticklabels(METRICS, fontsize=6.5, rotation=45, ha="right")
        ax.tick_params(axis="y", labelsize=6.5)
        ax.set_title(title, fontsize=7.5)
        ax.set_ylabel("ρ (vs ΔLAI)", fontsize=7)
        ax.legend(fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.40),
                  frameon=False, handlelength=1.1, columnspacing=0.8, handletextpad=0.35)
        ax.grid(axis="y", alpha=.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(out / "fig8_groups.png")
    plt.close(fig)


def fig12_mask(res, out):
    """掩膜边界鲁棒性：三种掩膜变体下 PSNR 与 SAM 的判别力。"""
    order = ["erode1", "orig", "dilate1"]
    labels = ["Erosion 1 px\n(-13.5% pixels)", "Original mask", "Dilation 1 px\n(+10.9% pixels)"]
    n_all = int(res.loc[res.scope == "all", "n_methods"].iloc[0])
    n_hq = int(res.loc[res.scope == "highq", "n_methods"].iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.67))
    for ax, scope, title in [(axes[0], "all", f"All {n_all} methods"),
                             (axes[1], "highq", f"High-quality subset ({n_hq} methods)")]:
        sub = res[res.scope == scope]
        x = np.arange(len(order))
        w = 0.36
        for i, (m, c) in enumerate([("PSNR", "#2f6fb3"), ("SAM", "#c0392b")]):
            vals = [abs(float(sub[(sub.variant == v) & (sub.metric == m)].rho.iloc[0]))
                    for v in order]
            bars = ax.bar(x + (i - .5) * w, vals, w, label=m, color=c)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + .01, f"{v:.2f}",
                        ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("|Spearman ρ| (vs ΔLAI)", fontsize=7.5)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.26),
                  ncol=2, frameon=False, handlelength=1.2, columnspacing=1.0)
        ax.grid(axis="y", alpha=.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(out / "fig12_mask_robust.png")
    plt.close(fig)


def fig9_failure(cfg, df, out):
    """
    失败案例图（投稿 checklist 要求）。
    取"最好的方法在最难的 patch 上"，说明它为什么失败——
    比再放一张成功案例有价值得多。
    """
    cache = Path(cfg["data"]["cache_dir"])
    g = df.groupby("method")["PSNR"].mean()
    best = g.idxmax()
    sub = df[df.method == best].nsmallest(1, "PSNR").iloc[0]
    d = load_patch(cache, sub["patch"])
    pred = METHODS[best](d["optical"], d["sar"], d["mask"])

    # 与 fig5 同一套显示定标：直接 clip(0,1) 在冬季低反射率场景下几乎全黑。
    # 上下限取自"有云影像的晴空像素"，三张影像面板共用，保证可比。
    _o = np.transpose(d["optical"][[3, 2, 1]], (1, 2, 0))
    _clear = ~d["mask"].astype(bool)
    _lo = np.percentile(_o[_clear], 2, axis=0)
    _hi = np.percentile(_o[_clear], 98, axis=0)

    def rgb(a):
        x = np.transpose(a[[3, 2, 1]], (1, 2, 0))
        y = np.clip((x - _lo) / np.maximum(_hi - _lo, 1e-6), 0, 1)
        return y ** 0.8

    err = np.abs(pred - d["target"]).mean(0)
    vmax = float(np.percentile(err, 99))

    DISP = {"uncrtaints_ft": "UnCRtainTS", "unet_sar": "U-Net+SAR", "dsen2cr_opt": "DSen2-CR"}
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.91))
    axes[0].imshow(rgb(d["optical"]))
    axes[0].set_title("Cloudy input", fontsize=7.5)
    axes[1].imshow(rgb(pred))
    axes[1].set_title(f"{DISP.get(best, best)} output", fontsize=7.5)
    axes[2].imshow(rgb(d["target"]))
    axes[2].set_title("Clear-sky ground truth", fontsize=7.5)
    im = axes[3].imshow(err, cmap="inferno", vmin=0, vmax=vmax)
    axes[3].set_title(f"Mean absolute error\nPSNR {sub['PSNR']:.2f} dB", fontsize=7.5)
    fig.colorbar(im, ax=axes[3], fraction=.046, pad=.03)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "fig9_failure.png")
    plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="",
                    help="只重绘指定图，逗号分隔（如 fig1,fig8）；缺省为全部")
    args = ap.parse_args()
    only = {x.strip() for x in args.only.replace(" ", "").split(",") if x.strip()}

    def want(n):
        return (not only) or n in only

    cfg = yaml.safe_load(open("configs/p1.yaml", encoding="utf-8"))
    tab = Path(cfg["output"]["tables"])
    fig = Path(cfg["output"]["figures"])
    fig.mkdir(parents=True, exist_ok=True)

    img = pd.read_csv(tab / "per_patch_metrics.csv")
    dn = pd.read_csv(tab / "per_patch_downstream.csv")
    df = img.merge(dn, on=["method", "patch"], how="inner")
    g = df.groupby("method")[METRICS + [TARGET]].mean()

    print(f"数据 {len(df)} 行 | 方法 {(g.index.size)} 个 | patch {df.patch.nunique()} 个"
          + (f" | 只画 {sorted(only)}" if only else ""))
    if want("fig1"):
        fig1_method_scatter(g, fig)
        print("  fig1 ✓")
    if want("fig2"):
        fig2_sample_corr(pd.read_csv(tab / "corr_sample_level.csv"), fig)
        print("  fig2 ✓")
    sp = tab / "corr_stratified.csv"
    if want("fig3") and sp.exists():
        fig3_stratified(pd.read_csv(sp), fig)
        print("  fig3 ✓")
    if want("fig4"):
        fig4_rank_compare(g, fig)
        print("  fig4 ✓")
    if want("fig5"):
        fig5_qualitative(cfg, fig)
        print("  fig5 ✓")
    hqp = tab / "corr_highq_subset.csv"
    if want("fig6") and hqp.exists():
        fig6_subset(pd.read_csv(tab / "corr_method_level.csv"), pd.read_csv(hqp), fig)
        print("  fig6 ✓")
    sep = tab / "corr_by_season.csv"
    if want("fig7") and sep.exists():
        fig7_season(pd.read_csv(sep), fig)
        print("  fig7 ✓")
    lcp, ctp = tab / "corr_by_landcover.csv", tab / "corr_by_cloudthick.csv"
    ctyp = tab / "corr_by_cloudtype.csv"
    if want("fig8") and lcp.exists() and ctp.exists():
        fig8_groups(pd.read_csv(lcp), pd.read_csv(ctp), fig,
                    pd.read_csv(ctyp) if ctyp.exists() else None)
        print("  fig8 ✓")
    if want("fig9"):
        try:
            fig9_failure(cfg, df, fig)
            print("  fig9 ✓")
        except Exception as e:
            print(f"  fig9 跳过：{type(e).__name__} {e}")
    msp = tab / "mask_sensitivity_summary.csv"
    if want("fig12") and msp.exists():
        fig12_mask(pd.read_csv(msp), fig)
        print("  fig12 ✓")
    print(f"图已写入 {fig}")


if __name__ == "__main__":
    main()
