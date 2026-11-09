"""
new_metric.py —— 把 P1 从"纯分析"升级为"分析 + 方法"的那一步

问题：如果论文只报告"PSNR 与 ΔLAI 相关性低"，审稿人会说这是负面结论、
没有可操作性（"那你让我用什么指标？"）。

做法：提出「任务一致性加权分数」TCWS（Task-Consistency Weighted Score）
    TCWS = -Σ w_m · z_m(quality)
其中 z_m 是把各指标按方向统一（越高越好）并标准化后的值，w_m ≥ 0 由
非负最小二乘拟合得到，使 TCWS 尽可能接近 -ΔLAI（即"越大越好、越接近 0 越准"）。

关键：权重不是拍脑袋给的，而是从"指标 → 下游误差"的关系里学出来的，
因此可以直接回答审稿人那句"为什么是这个权重"。

两种评估方式（都不许用测试数据拟合）：
  1) 打分集划分：一半 patch 拟合权重，另一半评估
  2) 留一方法交叉验证（LOMO）：留出一个方法，用其余方法拟合权重，
     预测被留出的方法 —— 这是更严格的检验：权重必须能迁移到没见过的去云方法
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import nnls
from scipy.stats import spearmanr

METRICS = ["PSNR", "SSIM", "SAM", "ERGAS", "MAE", "RMSE"]
HIGHER_BETTER = {"PSNR": True, "SSIM": True, "SAM": False,
                 "ERGAS": False, "MAE": False, "RMSE": False}
TARGET = "dLAI_mean"


def quality_matrix(df, cols=METRICS):
    """把各指标统一成"越大越好"的方向。"""
    X = df[cols].to_numpy(float).copy()
    for j, c in enumerate(cols):
        if not HIGHER_BETTER[c]:
            X[:, j] = -X[:, j]
    return X


def _z(X, ref=None):
    ref = X if ref is None else ref
    mu = np.nanmean(ref, axis=0)
    sd = np.nanstd(ref, axis=0) + 1e-12
    return (X - mu) / sd


def fit_weights(Xq, y):
    """
    非负最小二乘拟合 w，使 w·(−z(X)) ≈ y（y = ΔLAI，越大越差）。
    注意符号：z 是"质量"方向（越大越好），而 y 是"误差"方向（越大越差），
    两者相反。直接对 z 做非负回归会得到全零权重（不可行解），
    必须对 −z 回归，得到非负的"劣化权重"。
    """
    Z = _z(Xq)
    w, _ = nnls(-Z, y)
    return w


def score(Xq, w, ref):
    """TCWS：越大越好（= −预测的 ΔLAI）。"""
    return _z(Xq, ref) @ w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p1.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target", default="dLAI_mean",
                    choices=["dLAI_mean", "dCab_mean"],
                    help="下游误差变量。权重必须按任务分别拟合，不能跨任务复用")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    tab = Path(cfg["output"]["tables"])
    tgt = args.target
    tag = "cab" if tgt == "dCab_mean" else "lai"

    img = pd.read_csv(tab / "per_patch_metrics.csv")
    dn = pd.read_csv(tab / "per_patch_downstream.csv")
    if tgt not in dn.columns:
        raise SystemExit(f"下游表没有 {tgt}，请先用新版 downstream.py 重跑")
    df = img.merge(dn, on=["method", "patch"], how="inner").dropna(subset=METRICS + [tgt])
    print("=" * 72)
    print(f"目标变量 {tgt} | 样本 {len(df)} 行 | 方法 {df.method.nunique()} | patch {df.patch.nunique()}")
    print("=" * 72)

    Xq = quality_matrix(df)
    y = df[tgt].to_numpy(float)

    # ---------- 基线：单指标 / 等权平均 ----------
    rows = []
    for j, m in enumerate(METRICS):
        rho = spearmanr(Xq[:, j], y).correlation
        rows.append({"score": m, "kind": "单一指标", "spearman_vs_dLAI": rho})
    eq = Xq.sum(axis=1)
    rows.append({"score": "等权平均(方向统一后)", "kind": "朴素组合",
                 "spearman_vs_dLAI": spearmanr(eq, y).correlation})

    # ---------- 1) 打分集划分（按 patch 划分，避免同图泄漏） ----------
    rng = np.random.default_rng(args.seed)
    patches = df.patch.unique()
    rng.shuffle(patches)
    half = set(patches[: len(patches) // 2])
    tr = df.patch.isin(half).to_numpy()
    te = ~tr
    w = fit_weights(Xq[tr], y[tr])
    tcws_holdout = np.full(len(df), np.nan)
    tcws_holdout[te] = score(Xq[te], w, Xq[tr])
    rows.append({"score": "TCWS（打分集划分）", "kind": "本文方法",
                 "spearman_vs_dLAI": spearmanr(tcws_holdout[te], y[te]).correlation})

    print("\n【权重（打分集划分拟合）】")
    for m, wi in zip(METRICS, w):
        print(f"  {m:6s} {wi:7.4f}")
    print(f"  截距项（未约束）参考：ΔLAI 均值 {y[tr].mean():.3f}")

    # ---------- 2) 留一方法交叉验证 ----------
    print("\n【留一方法交叉验证 LOMO】")
    lomo_rows = []
    for mname in df.method.unique():
        m_tr = (df.method != mname).to_numpy()
        m_te = ~m_tr
        if m_te.sum() < 5:
            continue
        wm = fit_weights(Xq[m_tr], y[m_tr])
        s = score(Xq[m_te], wm, Xq[m_tr])
        rho = spearmanr(s, y[m_te]).correlation
        lomo_rows.append({"method": mname, "spearman": rho, "n": int(m_te.sum())})
        print(f"  留出 {mname:16s} ρ = {rho if np.isfinite(rho) else float('nan'):+.3f}  (n={m_te.sum()})")
    lomo = pd.DataFrame(lomo_rows)
    if not lomo.empty:
        print(f"  LOMO 中位数 ρ = {lomo.spearman.median():+.3f}")

    out = pd.DataFrame(rows)
    out["rho"] = out.spearman_vs_dLAI
    out = out.sort_values("rho")
    suffix = "" if tag == "lai" else "_cab"
    out.to_csv(tab / f"corr_new_metric{suffix}.csv", index=False)
    # 权重单独落盘，供论文/报告直接读取，避免从日志里手抄出错
    pd.DataFrame({"metric": METRICS, "weight": np.round(w, 6)}).to_csv(
        tab / f"tcws_weights{suffix}.csv", index=False)

    # ---------- 汇总对比 ----------
    print("\n" + "=" * 72)
    print("各项分数与 ΔLAI 的秩相关（越负越好：分数越高应代表误差越小）")
    print("=" * 72)
    print(out[["kind", "score", "spearman_vs_dLAI"]].to_string(index=False))
    best_single = out[out.kind == "单一指标"].spearman_vs_dLAI.min()
    tcws = float(out.loc[out.score.str.startswith("TCWS"), "spearman_vs_dLAI"].iloc[0])
    print(f"\n单指标最好：{best_single:+.3f} | TCWS：{tcws:+.3f} | 提升 {abs(tcws)-abs(best_single):+.3f}")
    if not lomo.empty:
        lomo.to_csv(tab / f"corr_new_metric_lomo{suffix}.csv", index=False)
    print(f"\n-> {tab/f'corr_new_metric{suffix}.csv'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
