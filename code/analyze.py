"""
analyze.py —— P1 的核心分析：指标与下游反演精度的一致性

三层分析（对应论文的三张主图）：
  1. 方法层：把每个方法在测试集上的平均指标，与它的平均 ΔLAI 做秩相关
     —— 回答"指标排名与下游排名是否一致"（n = 方法数，所以样本少，必须配置信区间）
  2. 样本层：把所有 (方法, patch) 当样本，逐条算指标与 ΔLAI 的相关
     —— 回答"单张图上指标高是否意味着下游好"（n 上千，统计上稳）
  3. 分层：按云量四分位分组重算
     —— 回答"结论在什么条件下成立"（这一层决定论文能走多高）

产出：
  results/tables/corr_method_level.csv
  results/tables/corr_sample_level.csv
  results/tables/corr_stratified.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import bootstrap as sci_bootstrap

METRICS = ["PSNR", "SSIM", "SAM", "ERGAS", "MAE", "RMSE"]
TARGET = "dLAI_mean"


def boot_ci(x, y, n=1000, seed=0):
    """秩相关的 bootstrap 95% CI。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 4:
        return np.nan, np.nan
    rho = spearmanr(x, y).correlation
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        if np.std(x[i]) < 1e-12 or np.std(y[i]) < 1e-12:
            continue
        r = spearmanr(x[i], y[i]).correlation
        if np.isfinite(r):
            vals.append(r)
    if not vals:
        return np.nan, np.nan
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def scene_descriptors(cache_dir: Path, names):
    """
    逐 patch 计算两个分层用的场景描述子。

    1) ndvi_clear —— 晴空区平均 NDVI，作为"地物类型"的代理：
         >0.45 植被；0.20–0.45 混合；<0.20 裸土/建成/水体
       NDVI 用 target（晴空真值）算，因为它是场景属性，不是方法输出，
       不会引入方法间的信息泄漏。

    2) cloud_bright —— 云区可见光平均反射率，作为"云厚度"的代理。
       严格说应该用云光学厚度(COT)，但 SEN12MS-CR 不提供 COT，
       而云的可见光亮度与光学厚度单调正相关（越厚越亮），
       是可用的近似。文中必须写明这是 proxy，不是 COT 本身。

    3) cirrus —— 云区 B10(1373nm) 平均反射率，作为"云类型"的代理。
       B10 是 Sentinel-2 的卷云波段：水云在 1.37μm 强烈吸收（值低），
       卷云（冰晶）在该波段散射明显（值高）。因此 B10 高 = 卷云型，
       低 = 水云/层积云型。这不是严格的云分类标签（需要 CloudSat/ CALIPSO），
       但它是传感器设计上的本职用途，比用亮度瞎猜可靠得多。
       注意：B10 恰好被排除在指标计算之外（均值仅 0.0008 会拉爆 ERGAS），
       所以这里用它做分层不会泄露到被评价的指标里。
    """
    import numpy as np
    from dataset import load_patch

    out = []
    for n in names:
        try:
            d = load_patch(cache_dir, n)
        except Exception:
            out.append({"patch": n, "ndvi_clear": np.nan,
                        "cloud_bright": np.nan, "cirrus": np.nan})
            continue
        t, o, m = d["target"], d["optical"], d["mask"].astype(bool)
        clear = ~m
        ndvi = np.nan
        if clear.sum() > 50:
            red, nir = t[3][clear], t[7][clear]
            nd = (nir - red) / (nir + red + 1e-6)
            ndvi = float(np.median(nd))
        cb = float(np.median(o[[1, 2, 3]].max(axis=0)[m])) if m.sum() > 20 else np.nan
        # B10 = 索引 9（13 波段顺序），只在云区统计
        cir = float(np.median(o[9][m])) if (m.sum() > 20 and o.shape[0] > 9) else np.nan
        out.append({"patch": n, "ndvi_clear": ndvi,
                    "cloud_bright": cb, "cirrus": cir})
    return pd.DataFrame(out)


def load_cfg(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p1.yaml")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    tab = Path(cfg["output"]["tables"])
    out = Path(cfg["output"]["tables"])
    out.mkdir(parents=True, exist_ok=True)

    img = pd.read_csv(tab / "per_patch_metrics.csv")
    dn = pd.read_csv(tab / "per_patch_downstream.csv")
    df = img.merge(dn, on=["method", "patch"], how="inner")
    print("=" * 70)
    print(f"合并完成：{len(df)} 行 | 方法 {df.method.nunique()} 个 | patch {df.patch.nunique()} 个")

    # 云量信息用于分层
    meta_path = Path(cfg["data"]["cache_dir"]) / "meta.json"
    if meta_path.exists():
        meta = pd.DataFrame(json.load(open(meta_path, encoding="utf-8")))
        df = df.merge(meta[["name", "cloud_fraction", "season"]],
                      left_on="patch", right_on="name", how="left")
        print(f"已挂接云量与季节信息（云量中位数 {df.cloud_fraction.median():.3f}）")

    # 场景描述子：地物类型（NDVI）与云厚度代理（云区亮度）
    desc = scene_descriptors(Path(cfg["data"]["cache_dir"]), df.patch.unique())
    df = df.merge(desc, on="patch", how="left")
    print(f"已挂接场景描述子（NDVI 中位 {df.ndvi_clear.median():.3f}，"
          f"云区亮度中位 {df.cloud_bright.median():.3f}）")
    print("=" * 70)

    # ---------------- 1) 方法层 ----------------
    g = df.groupby("method")[METRICS + [TARGET]].mean()
    print("\n【方法层】各方法平均表现（按 PSNR 降序）")
    print(g.sort_values("PSNR", ascending=False).round(4).to_string())

    rows = []
    for m in METRICS:
        rho, p = spearmanr(g[m], g[TARGET])
        lo, hi = boot_ci(g[m].values, g[TARGET].values, args.bootstrap)
        rows.append({"metric": m, "spearman_vs_dLAI": round(float(rho), 4),
                     "p_value": round(float(p), 4), "ci_low": round(lo, 4),
                     "ci_high": round(hi, 4), "n_methods": len(g)})
    ml = pd.DataFrame(rows).sort_values("spearman_vs_dLAI", key=abs, ascending=False)
    ml.to_csv(out / "corr_method_level.csv", index=False)
    print("\n【方法层】指标与 ΔLAI 的秩相关（n = 方法数，仅供参考，置信区间很宽）")
    print(ml.to_string(index=False))

    # ---------------- 2) 样本层 ----------------
    rows = []
    for m in METRICS:
        rho, p = spearmanr(df[m], df[TARGET])
        lo, hi = boot_ci(df[m].values, df[TARGET].values, args.bootstrap)
        rows.append({"metric": m, "spearman_vs_dLAI": round(float(rho), 4),
                     "p_value": round(float(p), 6), "ci_low": round(lo, 4),
                     "ci_high": round(hi, 4), "n_samples": len(df)})
    sl = pd.DataFrame(rows).sort_values("spearman_vs_dLAI", key=abs, ascending=False)
    sl.to_csv(out / "corr_sample_level.csv", index=False)
    print(f"\n【样本层】指标与 ΔLAI 的秩相关（n = {len(df)}，统计上稳定）")
    print(sl.to_string(index=False))

    # ---------------- 2b) 高质量子集：实践中最关心的一问 ----------------
    # "如果我只在几个已经不错的去云方法之间选，PSNR 的排序还可靠吗？"
    # 全量 11 个方法里混着 zero_fill 这种一眼就能看出很差的，会把相关性抬高。
    thr = 25.0
    hi_q = g[g["PSNR"] >= thr]
    if len(hi_q) >= 4:
        rows = []
        for m in METRICS:
            rho, p = spearmanr(hi_q[m], hi_q[TARGET])
            # 排名变化 >=2 位的方法个数，直接给出"排序不可靠"的可数证据
            r1 = hi_q[m].rank(ascending=(m in ("SAM", "ERGAS", "MAE", "RMSE")))
            r2 = hi_q[TARGET].rank(ascending=True)
            n_flip = int(((r1 - r2).abs() >= 2).sum())
            rows.append({"metric": m, "spearman_in_highq": round(float(rho), 4),
                         "p_value": round(float(p), 4), "n_methods": len(hi_q),
                         "n_rank_shift_ge2": n_flip})
        hq = pd.DataFrame(rows).sort_values("spearman_in_highq", key=abs, ascending=False)
        hq.to_csv(out / "corr_highq_subset.csv", index=False)
        print(f"\n【高质量子集】仅 PSNR≥{thr} dB 的 {len(hi_q)} 个方法："
              f"{list(hi_q.index)}")
        print(hq.to_string(index=False))

    # ---------------- 3) 分层 ----------------
    if "cloud_fraction" in df.columns:
        df["cf_bin"] = pd.qcut(df.cloud_fraction, 4, labels=["Q1低云", "Q2", "Q3", "Q4高云"],
                               duplicates="drop")
        rows = []
        for b in df.cf_bin.cat.categories:
            sub = df[df.cf_bin == b]
            for m in METRICS:
                rho, _ = spearmanr(sub[m], sub[TARGET])
                rows.append({"cloud_bin": str(b), "metric": m,
                             "spearman": round(float(rho), 4), "n": len(sub)})
        st = pd.DataFrame(rows)
        st.to_csv(out / "corr_stratified.csv", index=False)
        piv = st.pivot(index="metric", columns="cloud_bin", values="spearman")
        print("\n【分层】按云量四分位的秩相关（看结论是否随云量反转）")
        print(piv.round(3).to_string())

    # ---------------- 4) 季节分层：验证结论不是某一季的特性 ----------------
    if "season" in df.columns and df.season.nunique() >= 2:
        rows = []
        for s, sub in df.groupby("season"):
            for m in METRICS:
                rho, p = spearmanr(sub[m], sub[TARGET])
                rows.append({"season": s, "metric": m, "spearman": round(float(rho), 4),
                             "p_value": round(float(p), 4), "n": len(sub)})
        ss = pd.DataFrame(rows)
        ss.to_csv(out / "corr_by_season.csv", index=False)
        piv = ss.pivot(index="metric", columns="season", values="spearman")
        print("\n【季节分层】各季节内部的秩相关（符号一致 = 结论不依赖季节）")
        print(piv.round(3).to_string())

    # ---------------- 5) 地物类型分层（W6 要求） ----------------
    if "ndvi_clear" in df.columns and df.ndvi_clear.notna().sum() > 100:
        # 注意：实测这批场景 NDVI 集中在 0.45–0.74（几乎全为植被），
        # 用文献常用阈值 0.2/0.45 会退化成两组。改用三分位，保证三档且样本均衡；
        # 论文里要写明"本数据集以植被为主，裸土/水体样本不足"，不能假装有。
        try:
            df["landcover"] = pd.qcut(df.ndvi_clear, 3,
                                      labels=["低NDVI", "中NDVI", "高NDVI"],
                                      duplicates="drop")
        except ValueError:
            df["landcover"] = "全部"
        rows = []
        for b, sub in df.groupby("landcover", observed=True):
            for m in METRICS:
                rho, p = spearmanr(sub[m], sub[TARGET])
                rows.append({"landcover": str(b), "metric": m,
                             "spearman": round(float(rho), 4),
                             "p_value": round(float(p), 4), "n": len(sub)})
        lc = pd.DataFrame(rows)
        lc.to_csv(out / "corr_by_landcover.csv", index=False)
        piv = lc.pivot(index="metric", columns="landcover", values="spearman")
        print("\n【地物类型分层】按晴空 NDVI 三分位（检验结论是否依赖地表类型）")
        print(piv.round(3).to_string())

    # ---------------- 6) 云厚度代理分层（W6 要求） ----------------
    if "cloud_bright" in df.columns and df.cloud_bright.notna().sum() > 100:
        try:
            df["thick_bin"] = pd.qcut(df.cloud_bright, 3,
                                      labels=["薄云", "中云", "厚云"], duplicates="drop")
        except ValueError:
            df["thick_bin"] = "全部"
        rows = []
        for b, sub in df.groupby("thick_bin", observed=True):
            for m in METRICS:
                rho, p = spearmanr(sub[m], sub[TARGET])
                rows.append({"cloud_thickness": str(b), "metric": m,
                             "spearman": round(float(rho), 4),
                             "p_value": round(float(p), 4), "n": len(sub)})
        ct = pd.DataFrame(rows)
        ct.to_csv(out / "corr_by_cloudthick.csv", index=False)
        piv = ct.pivot(index="metric", columns="cloud_thickness", values="spearman")
        print("\n【云厚度分层】按云区可见光亮度三分位（亮度的单向代理，非真实 COT）")
        print(piv.round(3).to_string())

    # ---------------- 7) 云类型分层（W6 要求，用 B10 卷云波段） ----------------
    if "cirrus" in df.columns and df.cirrus.notna().sum() > 100:
        thr = float(df.cirrus.median())
        df["cloud_type"] = np.where(df.cirrus > thr, "卷云型", "水云型")
        rows = []
        for b, sub in df.groupby("cloud_type", observed=True):
            for m in METRICS:
                rho, p = spearmanr(sub[m], sub[TARGET])
                rows.append({"cloud_type": str(b), "metric": m,
                             "spearman": round(float(rho), 4),
                             "p_value": round(float(p), 4), "n": len(sub)})
        cty = pd.DataFrame(rows)
        cty.to_csv(out / "corr_by_cloudtype.csv", index=False)
        piv = cty.pivot(index="metric", columns="cloud_type", values="spearman")
        print(f"\n【云类型分层】按 B10(1373nm 卷云波段) 中位数切分"
              f"（阈值 {thr:.4f}；B10 高 = 卷云型，低 = 水云型）")
        print(piv.round(3).to_string())
        print("  说明：B10 是传感器专门用于探卷云的波段，用它判云类型是本职用途；"
              "但它只是代理，不是 CloudSat/CALIPSO 那种云分类标签。")

    print("\n" + "=" * 70)
    print(f"结果已写入 {out}")
    print("判读要点：")
    print("  · 方法层与样本层符号不一致 → 说明'方法排名'与'逐样本可靠性'不同，本身就是发现")
    print("  · 若 PSNR 相关低而 SAM 相关高 → 支持'应当用光谱指标评价去云'")
    print("  · 分层后若符号反转 → 论文升级为'指标在什么条件下失效'")
    print("=" * 70)


if __name__ == "__main__":
    main()
