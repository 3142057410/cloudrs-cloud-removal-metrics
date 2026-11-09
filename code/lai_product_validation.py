"""
lai_product_validation.py —— 第 2 步：把"指标与反演误差的一致性"升级为"指标与产品精度的一致性"

与 P1 原有下游分析的区别（这是本节存在的全部理由）：
    原来：ΔLAI = |LAI(pred) − LAI(target)|，target 是同一景的晴空影像，
          参照来自同一套 PROSAIL 检索，属于**自洽代理**。
    现在：LAI 参照换成外部产品 MCD15A3H（500 m，4 天，独立传感器 + 独立算法 +
          独立验证史），因此衡量的是"这个去云结果喂给产品链，产品会偏多少"。

三个口径必须分开报，否则数字会被误读：
    · _cld  只统计云掩膜内像元 —— 隔离"去云环节"的影响，用于**方法间排序**。
      可比性最好（所有方法用同一批像元、同一个参照），但绝对值受"云区本身
      与整块均值不同"污染，不能当产品精度读。
    · _all  统计整块 patch 像元 —— 产品实际交付的口径，用于**绝对偏差/RMSE**。
      晴空像元各方法完全相同，会稀释方法间差异，所以不用于排序。
    · 季节基线基线 break = LAI_pred − 同期 MODIS 季节趋势，衡量"这个估计值插进
      产品时序里会不会造成断点"。参照取自前后各一个合成期，剔除当期。

参照本身不确定度不忽略：
    用 LaiStdDev 逐像元算 σ_ref，并给出"方法间离散 ÷ σ_ref"的比值。比值 <1 时
    任何方法排序都不可信，这一条必须写在表里，而不是等审稿人来问。

用法：
    python src/lai_product_validation.py --processed data/s2_processed \
        --modis data/modis_lai --geo results_meta/patch_geo.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import METHODS
from dataset import USABLE_IDX
from downstream import DownstreamRetrieval
from metrics import evaluate_all

OPTICAL_ONLY = ["zero_fill", "mean_fill", "nearest_fill", "blur_fill",
                "inpaint_telea", "knn_spatial", "hist_match",
                "unet_opt", "dsen2cr_opt"]
IMG_METRICS = ["PSNR", "SSIM", "SAM", "ERGAS"]
GRID_M = 500.0
LAI_SCALE = 0.1
FILL_MIN = 249
COMPOSITE_DAYS = 4
BASELINE_HALF_WIN = 16      # 季节基线取 ±16 天内的合成期


# ------------------------------------------------------------------ MODIS 参照

def load_composites(modis_dir: Path):
    """读入全部合成期：{tile: {start_date: dict(lai, std, tr, shape)}}"""
    store = {}
    for f in sorted(modis_dir.glob("*.npz")):
        tile, ds = f.stem.split("_", 1)
        d = np.load(f)
        lai_dn, qc, std_dn = d["lai_dn"], d["qc"], d["std_dn"]
        tr = tuple(d["transform"])
        shape = tuple(int(x) for x in d["shape"])
        valid = lai_dn < FILL_MIN
        modland = (qc & 0b1) == 0
        scf = (qc >> 1) & 0b11
        good = valid & modland & (scf <= 1)
        store.setdefault(tile, {})[date.fromisoformat(ds)] = {
            "lai": np.where(good, lai_dn.astype(np.float32) * LAI_SCALE, np.nan),
            "std": np.where(good, std_dn.astype(np.float32) * LAI_SCALE, np.nan),
            "tr": tr, "shape": shape, "n_good": int(good.sum()),
        }
    return store


def match_composite(comp_dates, s2_date: date):
    """S2 日期落在哪个 4 天合成期内（起始日 <= D <= 起始日+3）。返回 (start, offset)。"""
    best = None
    for s in comp_dates:
        off = (s2_date - s).days
        if 0 <= off < COMPOSITE_DAYS:
            if best is None or abs(off) < abs(best[1]):
                best = (s, off)
    if best is None:                      # 合成期缺失时退到最近的（±2 天内）
        cand = [(abs((s2_date - s).days), s) for s in comp_dates]
        if cand:
            dist, s = min(cand)
            if dist <= 2:
                best = (s, (s2_date - s).days)
    return best


def patch_window(tr, shape, cx, cy, half):
    """求 patch 盒子覆盖的网格行列区间（按像元中心判定）。"""
    x0, y0 = cx - half, cy - half
    c0 = int(np.floor((x0 - tr[2]) / GRID_M))
    c1 = int(np.ceil((cx + half - tr[2]) / GRID_M))
    r0 = int(np.floor((tr[5] - (cy + half)) / GRID_M))
    r1 = int(np.ceil((tr[5] - y0) / GRID_M))
    c0, c1 = max(c0, 0), min(c1, shape[1])
    r0, r1 = max(r0, 0), min(r1, shape[0])
    if c1 <= c0 or r1 <= r0:
        return None
    # 只保留像元中心落在盒子内的
    cols = np.arange(c0, c1)
    rows = np.arange(r0, r1)
    xc = tr[2] + (cols + 0.5) * GRID_M
    yc = tr[5] - (rows + 0.5) * GRID_M
    keep_c = (xc >= x0) & (xc < cx + half)
    keep_r = (yc >= y0) & (yc < cy + half)
    if not keep_c.any() or not keep_r.any():
        return None
    return rows[keep_r][:, None], cols[keep_c][None, :]


def patch_ref(comp, geo_row):
    """取一个 patch 的参照值：均值、不确定度、参与像元数。"""
    idx = patch_window(comp["tr"], comp["shape"], geo_row["x_utm"], geo_row["y_utm"],
                       geo_row["footprint_m"] / 2.0)
    if idx is None:
        return np.nan, np.nan, 0
    rr, cc = idx
    v = comp["lai"][rr, cc]
    s = comp["std"][rr, cc]
    m = np.isfinite(v)
    if m.sum() == 0:
        return np.nan, np.nan, 0
    return float(np.nanmean(v)), float(np.nanmean(s[m])), int(m.sum())


# ------------------------------------------------------------------ 检索

def lai_of(ds, cube, mask, max_pixels=300, seed=0):
    """云区像元（cld）与整块 patch（all）分别抽样反演，返回各自均值。

    两个口径用途不同，不能混：
      cld —— 只取云掩膜内像元，隔离"去云环节"，用于方法间排序
      all —— 整块 patch，模拟产品实际交付值，用于绝对偏差
    早期版本把 all 写成了"只取晴空像元"，等于把各方法的差异抹平（所有方法的
    晴空部分完全相同），得到一堆一模一样的数 —— 那是错的。
    """
    m = mask.astype(bool).ravel()
    flat = cube.reshape(cube.shape[0], -1).T          # (H*W, C)
    rng = np.random.default_rng(seed)
    out = {}
    for tag, pool in (("cld", flat[m]), ("all", flat)):
        if len(pool) == 0:
            out[tag] = np.nan
            continue
        sel = pool if len(pool) <= max_pixels else pool[rng.choice(len(pool), max_pixels,
                                                                  replace=False)]
        # DownstreamRetrieval.select 的二维约定是 (M, C)，即每行一条光谱。
        # 早期这里传了 sel.transpose()（C, M），被当成 13 条 10 维"光谱"去查树，
        # 反演出来的是一堆错位样本的均值 —— 数值看着像样，其实毫无意义。
        out[tag] = float(np.mean(ds.invert(ds.select(sel))[0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/s2_processed")
    ap.add_argument("--modis", default="data/modis_lai")
    ap.add_argument("--geo", default="results_meta/patch_geo.csv")
    ap.add_argument("--lut", default="data/processed/prosail_lut.npz")
    ap.add_argument("--metrics", default="results/tables/per_patch_metrics_s2.csv")
    ap.add_argument("--out", default="results/tables")
    ap.add_argument("--figdir", default="results/figures")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-ref-px", type=int, default=1,
                    help="patch 内有效参照像元数下限；MODIS 500 m 下常见 4-9 个")
    a = ap.parse_args()

    geo = pd.read_csv(a.geo).set_index("name")
    comps = load_composites(Path(a.modis))
    ds = DownstreamRetrieval(a.lut)
    files = sorted(Path(a.processed).glob("*.npz"))
    if a.limit:
        files = files[: a.limit]
    print(f"patch {len(files)}  瓦片 {sorted(comps)}")

    # ---------- 1. 参照：每个 patch 在当期 + 季节基线 ----------
    ref_rows = []
    for f in files:
        name = f.stem
        g = geo.loc[name]
        s2d = date.fromisoformat(g["date"])
        tile = g["tile"]
        if tile not in comps:
            print(f"  跳过 {name}：无 {tile} 的 MODIS 数据")
            continue
        dates = sorted(comps[tile])
        mt = match_composite(dates, s2d)
        if mt is None:
            print(f"  跳过 {name}：{s2d} 无可用合成期")
            continue
        cstart, off = mt
        ref, sig, npx = patch_ref(comps[tile][cstart], g)

        # 季节基线：剔除当期，前后各取最近一个合成期，线性内插到 s2d
        others = [d for d in dates if d != cstart and abs((d - s2d).days) <= BASELINE_HALF_WIN]
        before = [d for d in others if d < s2d]
        after = [d for d in others if d >= s2d]
        vals = {}
        for tag, pool in (("b", before), ("a", after)):
            if pool:
                d0 = max(pool) if tag == "b" else min(pool)
                v0, _, _ = patch_ref(comps[tile][d0], g)
                vals[tag] = (d0, v0)
        if "b" in vals and "a" in vals and np.isfinite(vals["b"][1]) and np.isfinite(vals["a"][1]):
            (db, vb), (da, va) = vals["b"], vals["a"]
            w = (s2d - db).days / max((da - db).days, 1)
            base = vb + w * (va - vb)
        elif vals:
            base = list(vals.values())[0][1]
        else:
            base = np.nan

        ref_rows.append({"patch": name, "tile": tile, "date": str(s2d),
                         "comp_start": str(cstart), "comp_off_days": off,
                         "lai_ref": ref, "sigma_ref": sig, "n_ref_px": npx,
                         "lai_base": base,
                         "cloud_fraction": float(g["cloud_fraction"])})
    ref = pd.DataFrame(ref_rows).set_index("patch")
    if a.min_ref_px > 1:
        keep = ref.n_ref_px >= a.min_ref_px
        print(f"参照像元数 >={a.min_ref_px} 的 patch：{int(keep.sum())}/{len(ref)}")
        ref = ref[keep]
    ok = ref.lai_ref.notna()
    print(f"\n参照可用 {int(ok.sum())}/{len(ref)}；"
          f"参与像元数中位 {ref.n_ref_px[ok].median():.0f} "
          f"(5% {ref.n_ref_px[ok].quantile(.05):.0f})；"
          f"σ_ref 中位 {ref.sigma_ref[ok].median():.3f}")

    # ---------- 2. 逐方法检索 ----------
    rows = []
    for mi, mname in enumerate(OPTICAL_ONLY, 1):
        fn = METHODS[mname]
        for f in files:
            name = f.stem
            if name not in ref.index:
                continue
            d = np.load(f)
            opt, tgt, mask = d["optical"], d["target"], d["mask"]
            if mask.sum() < 20:
                continue
            pred = fn(opt, d["sar"], mask)
            met = evaluate_all(pred[USABLE_IDX], tgt[USABLE_IDX], mask=mask)
            lp = lai_of(ds, pred, mask)
            rows.append({"method": mname, "patch": name,
                         "LAI_ret_all": lp["all"], "LAI_ret_cld": lp["cld"],
                         **{k: met[k] for k in IMG_METRICS}})
        print(f"  [{mi}/{len(OPTICAL_ONLY)}] {mname}", flush=True)

    per = pd.DataFrame(rows)
    if per.empty:
        raise SystemExit("没有任何可用样本，检查 MODIS 目录与 patch 日期是否对得上")
    per = per.join(ref, on="patch")
    per["bias_all"] = per.LAI_ret_all - per.lai_ref
    per["err_all"] = per.LAI_ret_all - per.lai_ref
    per["abs_all"] = per.err_all.abs()
    per["bias_cld"] = per.LAI_ret_cld - per.lai_ref
    per["err_cld"] = per.LAI_ret_cld - per.lai_ref
    per["break"] = per.LAI_ret_all - per.lai_base
    # 全局偏差 c0：所有方法共有的系统偏移（来自 PROSAIL 检索本身，不是去云误差）。
    # 只在方法间比较时扣除，绝对精度一律报未扣除的原始值。
    c0 = float(np.nanmedian(per.err_all))
    per["err_db"] = per.err_all - c0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    per.to_csv(out / "product_validation_per_patch.csv", index=False)

    # ---------- 3. 方法级汇总 ----------
    def agg(g):
        n = len(g)
        rmse_all = float(np.sqrt(np.mean(g.err_all ** 2)))
        rmse_cld = float(np.sqrt(np.mean(g.err_cld ** 2)))
        sig2 = float(np.nanmean(g.sigma_ref ** 2))
        r2 = float(pearsonr(g.LAI_ret_all, g.lai_ref)[0] ** 2) if n > 3 else np.nan
        return pd.Series({
            "n": n,
            "PSNR": g.PSNR.mean(), "SSIM": g.SSIM.mean(),
            "SAM": g.SAM.mean(), "ERGAS": g.ERGAS.mean(),
            "bias_all": g.bias_all.mean(), "rmse_all": rmse_all,
            "mae_all": g.abs_all.mean(),
            "rmse_cld": rmse_cld,
            "rmse_cld_adj": float(np.sqrt(max(rmse_cld ** 2 - sig2, 0.0))),
            "rmse_all_db": float(np.sqrt(np.mean(g.err_db ** 2))),
            "r2": r2,
            "break_med": float(np.nanmedian(np.abs(g["break"]))),
            "sigma_ref": float(np.nanmean(g.sigma_ref)),
        })

    # 不用 groupby.apply：pandas 2.x 会把分组列并进结果，行列会转置
    ms = pd.DataFrame({m: agg(g) for m, g in per.groupby("method")}).T
    ms = ms.loc[[m for m in OPTICAL_ONLY if m in ms.index]]
    ms["rank_rmse_cld"] = ms.rmse_cld.rank()
    ms["rank_psnr"] = ms.PSNR.rank(ascending=False)
    ms["rank_sam"] = ms.SAM.rank()
    ms.round(4).to_csv(out / "product_validation_method.csv")
    print("\n【方法级：图像指标 vs 产品精度】")
    print(ms[["PSNR", "SSIM", "SAM", "ERGAS", "bias_all", "rmse_all", "rmse_all_db",
              "rmse_cld", "rmse_cld_adj", "r2", "break_med"]].round(3).to_string())

    # ---------- 4. 一致性：图像指标能否排序产品精度 ----------
    corr = []
    for metric in IMG_METRICS:
        for target, sign in (("rmse_cld", -1), ("rmse_all_db", -1), ("rmse_all", -1),
                             ("mae_all", -1), ("r2", +1), ("break_med", -1)):
            c = ms[[metric, target]].dropna()
            if len(c) < 4:
                continue
            rho, p = spearmanr(c[metric], c[target])
            corr.append({"metric": metric, "target": target, "n": len(c),
                         "rho": round(float(rho), 4), "p": round(float(p), 4),
                         "expected_sign": sign,
                         "consistent": bool(np.sign(rho) == sign)})
    corr = pd.DataFrame(corr)
    corr.to_csv(out / "product_validation_corr.csv", index=False)
    print("\n【图像指标 → 产品精度 的秩相关】")
    print(corr.pivot_table(index="metric", columns="target", values="rho")
          .round(3).to_string())

    # ---------- 5. 误差预算 ----------
    spread = float(ms.rmse_cld.max() - ms.rmse_cld.min())
    sig_med = float(ms.sigma_ref.median())
    ratio = spread / sig_med if sig_med > 0 else np.nan
    budget = {"global_bias_c0": c0,
              # 两个口径都给：方法级表里的 sigma_ref 是逐 patch 的均值再跨方法取中位，
              # 等价于总体均值；逐 patch 的中位数明显更小（分布右偏），两种都报以免被挑口径
              "sigma_ref_mean": sig_med,
              "sigma_ref_median_patch": float(per.sigma_ref.median()) if "sigma_ref" in per else None,
              "inter_method_rmse_spread": spread,
              "ratio_spread_over_sigma": ratio,
              "interpretation": ("信号大于参照噪声，方法间排序可信" if ratio > 1
                                 else "方法间差异小于参照噪声，排序需谨慎解释")}
    json.dump(budget, open(out / "product_validation_budget.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"\n【误差预算】σ_ref 中位 {sig_med:.3f}；方法间 RMSE 极差 {spread:.3f}；"
          f"比值 {ratio:.2f} → {budget['interpretation']}")

    # ---------- 6. 站级复核（6 个站点，绝对精度更稳） ----------
    site = (per.groupby(["method", "tile", "date"])
            .agg(LAI_ret_all=("LAI_ret_all", "mean"),
                 lai_ref=("lai_ref", "mean"),
                 n_patch=("patch", "size")).reset_index())
    site["err"] = site.LAI_ret_all - site.lai_ref
    site.to_csv(out / "product_validation_site.csv", index=False)
    s = site.groupby("method").agg(n_site=("err", "size"),
                                   bias=("err", "mean"),
                                   rmse=("err", lambda x: float(np.sqrt(np.mean(x ** 2)))))
    s = s.loc[[m for m in OPTICAL_ONLY if m in s.index]]
    print("\n【站级复核】")
    print(s.round(3).to_string())
    s.round(4).to_csv(out / "product_validation_site_summary.csv")

    # ---------- 7. 图 ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fd = Path(a.figdir); fd.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({"font.size": 9, "axes.grid": True,
                             "grid.alpha": .25, "figure.dpi": 200})

        # Fig 9：产品精度散点（每方法一色）+ 1:1 线
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        cmap = plt.get_cmap("tab10")
        for i, mname in enumerate(ms.index):
            g = per[per.method == mname]
            ax.scatter(g.lai_ref, g.LAI_ret_all, s=9, alpha=.45,
                       color=cmap(i % 10), label=f"{mname} ({ms.rmse_all[mname]:.2f})")
        lim = [0, max(per.lai_ref.max(), per.LAI_ret_all.max()) * 1.05]
        ax.plot(lim, lim, "k--", lw=.8)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("MCD15A3H LAI (reference)")
        ax.set_ylabel("Retrieved LAI from cloud-removed S2")
        ax.legend(fontsize=5.5, ncol=2, loc="upper left", frameon=False)
        ax.set_title("Patch-level LAI against an operational product", fontsize=9)
        fig.tight_layout(); fig.savefig(fd / "fig9_lai_product_scatter.png"); plt.close(fig)

        # Fig 10：偏差随云量分级
        bins = [0, .2, .4, .6, .8, 1.01]
        lab = ["0-20", "20-40", "40-60", "60-80", "80-100"]
        per["cf_bin"] = pd.cut(per.cloud_fraction, bins, labels=lab, right=False)
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        for i, mname in enumerate(["zero_fill", "mean_fill", "hist_match",
                                   "unet_opt", "dsen2cr_opt"]):
            g = per[per.method == mname].groupby("cf_bin", observed=True)
            y = g.bias_all.mean(); lo = g.bias_all.std()
            ax.errorbar(range(len(y)), y, yerr=lo, marker="o", ms=3, lw=1,
                        capsize=2, label=mname)
        ax.axhline(0, color="k", lw=.8, ls=":")
        ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab)
        ax.set_xlabel("Cloud fraction of patch (%)")
        ax.set_ylabel("LAI bias vs MCD15A3H")
        ax.legend(fontsize=5.5, ncol=2, frameon=False)
        ax.set_title("Product-level bias grows with cloud cover", fontsize=9)
        fig.tight_layout(); fig.savefig(fd / "fig10_bias_vs_cloud.png"); plt.close(fig)
        print(f"\n图 -> {fd}/fig9_lai_product_scatter.png, fig10_bias_vs_cloud.png")
    except Exception as e:
        print(f"绘图跳过：{type(e).__name__} {e}")

    print(f"\n表 -> {out}/product_validation_*.csv")


if __name__ == "__main__":
    main()
