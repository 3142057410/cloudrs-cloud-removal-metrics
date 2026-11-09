"""
prodval_bootstrap.py —— 产品级秩相关的 patch 级 bootstrap 置信区间

为什么必须做：只有 9 个方法，Spearman 的标准误约 0.35，两个 ρ 相差 0.2 完全可能是
抽样噪声。不给区间就断言"PSNR 比 SAM 更懂产品精度"是过度解读；给了区间才能说
"符号可转移，量级不可分"。

注意：per-patch 表里没有 rmse_cld 这类方法级量，它只存逐 patch 的误差。
所以每次重采样后都要**重新聚合成方法级指标**再求秩相关，不能直接对逐 patch 列求均值
（那样算的是 MAE，不是 RMSE）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

METRICS = ["PSNR", "SSIM", "SAM", "ERGAS"]
TARGETS = ["rmse_cld", "rmse_all_db", "r2", "break_med"]
NBOOT = 1000


def wide_tables(per: pd.DataFrame) -> dict[str, np.ndarray]:
    """把长表转成 (method, patch) 的矩阵，patch 顺序在各指标间严格一致。"""
    cols = ["PSNR", "SSIM", "SAM", "ERGAS", "err_cld", "err_db", "break",
            "lai_ref", "LAI_ret_cld"]
    out = {}
    for c in cols:
        out[c] = per.pivot(index="method", columns="patch", values=c).to_numpy(float)
    return out


def aggregate(w: dict[str, np.ndarray], idx: np.ndarray) -> pd.DataFrame:
    """给定重采样后的 patch 下标（可重复），重算方法级指标。"""
    def mnan(x):
        return np.nanmean(x, axis=1)

    def rnan(x):
        return np.sqrt(np.nanmean(x ** 2, axis=1))

    ref = w["lai_ref"][:, idx]                      # (M, B) 各方法行相同
    ret = w["LAI_ret_cld"][:, idx]
    e = w["err_cld"][:, idx]
    edb = w["err_db"][:, idx]
    brk = w["break"][:, idx]

    ss_res = np.nansum((ret - ref) ** 2, axis=1)
    ss_tot = np.nansum((ref - np.nanmean(ref)) ** 2)
    return pd.DataFrame({
        "PSNR": mnan(w["PSNR"][:, idx]),
        "SSIM": mnan(w["SSIM"][:, idx]),
        "SAM": mnan(w["SAM"][:, idx]),
        "ERGAS": mnan(w["ERGAS"][:, idx]),
        "rmse_cld": rnan(e),
        "rmse_all_db": rnan(edb),
        "r2": 1.0 - ss_res / ss_tot,
        "break_med": np.nanmedian(np.abs(brk), axis=1),
    })


def rho_of(tab: pd.DataFrame, metric: str, tgt: str) -> float:
    x, y = tab[metric].to_numpy(float), tab[tgt].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return np.nan
    return float(spearmanr(x[ok], y[ok]).correlation)


def main():
    per = pd.read_csv("results/tables_boa/product_validation_per_patch.csv")
    w = wide_tables(per)
    n = w["PSNR"].shape[1]
    rng = np.random.default_rng(0)
    full = np.arange(n)

    base = aggregate(w, full)
    rows = []
    for metric in METRICS:
        for tgt in TARGETS:
            point = rho_of(base, metric, tgt)
            boot = []
            for _ in range(NBOOT):
                pick = rng.integers(0, n, n)
                v = rho_of(aggregate(w, pick), metric, tgt)
                if np.isfinite(v):
                    boot.append(v)
            lo, hi = (np.percentile(boot, [2.5, 97.5]) if boot
                      else (np.nan, np.nan))
            rows.append({"metric": metric, "target": tgt,
                         "rho": round(point, 4),
                         "ci_lo": round(float(lo), 4),
                         "ci_hi": round(float(hi), 4),
                         "ci_excludes_zero": bool(lo * hi > 0),
                         "n_boot": len(boot)})
    df = pd.DataFrame(rows)
    df.to_csv("results/tables_boa/product_validation_corr_boot.csv", index=False)

    print("Spearman rho（方法级，n=9）")
    print(df.pivot_table(index="metric", columns="target",
                         values="rho").round(3).to_string())
    print("\n95% bootstrap CI（target = rmse_cld，负号才对）")
    for _, r in df[df.target == "rmse_cld"].iterrows():
        star = "*" if r.ci_excludes_zero else " "
        print(f"  {r.metric:6s} {r.rho:+.3f}  [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]{star}")
    print("\n* 表示 95% 区间不含 0")
    print("\n方法级基础量（全样本）：")
    print(base.round(3).to_string())


if __name__ == "__main__":
    main()
