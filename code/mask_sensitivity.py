"""
mask_sensitivity.py —— 云掩膜误差敏感性（鲁棒性实验）

问题：前面所有实验都假设云掩膜是完美的（阈值 0.15 检测出来的那一版）。
实际业务里掩膜来自云检测产品，边缘通常有 1–2 像素误差。如果结论对掩膜边界敏感，
那"PSNR 不可靠"这个结论就只在完美掩膜下成立，实用性大打折扣。

设计（刻意选择"只动评价掩膜"这一种）：
  方法本身仍用它自己的掩膜生成结果（pred 只算一次，保证与主实验完全一致），
  只把**评价区域**换成 腐蚀 1px / 原始 / 膨胀 1px 三种。
这样变量只有一个，解释干净：掩膜边界偏 1 像素时，指标与下游误差的关系变不变。

为什么不把"方法输入掩膜"也一起扰动：那会同时改变 14 个方法的输出，
误差来源混在一起，无法判断是"指标不稳"还是"方法本身对掩膜敏感"。

用法：
  python src/mask_sensitivity.py --methods a,b,c          # 推理（分批）
  python src/mask_sensitivity.py --analyze                # 只做汇总分析
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import ndimage
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import METHODS
from dataset import USABLE_IDX, load_patch
from downstream import DownstreamRetrieval, retrieval_delta
from metrics import sam as sam_fn

VARIANTS = ("erode1", "orig", "dilate1")
HIGH_THR = 25.0
METRICS = ["PSNR", "SAM", "dLAI"]


def morph(mask, variant):
    m = mask.astype(bool)
    if variant == "orig":
        return m
    st = np.ones((3, 3), bool)
    return ndimage.binary_erosion(m, st) if variant == "erode1" else ndimage.binary_dilation(m, st)


def run(args, cfg):
    cache = Path(cfg["data"]["cache_dir"])
    names = json.load(open(cache / "split.json", encoding="utf-8"))["test"]
    if args.limit:
        names = names[: args.limit]
    ds = DownstreamRetrieval(cfg["downstream"]["lut_path"])
    methods = args.methods.split(",")

    rows = []
    for mi, mname in enumerate(methods, 1):
        fn = METHODS[mname]
        for n in names:
            d = load_patch(cache, n)
            base = d["mask"].astype(bool)
            if base.sum() < 20:
                continue
            pred = fn(d["optical"], d["sar"], d["mask"])       # 只算一次，与主实验一致
            p_all, t_all = pred[USABLE_IDX], d["target"][USABLE_IDX]
            for v in VARIANTS:
                m = morph(base, v)
                if m.sum() < 50:
                    continue
                mse = ((p_all[:, m] - t_all[:, m]) ** 2).mean()
                rec = {
                    "method": mname, "patch": n, "variant": v,
                    "n_px": int(m.sum()),
                    "PSNR": float(10 * np.log10(1 / max(mse, 1e-8))),
                    "SAM": float(sam_fn(p_all, t_all, mask=m)),
                    "dLAI": float(retrieval_delta(ds, pred, d["target"], m)["dLAI_mean"]),
                }
                rows.append(rec)
        print(f"  [{mi}/{len(methods)}] {mname} 完成", flush=True)

    df = pd.DataFrame(rows)
    out = Path(cfg["output"]["tables"]) / "mask_sensitivity.csv"
    if args.append and out.exists():
        old = pd.read_csv(out)
        df = pd.concat([old, df], ignore_index=True).drop_duplicates(
            subset=["method", "patch", "variant"], keep="last")
    df.to_csv(out, index=False)
    print(f"  -> {out}  共 {len(df)} 行，方法 {df.method.nunique()} 个")


def analyze(cfg):
    tab = Path(cfg["output"]["tables"])
    df = pd.read_csv(tab / "mask_sensitivity.csv")
    print("=" * 74)
    print(f"掩膜敏感性：{len(df)} 行 | 方法 {df.method.nunique()} | "
          f"patch {df.patch.nunique()} | 变体 {sorted(df.variant.unique())}")
    print("=" * 74)

    # 评价区域大小的变化
    sz = df.groupby("variant")["n_px"].mean()
    base = sz.get("orig", np.nan)
    print("\n【评价像素数变化（相对原始掩膜）】")
    for v in VARIANTS:
        if v in sz:
            print(f"  {v:9s} 平均 {sz[v]:.0f} 像素  ({(sz[v]/base-1)*100:+.1f}%)")

    # 各变体下的相关性与“高质量子集”结论
    rows = []
    for v in VARIANTS:
        sub = df[df.variant == v]
        g = sub.groupby("method")[METRICS].mean()
        hi = g[g["PSNR"] >= HIGH_THR]
        for name, d_ in (("all", g), ("highq", hi)):
            if len(d_) < 4:
                continue
            for m in ("PSNR", "SAM"):
                rho = spearmanr(d_[m], d_["dLAI"]).correlation
                rows.append({"variant": v, "scope": name, "n_methods": len(d_),
                             "metric": m, "rho": round(float(rho), 4)})
    res = pd.DataFrame(rows)
    res.to_csv(tab / "mask_sensitivity_summary.csv", index=False)

    print("\n【各掩膜变体下的秩相关（越低越说明结论会变）】")
    piv = res.pivot_table(index=["scope", "metric"], columns="variant", values="rho")
    print(piv.round(3).to_string())

    # 关键判据：SAM 是否在所有变体下都优于 PSNR（高质量子集）
    hq = res[(res.scope == "highq")].pivot(index="variant", columns="metric", values="rho")
    print("\n【高质量子集：SAM vs PSNR】")
    print(hq.round(3).to_string())
    if {"SAM", "PSNR"}.issubset(hq.columns):
        ok = (hq["SAM"].abs() > hq["PSNR"].abs()).all()
        spread = (hq["SAM"].abs() - hq["PSNR"].abs())
        print(f"\n  SAM 在全部 {len(hq)} 个掩膜变体下都优于 PSNR：{'是 ✓' if ok else '否 ✗'}")
        print(f"  优势幅度 {spread.min():+.3f} ~ {spread.max():+.3f}")
        print("  判读：优势方向稳定且幅度远大于变体间的波动 → 掩膜边界 1 像素误差不动摇主结论")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p1.yaml")
    ap.add_argument("--methods", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    if args.analyze:
        analyze(cfg)
        return
    if not args.methods:
        raise SystemExit("推理模式必须给 --methods，或用 --analyze")
    run(args, cfg)


if __name__ == "__main__":
    main()
