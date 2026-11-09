"""
ndvi_task.py —— 第二个下游任务：NDVI 分类（离散决策，机制与参数反演不同）

为什么需要它：
  论文现在的两个下游变量（LAI、Cab）都来自同一套 PROSAIL 查找表，
  审稿人会说"你只是换了输出通道，不是换了下游任务"。
  NDVI 分类是**离散判决**任务：误差以错分率计，而不是连续参数偏差，
  误差机制完全不同。若指标与它的关系同样成立，结论就不再依赖单一反演框架。

做法：
  在云区像素上，用 B8/B4 计算 NDVI，按阈值分为 3 类（非植被 <0.2、稀疏 0.2–0.5、密植被 >0.5），
  比较重建影像与晴空真值的分类一致性（总体精度 OA）。
  下游误差记为 Δclass = 1 − OA，数值越大越差。

输出：results/tables/ndvi_task.csv（长表，可 --append 分批）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import METHODS
from dataset import load_patch

THR = (0.20, 0.50)          # NDVI 分级阈值


def classify(nd):
    """三分类：0 非植被 / 1 稀疏 / 2 密植被。"""
    c = np.zeros(nd.shape, np.int8)
    c[nd >= THR[0]] = 1
    c[nd >= THR[1]] = 2
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p1.yaml")
    ap.add_argument("--methods", required=True)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    cache = Path(cfg["data"]["cache_dir"])
    names = json.load(open(cache / "split.json", encoding="utf-8"))["test"]
    if args.limit:
        names = names[: args.limit]
    methods = args.methods.split(",")

    rows = []
    for mi, m in enumerate(methods, 1):
        fn = METHODS[m]
        for n in names:
            d = load_patch(cache, n)
            mask = d["mask"].astype(bool)
            if mask.sum() < 50:
                continue
            pred = fn(d["optical"], d["sar"], d["mask"])
            # NDVI = (NIR - R) / (NIR + R)，B8 索引 7，B4 索引 3
            def ndvi(img):
                nir, red = img[7][mask], img[3][mask]
                return (nir - red) / (nir + red + 1e-6)
            cp = classify(ndvi(pred))
            ct = classify(ndvi(d["target"]))
            rows.append({
                "method": m, "patch": n, "n_px": int(mask.sum()),
                "OA": float((cp == ct).mean()),
                "dclass": float(1.0 - (cp == ct).mean()),
                "dNDVI": float(np.mean(np.abs(ndvi(pred) - ndvi(d["target"])))),
            })
        print(f"  [{mi}/{len(methods)}] {m} 完成", flush=True)

    df = pd.DataFrame(rows)
    out = Path(cfg["output"]["tables"]) / "ndvi_task.csv"
    if args.append and out.exists():
        old = pd.read_csv(out)
        df = pd.concat([old, df], ignore_index=True).drop_duplicates(
            subset=["method", "patch"], keep="last")
    df.to_csv(out, index=False)
    print(f"  -> {out}  {len(df)} 行，方法 {df.method.nunique()} 个")


if __name__ == "__main__":
    main()
