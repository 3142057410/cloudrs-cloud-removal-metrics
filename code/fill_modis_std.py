"""
fill_modis_std.py —— 补齐 LaiStdDev（逐像元不确定度）

背景：fetch_modis_lai.py 早期版本用"S2 日期 == 合成日期"判断目标时相，而 MCD15A3H
按合成期起始日命名，两者永不相等，于是 LaiStdDev 一次都没被读过，npz 里全是 255
填充值，误差预算会算出 σ_ref = 25.5（荒谬）。这里按"合成期是否覆盖某个 S2 日期"
重新判定，只补这些合成期的 Std 波段。

用法：python src/fill_modis_std.py --modis data/modis_lai --geo results_meta/patch_geo.csv
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_modis_lai import (COMPOSITE_DAYS, FILL_MIN, GRID_M, CRS_UTM,
                             read_to_grid, search, sign, _fetch)
import rasterio
from rasterio.warp import transform as warp_transform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modis", default="data/modis_lai")
    ap.add_argument("--geo", default="results_meta/patch_geo.csv")
    a = ap.parse_args()

    md = Path(a.modis)
    geo = pd.read_csv(a.geo)
    s2dates = sorted(set(geo["date"].unique()))
    cache = {}
    cf = md / "_sign_cache.json"
    if cf.exists():
        cache = json.loads(cf.read_text())

    # index 里同一 (tile, date) 可能有多行（AOI 跨 MODIS 分幅时），全部留着逐个试
    idx = pd.read_csv(md / "modis_index.csv")
    cand = {}
    for _, r in idx.iterrows():
        cand.setdefault((r["tile"], r["date"]), []).append(r["item"])
    todo = [(k, v) for k, v in cand.items()
            if any(0 <= (date.fromisoformat(x) - date.fromisoformat(k[1])).days < COMPOSITE_DAYS
                   for x in s2dates)]
    print(f"需要补 Std 的合成期：{len(todo)} 片")

    done = 0
    for (tile, ds), items in todo:
        f = md / f"{tile}_{ds}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        tr9 = np.array(d["transform"]).ravel()
        tr_aff = rasterio.Affine(*[float(x) for x in tr9[:6]])
        shape = tuple(int(x) for x in d["shape"])
        arr = None
        for iid in items:
            try:
                it = json.loads(_fetch(urllib.request.Request(
                    "https://planetarycomputer.microsoft.com/api/stac/v1/search",
                    data=json.dumps({"collections": ["modis-15A3H-061"],
                                     "ids": [iid], "limit": 1}).encode(),
                    headers={"Content-Type": "application/json"})))["features"][0]
                a = read_to_grid(sign(it["assets"]["LaiStdDev_500m"]["href"], cache),
                                 tr_aff, shape)
            except Exception:
                continue
            if a is not None:
                arr = a
                break
        if arr is None:
            print(f"  {tile} {ds} 无覆盖分幅，跳过")
            continue
        np.savez_compressed(f, lai_dn=d["lai_dn"], qc=d["qc"], std_dn=arr.astype(np.uint8),
                            transform=np.array(d["transform"]), shape=np.array(shape))
        done += 1
        print(f"  {tile} {ds}  Std 有效像元 {int((arr < FILL_MIN).sum())}", flush=True)
    cf.write_text(json.dumps(cache))
    print(f"补齐 {done} 片")


if __name__ == "__main__":
    main()
