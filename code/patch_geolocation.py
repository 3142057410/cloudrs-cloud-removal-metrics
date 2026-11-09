"""
patch_geolocation.py —— 还原 data/s2_processed 里每个 patch 的地理位置

为什么需要这一步：
    build_s2_dataset.py 只把 cube/mask 存进 npz，没有存 (yy, xx) 或经纬度。
    而 patch 编号 k 是在"云量过滤之后"才递增的，所以 k 不直接等于网格序号，
    无法用 k // 8、k % 8 反推。唯一可靠的办法是按原样重放一遍那个循环：
    同样的 pairs 顺序、同样的 20m SCL -> 10m 最近邻重采样、同样的中心裁剪、
    同样的云量阈值 (0.02, 0.95)。

    重放完必须与 data/s2_processed/meta.json 的 cloud_fraction 逐条比对，
    对不上就说明重放逻辑与原构建不一致，后面的 MODIS 取样全是错的 —— 所以
    这里把校验做成硬性断言，不通过直接退出。

输出：results_meta/patch_geo.csv
    name, tile, pair_idx, date, yy, xx, x_utm, y_utm, lon, lat,
    px_size, patch_px, footprint_m
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform as warp_transform

CLOUD_SCL = (8, 9, 10)
PATCH_PX = 128
CROP = 1024
MIN_CLOUD, MAX_CLOUD = 0.02, 0.95


def _open_band(path: Path, ref, is_scl: bool):
    """读一个波段并对齐到 ref 格网。ref=None 时用它建立格网。"""
    with rasterio.open(path) as ds:
        if ref is None:
            arr = ds.read(1).astype(np.float32)
            return arr, (ds.transform, ds.width, ds.height, ds.crs)
        tr, W, H, crs = ref
        if ds.width == W and ds.height == H and ds.transform == tr:
            return ds.read(1).astype(np.float32), ref
        arr = np.zeros((H, W), np.float32)
        reproject(
            source=rasterio.band(ds, 1), destination=arr,
            src_transform=ds.transform, src_crs=ds.crs,
            dst_transform=tr, dst_crs=crs,
            resampling=Resampling.nearest if is_scl else Resampling.bilinear,
        )
        return arr, ref


def replay(raw: Path, pairs: pd.DataFrame, crop: int, patch: int):
    rows = []
    for i, r in pairs.iterrows():
        cname = str(r.cloudy_name).replace(".SAFE", "")
        cdir = raw / cname
        if not (cdir / "B2.jp2").exists() or not (cdir / "SCL.jp2").exists():
            print(f"[{i+1}/{len(pairs)}] 缺 {cname}，跳过", flush=True)
            continue

        b2, ref = _open_band(cdir / "B2.jp2", None, False)
        scl, _ = _open_band(cdir / "SCL.jp2", ref, True)
        tr, W, H, crs = ref

        y0, x0 = (H - crop) // 2, (W - crop) // 2
        scl = scl[y0:y0 + crop, x0:x0 + crop]
        cloud = np.isin(scl.astype(np.int16), CLOUD_SCL)

        k = 0
        for yy in range(0, crop - patch + 1, patch):
            for xx in range(0, crop - patch + 1, patch):
                m = cloud[yy:yy + patch, xx:xx + patch]
                cf = float(m.mean())
                if cf < MIN_CLOUD or cf > MAX_CLOUD:
                    continue
                # patch 中心在整景中的像素坐标
                cy = y0 + yy + patch / 2.0
                cx = x0 + xx + patch / 2.0
                xu = tr.c + cx * tr.a
                yu = tr.f + cy * tr.e
                lon, lat = warp_transform(crs, "EPSG:4326", [xu], [yu])
                rows.append({
                    "name": f"{r.tile}_{cname.split('_')[2]}_{k:03d}",
                    "tile": r.tile,
                    "pair_idx": i,
                    "date": str(r.cloudy_date)[:10],
                    "yy": yy, "xx": xx,
                    "x_utm": round(xu, 1), "y_utm": round(yu, 1),
                    "lon": round(float(lon[0]), 6), "lat": round(float(lat[0]), 6),
                    "px_size": 10.0,
                    "patch_px": patch,
                    "footprint_m": patch * abs(tr.a),
                    "cloud_fraction": round(cf, 4),
                })
                k += 1
        print(f"[{i+1}/{len(pairs)}] {r.tile} {str(r.cloudy_date)[:10]} -> {k} patch", flush=True)
    return pd.DataFrame(rows)


def verify(geo: pd.DataFrame, meta_path: Path):
    """与构建时的 meta.json 逐条对齐：名字、顺序、云量都必须一致。"""
    meta = json.load(open(meta_path, encoding="utf-8"))
    mdf = pd.DataFrame(meta)
    if len(mdf) != len(geo):
        raise SystemExit(f"数量不一致：meta {len(mdf)} vs 重放 {len(geo)}")
    if list(mdf["name"]) != list(geo["name"]):
        bad = [(a, b) for a, b in zip(mdf["name"], geo["name"]) if a != b][:5]
        raise SystemExit(f"patch 名字序列不一致，前 5 个差异：{bad}")
    diff = (mdf["cloud_fraction"].astype(float) - geo["cloud_fraction"].astype(float)).abs()
    if diff.max() > 5e-4:
        j = int(diff.idxmax())
        raise SystemExit(
            f"云量校验失败：最大偏差 {diff.max():.6f} @ {mdf['name'][j]} "
            f"(meta {mdf['cloud_fraction'][j]} vs 重放 {geo['cloud_fraction'][j]})"
        )
    print(f"校验通过：{len(geo)} 个 patch，名字序列与云量全部吻合（最大偏差 {diff.max():.2e}）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="results_meta/s2_pairs.csv")
    ap.add_argument("--raw", default="data/s2_raw")
    ap.add_argument("--processed", default="data/s2_processed")
    ap.add_argument("--out", default="results_meta/patch_geo.csv")
    ap.add_argument("--crop", type=int, default=CROP)
    ap.add_argument("--patch", type=int, default=PATCH_PX)
    a = ap.parse_args()

    pairs = pd.read_csv(a.pairs)
    geo = replay(Path(a.raw), pairs, a.crop, a.patch)
    verify(geo, Path(a.processed) / "meta.json")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    geo.to_csv(out, index=False)
    print(f"\n{len(geo)} 个 patch -> {out}")
    print(f"经纬度范围：lon {geo.lon.min():.3f}–{geo.lon.max():.3f}  "
          f"lat {geo.lat.min():.3f}–{geo.lat.max():.3f}")
    print(f"patch 足迹 {geo.footprint_m.iloc[0]:.0f} m，日期 {geo.date.nunique()} 个："
          f"{sorted(geo.date.unique())}")


if __name__ == "__main__":
    main()
