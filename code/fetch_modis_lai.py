"""
fetch_modis_lai.py —— 抓取 MCD15A3H（MODIS LAI/FPAR，4 天合成，500 m）作为产品级参照

为什么走 Microsoft Planetary Computer：
    本机没有 Earthdata 令牌，也没有 GEE 账号；LP DAAC 的 HTTPS 下载必须登录。
    PC 的 STAC 目录匿名可读，且 modis-15A3H-061 就是 MCD15A3H 原产品
    （Collection 6.1，500 m，4 天），带 Lai_500m / FparLai_QC / LaiStdDev_500m。
    第三条最有用：LaiStdDev 是逐像元不确定度，误差预算不必再套文献常数。

踩过的坑（别回头再踩）：
    /api/sas/v1/token/{collection} 的容器级令牌在本机代理下签名校验必然失败
    （Azure 报 "Signature did not match"；令牌 sig 含 %2B，经代理后编码被改写）。
    改用逐 URL 的 /api/sas/v1/sign?href=... 后正常，本脚本只用后者。

投影处理：
    MODIS 原生 Sinusoidal。重投影到各 S2 瓦片所在 UTM 50N 的 **固定 500 m 网格**
    （原点按 500 m 取整），所有日期共用同一网格，时序分析无需再对齐。
    重采样用 nearest —— 500 m 之间插值等于凭空造值，只会把不确定度藏起来。

用法：
    python src/fetch_modis_lai.py --geo results_meta/patch_geo.csv \
        --start 2022-05-20 --end 2022-09-10 --out data/modis_lai
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform as warp_transform
from rasterio.windows import from_bounds, Window

COLLECTION = "modis-15A3H-061"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
SIGN = "https://planetarycomputer.microsoft.com/api/sas/v1/sign?href="
CRS_UTM = "EPSG:32650"
GRID_M = 500.0
LAI_SCALE = 0.1
FILL_MIN = 249          # 249/250/255 为填充或未生产
SIGN_TTL = 1500         # 签名 URL 约 1 h 有效，缓存 25 min
COMPOSITE_DAYS = 4      # MCD15A3H 合成期长度，用于 S2 日期配对


# --------------------------------------------------------------------- 网络

def _fetch(req, timeout=90, retries=4):
    last = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:                      # 代理偶发 502，退避重试
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"请求失败：{last}")


def sign(href: str, cache: dict) -> str:
    now = time.time()
    hit = cache.get(href)
    if hit and now - hit[0] < SIGN_TTL:
        return hit[1]
    raw = _fetch(SIGN + urllib.parse.quote(href, safe=""))
    signed = json.loads(raw)["href"]
    cache[href] = [now, signed]
    return signed


def search(bbox, start: str, end: str):
    body = {"collections": [COLLECTION], "bbox": [float(v) for v in bbox],
            "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": 300}
    req = urllib.request.Request(STAC + "/search", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(_fetch(req))["features"]


def item_date(item):
    """MCD15A3H.A2022209.h27v05.061.xxx -> 2022-07-28"""
    doy = item["id"].split(".")[1]
    return date(int(doy[1:5]), 1, 1) + timedelta(days=int(doy[5:]) - 1)


# --------------------------------------------------------------------- 读窗

def read_to_grid(url: str, dst_transform, dst_shape, is_qc=False):
    """读 COG 中覆盖目标网格的窗口，最近邻重投影到目标网格。

    坑：目标网格可能整体落在该 MODIS 分幅之外（AOI 跨越 h/v 接缝时常见），
    此时 from_bounds 得到的窗口虽然有 1 像素宽，实际 read 会返回 0 列数组，
    reproject 就抛 "Invalid dataset dimensions : 0 x N"。所以必须先把窗口
    裁进数据范围内再判空，不能只判 width<1。
    """
    with rasterio.open(url) as ds:
        left, top = dst_transform.c, dst_transform.f
        right = left + dst_transform.a * dst_shape[1]
        bottom = top + dst_transform.e * dst_shape[0]
        xs, ys = warp_transform(CRS_UTM, ds.crs,
                                [left, right, left, right],
                                [bottom, bottom, top, top])
        win = from_bounds(min(xs), min(ys), max(xs), max(ys), ds.transform)
        win = win.round_offsets().round_lengths()
        win = win.intersection(Window(0, 0, ds.width, ds.height))
        if win.width < 1 or win.height < 1:
            return None
        src = ds.read(1, window=win, boundless=False)
        if src.size == 0 or src.shape[0] == 0 or src.shape[1] == 0:
            return None
        src_tr = rasterio.windows.transform(win, ds.transform)
        # reproject 的 destination 必须是实数组 —— 传 .astype() 的副本会静默丢结果
        out = np.full(dst_shape, 255, np.int16)
        reproject(
            source=src.astype(np.int16), destination=out,
            src_transform=src_tr, src_crs=ds.crs,
            dst_transform=dst_transform, dst_crs=CRS_UTM,
            src_nodata=ds.nodata, dst_nodata=255,
            resampling=Resampling.nearest,
        )
        return out.astype(np.uint8 if is_qc else np.int16)


def build_grid(g: pd.DataFrame, margin_m: float):
    """以该瓦片 patch 足迹的并集建固定 500 m 网格。"""
    half = float(g["footprint_m"].iloc[0]) / 2.0
    x0 = np.floor((g.x_utm.min() - half - margin_m) / GRID_M) * GRID_M
    x1 = g.x_utm.max() + half + margin_m
    y0 = np.floor((g.y_utm.min() - half - margin_m) / GRID_M) * GRID_M
    y1 = g.y_utm.max() + half + margin_m
    w = int(np.ceil((x1 - x0) / GRID_M))
    h = int(np.ceil((y1 - y0) / GRID_M))
    tr = rasterio.Affine(GRID_M, 0.0, x0, 0.0, -GRID_M, y0 + h * GRID_M)
    return tr, (h, w)


# --------------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geo", default="results_meta/patch_geo.csv")
    ap.add_argument("--start", default="2022-05-20")
    ap.add_argument("--end", default="2022-09-10")
    ap.add_argument("--out", default="data/modis_lai")
    ap.add_argument("--margin", type=float, default=1500.0)
    a = ap.parse_args()

    geo = pd.read_csv(a.geo)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cf = out / "_sign_cache.json"
    cache = json.loads(cf.read_text()) if cf.exists() else {}

    target_dates = set(geo["date"].unique())
    rows, written = [], 0

    for tile, g in geo.groupby("tile"):
        tr, shape = build_grid(g, a.margin)
        xs, ys = warp_transform(CRS_UTM, "EPSG:4326",
                                [tr.c, tr.c + tr.a * shape[1]],
                                [tr.f + tr.e * shape[0], tr.f])
        bbox = (min(xs), min(ys), max(xs), max(ys))
        items = search(bbox, a.start, a.end)
        by_date = defaultdict(list)
        for it in items:
            by_date[item_date(it)].append(it)
        print(f"\n=== {tile}  网格 {shape[0]}x{shape[1]} @500m  "
              f"bbox=({bbox[0]:.3f},{bbox[1]:.3f},{bbox[2]:.3f},{bbox[3]:.3f})  "
              f"时相 {len(by_date)}", flush=True)

        for d in sorted(by_date):
            lai = np.full(shape, 255, np.uint8)
            qc = np.full(shape, 255, np.uint8)
            std = np.full(shape, 255, np.uint8)
            ok = False
            for it in by_date[d]:
                try:
                    a_lai = read_to_grid(sign(it["assets"]["Lai_500m"]["href"], cache), tr, shape)
                    a_qc = read_to_grid(sign(it["assets"]["FparLai_QC"]["href"], cache),
                                        tr, shape, is_qc=True)
                except Exception as e:
                    print(f"    {d} 读取失败 {it['id']}: {type(e).__name__} {e}", flush=True)
                    continue
                if a_lai is None:
                    continue
                m = a_lai < FILL_MIN
                lai[m] = a_lai[m].astype(np.uint8)
                qc[m] = a_qc[m].astype(np.uint8)
                ok = True
                if any(0 <= (date.fromisoformat(x) - d).days < COMPOSITE_DAYS
                       for x in target_dates):
                    try:
                        a_std = read_to_grid(sign(it["assets"]["LaiStdDev_500m"]["href"], cache),
                                             tr, shape)
                        ms = (a_std < FILL_MIN) & m
                        std[ms] = a_std[ms].astype(np.uint8)
                    except Exception:
                        pass
            if not ok:
                continue

            valid = lai < FILL_MIN
            modland = (qc & 0b1) == 0                    # 主算法（含饱和）
            scf = (qc >> 1) & 0b11                       # 0 最好 … 3 未生产
            good = valid & modland & (scf <= 1)          # 只保留 best / good
            np.savez_compressed(out / f"{tile}_{d}.npz",
                                lai_dn=lai, qc=qc, std_dn=std,
                                transform=np.array(tr, dtype=np.float64),
                                shape=np.array(shape))
            written += 1
            covers = [x for x in target_dates
                      if 0 <= (date.fromisoformat(x) - d).days < COMPOSITE_DAYS]
            rows.append({"tile": tile, "date": str(d), "item": it["id"],
                         "n_px": int(valid.sum()), "n_good": int(good.sum()),
                         "frac_good": round(float(good.sum()) / max(int(valid.sum()), 1), 4),
                         "lai_med": round(float(np.median(lai[good])) * LAI_SCALE, 2)
                                     if good.any() else np.nan,
                         "is_target": bool(covers),
                         "covers_s2": ",".join(sorted(covers))})
            if written % 15 == 0:
                cf.write_text(json.dumps(cache))
                print(f"    … 已写 {written} 片", flush=True)
    cf.write_text(json.dumps(cache))

    df = pd.DataFrame(rows)
    df.to_csv(out / "modis_index.csv", index=False)
    print(f"\n写入 {written} 片 -> {out}")
    if len(df):
        tg = df[df.is_target]
        print(f"\n目标日期覆盖：{len(tg)} 条")
        print(tg[["tile", "date", "frac_good", "lai_med"]].to_string(index=False))
        print(f"\n全序列 good 占比：中位 {df.frac_good.median():.3f}  最低 {df.frac_good.min():.3f}")


if __name__ == "__main__":
    main()
