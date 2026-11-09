"""
search_s2_pairs.py —— 在 Copernicus Data Space 上检索 Sentinel-2 L2A 的"有云/晴空"配对

只做检索与选景，不需要账号（catalogue 的 OData 接口是公开的）。
真正下载需要登录 token，见 download_s2.py。

配对逻辑（云去除数据集的标准做法）：
  同一个 tile、同一片区域，挑一景「云量高」和一景「云量低」，两者日期尽量接近。
  日期越近，地表变化（物候、耕作）越小，晴空景才配得上"真值"这个身份。

用法：
    # 北京一带，2022 年 6–9 月
    python search_s2_pairs.py --bbox 115.8,39.5,116.9,40.4 --start 2022-06-01 --end 2022-09-30
    # 换区域：长江中下游
    python search_s2_pairs.py --bbox 113.0,29.5,115.0,31.5 --start 2022-03-01 --end 2022-11-30

输出：
    results_meta/s2_products.csv     所有检索到的产品（含云量）
    results_meta/s2_pairs.csv        选出的配对（每行一对）
"""
from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
UA = {"User-Agent": "cloudrs-research/1.0"}


def query(bbox, start, end, max_cloud=100.0, limit=1000):
    """按 bbox + 时间 + 云量上限检索 S2 L2A。分页取完。"""
    w, s, e, n = bbox
    poly = f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"
    flt = (
        "Collection/Name eq 'SENTINEL-2' "
        "and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
        "and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A') "
        f"and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' "
        f"and att/OData.CSC.DoubleAttribute/Value le {max_cloud:.1f}) "
        f"and ContentDate/Start gt {start}T00:00:00.000Z "
        f"and ContentDate/Start lt {end}T23:59:59.000Z "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{poly}')"
    )
    out, skip = [], 0
    while skip < limit:
        q = urllib.parse.urlencode({
            "$filter": flt, "$top": min(100, limit - skip), "$skip": skip,
            "$orderby": "ContentDate/Start asc",
            "$expand": "Attributes",
        })
        req = urllib.request.Request(f"{BASE}?{q}", headers=UA)
        with urllib.request.urlopen(req, timeout=90) as r:
            d = json.load(r)
        v = d.get("value", [])
        if not v:
            break
        for x in v:
            attrs = {a["Name"]: a.get("Value") for a in x.get("Attributes", [])}
            out.append({
                "name": x["Name"],
                "id": x["Id"],
                "start": x["ContentDate"]["Start"][:19],
                "cloud": float(attrs.get("cloudCover", 100.0) or 100.0),
                "tile": attrs.get("tileId") or x["Name"].split("_")[-2],
                "size_mb": round((x.get("ContentLength") or 0) / 1e6, 1),
                "online": x.get("Online", True),
            })
        skip += len(v)
        print(f"  已取 {len(out)} 条 ...", flush=True)
        if len(v) < 100:
            break
    return pd.DataFrame(out)


def pick_pairs(df, cloud_hi=50.0, cloud_lo=10.0, max_gap=15, per_tile=3):
    """
    每个 tile 内挑配对：a=高云（≥cloud_hi），b=低云（≤cloud_lo），日期差 ≤ max_gap 天。
    优先选日期差最小的组合；同一 tile 最多输出 per_tile 对，避免全挤在一个地区。
    """
    rows = []
    if df.empty:
        return pd.DataFrame(rows)
    df = df.copy()
    df["t"] = pd.to_datetime(df.start)
    for tile, g in df.groupby("tile"):
        cloudy = g[g.cloud >= cloud_hi].sort_values("t")
        clear = g[g.cloud <= cloud_lo].sort_values("t")
        if cloudy.empty or clear.empty:
            continue
        cands = []
        for _, a in cloudy.iterrows():
            for _, b in clear.iterrows():
                gap = abs((a.t - b.t).days)
                if gap <= max_gap:
                    cands.append((gap, a, b))
        cands.sort(key=lambda x: x[0])
        used = set()
        for gap, a, b in cands:
            if a["id"] in used or b["id"] in used:
                continue
            used.add(a["id"]); used.add(b["id"])
            rows.append({
                "tile": tile,
                "cloudy_name": a["name"], "cloudy_date": a["start"], "cloudy_cover": a["cloud"],
                "clear_name": b["name"], "clear_date": b["start"], "clear_cover": b["cloud"],
                "gap_days": gap, "cloudy_id": a["id"], "clear_id": b["id"],
            })
            if len([r for r in rows if r["tile"] == tile]) >= per_tile:
                break
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", required=True, help="west,south,east,north（经纬度）")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--max-cloud", type=float, default=100.0)
    ap.add_argument("--cloud-hi", type=float, default=50.0)
    ap.add_argument("--cloud-lo", type=float, default=10.0)
    ap.add_argument("--max-gap", type=int, default=15)
    ap.add_argument("--per-tile", type=int, default=3)
    ap.add_argument("--out", default="results_meta")
    args = ap.parse_args()

    bbox = [float(x) for x in args.bbox.split(",")]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    print(f"检索 bbox={bbox} {args.start} ~ {args.end} ...")
    df = query(bbox, args.start, args.end, args.max_cloud)
    if df.empty:
        print("没检索到任何产品。放宽时间范围或云量条件再试。")
        return
    df.to_csv(out / "s2_products.csv", index=False)
    print(f"\n共 {len(df)} 景 | tile {df.tile.nunique()} 个 | "
          f"云量中位 {df.cloud.median():.1f}%")
    print(df.groupby("tile").agg(景数=("name", "size"), 最低云量=("cloud", "min"),
                                 最高云量=("cloud", "max")).to_string())

    pairs = pick_pairs(df, args.cloud_hi, args.cloud_lo, args.max_gap, args.per_tile)
    if pairs.empty:
        print("\n没凑出配对。建议：--cloud-hi 降到 40，或 --max-gap 放宽到 30。")
        return
    pairs.to_csv(out / "s2_pairs.csv", index=False)
    print(f"\n选出 {len(pairs)} 对：")
    print(pairs[["tile", "cloudy_date", "cloudy_cover", "clear_date",
                 "clear_cover", "gap_days"]].to_string(index=False))
    print(f"\n-> {out/'s2_pairs.csv'}")
    print("下一步：注册 Copernicus 账号，把 s2_pairs.csv 交给 download_s2.py 下载。")


if __name__ == "__main__":
    main()
