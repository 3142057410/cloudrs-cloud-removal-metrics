"""
build_s2_dataset.py —— 把下载好的 Sentinel-2 L2A 波段切成与外采缓存同格式的 patch

做四件事：
  1) 同一对（有云景 / 晴空景）的波段对齐到统一的 10m 栅格
     10m 波段（B2/B3/B4/B8）作为参考格网，20m 波段（B5/B6/B7/B8A/B11/B12/SCL）重采样上来。
  2) 从影像中心裁一块 1024×1024（10m，约 10 km），避免边缘的 NaN 与投影缝隙
  3) 切成 128×128 的不重叠 patch
  4) 用 SCL 云分类生成掩膜（8/9/10 = 云），存成与现有缓存一致的 npz

关键的兼容处理：
  现有 dataset.py 期望 13 个波段（B1..B12 顺序）。我们只下 10 个可用波段，
  而 B1/B9/B10 恰好是**被指标和反演排除**的三个（大气/水汽/卷云波段）。
  所以这里把 13 通道数组的这三个位置填 0，下游代码一行都不用改。

用法：
    python src/build_s2_dataset.py --pairs results_meta/s2_pairs.csv \
        --raw data/s2_raw --out data/s2_processed --crop 1024
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.warp import reproject

# 13 波段顺序里，我们实际拥有的是这 10 个（索引与 dataset.py 的 S2_BAND_NAMES 对齐）
S2_ORDER = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
HAVE = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
CLOUD_SCL = (8, 9, 10)      # 中等概率云 / 高概率云 / 薄卷云
# 处理基线 >= 04.00 的 L2A 产品需加回 -1000 的偏移（DN 单位）
BOA_OFFSET_DN = -1000.0
SHADOW_SCL = (3,)


def load_product(path: Path, crop: int, boa_off: float = BOA_OFFSET_DN):
    """读一个产品，只读中心裁剪窗口，不拼整幅 13 波段立方体。

    踩过的坑：整幅 10980x10980 的 13 波段 float32 立方体要 5.84 GiB，
    连着处理几对就 MemoryError，后四对直接丢光（数据集从 156 个 patch 掉到 79 个）。
    改成按裁剪窗口读之后，峰值内存降到百 MB 量级。
    """
    files = {b: path / f"{b}.jp2" for b in HAVE + ["SCL"]}
    for b, f in files.items():
        if not f.exists():
            raise FileNotFoundError(f)

    with rasterio.open(files["B2"]) as ds:          # B2 是 10 m，作参考格网
        ref_tr, W, H, crs = ds.transform, ds.width, ds.height, ds.crs
    if H < crop or W < crop:
        raise ValueError(f"影像小于裁剪尺寸 {H}x{W} < {crop}")
    y0, x0 = (H - crop) // 2, (W - crop) // 2
    crop_tr = ref_tr * rasterio.Affine.translation(x0, y0)
    win = Window(x0, y0, crop, crop)

    cube = np.zeros((13, crop, crop), np.float32)
    scl = None
    for b, f in files.items():
        with rasterio.open(f) as ds:
            if ds.width == W and ds.height == H and ds.transform == ref_tr:
                arr = ds.read(1, window=win).astype(np.float32)
            else:
                # 20 m 波段直接重投影到裁剪窗口，不经过整幅中间数组
                arr = np.zeros((crop, crop), np.float32)
                reproject(
                    source=rasterio.band(ds, 1), destination=arr,
                    src_transform=ds.transform, src_crs=ds.crs,
                    dst_transform=crop_tr, dst_crs=crs,
                    resampling=Resampling.nearest if b == "SCL" else Resampling.bilinear,
                )
        if b == "SCL":
            scl = arr.astype(np.int16)
        else:
            # L2A 反射率 = (DN + BOA_ADD_OFFSET) / QUANTIFICATION_VALUE。
            # 处理基线 04.00（2022-01-25）起 BOA_ADD_OFFSET = -1000，此前为 0。
            # 只除 10000 会把所有波段整体抬高 0.1，植被红波段从 0.03 变成 0.13，
            # 送进 PROSAIL 检索会一路塌到 LAI ~ 0.1 —— 这个坑踩过一次，整条反演链失真。
            cube[S2_ORDER.index(b)] = (arr + boa_off) / 10000.0
    return cube, scl


def center_crop(cube, scl, size):
    """中心裁剪。scl 允许为 None（晴空景不需要云分类），此时只回 cube。

    这里踩过一次：对晴空景传 None，原来直接 scl[...] 切片，全 8 对都抛
    TypeError: 'NoneType' object is not subscriptable，夜里整条链路空转。
    """
    _, H, W = cube.shape
    if H < size or W < size:
        raise ValueError(f"影像小于裁剪尺寸 {H}×{W} < {size}")
    y0, x0 = (H - size) // 2, (W - size) // 2
    cube_out = cube[:, y0:y0 + size, x0:x0 + size]
    scl_out = scl[y0:y0 + size, x0:x0 + size] if scl is not None else None
    return cube_out, scl_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="results_meta/s2_pairs.csv")
    ap.add_argument("--raw", default="data/s2_raw")
    ap.add_argument("--out", default="data/s2_processed")
    ap.add_argument("--crop", type=int, default=1024, help="中心裁剪边长（10m 像素）")
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--min-cloud", type=float, default=0.02)
    ap.add_argument("--max-cloud", type=float, default=0.95)
    ap.add_argument("--append-meta", action="store_true",
                    help="分块构建时把结果并入已有 meta.json，而不是覆盖")
    ap.add_argument("--boa-offset-dn", type=float, default=BOA_OFFSET_DN,
                    help="L2A 偏移，基线>=04.00 为 -1000；老产品填 0")
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(args.pairs)

    meta, n_ok, n_skip = [], 0, 0
    for i, row in pairs.iterrows():
        cname = row.cloudy_name.replace(".SAFE", "")
        lname = row.clear_name.replace(".SAFE", "")
        cd, ld = raw / cname, raw / lname
        if not cd.exists() or not ld.exists():
            print(f"[{i+1}/{len(pairs)}] 缺数据，跳过 {row.tile}")
            n_skip += 1
            continue
        try:
            cube_c, scl_c = load_product(cd, args.crop, args.boa_offset_dn)
            cube_l, _ = load_product(ld, args.crop, args.boa_offset_dn)
        except Exception as e:
            import traceback
            print(f"[{i+1}/{len(pairs)}] {row.tile} 读取失败：{type(e).__name__} {e}")
            traceback.print_exc()
            n_skip += 1
            continue

        cloud = np.isin(scl_c, CLOUD_SCL)
        P = args.patch
        k = 0
        for yy in range(0, args.crop - P + 1, P):
            for xx in range(0, args.crop - P + 1, P):
                m = cloud[yy:yy + P, xx:xx + P]
                cf = float(m.mean())
                if cf < args.min_cloud or cf > args.max_cloud:
                    continue
                name = f"{row.tile}_{cname.split('_')[2]}_{k:03d}"
                np.savez_compressed(
                    out / f"{name}.npz",
                    optical=cube_c[:, yy:yy + P, xx:xx + P].astype(np.float32),
                    target=cube_l[:, yy:yy + P, xx:xx + P].astype(np.float32),
                    mask=m.astype(np.uint8),
                    # 没有下 S1，这里给全零占位，方便沿用同一套加载代码
                    sar=np.zeros((2, P, P), np.float32),
                )
                meta.append({"name": name, "tile": row.tile,
                             "cloudy_date": row.cloudy_date, "clear_date": row.clear_date,
                             "gap_days": int(row.gap_days), "cloud_fraction": round(cf, 4),
                             "season": "s2_l2a"})
                k += 1
        n_ok += 1
        print(f"[{i+1}/{len(pairs)}] {row.tile} {row.cloudy_date[:10]} -> 产出 {k} 个 patch")
        # 每对处理完显式释放：GDAL 的块缓存会随对次累积，不清理后几对会 MemoryError
        del cube_c, cube_l, scl_c, cloud
        gc.collect()

    mf = out / "meta.json"
    if args.append_meta and mf.exists():
        old = json.load(open(mf, encoding="utf-8"))
        have = {x["name"] for x in old}
        meta = old + [x for x in meta if x["name"] not in have]
    json.dump(meta, open(mf, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n完成：{n_ok} 对可用 / {n_skip} 对跳过 | 共 {len(meta)} 个 patch")
    if meta:
        cf = np.array([m["cloud_fraction"] for m in meta])
        print(f"云量分布：中位 {np.median(cf):.3f}  5% {np.percentile(cf,5):.3f}  95% {np.percentile(cf,95):.3f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
