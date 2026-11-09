"""
dataset.py —— SEN12MS-CR parquet 读取、裁剪、缓存、划分

设计要点：
1. 自动识别列名（不同镜像命名不同），只依赖 *_shape 配对规则。
2. 逐 batch 解码，避免把 2.4GB 分片一次性读进内存（本机可用内存仅 9.6GB）。
3. 云掩膜只由"有云影像"本身计算，不使用 target —— 否则推理时拿不到，属于数据泄漏。
4. 产出：
     data/processed/*.npz   每个 patch 一个（optical, sar, mask, target, meta）
     data/processed/split.json  固定划分，之后永不修改
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# Sentinel-2 13 波段的常见顺序（SEN12MS-CR 为 13 波段）
S2_BAND_NAMES = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8",
                 "B8A", "B9", "B10", "B11", "B12"]
# 指标计算可用的波段：排除 B1/B9/B10 三个大气波段。
# B10（1373nm，卷云）在 L1C TOA 里均值仅 0.0008，ERGAS 的分母 μ_b 会把它放大成
# 10^4 量级（实测 ERGAS 97841），PSNR/SAM 也被近零波段稀释，指标完全失去意义。
USABLE_IDX = [i for i, n in enumerate(S2_BAND_NAMES) if n not in ("B1", "B9", "B10")]
BAND_INDEX = {"B2": 1, "B3": 2, "B4": 3, "B8": 7}   # 13 波段全量时的索引
RGB_INDICES = [1, 2, 3, 7]                          # B2,B3,B4,B8


# ------------------------------------------------------------------ 工具
DTYPE_BY_ITEMSIZE = {1: np.uint8, 2: np.int16, 4: np.float32, 8: np.float64}


def _decode(buf, shape, dtype=None):
    """
    解码二进制列。dtype 若为 None 则按 字节数/prod(shape) 反推 itemsize，
    避免写死 int16 —— 不同镜像可能存 uint8 / int16 / float32。
    """
    a = np.frombuffer(buf, dtype=np.uint8)
    if shape is not None:
        n = int(np.prod([int(s) for s in shape]))
        if n > 0 and a.size % n == 0:
            isz = a.size // n
            real = DTYPE_BY_ITEMSIZE.get(isz)
            if real is not None:
                a = a.view(real)
            elif dtype is not None:
                a = a.view(dtype)
            else:
                raise RuntimeError(f"无法推断 dtype: itemsize={isz}")
            return a.reshape(tuple(int(s) for s in shape))
    return a.view(dtype) if dtype is not None else a


def _find_col(cols, *keys):
    """按关键字模糊匹配列名。"""
    for k in keys:
        for c in cols:
            if c.lower() == k.lower():
                return c
    for k in keys:
        for c in cols:
            if k.lower() in c.lower() and not c.endswith("_shape"):
                return c
    return None


def _shape_of(row, cols, base):
    for cand in (f"{base}_shape", "opt_shape", "img_shape"):
        if cand in cols:
            v = row[cand]
            return v if isinstance(v, (list, tuple, np.ndarray)) else None
    return None


# ------------------------------------------------------------------ 读取
def iter_samples(parquet_paths, batch_size=4):
    """
    逐条产出 dict：optical(C,H,W) float32 [0,1]、sar(2,H,W)、target(C,H,W)、
    season/scene/patch_id 元数据。数值已除以 scale 转成反射率。
    """
    for path in parquet_paths:
        try:
            pf = pq.ParquetFile(path)
        except Exception as e:
            print(f"  跳过无法读取的分片 {Path(path).name}（{type(e).__name__}，可能仍在下载）", flush=True)
            continue
        cols = pf.schema_arrow.names
        c_cloudy = _find_col(cols, "cloudy", "cloud", "optical_cloudy")
        c_target = _find_col(cols, "target", "clean", "clear", "gt")
        c_sar = _find_col(cols, "sar", "radar")
        if not (c_cloudy and c_target):
            raise RuntimeError(f"{path} 未找到 cloudy/target 列，实际列：{cols}")

        season = Path(path).stem.split("_")[0] if "scene" in Path(path).name else "unknown"
        season = Path(path).parent.name or season
        c_season = _find_col(cols, "season")
        c_patch = _find_col(cols, "patch", "patch_id", "name")

        gid = 0   # 全局序号：不能用批内索引 i —— batch_size=4 时 i 只在 0~3 循环，
                  # 会让 1500 个 patch 只生成 8 个不同文件名而互相覆盖（已踩过）
        for batch in pf.iter_batches(batch_size=batch_size):
            d = batch.to_pydict()
            n = len(d[cols[0]])
            for i in range(n):
                row = {k: d[k][i] for k in cols}
                shp = _shape_of(row, cols, c_cloudy)
                opt = _decode(row[c_cloudy], shp).astype(np.float32)
                tgt = _decode(row[c_target], _shape_of(row, cols, c_target)).astype(np.float32)
                sar = None
                if c_sar:
                    sar = _decode(row[c_sar], _shape_of(row, cols, c_sar)).astype(np.float32)

                # (H,W,C) 统一成 (C,H,W)
                if opt.ndim == 3 and opt.shape[0] > opt.shape[-1]:
                    opt = np.transpose(opt, (2, 0, 1))
                    tgt = np.transpose(tgt, (2, 0, 1))
                if sar is not None and sar.ndim == 3 and sar.shape[0] > sar.shape[-1]:
                    sar = np.transpose(sar, (2, 0, 1))

                row_season = str(row[c_season]) if c_season else season
                row_patch = str(row[c_patch]) if c_patch else f"{gid:05d}"
                gid += 1

                yield {
                    "optical": opt,
                    "target": tgt,
                    "sar": sar,
                    "season": row_season,
                    "scene": Path(path).stem,
                    "idx": row_patch,
                }


def warp_resize(a, size):
    """
    统一到 size×size。

    优先中心裁剪而不是缩放：256→128 的双线性缩放会把 2×2 邻域平均掉，
    相当于给所有方法都做了一次低通滤波，会系统性抬高 PSNR 并抹平方法间差异
    ——这对"比较指标之间是否一致"的实验是致命的。中心裁剪保留原始分辨率。
    """
    import cv2
    h, w = a.shape[-2], a.shape[-1]
    if h == size and w == size:
        return a
    if h >= size and w >= size:
        top, left = (h - size) // 2, (w - size) // 2
        return np.ascontiguousarray(a[:, top:top + size, left:left + size])
    out = np.empty((a.shape[0], size, size), np.float32)
    for c in range(a.shape[0]):
        out[c] = cv2.resize(a[c], (size, size), interpolation=cv2.INTER_LINEAR)
    return out


def make_cloud_mask(optical, threshold=0.15, band_for_mask=None):
    """
    只由有云影像生成云掩膜：可见光反射率超过阈值即判为云。
    不使用 target，保证推理时可用（无泄漏）。

    阈值 0.15 是在 scene_31 的 30 个 patch 上标定的（以 target 为参照，
    真云定义为 cloudy−target 的可见光最大反射率差 > 0.05）：
        0.12 → 精度 0.943 召回 1.000（过度分割 6%，会把晴空像元也交给基线去填）
        0.15 → 精度 1.000 召回 0.897   ← 采用
        0.20 → 精度 1.000 召回 0.719
    选精度 1.0 的一侧，是为了保证"被替换的像元确实是被云污染的"，
    否则基线会把晴空像元也改掉，人为制造误差，污染后续所有结论。

    早先版本用 0.55*p99 的自适应阈值，在低对比度 patch 上会把整幅判成云
    （实测出现 100% 云占比），已废弃。

    波段选择：13 波段时取 B2/B3/B4（索引 1,2,3）；索引 0 是 B1 海岸带波段，
    受瑞利散射影响大，不适合判云。
    """
    c = optical.shape[0]
    if c >= 13:
        idx = [1, 2, 3]
    elif c >= 4:
        idx = [0, 1, 2]
    else:
        idx = list(range(c))
    b = optical[idx].max(axis=0)
    m = b > threshold
    return m.astype(np.uint8)


def build_cache(raw_dir, cache_dir, patch_size, scale, max_patches, cloud_threshold):
    """把 parquet 转成 npz 缓存。返回 meta 列表。"""
    raw_dir, cache_dir = Path(raw_dir), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pqs = sorted(raw_dir.glob("*.parquet"))
    if not pqs:
        raise FileNotFoundError(f"{raw_dir} 下没有 parquet 文件")

    metas, count = [], 0
    seen = set()
    for s in iter_samples(pqs):
        if count >= max_patches:
            break
        try:
            opt = s["optical"] / scale
            tgt = s["target"] / scale
        except Exception as e:
            print(f"  跳过损坏样本 {s.get('scene')}:{s.get('idx')} ({type(e).__name__})", flush=True)
            continue
        sar = s["sar"] if s["sar"] is not None else np.zeros((2,) + opt.shape[1:], np.float32)

        # 保留全部波段：PSNR/SSIM 用可见光+近红外子集，下游反演需要红边 B5/B6/B7
        # （只用 B2/B3/B4/B8 时 LAI 与 Cab 不可分离）。波段顺序在 meta 里记录。
        n_band = opt.shape[0]

        opt = warp_resize(opt, patch_size)
        tgt = warp_resize(tgt, patch_size)
        # SAR 必须与光学同步裁剪，否则 sar.shape[-2:] != optical.shape[-2:]，
        # 会让所有 SAR 引导方法静默退化回 mean_fill（实测三个方法指标完全相同）。
        if sar.shape[-2:] != opt.shape[-2:]:
            sar = warp_resize(sar, patch_size)
        mask = make_cloud_mask(opt, cloud_threshold)

        # 名字必须全局唯一：season 取自数据自带的列，patch 取自数据自带的 patch 列，
        # 再拼上分片名，避免两个分片里同名 patch 互相覆盖。
        name = f"{s['season']}_{s['scene']}_{s['idx']}"
        assert name not in seen, f"缓存名重复：{name}"
        seen.add(name)
        np.savez_compressed(
            cache_dir / f"{name}.npz",
            optical=opt.astype(np.float32),
            target=tgt.astype(np.float32),
            mask=mask,
            sar=sar.astype(np.float32),
        )
        metas.append({
            "name": name,
            "season": s["season"],
            "scene": s["scene"],
            "n_band": int(n_band),
            "cloud_fraction": float(mask.mean()),
            "mean_brightness": float(opt.mean()),
        })
        count += 1
        if count % 50 == 0:
            print(f"  已处理 {count} 个 patch ...", flush=True)

    print(f"缓存完成，共 {count} 个 patch -> {cache_dir}")
    return metas


def make_split(metas, out_json, val_ratio, test_ratio, seed):
    """按固定 seed 划分 train/val/test，写入 json。"""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(metas))
    n = len(metas)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    split = {
        "test": [metas[i]["name"] for i in idx[:n_test]],
        "val": [metas[i]["name"] for i in idx[n_test:n_test + n_val]],
        "train": [metas[i]["name"] for i in idx[n_test + n_val:]],
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2, ensure_ascii=False)
    print(f"划分完成：train {len(split['train'])} / val {len(split['val'])} / test {len(split['test'])}")
    print(f"  -> {out_json}")
    return split


def filter_metas(metas, lo, hi, verbose=True):
    """
    剔除云量过少（几乎无云，去云任务无从评价）和过多（无晴空锚点，
    基线方法全部退化为常量填充，误差被人为放大）的 patch。

    实测 1500 个 patch 中：云量<2% 有 106 个、>95% 有 171 个，
    两端都留着会让"云量分层"这一层分析失去意义。
    """
    keep = [m for m in metas if lo <= m["cloud_fraction"] <= hi]
    if verbose:
        print(f"云量过滤 [{lo:.2f}, {hi:.2f}]：{len(metas)} -> {len(keep)}"
              f"（剔除 {len(metas)-len(keep)} 个）")
    return keep


# ------------------------------------------------------------------ 读缓存
def load_patch(cache_dir, name):
    z = np.load(Path(cache_dir) / f"{name}.npz")
    return {k: z[k] for k in z.files}
