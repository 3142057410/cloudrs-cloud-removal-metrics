"""
build_cache.py —— 把 parquet 分片转成 npz 缓存并生成固定划分

用法：
  python src/build_cache.py --config configs/p1.yaml
  python src/build_cache.py --config configs/p1.yaml --max-patches 50   # 冒烟测试
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from dataset import build_cache, filter_metas, make_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p1.yaml")
    ap.add_argument("--max-patches", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    cache_dir = Path(cfg["data"]["cache_dir"])
    max_patches = args.max_patches or cfg["data"]["max_patches"]

    metas = build_cache(
        raw_dir=cfg["data"]["raw_dir"],
        cache_dir=cache_dir,
        patch_size=cfg["data"]["patch_size"],
        scale=cfg["data"]["scale"],
        max_patches=max_patches,
        cloud_threshold=cfg["data"]["cloud_threshold"],
    )

    with open(cache_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(metas, f, indent=2, ensure_ascii=False)

    metas = filter_metas(
        metas,
        cfg["data"].get("min_cloud_fraction", 0.02),
        cfg["data"].get("max_cloud_fraction", 0.95),
    )
    make_split(
        metas,
        cache_dir / "split.json",
        val_ratio=cfg["split"]["val_ratio"],
        test_ratio=cfg["split"]["test_ratio"],
        seed=cfg["split"]["seed"],
    )


if __name__ == "__main__":
    main()
