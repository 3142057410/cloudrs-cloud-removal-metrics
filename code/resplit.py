"""
resplit.py —— 只根据已有 meta.json 重新划分，不重建 npz 缓存

用途：改了云量过滤范围或划分比例后，不必重跑 3 分钟的缓存构建。
用法：python src/resplit.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from dataset import filter_metas, make_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p1.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    cache = Path(cfg["data"]["cache_dir"])

    metas = json.load(open(cache / "meta.json", encoding="utf-8"))
    metas = filter_metas(
        metas,
        cfg["data"].get("min_cloud_fraction", 0.02),
        cfg["data"].get("max_cloud_fraction", 0.95),
    )
    make_split(
        metas,
        cache / "split.json",
        val_ratio=cfg["split"]["val_ratio"],
        test_ratio=cfg["split"]["test_ratio"],
        seed=cfg["split"]["seed"],
    )


if __name__ == "__main__":
    main()
