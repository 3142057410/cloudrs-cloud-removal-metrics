"""
run_baselines.py —— 对所有基线方法做推理并计算指标

产出：
  results/tables/per_patch_metrics.csv   每个方法 × 每个 patch 的全部指标
  results/tables/method_summary.csv      方法级别汇总（mean ± std）

用法：
  python src/run_baselines.py --config configs/p1.yaml
  python src/run_baselines.py --config configs/p1.yaml --methods zero_fill,mean_fill
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import METHODS
from dataset import USABLE_IDX, load_patch
from metrics import evaluate_all


def load_cfg(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def band_selector(cfg):
    """波段选择器：usable 排除 B1/B9/B10 三个大气波段，all 保留全部 13 个。"""
    if cfg["metrics"].get("bands", "usable") == "all":
        return lambda a: a
    idx = USABLE_IDX
    return lambda a: a[idx] if a.shape[0] > max(idx) else a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p1.yaml")
    ap.add_argument("--methods", default=None, help="逗号分隔，缺省用配置里的全部")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 个 patch，用于冒烟测试")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    cache_dir = Path(cfg["data"]["cache_dir"])
    split = json.load(open(cache_dir / "split.json", encoding="utf-8"))
    names = split["test"][: args.limit] if args.limit else split["test"]

    methods = args.methods.split(",") if args.methods else cfg["methods"]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise SystemExit(f"未知方法 {unknown}，可选：{list(METHODS)}")

    print("=" * 68)
    print(f"测试 patch 数 {len(names)} | 方法 {len(methods)} 个 | 分辨率 {cfg['data']['patch_size']}")
    print(f"指标波段：{cfg['metrics'].get('bands', 'usable')}")
    print("=" * 68)

    sel = band_selector(cfg)
    rows = []
    t_all = time.time()
    for mi, mname in enumerate(methods, 1):
        fn = METHODS[mname]
        t0 = time.time()
        for i, name in enumerate(names, 1):
            d = load_patch(cache_dir, name)
            pred = fn(d["optical"], d["sar"], d["mask"])
            res = evaluate_all(sel(pred), sel(d["target"]), mask=d["mask"],
                               data_range=cfg["metrics"]["data_range"])
            res.update({"method": mname, "patch": name})
            rows.append(res)
            if i % 200 == 0:
                print(f"  [{mi}/{len(methods)}] {mname}: {i}/{len(names)}", flush=True)
        dt = time.time() - t0
        print(f"  [{mi}/{len(methods)}] {mname} 完成，用时 {dt:.1f}s "
              f"（{dt/max(len(names),1)*1000:.1f} ms/张）", flush=True)

    df = pd.DataFrame(rows)
    out_dir = Path(cfg["output"]["tables"])
    out_dir.mkdir(parents=True, exist_ok=True)
    per_patch = out_dir / "per_patch_metrics.csv"
    df.to_csv(per_patch, index=False)

    metric_cols = ["PSNR", "SSIM", "SAM", "ERGAS", "MAE", "RMSE"]
    summ = df.groupby("method")[metric_cols].agg(["mean", "std"]).round(4)
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(out_dir / "method_summary.csv", index=False)

    print("\n" + "=" * 68)
    print("方法级汇总（mean）")
    print("=" * 68)
    show = df.groupby("method")[metric_cols].mean().round(4)
    print(show.to_string())
    print(f"\n按 PSNR 排序：")
    print(show.sort_values("PSNR", ascending=False).to_string())
    print(f"\n总用时 {(time.time()-t_all)/60:.1f} 分钟")
    print(f"输出：{per_patch}")
    print("=" * 68)


if __name__ == "__main__":
    main()
