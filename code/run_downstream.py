"""
run_downstream.py —— 对每个方法的去云结果做下游 LAI 反演，量化参数偏移

产出：results/tables/per_patch_downstream.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import METHODS
from dataset import load_patch
from downstream import DownstreamRetrieval, retrieval_delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p1.yaml")
    ap.add_argument("--methods", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-pixels", type=int, default=300)
    ap.add_argument("--append", action="store_true",
                    help="把本次结果追加到已有 CSV（分批跑时用，按 method+patch 去重）")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    cache_dir = Path(cfg["data"]["cache_dir"])
    split = json.load(open(cache_dir / "split.json", encoding="utf-8"))
    names = split["test"][: args.limit] if args.limit else split["test"]
    methods = args.methods.split(",") if args.methods else cfg["methods"]

    ds = DownstreamRetrieval(cfg["downstream"]["lut_path"])
    med, p90 = ds.lut_ambiguity()
    print("=" * 68)
    print(f"LUT: {ds.refl.shape[0]} 条 | 波段 {ds.band_names}")
    print(f"LUT 病态程度：k近邻 LAI 标准差 中位数 {med:.3f} / 90分位 {p90:.3f}")
    print(f"测试 patch {len(names)} | 方法 {len(methods)} | 每图抽样 {args.max_pixels} 像素")
    print("=" * 68)

    rows = []
    t_all = time.time()
    for mi, mname in enumerate(methods, 1):
        fn = METHODS[mname]
        t0 = time.time()
        for name in names:
            d = load_patch(cache_dir, name)
            pred = fn(d["optical"], d["sar"], d["mask"])
            res = retrieval_delta(ds, pred, d["target"], d["mask"],
                                  max_pixels=args.max_pixels)
            res.update({"method": mname, "patch": name})
            rows.append(res)
        print(f"  [{mi}/{len(methods)}] {mname} 完成 {time.time()-t0:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    out = Path(cfg["output"]["tables"]) / "per_patch_downstream.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.append and out.exists():
        old = pd.read_csv(out)
        df = pd.concat([old, df], ignore_index=True)
        df = df.drop_duplicates(subset=["method", "patch"], keep="last")
        print(f"  追加模式：合并后 {len(df)} 行（{df.method.nunique()} 个方法）")
    df.to_csv(out, index=False)

    print("\n" + "=" * 68)
    print("下游影响（越小越好）")
    print("=" * 68)
    print(df.groupby("method")[["dLAI_mean", "dLAI_bias", "dLAI_rel"]].mean().round(4).to_string())
    print(f"\n总用时 {(time.time()-t_all)/60:.1f} 分钟 -> {out}")


if __name__ == "__main__":
    main()
