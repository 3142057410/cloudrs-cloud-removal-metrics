"""
build_s2_all.py —— 分块构建 L2A 数据集

为什么分块：一次性处理 8 对时，GDAL 的 JP2 块缓存会随对次累积（每对 22 个
10980² 波段），到第 4 对就 MemoryError，数据集从 156 个 patch 掉到 79 个。
每块 2 对、独立进程，块间内存自然回收；meta.json 用 --append-meta 合并。

用法：
    python src/build_s2_all.py --out data/s2_processed --chunk 2
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="results_meta/s2_pairs.csv")
    ap.add_argument("--raw", default="data/s2_raw")
    ap.add_argument("--out", default="data/s2_processed")
    ap.add_argument("--crop", type=int, default=1024)
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--boa-offset-dn", type=float, default=-1000.0)
    a = ap.parse_args()

    pairs = pd.read_csv(ROOT / a.pairs)
    out = ROOT / a.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = ROOT / "results_meta" / "_pairs_chunk.csv"

    n = len(pairs)
    for s in range(0, n, a.chunk):
        seg = pairs.iloc[s:s + a.chunk]
        seg.to_csv(tmp, index=False)
        cmd = [sys.executable, str(ROOT / "src/build_s2_dataset.py"),
               "--pairs", str(tmp), "--raw", a.raw, "--out", a.out,
               "--crop", str(a.crop), "--boa-offset-dn", str(a.boa_offset_dn),
               "--append-meta"]
        print(f"\n### 块 {s // a.chunk + 1}: pairs {s}-{s + len(seg) - 1}", flush=True)
        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode != 0:
            print(f"块 {s // a.chunk + 1} 返回码 {r.returncode}", flush=True)
    tmp.unlink(missing_ok=True)

    npx = len(list(out.glob("*.npz")))
    print(f"\n合计 {npx} 个 patch -> {a.out}")


if __name__ == "__main__":
    main()
