"""
noise_floor.py -- 阴性对照：在同一幅晴空影像上制造"应为零差异"的比较。

目的：方法间的 ΔLAI 差（最好 0.393）是否大于反演链自身的噪声地板？

两种地板，都只依赖已有数据与已有查找表：
  1) 抽样地板 split : 从晴空区随机抽 600 个像元，均分两组各 300，
     分别反演后逐像元求 |ΔLAI| 的均值。等价于"同一个场景、同一个方法、
     换一批 300 个像元"时会出现的差异，是任何方法对不可能低于的量。
  2) 辐射地板 noise : 同一批 300 像元，反射率乘 (1 + N(0, 0.02)) 的
     独立乘性噪声（与 LUT 校验时所用噪声同量级），逐像元 |ΔLAI| 均值。
     它量化"共模误差抵消"之后仍残留的那一部分。

输出：
  tables/repro/noise_floor.csv     逐 patch
  tables/repro/noise_floor.json    汇总

用法：python noise_floor.py <项目根目录>   # 需含 data/processed/*.npz 与 src/downstream.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

# Data root: the directory that holds data/processed/*.npz. Pass it as argv[1]
# (or set CLOUDRS_ROOT). The processed cubes are not redistributed with the
# archive, so this script is only runnable in a full working copy.
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CLOUDRS_ROOT", ".")).resolve()
sys.path.insert(0, str(ROOT / "src"))
from downstream import DownstreamRetrieval  # noqa: E402

LUT = ROOT / "data" / "processed" / "prosail_lut.npz"
PROC = ROOT / "data" / "processed"
JAG = Path(__file__).resolve().parents[1]
OUTDIR = JAG / "tables" / "repro"

N_SIDE = 300      # 每个子样本的像元数，与主实验一致（max_pixels=300）
NOISE = 0.02      # 乘性辐射噪声标准差


def main():
    ds = DownstreamRetrieval(LUT)
    import csv
    with open(JAG / "_repo" / "tables" / "main" / "per_patch_metrics.csv", encoding="utf-8") as fh:
        patches = sorted({r["patch"] for r in csv.DictReader(fh) if r.get("patch")})
    print(f"patches: {len(patches)}")

    rows = []
    for i, name in enumerate(patches):
        f = PROC / f"{name}.npz"
        if not f.is_file():
            continue
        d = np.load(f)
        tgt = ds.select(d["target"].astype(np.float32))
        clear = ~d["mask"].astype(bool)
        m = clear.ravel()
        spec_all = tgt.reshape(tgt.shape[0], -1).T[m]
        if len(spec_all) < 2 * N_SIDE:
            continue
        rng = np.random.default_rng(1000 + i)
        pick = rng.choice(len(spec_all), 2 * N_SIDE, replace=False)
        a, b = spec_all[pick[:N_SIDE]], spec_all[pick[N_SIDE:]]
        la, lb = ds.invert_all(a)["lai"], ds.invert_all(b)["lai"]
        split_floor = float(np.mean(np.abs(la - lb)))

        noisy = a * (1.0 + rng.normal(0.0, NOISE, a.shape).astype(np.float32))
        ln_ = ds.invert_all(noisy)["lai"]
        noise_floor = float(np.mean(np.abs(la - ln_)))

        rows.append({"patch": name, "n_clear": int(len(spec_all)),
                     "dLAI_split": round(split_floor, 4),
                     "dLAI_radiometric": round(noise_floor, 4)})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(patches)}")

    sp = np.array([r["dLAI_split"] for r in rows])
    nf = np.array([r["dLAI_radiometric"] for r in rows])
    summary = {
        "n_patches": len(rows),
        "pixels_per_subsample": N_SIDE,
        "multiplicative_noise_sigma": NOISE,
        "split_half": {"mean": round(float(sp.mean()), 4), "median": round(float(np.median(sp)), 4),
                       "p90": round(float(np.percentile(sp, 90)), 4), "max": round(float(sp.max()), 4)},
        "radiometric": {"mean": round(float(nf.mean()), 4), "median": round(float(np.median(nf)), 4),
                        "p90": round(float(np.percentile(nf, 90)), 4), "max": round(float(nf.max()), 4)},
        "frac_split_above_best_method": round(float((sp > 0.3926).mean()), 4),
        "frac_radiometric_above_best_method": round(float((nf > 0.3926).mean()), 4),
        "frac_split_above_median_method": round(float((sp > 0.58).mean()), 4),
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    import csv
    with open(OUTDIR / "noise_floor.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump(summary, open(OUTDIR / "noise_floor.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
