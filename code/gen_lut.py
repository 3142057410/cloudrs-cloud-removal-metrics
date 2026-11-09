"""
gen_lut.py —— 用 PROSAIL 生成 LAI 反演查找表（下游反演器）

为什么用 LUT 而不是训练一个反演网络：
  · 确定性、可复现，审稿人无法质疑"你的下游器是不是被调过"
  · 不引入额外训练，P1 全程零训练
  · 物理可解释：LUT 本身就是辐射传输模型的离散采样

关于参数空间的取舍（这是本文件最关键的设计，实测得出）：
  第一版让 11 个 PROSAIL 参数全部独立均匀采样，结果 LAI 在光谱空间里
  几乎不可辨识——最近 50 个邻居的 LAI 标准差中位数 1.55，接近全域 2.31。
  根因是叶片结构参数 N 与 LAI 在近红外上高度混叠（N=2.5,LAI=2 与
  N=1.0,LAI=8 可给出几乎相同的 NIR）。加红边波段、加样本量都无效
  （13 波段仍是 1.53）。
  改为固定叶片结构与观测几何、只让 LAI/Cab/叶倾角/土壤亮度变化后，
  邻居 LAI 标准差降到 0.47（含 psoil）——这是标准 LUT 反演做法。

输出：data/processed/prosail_lut.npz
  refl (N,13)   Sentinel-2 B1..B12 反射率
  lai  (N,)     对应 LAI
  params (N,4)  [lai, cab, lidfa, psoil]
  band_names (13,)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

try:
    import prosail
except ImportError as e:
    raise SystemExit("需要 prosail：pip install prosail") from e

# Sentinel-2A 全部 13 个波段（中心波长 nm / 半高宽 nm）
S2_BANDS = {
    "B1":  (443.9, 27.0),
    "B2":  (496.6, 98.0),
    "B3":  (560.0, 45.0),
    "B4":  (664.5, 38.0),
    "B5":  (703.9, 19.0),
    "B6":  (740.2, 18.0),
    "B7":  (782.5, 28.0),
    "B8":  (835.1, 145.0),
    "B8A": (864.8, 33.0),
    "B9":  (945.0, 26.0),
    "B10": (1373.5, 75.0),
    "B11": (1613.7, 143.0),
    "B12": (2202.4, 242.0),
}
WL = np.arange(400, 2501, dtype=np.float64)


def gaussian_srf(center, fwhm, wl=WL):
    sigma = fwhm / 2.3548
    g = np.exp(-0.5 * ((wl - center) / sigma) ** 2)
    return g / g.sum()


_SRF = [gaussian_srf(c, f) for c, f in S2_BANDS.values()]


def to_s2(refl_1nm):
    return np.stack([refl_1nm @ s for s in _SRF], axis=1)


# 变化的参数（决定反演的可辨识性，不要随意增加）
VARY = {
    "lai":   (0.0, 8.0),
    "cab":   (5.0, 100.0),
    "lidfa": (30.0, 60.0),
    "psoil": (0.0, 1.0),
}

# 固定的叶片结构、含水/干物质、热点与观测几何
FIXED = {
    "n": 1.5,
    "car": 8.0,
    "cbrown": 0.0,
    "cw": 0.01,
    "cm": 0.01,
    "hspot": 0.01,
    "psi": 0.0,
    "tts": 30.0,
    "tto": 0.0,
    "rsoil": 1,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p1.yaml")
    ap.add_argument("--samples", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    n_total = args.samples or cfg["downstream"]["lut_samples"]
    out_path = Path(cfg["downstream"]["lut_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(2026)
    print("=" * 70)
    print(f"PROSAIL 查找表生成 | {n_total} 条 | 变化参数：{list(VARY)}")
    print("=" * 70)

    V = {k: rng.uniform(lo, hi, n_total) for k, (lo, hi) in VARY.items()}
    refl = np.empty((n_total, 2101), np.float32)
    t0 = time.time()

    for i in range(n_total):
        kw = dict(FIXED)
        kw.update({k: float(V[k][i]) for k in VARY})
        refl[i] = prosail.run_prosail(**kw)
        if (i + 1) % 25000 == 0:
            el = time.time() - t0
            eta = el / (i + 1) * (n_total - i - 1)
            print(f"  {i+1}/{n_total}  已用 {el/60:.1f} min  预计剩余 {eta/60:.1f} min", flush=True)

    s2 = to_s2(refl)
    keep = np.isfinite(s2).all(axis=1) & (s2 > 0).all(axis=1)
    print(f"\n有效样本 {keep.sum()} / {n_total}")

    params = np.stack([V[k] for k in VARY], axis=1)
    np.savez_compressed(
        out_path,
        refl=s2[keep].astype(np.float32),
        lai=V["lai"][keep].astype(np.float32),
        params=params[keep].astype(np.float32),
        param_names=np.array(list(VARY)),
        band_names=np.array(list(S2_BANDS.keys())),
    )

    # --- 可辨识性自检：最近 50 个邻居的 LAI 离散度 ---
    from scipy.spatial import cKDTree
    tree = cKDTree(s2[keep])
    r = np.random.default_rng(1)
    idx = r.choice(keep.sum(), min(300, keep.sum()), replace=False)
    sp = np.array([V["lai"][keep][tree.query(s2[keep][i], k=50)[1]].std() for i in idx])
    print(f"可辨识性自检：邻居 LAI 标准差 中位 {np.median(sp):.2f} / 90%分位 "
          f"{np.percentile(sp, 90):.2f}（全域 {V['lai'][keep].std():.2f}，越小越好）")

    print(f"\n保存 -> {out_path}")
    print(f"refl {s2[keep].shape} | LAI {V['lai'][keep].min():.2f}~{V['lai'][keep].max():.2f}")
    print("波段均值：" + "  ".join(f"{k}={v:.4f}" for k, v in zip(S2_BANDS, s2[keep].mean(axis=0))))
    print(f"总用时 {(time.time()-t0)/60:.1f} 分钟")
    print("=" * 70)


if __name__ == "__main__":
    main()
