# -*- coding: utf-8 -*-
"""反演器自验证：留一法（LOO）测 PROSAIL-LUT + 8 近邻反演的精度。

为什么必须做：论文用 ΔLAI 当"下游可用性"的真值，但从未交代反演器自身精度。
审稿人第一问必然是"ΔLAI 里有多少是反演器误差"。

做法：从 30 万条 LUT 中随机抽 n 条做测试，对每条在**其余样本**中找 k=8 近邻
（与 downstream.py 的 invert() 完全一致：排除 B1/B9/B10、距离倒数加权），
用近邻的加权均值预测 LAI/Cab，与真值比较。
另做两档乘性噪声（2%、5%），模拟真实影像的辐射噪声——无噪声的 LOO 偏乐观。
"""
import numpy as np
from scipy.spatial import cKDTree

LUT = "data/processed/prosail_lut.npz"
EXCLUDE = ("B1", "B9", "B10")
K = 8
N = 3000
rng = np.random.default_rng(0)

d = np.load(LUT)
names = [str(x) for x in d["band_names"]]
keep = [i for i, n in enumerate(names) if n not in EXCLUDE]
refl = d["refl"][:, keep].astype(np.float32)
lai = d["lai"].astype(np.float64)
pn = [str(x) for x in d["param_names"]]
cab = d["params"][:, pn.index("cab")].astype(np.float64)

print(f"LUT {len(refl)} 条 × {len(keep)} 波段；LAI {lai.min():.1f}–{lai.max():.1f}，"
      f"Cab {cab.min():.0f}–{cab.max():.0f}")

tree = cKDTree(refl)
idx_test = rng.choice(len(refl), size=N, replace=False)
# 全库检索 k+1，再剔除自身
allidx, alldist = tree.query(refl[idx_test], k=K + 1, workers=-1)


def loo_predict(spectra, noise=0.0):
    """对给定光谱做 k 近邻反演；noise>0 时先加乘性噪声。"""
    q = spectra
    if noise:
        q = q * (1.0 + rng.normal(0, noise, q.shape)).astype(np.float32)
    dist, idx = tree.query(q, k=K + 1, workers=-1)
    # 若最近邻距离≈0 且是本样本，剔除；否则用前 k 个
    out_lai = np.empty(len(q))
    out_cab = np.empty(len(q))
    out_std = np.empty(len(q))
    for i in range(len(q)):
        di, ii = dist[i], idx[i]
        if di[0] < 1e-8:
            di, ii = di[1:], ii[1:]
        di, ii = di[:K], ii[:K]
        w = 1.0 / (di + 1e-6)
        w = w / w.sum()
        out_lai[i] = (lai[ii] * w).sum()
        out_cab[i] = (cab[ii] * w).sum()
        out_std[i] = np.sqrt(((lai[ii] - out_lai[i]) ** 2 * w).sum())
    return out_lai, out_cab, out_std


def report(tag, pred, true, unit, lo, hi):
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    bias = float(np.mean(pred - true))
    r2 = 1 - np.sum((pred - true) ** 2) / np.sum((true - true.mean()) ** 2)
    mae = float(np.mean(np.abs(pred - true)))
    print(f"  {tag:14s} RMSE {rmse:7.3f} {unit} | MAE {mae:7.3f} | 偏差 {bias:+6.3f} | "
          f"R² {r2:6.3f} | 相对量程 {rmse / (hi - lo) * 100:5.1f}%")
    return rmse, r2


print("\n=== 留一法精度 ===")
for noise, tag in [(0.0, "无噪声"), (0.02, "噪声 2%"), (0.05, "噪声 5%")]:
    p_lai, p_cab, p_std = loo_predict(refl[idx_test], noise)
    print(f"[{tag}]  近邻 LAI 加权标准差中位数 {np.median(p_std):.2f}")
    report("LAI", p_lai, lai[idx_test], "", 0, 8)
    report("Cab", p_cab, cab[idx_test], "μg/cm²", cab.min(), cab.max())

print("\n=== 可辨识性（全库随机 5000 点的近邻 LAI 离散度）===")
sub = rng.choice(len(refl), 5000, replace=False)
_, _, std = loo_predict(refl[sub])
print(f"  近邻 LAI 标准差：中位数 {np.median(std):.2f}，90 分位 {np.percentile(std, 90):.2f}；"
      f"全域标准差 {lai.std():.2f}")
