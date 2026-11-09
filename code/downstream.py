"""
downstream.py —— 下游 LAI 反演器（PROSAIL 查找表 + KD 树最近邻）

关键设计：P1 衡量的是"去云结果导致的参数偏移"，而不是"反演绝对精度"。
  对同一片云区像素：
      LAI_from_pred  = invert(pred 的反射率)
      LAI_from_target= invert(target 的反射率)   ← 用晴空影像作为"无云情形"的参照
  ΔLAI = |LAI_from_pred − LAI_from_target|
这样不需要地面实测站点，也能量化"去云误差如何传到参数"，且逻辑闭环、无数据泄漏。

用法：
    ds = DownstreamRetrieval("data/processed/prosail_lut.npz")
    lai = ds.invert(spectra)          # spectra: (M, 4)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

# 默认排除的大气波段：B1（443nm，瑞利散射主导）、B9（945nm，水汽吸收）、
# B10（1373nm，卷云波段）。SEN12MS-CR 提供的是 L1C TOA 反射率，而 PROSAIL
# 输出地表反射率，这三个波段 TOA 与地表的偏差远大于地表信号本身，
# 放进 KD 树会污染近邻搜索。
ATMO_BANDS = ("B1", "B9", "B10")


class DownstreamRetrieval:
    def __init__(self, lut_path, k=8, exclude=ATMO_BANDS):
        d = np.load(Path(lut_path))
        names = [str(x) for x in d["band_names"]]
        self.keep = [i for i, n in enumerate(names) if n not in exclude]
        self.band_names = [names[i] for i in self.keep]
        self.refl = d["refl"][:, self.keep].astype(np.float32)
        self.lai = d["lai"].astype(np.float32)
        # LUT 的 params 列为 [lai, cab, lidfa, psoil]，Cab 现成可用，
        # 因此"第二个下游变量"不需要重跑 LUT（原本要 20 分钟的 PROSAIL 批量计算）。
        self.cab = None
        if "params" in d.files and "param_names" in d.files:
            pn = [str(x) for x in d["param_names"]]
            if "cab" in pn:
                self.cab = d["params"][:, pn.index("cab")].astype(np.float32)
        self.k = k
        self.tree = cKDTree(self.refl)

    def select(self, img):
        """把 13 波段影像裁到反演所用的波段。img: (13,H,W) 或 (M,13)。"""
        if img.shape[0] == len(self.keep) and img.shape[0] != 13:
            return img
        return img[self.keep] if img.ndim == 3 else img[:, self.keep]

    def _neighbors(self, spectra, workers=-1):
        spectra = np.asarray(spectra, np.float32)
        if spectra.ndim == 1:
            spectra = spectra[None, :]
        dist, idx = self.tree.query(spectra, k=self.k, workers=workers)
        if self.k == 1:
            return idx[:, None], np.ones((len(idx), 1), np.float32)
        w = 1.0 / (dist + 1e-6)
        return idx, (w / w.sum(axis=1, keepdims=True)).astype(np.float32)

    def _weighted(self, idx, w, values):
        vals = values[idx]
        if vals.ndim == 1:
            vals = vals[:, None]
        mean = (vals * w).sum(axis=1)
        std = np.sqrt(((vals - mean[:, None]) ** 2 * w).sum(axis=1))
        return mean.astype(np.float32), std.astype(np.float32)

    def invert(self, spectra, workers=-1):
        """
        最近邻反演。返回 (lai_mean, lai_std)：
        用 k 个近邻的加权平均作为估计，其标准差作为可辨识性指标
        —— 逆问题病态时近邻的 LAI 差异大，标准差自然变大。
        """
        idx, w = self._neighbors(spectra, workers)
        return self._weighted(idx, w, self.lai)

    def invert_all(self, spectra, workers=-1):
        """一次近邻搜索同时给出 LAI 与 Cab，保证两个参数用同一组近邻（可比性强）。"""
        idx, w = self._neighbors(spectra, workers)
        out = {"lai": self._weighted(idx, w, self.lai)[0]}
        if self.cab is not None:
            out["cab"] = self._weighted(idx, w, self.cab)[0]
        return out

    def lut_ambiguity(self, n=5000, seed=0):
        """检查 LUT 本身的病态程度：随机采样点，看 k 近邻的 LAI 分散度。"""
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.refl), size=min(n, len(self.refl)), replace=False)
        _, std = self.invert(self.refl[idx])
        return float(np.median(std)), float(np.percentile(std, 90))


def patch_spectra(img, mask, max_pixels=300, seed=0):
    """从云区像素里抽 (M,4) 光谱；像素太多时随机抽样，保证速度。"""
    m = mask.astype(bool).ravel()
    spec = img.reshape(img.shape[0], -1).T[m]        # (M,4)
    if len(spec) > max_pixels:
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(spec), max_pixels, replace=False)
        spec = spec[sel]
    return spec


def retrieval_delta(ds, pred, target, mask, max_pixels=300, seed=0):
    """
    返回 dict：预测与真值反演结果的差异。
      dLAI_*  LAI 的偏差（主指标）
      dCab_*  叶绿素 Cab 的偏差（第二个下游变量，用于检验结论是否只对 LAI 成立）
    两者共用同一组近邻，因此差异只来自"参数对光谱的敏感方式不同"，
    不会混入近邻搜索本身的随机性。
    """
    sp = patch_spectra(ds.select(pred), mask, max_pixels, seed)
    st = patch_spectra(ds.select(target), mask, max_pixels, seed)
    if len(sp) == 0:
        return {"dLAI_mean": np.nan, "dLAI_bias": np.nan, "dLAI_rel": np.nan,
                "LAI_pred": np.nan, "LAI_true": np.nan,
                "dCab_mean": np.nan, "dCab_bias": np.nan, "dCab_rel": np.nan}
    pp = ds.invert_all(sp)
    pt = ds.invert_all(st)
    lp, lt = pp["lai"], pt["lai"]
    d = lp - lt
    out = {
        "dLAI_mean": float(np.mean(np.abs(d))),
        "dLAI_bias": float(np.mean(d)),
        "dLAI_rel": float(np.mean(np.abs(d)) / (float(np.mean(lt)) + 1e-6)),
        "LAI_pred": float(np.mean(lp)),
        "LAI_true": float(np.mean(lt)),
    }
    if "cab" in pp and "cab" in pt:
        cp, ct_ = pp["cab"], pt["cab"]
        dc = cp - ct_
        out.update({
            "dCab_mean": float(np.mean(np.abs(dc))),
            "dCab_bias": float(np.mean(dc)),
            "dCab_rel": float(np.mean(np.abs(dc)) / (float(np.mean(ct_)) + 1e-6)),
        })
    else:
        out.update({"dCab_mean": np.nan, "dCab_bias": np.nan, "dCab_rel": np.nan})
    return out
