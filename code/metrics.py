"""
metrics.py —— 统一指标库（P1/P2/P4 共用）

设计原则：
1. 所有函数接受 (C,H,W) 或 (H,W) 的 numpy 数组，值域 [0,1] 的地表反射率。
2. 统一支持 mask：只在云区/有效区计算，避免无效像元污染指标。
3. 每个指标单独一个函数，方便写消融；evaluate_all 一次性返回全部。
4. 不依赖任何框架，纯 numpy + scikit-image，跑得快、好调试。

用法：
    from metrics import evaluate_all, bootstrap_spearman
    res = evaluate_all(pred, gt, mask=predict_mask)
"""
from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity as _ssim

EPS = 1e-8


# ---------------------------------------------------------------- 基础工具
def _as_hwc(a: np.ndarray) -> np.ndarray:
    """(C,H,W) -> (H,W,C)，已经是 (H,W,C) 就原样返回。"""
    if a.ndim == 2:
        return a[..., None]
    if a.ndim == 3:
        # 波段数很小的情形按 CHW 处理，否则按 HWC
        if a.shape[0] <= 32 and a.shape[0] < a.shape[-1]:
            return np.transpose(a, (1, 2, 0))
        return a
    raise ValueError(f"期望 2D 或 3D 数组，得到 {a.ndim}D")


def _select(a: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """按 mask 取出有效像元（返回 (N,C)）。mask 为 None 时取全部。"""
    a = _as_hwc(a).astype(np.float64)
    if mask is None:
        return a.reshape(-1, a.shape[-1])
    m = np.asarray(mask).squeeze().astype(bool)
    if m.shape != a.shape[:2]:
        raise ValueError(f"mask 尺寸 {m.shape} 与影像 {a.shape[:2]} 不匹配")
    return a[m]


# ---------------------------------------------------------------- 像素级指标
def rmse(pred, gt, mask=None) -> float:
    d = _select(pred, mask) - _select(gt, mask)
    return float(np.sqrt(np.mean(d ** 2)))


def mae(pred, gt, mask=None) -> float:
    d = _select(pred, mask) - _select(gt, mask)
    return float(np.mean(np.abs(d)))


def psnr(pred, gt, mask=None, data_range: float = 1.0) -> float:
    """PSNR。注意：加 mask 后是"仅在有效区"的 PSNR，与不加 mask 不可直接比较。"""
    mse = rmse(pred, gt, mask) ** 2
    return float(10.0 * np.log10(data_range ** 2 / max(mse, EPS)))


def ssim(pred, gt, mask=None, data_range: float = 1.0) -> float:
    """
    SSIM。skimage 不支持 mask，因此做法是：
    无 mask 直接算；有 mask 时把无效区替换为 gt（使其局部统计不产生误差），
    再在全图算 SSIM —— 这是文献中常见的近似，需在论文里注明。
    """
    p, g = _as_hwc(pred).astype(np.float64), _as_hwc(gt).astype(np.float64)
    if mask is not None:
        m = np.asarray(mask).squeeze().astype(bool)[..., None]
        p = np.where(m, p, g)
    if p.shape[-1] == 1:
        p, g = p[..., 0], g[..., 0]
    return float(_ssim(g, p, data_range=data_range, channel_axis=(None if p.ndim == 2 else -1)))


# ---------------------------------------------------------------- 光谱级指标
def sam(pred, gt, mask=None) -> float:
    """
    光谱角（度）。衡量光谱形状是否失真，对亮度整体偏移不敏感。
    仅在高亮度像元上统计更稳（可通过 mask 传入，如 gt 近红外 > 0.05）。
    """
    p, g = _select(pred, mask), _select(gt, mask)
    num = np.sum(p * g, axis=1)
    den = np.linalg.norm(p, axis=1) * np.linalg.norm(g, axis=1)
    cos = np.clip(num / np.maximum(den, EPS), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)).mean())


def ergas(pred, gt, mask=None, ratio: float = 4.0) -> float:
    """
    ERGAS（无量纲，越小越好）。ratio = 高/低分辨率之比（Sentinel-2 常用 4）。
    同时惩罚亮度偏差与波段间相对误差，是 pansharpening/去云领域的常用指标。
    """
    p, g = _select(pred, mask), _select(gt, mask)
    mu_g = g.mean(axis=0)
    rmse_b = np.sqrt(((p - g) ** 2).mean(axis=0))
    val = np.sqrt(np.mean((rmse_b / np.maximum(mu_g, EPS)) ** 2))
    return float(100.0 * ratio * val)


# ---------------------------------------------------------------- 汇总
def evaluate_all(pred, gt, mask=None, data_range: float = 1.0) -> dict:
    """一次性算全部指标，返回 dict（键名直接可用作表头）。"""
    return {
        "PSNR": psnr(pred, gt, mask, data_range),
        "SSIM": ssim(pred, gt, mask, data_range),
        "SAM": sam(pred, gt, mask),
        "ERGAS": ergas(pred, gt, mask),
        "MAE": mae(pred, gt, mask),
        "RMSE": rmse(pred, gt, mask),
    }


# ---------------------------------------------------------------- 统计工具
def bootstrap_spearman(x, y, n_boot: int = 1000, seed: int = 0):
    """
    秩相关的 bootstrap 置信区间。
    P1 的核心统计：方法数只有 8 个时，单点 Spearman 极不稳定，
    必须给置信区间，否则审稿人一句 "method count too small" 就否了。

    返回 (rho, ci_low, ci_high)。
    """
    from scipy.stats import spearmanr

    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3:
        raise ValueError("样本太少，无法计算秩相关")
    rho = float(spearmanr(x, y).correlation)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    rhos = []
    for i in idx:
        r = spearmanr(x[i], y[i]).correlation
        if np.isfinite(r):
            rhos.append(r)
    rhos = np.asarray(rhos)
    lo, hi = np.percentile(rhos, [2.5, 97.5]) if rhos.size else (np.nan, np.nan)
    return rho, float(lo), float(hi)


# ---------------------------------------------------------------- 自测
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    gt = rng.random((4, 64, 64)).astype(np.float32)
    pred = np.clip(gt + 0.02 * rng.standard_normal(gt.shape), 0, 1).astype(np.float32)
    mask = np.zeros((64, 64), bool)
    mask[16:48, 16:48] = True

    print("=" * 58)
    print("metrics.py 自测")
    print("=" * 58)
    print("无 mask：")
    for k, v in evaluate_all(pred, gt).items():
        print(f"  {k:6s} {v:10.4f}")
    print("有 mask（仅中心 32x32）：")
    for k, v in evaluate_all(pred, gt, mask).items():
        print(f"  {k:6s} {v:10.4f}")

    print("-" * 58)
    print("自洽性检查：")
    print(f"  完美重建 PSNR = {psnr(gt, gt):.1f} dB（应为 inf 或极大）")
    print(f"  完美重建 SAM  = {sam(gt, gt):.4f} 度（应为 0）")
    # 注意：SAM 只对“乘性增益”不敏感，对“加性偏移”敏感 —— 这正好是 P4 误差分解要区分的两类误差
    print(f"  加性偏移 +0.1：SAM = {sam(np.clip(gt+0.1,0,1), gt):.4f} 度（变大，加性偏移会被 SAM 捕捉）")
    print(f"  加性偏移 +0.1：PSNR = {psnr(np.clip(gt+0.1,0,1), gt):.1f} dB（明显下降）")
    print(f"  乘性增益 x1.1：SAM = {sam(np.clip(gt*1.1,0,1), gt):.4f} 度（接近 0，SAM 对增益不敏感）")
    print(f"  乘性增益 x1.1：PSNR = {psnr(np.clip(gt*1.1,0,1), gt):.1f} dB（下降）")

    rho, lo, hi = bootstrap_spearman([5, 4, 3, 2, 1], [5.1, 4.0, 2.9, 2.1, 1.0])
    print("-" * 58)
    print(f"bootstrap_spearman 演示：rho={rho:.3f}  95%CI=[{lo:.3f}, {hi:.3f}]")
    print("=" * 58)
