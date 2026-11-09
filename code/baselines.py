"""
baselines.py —— 无需训练的去云基线

这些方法看似简单，但对 P1 恰好合适：它们会产生差异很大的误差分布
（有的 PSNR 高但光谱失真，有的相反），正好用来检验"指标之间是否一致"。

若要加学习型方法，只需在本文件按同样签名
    fn(optical, sar, mask) -> pred(float32, C,H,W, [0,1])
注册到 METHODS 即可，主流程无需改动。
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def _normalize_out(pred, optical, mask):
    """把结果裁剪到合理范围，并保持云区外与原图一致（去云不应改动晴空区）。"""
    pred = np.clip(pred, 0.0, 1.5)
    out = optical.copy()
    m = mask.astype(bool)
    out[:, m] = pred[:, m]
    return out.astype(np.float32)


def zero_fill(optical, sar, mask):
    """云区置零：最弱基线，用来锚定指标下界。"""
    pred = optical.copy()
    pred[:, mask.astype(bool)] = 0.0
    return pred.astype(np.float32)


def mean_fill(optical, sar, mask):
    """云区用晴空像元的波段均值填充：只保留场景整体亮度，无空间结构。"""
    m = mask.astype(bool)
    pred = optical.copy()
    if (~m).sum() < 10:
        pred[:, m] = 0.0
        return pred.astype(np.float32)
    mus = optical[:, ~m].mean(axis=1)
    pred[:, m] = mus[:, None]
    return pred.astype(np.float32)


def nearest_fill(optical, sar, mask):
    """最近晴空像元填充（距离变换）：保留一定空间连续性，但会拖影。"""
    m = mask.astype(bool)
    if (~m).sum() < 10:
        return zero_fill(optical, sar, mask)
    pred = optical.copy()
    # 用晴空区做源，找每个云像元最近的晴空像元索引
    _, (iy, ix) = ndimage.distance_transform_edt(m, return_indices=True)
    for c in range(optical.shape[0]):
        ch = optical[c]
        filled = ch[iy, ix]
        pred[c, m] = filled[m]
    return _normalize_out(pred, optical, mask)


def blur_fill(optical, sar, mask):
    """均值填充后做高斯模糊：视觉更平滑，但会抹掉地物边界。"""
    base = mean_fill(optical, sar, mask)
    m = mask.astype(bool)
    out = base.copy()
    for c in range(base.shape[0]):
        sm = ndimage.gaussian_filter(base[c], sigma=3.0)
        out[c, m] = sm[m]
    return _normalize_out(out, optical, mask)


def sar_guided(optical, sar, mask):
    """
    SAR 引导填充：按 SAR 强度分箱，每个箱用该箱内晴空像元的光学均值。
    比 mean_fill 多了"结构信息"，但 SAR 与光学的物理量不同，仍会有偏差。
    """
    m = mask.astype(bool)
    if (~m).sum() < 50 or sar is None or sar.shape[-2:] != optical.shape[-2:]:
        return mean_fill(optical, sar, mask)

    # SAR: 取两通道均值，按分位数分 8 箱
    s = sar.mean(axis=0)
    s_valid = s[~m]
    if s_valid.size < 20:
        return mean_fill(optical, sar, mask)
    qs = np.quantile(s_valid, np.linspace(0, 1, 9))
    qs = np.unique(qs)
    if qs.size < 3:
        return mean_fill(optical, sar, mask)

    bins = np.digitize(s, qs[1:-1])
    pred = optical.copy()
    for b in np.unique(bins[m]):
        sel = (bins == b) & (~m)
        target = (bins == b) & m
        if sel.sum() >= 5:
            pred[:, target] = optical[:, sel].mean(axis=1)[:, None]
        else:
            pred[:, target] = optical[:, ~m].mean(axis=1)[:, None]
    return _normalize_out(pred, optical, mask)


def inpaint_telea(optical, sar, mask):
    """
    OpenCV Telea 结构修复。
    必须走 uint8：cv2.inpaint 对 float32 输入会输出越界值（实测 −1.86~1.88），
    转成 0~255 再除回来。量化步长 1/255≈0.0039，对应 PSNR 上限约 59 dB，
    远高于本实验的 25~35 dB 区间，不影响方法间比较。
    """
    import cv2
    m = mask.astype(bool)
    if m.sum() == 0:
        return optical.copy()
    pred = optical.copy()
    msk = m.astype(np.uint8) * 255
    for c in range(optical.shape[0]):
        band = np.clip(optical[c], 0, 1) * 255.0
        out = cv2.inpaint(band.astype(np.uint8), msk, 5, cv2.INPAINT_TELEA)
        pred[c] = out.astype(np.float32) / 255.0
    return _normalize_out(pred, optical, mask)


def sar_regression(optical, sar, mask):
    """
    SAR→光学线性回归填充：在晴空像元上逐波段拟合 optical ~ [1, SAR]，
    再对云区预测。与 sar_guided 的区别是参数化模型而非分箱均值，
    结果更平滑、典型地低估高反射率地物。
    """
    m = mask.astype(bool)
    if (~m).sum() < 100 or sar is None or sar.shape[-2:] != optical.shape[-2:]:
        return mean_fill(optical, sar, mask)
    s = sar.mean(axis=0).ravel().astype(np.float64)
    X = np.stack([np.ones_like(s), s], axis=1)
    clear = (~m).ravel()
    pred = optical.copy()
    for c in range(optical.shape[0]):
        y = optical[c].ravel().astype(np.float64)
        try:
            beta, *_ = np.linalg.lstsq(X[clear], y[clear], rcond=None)
        except np.linalg.LinAlgError:
            continue
        est = (X @ beta).reshape(optical.shape[1:])
        pred[c][m] = est[m]
    return _normalize_out(pred, optical, mask)


def knn_spatial(optical, sar, mask):
    """k 近邻空间平均（k=16）：比最近邻更平滑，但会模糊地物边界。"""
    from scipy.spatial import cKDTree
    m = mask.astype(bool)
    if (~m).sum() < 20:
        return mean_fill(optical, sar, mask)
    yy, xx = np.mgrid[0:optical.shape[1], 0:optical.shape[2]]
    pts = np.stack([yy.ravel(), xx.ravel()], 1).astype(np.float32)
    clear = (~m).ravel()
    tree = cKDTree(pts[clear])
    k = min(16, int(clear.sum()))
    _, idx = tree.query(pts[m.ravel()], k=k)
    src = optical.reshape(optical.shape[0], -1)[:, clear]     # (C, Nclear)
    vals = src[:, idx].mean(axis=2)                            # (C, Ncloud)
    pred = optical.copy()
    pred[:, m] = vals
    return _normalize_out(pred, optical, mask)


def hist_match(optical, sar, mask):
    """
    直方图匹配：把云区各波段的分布映射到晴空区的分布。
    保留空间纹理但辐射值整体错位——最容易产生"PSNR 尚可、光谱失真严重"的方法，
    对检验指标一致性特别有价值。

    实现用固定 256 档分位 + np.interp，而不是对每个像元调用 np.quantile
    （后者在 128×128×13 波段下约 0.8 s/张，525 张要 400 s，曾导致下游流程超时）。
    """
    m = mask.astype(bool)
    if m.sum() == 0 or (~m).sum() < 50:
        return optical.copy()
    grid = np.linspace(0.0, 1.0, 257)
    pred = optical.copy()
    for c in range(optical.shape[0]):
        src = optical[c][m]
        ref = optical[c][~m]
        if src.size == 0:
            continue
        ref_q = np.quantile(ref, grid)
        order = np.argsort(src, kind="stable")
        q = (np.arange(src.size) + 0.5) / src.size
        mapped = np.interp(q, grid, ref_q)
        tmp = np.empty_like(src)
        tmp[order] = mapped
        pred[c][m] = tmp
    return _normalize_out(pred, optical, mask)


# ---------------------------------------------------------------- 学习型方法
_MODEL_CACHE = {}


def _predict_learned(name, optical, sar, mask):
    """
    调用已训练好的 U-Net（见 models.py / train_unet.py）。
    模型第一次使用时加载并缓存，避免每个 patch 都读盘。
    """
    import torch
    from models import load_model, predict

    if name not in _MODEL_CACHE:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL_CACHE[name] = load_model(name, device)
    net, use_sar = _MODEL_CACHE[name]
    device = next(net.parameters()).device
    return predict(net, use_sar, optical, sar, mask, device)


def unet_opt(optical, sar, mask):
    """学习方法：只输入有云光学影像的轻量 U-Net（1.95M 参数）。"""
    return _predict_learned("unet_opt", optical, sar, mask)


def unet_sar(optical, sar, mask):
    """学习方法：光学 + SAR 融合的轻量 U-Net，与 unet_opt 只差输入，便于消融。"""
    return _predict_learned("unet_sar", optical, sar, mask)


def dsen2cr_opt(optical, sar, mask):
    """DSen2-CR 结构（Meraner et al. 2020），只输入光学。"""
    return _predict_learned("dsen2cr_opt", optical, sar, mask)


def dsen2cr_sar(optical, sar, mask):
    """DSen2-CR 结构，光学 + SAR。原文主模型即该配置，全程不下采样。"""
    return _predict_learned("dsen2cr_sar", optical, sar, mask)


def diffusion(optical, sar, mask):
    """
    生成族：条件扩散（2.55M 参数，DDIM 20 步）。
    规模与训练步数都远小于文献里的扩散去云（通常是几百 M、十万级步），
    因此指标偏弱是预期内的；它的作用是把"生成式"这一族纳入评测，
    以便观察生成式方法在像素指标与下游指标上是否会不一致。
    """
    import torch
    from models import load_diffusion, predict_diffusion

    if "diffusion" not in _MODEL_CACHE:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL_CACHE["diffusion"] = (load_diffusion(device), device)
    (bundle, device) = _MODEL_CACHE["diffusion"]
    return predict_diffusion(bundle, optical, sar, mask, device)


def uncrtaints_ft(optical, sar, mask):
    """
    学习方法（外部权重）：UnCRtainTS 官方架构 + 官方预训练权重 + 本数据集域内微调。

    为什么必须微调：开箱直接用只有 26.5 dB，低于均值填充。核对作者源码后确认接入口径无误
    （权重完整加载、SAR 归一化与通道顺序、输入 ×scale_by=10、输出 ÷10 全部对齐），
    但输出整体泛白、云区与晴空区误差几乎相等——是本数据集的强云影条件造成的域偏移。
    在本数据 train 划分上微调 10 轮（冻结 BN、只更新卷积）后升到 34 dB 量级。

    推理走 128 缓存口径（微调也是 128 做的），与其余方法完全一致。
    """
    from uncrtaints_adapter import predict_ft
    return predict_ft(optical, sar, mask)


METHODS = {
    "zero_fill": zero_fill,
    "mean_fill": mean_fill,
    "nearest_fill": nearest_fill,
    "blur_fill": blur_fill,
    "sar_guided": sar_guided,
    "inpaint_telea": inpaint_telea,
    "sar_regression": sar_regression,
    "knn_spatial": knn_spatial,
    "hist_match": hist_match,
    "unet_opt": unet_opt,
    "unet_sar": unet_sar,
    "dsen2cr_opt": dsen2cr_opt,
    "dsen2cr_sar": dsen2cr_sar,
    "diffusion": diffusion,
    "uncrtaints_ft": uncrtaints_ft,
}
