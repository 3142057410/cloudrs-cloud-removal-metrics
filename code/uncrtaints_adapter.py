"""
uncrtaints_adapter.py —— UnCRtainTS（monotemporalL2 官方权重）的推理适配器

为什么要标定：
  UnCRtainTS 训练时对两类输入做了特定归一化（SAR: dB→[0,1] 再映射到 [-1,+1]；
  S2: 反射率 [0,1]，模型内部再乘 scale_by=10）。我们的缓存里
  S2 已是 [0,1]，SAR 是原始 dB（[-25,0]）。归一化或通道顺序差一点，
  输出就会偏离但**不会报错**——必须用重建精度反过来挑。

用法：
    python src/uncrtaints_adapter.py --calibrate --n 6      # 标定，输出各组合的 PSNR
    python src/uncrtaints_adapter.py --pred --n 3           # 看单个 patch 的重建范围

作为方法接入 baselines 时，直接调用 predict(optical, sar, mask)。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PROJ = Path(__file__).resolve().parent.parent
UR = PROJ / "third_party" / "uncrtaints"
W = PROJ / "third_party" / "weights" / "uncrtaints"
sys.path.insert(0, str(UR))
sys.path.insert(0, str(UR / "model"))

S1_BANDS, S2_BANDS = 2, 13
_MODEL = None
_CFG = None
_LOADED = None            # 当前已载入的权重路径（用于 ensure_weights 判断是否需重载）

# 标定得到的最优组合（由 --calibrate 决定后写死，避免每次推理都试）
SAR_MODE = "rescale01"      # rescale01 | minus1_plus1 | raw_db
ORDER = "sar_first"         # sar_first | s2_first


def _remap(k: str) -> str:
    """权重键名对齐：去 netG. 前缀，并把 in_block1/out_block1..5 映射为 .0/.1.."""
    k = k.replace("netG.", "").replace("module.", "")
    k = re.sub(r"\bin_block(\d+)", lambda m: f"in_block.{int(m.group(1)) - 1}", k)
    k = re.sub(r"\bout_block(\d+)", lambda m: f"out_block.{int(m.group(1)) - 1}", k)
    return k


def get_model(device="cuda"):
    """构建并加载官方权重（单例，重复调用不重复加载）。"""
    global _MODEL, _CFG
    if _MODEL is not None:
        return _MODEL
    from model.src.backbones import uncrtaints as U

    conf = json.load(open(W / "conf.json", encoding="utf-8"))
    conf.setdefault("block_type", "mbconv")
    conf.setdefault("var_nonLinearity", "softplus")
    _CFG = SimpleNamespace(**conf)

    m = U.UNCRTAINTS(
        input_dim=S1_BANDS * int(_CFG.use_sar) + S2_BANDS,
        encoder_widths=_CFG.encoder_widths, decoder_widths=_CFG.decoder_widths,
        out_conv=_CFG.out_conv, out_nonlin_mean=_CFG.mean_nonLinearity,
        out_nonlin_var=_CFG.var_nonLinearity, agg_mode=_CFG.agg_mode,
        encoder_norm=_CFG.encoder_norm, decoder_norm=_CFG.decoder_norm,
        n_head=_CFG.n_head, d_model=_CFG.d_model, d_k=_CFG.d_k,
        pad_value=_CFG.pad_value, padding_mode=_CFG.padding_mode,
        positional_encoding=_CFG.positional_encoding, covmode=_CFG.covmode,
        scale_by=_CFG.scale_by, separate_out=_CFG.separate_out, use_v=_CFG.use_v,
        block_type=_CFG.block_type, is_mono=_CFG.pretrain,
    ).to(device)

    ck = torch.load(W / "model.pth.tar", map_location="cpu")
    sd = ck.get("state_dict_G") or ck.get("state_dict") or ck
    sd = {_remap(k): v for k, v in sd.items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    # 缺失应为 0；unexpected 只应是 temporal_encoder（is_mono 时不使用）
    if missing:
        raise RuntimeError(f"权重未完整加载，缺失 {len(missing)} 项：{missing[:4]}")
    m.eval()
    _MODEL = m
    return m


def load_weights(path, device="cuda"):
    """
    把指定权重（例如域内微调得到的 checkpoint）加载进单例模型。

    微调 checkpoint 由 finetune_uncrtaints.py 保存，结构为
    {'state_dict_G': model.state_dict(), 'epoch': ..., 'val_psnr': ...}，
    键名已经是模型侧命名，无需再做 _remap。
    """
    global _MODEL
    m = get_model(device)
    ck = torch.load(Path(path), map_location=device)
    sd = ck.get("state_dict_G", ck)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"微调权重未完整加载，缺失 {len(missing)} 项：{missing[:4]}")
    m.eval()
    print(f"  [uncrtaints] 已加载 {path}"
          f"（epoch {ck.get('epoch', '?')}，val {ck.get('val_psnr', float('nan')):.2f} dB）",
          flush=True)
    return m


def ensure_weights(path, device="cuda"):
    """按路径懒加载：首次调用时加载，重复调用不重复读盘。"""
    global _LOADED
    key = str(Path(path).resolve())
    if _LOADED != key:
        load_weights(path, device)
        _LOADED = key
    return get_model(device)


def _norm_sar(sar: np.ndarray, mode: str) -> np.ndarray:
    s = np.clip(sar.astype(np.float32), -25.0, 0.0)
    if mode == "rescale01":
        return (s + 25.0) / 25.0
    if mode == "minus1_plus1":
        return 2.0 * ((s + 25.0) / 25.0) - 1.0
    if mode == "raw_db":
        return s
    raise ValueError(mode)


@torch.no_grad()
def _forward(m, x):
    """x: (C,H,W) 已是模型输入尺度（即已乘 scale_by）。返回 (13,H,W) 反射率 [0,1]。"""
    h, w = x.shape[1:]
    ph, pw = (-h) % 32, (-w) % 32
    if ph or pw:
        x = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="reflect")
    t = torch.from_numpy(np.ascontiguousarray(x))[None, None].cuda()   # (1,1,C,H,W)
    y = m(t)
    y = y[0] if isinstance(y, (tuple, list)) else y                    # (1,1,13,H,W)
    y = y[0, 0].float().cpu().numpy()
    y = y / float(_CFG.scale_by)          # out = scale_by * sigmoid(...)，除回来得 [0,1]
    return np.clip(y[:, :h, :w], 0.0, 1.0)


@torch.no_grad()
def predict_native(optical, sar, device="cuda"):
    """
    在「原生分辨率」上推理，用于权重训练尺度与数据尺度一致的情形。

    官方权重在 256×256 上训练；把 128 的裁剪块直接喂进去会让输出严重过平滑
    （实测云区与晴空区的误差几乎相等，PSNR 只有 27 dB，而原生 256 为 31.7 dB）。
    optical/sar 必须是未降采样的原始数组。
    """
    m = get_model(device)
    s = _norm_sar(sar, "rescale01")                      # 与 process_SAR('default') 一致
    x = np.concatenate([s, optical.astype(np.float32)], 0)    # SAR 在前，S2 在后（共 15 通道）
    return _forward(m, x * float(_CFG.scale_by))              # set_input 里乘 scale_by


@torch.no_grad()
def predict(optical, sar, mask, device="cuda", sar_mode=None, order=None,
            target_size=None, fill_missing=False):
    """
    从 128 缓存块推理。仅用于对照实验：分辨率降到 128 后模型会过平滑，
    指标明显偏低（约 27 dB），不作为主结果使用。

    optical: (13,H,W) 反射率 [0,1]；sar: (2,H,W) dB；mask: (H,W)
    """
    sar_mode = sar_mode or SAR_MODE
    order = order or ORDER
    m = get_model(device)

    opt = optical.astype(np.float32).copy()
    if fill_missing:
        # B1(443nm)≈0.8×B2；B9(945nm 水汽)≈0.9；B10(1373nm 卷云)≈0.02
        opt[0] = 0.8 * opt[1]
        opt[9] = 0.9
        opt[10] = 0.02

    s = _norm_sar(sar, sar_mode)
    x = np.concatenate([s, opt], 0) if order == "sar_first" else np.concatenate([opt, s], 0)
    x = x * float(_CFG.scale_by)                     # ← 关键：set_input 里乘 scale_by

    H, Wd = optical.shape[1:]
    if target_size and target_size != H:
        import torch.nn.functional as F
        x = F.interpolate(torch.from_numpy(x)[None], size=(target_size, target_size),
                          mode="bilinear", align_corners=False)[0].numpy()

    y = _forward(m, x)

    if target_size and target_size != H:
        import torch.nn.functional as F
        y = F.interpolate(torch.from_numpy(y)[None], size=(H, Wd),
                          mode="bilinear", align_corners=False)[0].numpy()
    return np.clip(y[:, :H, :Wd], 0.0, 1.0)


# 域内微调权重（由 finetune_uncrtaints.py 产出）。改这里即可切换接入的权重。
FT_WEIGHTS = PROJ / "third_party" / "weights" / "uncrtaints_ft" / "ft10.pth.tar"


def predict_ft(optical, sar, mask=None):
    """域内微调权重的推理入口（128 缓存口径），供 baselines.METHODS 直接调用。"""
    ensure_weights(FT_WEIGHTS)
    return predict(optical, sar, mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--pred", action="store_true")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    from dataset import USABLE_IDX, load_patch
    from metrics import psnr

    cache = PROJ / "data" / "processed"
    names = json.load(open(cache / "split.json", encoding="utf-8"))["test"][: args.n]
    patches = [load_patch(cache, n) for n in names]

    if args.calibrate:
        print(f"在 {len(patches)} 个 patch 上标定归一化与通道顺序：")
        best = None
        for mode in ("rescale01", "minus1_plus1", "raw_db"):
            for order in ("sar_first", "s2_first"):
                vals = []
                for d in patches:
                    m = d["mask"].astype(bool)
                    if m.sum() < 20:
                        continue
                    y = predict(d["optical"], d["sar"], d["mask"], sar_mode=mode, order=order)
                    vals.append(psnr(y[USABLE_IDX], d["target"][USABLE_IDX], mask=d["mask"]))
                v = float(np.mean(vals))
                print(f"  {mode:14s} {order:10s} PSNR {v:6.2f} dB")
                if best is None or v > best[0]:
                    best = (v, mode, order)
        print(f"\n最优：{best[1]} + {best[2]}  ({best[0]:.2f} dB)")
        print("把这两个值写进 uncrtaints_adapter.py 的 SAR_MODE / ORDER")
        return

    if args.pred:
        d = patches[0]
        y = predict(d["optical"], d["sar"], d["mask"])
        m = d["mask"].astype(bool)
        print(f"输入 optical [{d['optical'].min():.3f},{d['optical'].max():.3f}]  "
              f"输出 [{y.min():.3f},{y.max():.3f}]")
        print(f"该 patch PSNR {psnr(y[USABLE_IDX], d['target'][USABLE_IDX], mask=d['mask']):.2f} dB"
              f"（有云输入本身为 "
              f"{psnr(d['optical'][USABLE_IDX], d['target'][USABLE_IDX], mask=d['mask']):.2f} dB）")


if __name__ == "__main__":
    main()
