"""
models.py —— 轻量 U-Net 去云模型（本地 4GB 显存可训练）

为什么要补学习方法：
  前面 9 个方法全部属于"非学习型局部统计/插值"这一族，误差结构高度相似，
  如果论文只测这一族，"指标与下游不一致"可能只是这一族的特性。
  必须至少加一族"学习方法"，才能说明结论不是方法族选择造成的。

两个变体（刻意只差在输入，便于做消融）：
  unet_opt : 只输入有云光学影像
  unet_sar : 输入有云光学 + SAR（SAR 以零填充方式与光学拼接）
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CKPT_DIR = Path("models")


def _blk(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, cin=13, cout=13, base=32):
        super().__init__()
        self.d1 = _blk(cin, base)
        self.d2 = _blk(base, base * 2)
        self.d3 = _blk(base * 2, base * 4)
        self.bott = _blk(base * 4, base * 8)
        self.u3 = _blk(base * 8 + base * 4, base * 4)
        self.u2 = _blk(base * 4 + base * 2, base * 2)
        self.u1 = _blk(base * 2 + base, base)
        self.out = nn.Conv2d(base, cout, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        s1 = self.d1(x)
        s2 = self.d2(self.pool(s1))
        s3 = self.d3(self.pool(s2))
        b = self.bott(self.pool(s3))
        u3 = self.u3(torch.cat([F.interpolate(b, scale_factor=2, mode="nearest"), s3], 1))
        u2 = self.u2(torch.cat([F.interpolate(u3, scale_factor=2, mode="nearest"), s2], 1))
        u1 = self.u1(torch.cat([F.interpolate(u2, scale_factor=2, mode="nearest"), s1], 1))
        return self.out(u1)


def build_input(optical, sar, use_sar):
    """
    组装模型输入。
    关键：把 cloud mask 当作额外通道喂进去。否则模型不知道要修哪里，
    会把整幅图（包括本来正确的晴空区）都改一遍，人为拉低所有指标。
    """
    c, h, w = optical.shape
    br = optical[1:4].max(axis=0, keepdims=True)     # 可见光亮度，用于生成掩膜通道
    m = (br > 0.15).astype(np.float32)
    parts = [optical, m]
    if use_sar and sar is not None:
        s = (sar - sar.mean()) / (sar.std() + 1e-6)   # dB 值量纲差异大，标准化
        parts.append(s.astype(np.float32))
    return np.concatenate(parts, 0).astype(np.float32)


def load_model(name, device="cuda"):
    """
    按名字加载权重。名字形如 unet_sar / unet_opt_s1 / dsen2cr_sar / dsen2cr_opt_s2。
    用 "_sar" in name 而不是 endswith("_sar")：带种子后缀时 endswith 会判错，
    导致输入通道数不匹配。
    """
    use_sar = "_sar" in name
    cin = 13 + 1 + (2 if use_sar else 0)
    if name.startswith("dsen2cr"):
        net = DSen2CRNet(cin=cin, cout=13, F=64).to(device)
    else:
        net = UNet(cin=cin, cout=13, base=32).to(device)
    ck = CKPT_DIR / f"{name}.pt"
    if not ck.exists():
        raise FileNotFoundError(f"缺少权重 {ck}")
    net.load_state_dict(torch.load(ck, map_location=device))
    net.eval()
    return net, use_sar


# ---------------------------------------------------------------- DSen2-CR 结构
class _DSen2Block(nn.Module):
    """DSen2-CR 的残差块：conv-ReLU-conv，残差缩放 0.1（与原仓库一致）。"""

    def __init__(self, dim, res_scale=0.1):
        super().__init__()
        self.res_scale = res_scale
        self.conv_block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, padding=1, bias=True),
        )

    def forward(self, x):
        return x + self.res_scale * self.conv_block(x)


class DSen2CRNet(nn.Module):
    """
    DSen2-CR（Meraner et al., ISPRS J 2020）官方结构的复现。

    原结构：Conv(15→256) + ReLU → 16×ResnetBlock(256) → Conv(256→13)，
    外加一条长跳跃：输出 = 有云光学 13 波段 + 网络残差。
    与 U-Net 的关键差别是**全程不下采样**，因此不会产生上采样带来的平滑，
    误差特性与 U-Net 明显不同——这正是加它作为独立方法的价值。

    本实现把通道数从 256 降到 64：全尺寸（256 通道、不下采样）在 128×128、
    4GB 显存下无法训练（单层激活就 ~16MB×50 层）。
    这是规模缩减，不是结构改动，论文里必须写明。

    官方预训练权重（Google Drive）在国内网络不可达，因此这里按原结构自行训练；
    若后续拿到 carl.h5，可用 convert_dsen2cr_weights.py 转换后替换。
    """

    def __init__(self, cin=16, cout=13, F=64, B=16):
        super().__init__()
        layers = [nn.Conv2d(cin, F, 3, padding=1, bias=True), nn.ReLU(True)]
        layers += [_DSen2Block(F) for _ in range(B)]
        layers += [nn.Conv2d(F, cout, 3, padding=1, bias=True)]
        self.model = nn.Sequential(*layers)
        self.cout = cout

    def forward(self, x):
        # 长跳跃：输出 = 有云光学（build_input 的前 13 通道）+ 网络残差
        return x[:, : self.cout, ...] + self.model(x)


@torch.no_grad()
def predict(net, use_sar, optical, sar, mask, device="cuda"):
    """推理整幅 patch，并只把云区替换掉（云区外保持原值）。"""
    x = build_input(optical, sar, use_sar)
    t = torch.from_numpy(x)[None].to(device)
    y = net(t)[0].cpu().numpy()
    m = mask.astype(bool)
    out = optical.copy()
    out[:, m] = y[:, m]
    return np.clip(out, 0, 1.5).astype(np.float32)


# ---------------------------------------------------------------- 条件扩散
def load_diffusion(device="cuda", base=32, ddim_steps=20, name="diffusion"):
    """加载条件扩散模型（结构与 train_diffusion.py 一致）。"""
    from diffusers import DDIMScheduler
    import importlib
    td = importlib.import_module("train_diffusion")
    net = td.make_unet(base).to(device)
    ck = CKPT_DIR / f"{name}.pt"
    if not ck.exists():
        raise FileNotFoundError(f"缺少权重 {ck}")
    net.load_state_dict(torch.load(ck, map_location=device))
    net.eval()
    sched = DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2",
                          clip_sample=True, set_alpha_to_one=False)
    return net, sched, ddim_steps


@torch.no_grad()
def predict_diffusion(bundle, optical, sar, mask, device="cuda"):
    import importlib
    td = importlib.import_module("train_diffusion")
    net, sched, steps = bundle
    cond = torch.from_numpy(build_input(optical, sar, True)[None]).to(device)
    gen = td.sample(net, cond, sched, steps)[0].cpu().numpy()
    m = mask.astype(bool)
    out = optical.copy()
    out[:, m] = gen[:, m]
    return np.clip(out, 0, 1.5).astype(np.float32)
