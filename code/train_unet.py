"""
train_unet.py —— 在 train 划分上训练轻量 U-Net 去云模型

用法：
  python src/train_unet.py --variant opt --epochs 40
  python src/train_unet.py --variant sar --epochs 40

设计要点：
  · 只用 cloud 区算损失（云区外本来就对，不该主导梯度）
  · 随机翻转/旋转做增强（训练集只有 612 个 patch，必须增强）
  · 验证集按 PSNR 选最优 epoch，避免过拟合
  · 4GB 显存：128×128、batch 4，约 2.5GB 峰值
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import load_patch
from models import CKPT_DIR, UNet, build_input


def _flip(a, axis):
    """mask 是 (H,W)、影像是 (C,H,W)，轴号要跟着维度走。"""
    return np.flip(a, axis) if a.ndim == 3 else np.flip(a, axis - 1)


def _rot90(a, k):
    return np.rot90(a, k, (1, 2)) if a.ndim == 3 else np.rot90(a, k)


def aug(opt, tgt, sar, mask, rng):
    if rng.random() < .5:
        opt, tgt, sar, mask = (_flip(a, 2) for a in (opt, tgt, sar, mask))
    if rng.random() < .5:
        opt, tgt, sar, mask = (_flip(a, 1) for a in (opt, tgt, sar, mask))
    k = int(rng.integers(0, 4))
    if k:
        opt, tgt, sar, mask = (_rot90(a, k) for a in (opt, tgt, sar, mask))
    return (np.ascontiguousarray(a) for a in (opt, tgt, sar, mask))


def masked_l1(pred, tgt, mask):
    """云区损失 + 全图损失（各占一半），权重固定，不做手调。"""
    m = mask[:, None].bool()
    l_cl = (pred * m - tgt * m).abs().sum() / (m.sum() * pred.shape[1] + 1e-6)
    return 0.5 * l_cl + 0.5 * F.l1_loss(pred, tgt)


@torch.no_grad()
def evaluate(net, names, cache, use_sar, device):
    psnrs = []
    for n in names:
        d = load_patch(cache, n)
        x = build_input(d["optical"], d["sar"], use_sar)
        y = net(torch.from_numpy(x)[None].to(device))[0].cpu().numpy()
        m = d["mask"].astype(bool)
        pred = d["optical"].copy()
        pred[:, m] = y[:, m]
        mse = ((pred[:, m] - d["target"][:, m]) ** 2).mean()
        psnrs.append(10 * np.log10(1 / max(mse, 1e-8)))
    return float(np.mean(psnrs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p1.yaml")
    ap.add_argument("--variant", choices=["opt", "sar"], default="sar")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="权重文件名后缀，如 _s1；空则用默认名")
    ap.add_argument("--arch", choices=["unet", "dsen2cr"], default="unet",
                    help="unet = 本文轻量 U-Net；dsen2cr = 复现 Meraner2020 的残差结构")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    cache = Path(cfg["data"]["cache_dir"])
    split = json.load(open(cache / "split.json", encoding="utf-8"))
    tr, va = split["train"], split["val"]
    if args.limit:
        tr, va = tr[: args.limit], va[: max(1, args.limit // 4)]

    use_sar = args.variant == "sar"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cin = 13 + 1 + (2 if use_sar else 0)
    if args.arch == "dsen2cr":
        from models import DSen2CRNet
        net = DSen2CRNet(cin=cin, cout=13, F=64).to(device)
    else:
        net = UNet(cin=cin, cout=13, base=32).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    n_par = sum(p.numel() for p in net.parameters()) / 1e6

    name = f"{args.arch}_{args.variant}{args.tag}"
    CKPT_DIR.mkdir(exist_ok=True)
    print("=" * 68)
    print(f"训练 {name} | 参数 {n_par:.2f}M | 输入 {cin} 通道 | 训练 {len(tr)} / 验证 {len(va)}")
    print(f"设备 {device} | batch {args.batch} | epochs {args.epochs} | seed {args.seed}")
    print("=" * 68)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    best = -1e9
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        net.train()
        perm = rng.permutation(len(tr))
        tot, nb = 0.0, 0
        for i in range(0, len(tr), args.batch):
            idx = perm[i:i + args.batch]
            xs, ys, ms = [], [], []
            for j in idx:
                d = load_patch(cache, tr[j])
                o, t, s, m = aug(d["optical"], d["target"], d["sar"], d["mask"].astype(np.float32), rng)
                xs.append(build_input(o, s, use_sar))
                ys.append(t)
                ms.append(m)
            x = torch.from_numpy(np.stack(xs)).to(device)
            y = torch.from_numpy(np.stack(ys)).to(device)
            mk = torch.from_numpy(np.stack(ms)).to(device)
            pred = net(x)
            loss = masked_l1(pred, y, mk)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        sched.step()
        ps = evaluate(net, va, cache, use_sar, device)
        flag = ""
        if ps > best:
            best = ps
            torch.save(net.state_dict(), CKPT_DIR / f"{name}.pt")
            flag = "  <- best"
        print(f"  ep {ep:3d}/{args.epochs}  loss {tot/max(nb,1):.4f}  valPSNR {ps:.2f} dB{flag}", flush=True)

    print(f"\n完成，用时 {(time.time()-t0)/60:.1f} 分钟 | 最优 valPSNR {best:.2f} dB")
    print(f"权重 -> {CKPT_DIR/f'{name}.pt'}")


if __name__ == "__main__":
    main()
