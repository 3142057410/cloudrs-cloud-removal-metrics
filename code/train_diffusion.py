"""
train_diffusion.py —— 小型条件扩散去云模型（本地 4GB 可训练/推理）

为什么必须补这一族：
  云去除领域当前的主流是生成式（GAN / 扩散），而生成式方法恰恰最容易出现
  "像素指标好、但云下地物是编出来的"这一现象。如果评测里没有生成式方法，
  论文关于"指标是否足以反映可用性"的结论就缺了最关键的证据。

实现（刻意做小，4GB 显存可跑）：
  条件 = cat[有云光学, 云掩膜, SAR标准化]   （16 通道）
  去噪目标 = 晴空光学                       （13 通道）
  UNet 输入 = cat[x_t, 条件]，即 29 → 13 通道
  训练用 DDPM（预测噪声），采样用 DDIM 20 步
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
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import load_patch
from models import CKPT_DIR, build_input


def make_unet(base=32, cond_ch=16, out_ch=13):
    return UNet2DModel(
        sample_size=128,
        in_channels=out_ch + cond_ch,
        out_channels=out_ch,
        layers_per_block=1,
        block_out_channels=(base, base * 2, base * 4),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
        norm_num_groups=8,
    )


@torch.no_grad()
def sample(net, cond, sched_ddim, steps=20, generator=None):
    """DDIM 采样，条件固定的情况下从噪声生成晴空影像。"""
    sched_ddim.set_timesteps(steps)
    x = torch.randn(cond.shape[0], 13, cond.shape[2], cond.shape[3],
                    device=cond.device, generator=generator)
    for t in sched_ddim.timesteps:
        model_in = torch.cat([x, cond], 1)
        eps = net(model_in, t).sample
        x = sched_ddim.step(eps, t, x).prev_sample
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p1.yaml")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0, help="随机种子；不同种子用于稳定性评估")
    ap.add_argument("--tag", default="", help="权重文件名后缀，如 _s1")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    cache = Path(cfg["data"]["cache_dir"])
    split = json.load(open(cache / "split.json", encoding="utf-8"))
    tr, va = split["train"], split["val"]
    if args.limit:
        tr, va = tr[: args.limit], va[: max(1, args.limit // 4)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = make_unet(args.base).to(device)
    n_par = sum(p.numel() for p in net.parameters()) / 1e6
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched_noise = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    sched_ddim = DDIMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2",
                               clip_sample=True, set_alpha_to_one=False)

    name = f"diffusion{args.tag}"
    CKPT_DIR.mkdir(exist_ok=True)
    print("=" * 68)
    print(f"训练条件扩散 {name} | 参数 {n_par:.2f}M | 训练 {len(tr)} / 验证 {len(va)}")
    print(f"device {device} | batch {args.batch} | steps {args.steps}")
    print("=" * 68)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    t0 = time.time()
    best = -1e9
    n = len(tr)
    step = 0
    ep = 0
    while step < args.steps:
        ep += 1
        perm = rng.permutation(n)
        for i in range(0, n, args.batch):
            if step >= args.steps:
                break
            idx = perm[i:i + args.batch]
            conds, x0s = [], []
            for j in idx:
                d = load_patch(cache, tr[j])
                conds.append(build_input(d["optical"], d["sar"], True))
                x0s.append(d["target"])
            cond = torch.from_numpy(np.stack(conds)).to(device)
            x0 = torch.from_numpy(np.stack(x0s)).to(device)
            eps = torch.randn_like(x0)
            t = torch.randint(0, 1000, (x0.shape[0],), device=device).long()
            xt = sched_noise.add_noise(x0, eps, t)
            pred = net(torch.cat([xt, cond], 1), t).sample
            loss = F.mse_loss(pred, eps)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 250 == 0:
                el = time.time() - t0
                print(f"  step {step}/{args.steps}  loss {loss.item():.4f}  "
                      f"用时 {el/60:.1f} min", flush=True)
            if step % 1000 == 0 or step == args.steps:
                net.eval()
                ps = []
                for vn in va[:12]:
                    d = load_patch(cache, vn)
                    c = torch.from_numpy(build_input(d["optical"], d["sar"], True)[None]).to(device)
                    gen = sample(net, c, sched_ddim, 20)
                    m = d["mask"].astype(bool)
                    pred_np = d["optical"].copy()
                    pred_np[:, m] = gen[0].cpu().numpy()[:, m]
                    mse = ((pred_np[:, m] - d["target"][:, m]) ** 2).mean()
                    ps.append(10 * np.log10(1 / max(mse, 1e-8)))
                cur = float(np.mean(ps))
                flag = ""
                if cur > best:
                    best = cur
                    torch.save(net.state_dict(), CKPT_DIR / f"{name}.pt")
                    flag = "  <- best"
                print(f"  step {step}  验证 PSNR {cur:.2f} dB{flag}", flush=True)
                net.train()

    print(f"\n完成，用时 {(time.time()-t0)/60:.1f} 分钟 | 最优 valPSNR {best:.2f} dB")
    print(f"权重 -> {CKPT_DIR/f'{name}.pt'}")
    print("=" * 68)


if __name__ == "__main__":
    main()
