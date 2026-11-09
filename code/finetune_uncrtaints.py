"""
finetune_uncrtaints.py —— UnCRtainTS 官方权重的域内微调（诊断用）

目的不是刷分，是判定「26.5 dB 到底是接入 bug 还是域偏移」：
  - 若微调能到 31 dB 以上 → 数据管线正确，差距来自云影/辐射条件差异
  - 若卡在 27 左右不上 → 接入仍有问题，梯度学不动，应立即停手

保守设置：
  · 全程 model.eval()（冻结 BatchNorm 的 running stats 与 dropout），
    只更新卷积/线性权重 —— 这样任何提升都来自权重适配，不会因 BN 统计漂移而虚高
  · 训练集与验证集来自 split.json 的 train / val，测试集只在最后评估一次，不参与选模型
  · 损失用 L1（对薄云边缘更稳），只在可用 10 波段上回传

用法：
  python src/finetune_uncrtaints.py --epochs 8
  python src/finetune_uncrtaints.py --epochs 2 --limit 32      # 冒烟
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import USABLE_IDX, load_patch
from metrics import psnr
from uncrtaints_adapter import get_model, _norm_sar

PROJ = Path(__file__).resolve().parent.parent
CACHE = PROJ / "data" / "processed"
OUT = PROJ / "third_party" / "weights" / "uncrtaints_ft"


def make_input(d):
    """缓存块 → 模型输入 (C,H,W)。与 uncrtaints_adapter.predict 完全同一口径。"""
    sar = _norm_sar(d["sar"], "rescale01")
    opt = d["optical"].astype(np.float32)
    return np.concatenate([sar, opt], 0)


def to_batch(items):
    """items: list of 输入数组 → (B,1,C,H,W) tensor，已乘 scale_by。"""
    x = np.stack(items, 0) * 10.0
    return torch.from_numpy(x)[:, None].cuda()


@torch.no_grad()
def evaluate(model, names, verbose=True):
    model.eval()
    vals, sams = [], []
    for n in names:
        d = load_patch(CACHE, n)
        y = model(to_batch([make_input(d)]))[0, 0].cpu().numpy() / 10.0
        y = np.clip(y, 0, 1)
        vals.append(psnr(y[USABLE_IDX], d["target"][USABLE_IDX], mask=d["mask"]))
    if verbose:
        print(f"    评估 PSNR {np.mean(vals):.2f} dB（{len(vals)} patch）", flush=True)
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="ft", help="权重文件名（不含扩展名）")
    ap.add_argument("--seed", type=int, default=1, help="数据顺序种子")
    args = ap.parse_args()

    split = json.load(open(CACHE / "split.json", encoding="utf-8"))
    tr, va, te = split["train"], split["val"], split["test"]
    if args.limit:
        tr, va = tr[: args.limit], va[: max(8, args.limit // 4)]

    model = get_model()
    model.eval()                                  # 冻结 BN / dropout
    for p in model.parameters():
        p.requires_grad_(True)

    print("=" * 66)
    print(f"训练 {len(tr)} / 验证 {len(va)} / 测试 {len(te)}（测试集不参与选模型）")
    print(f"epochs {args.epochs} | bs {args.bs} | lr {args.lr} | 冻结 BN")
    print("=" * 66)
    base = evaluate(model, va)
    print(f"  起点（官方权重）验证 PSNR {base:.2f} dB\n")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    best = {"psnr": base, "epoch": 0}
    OUT.mkdir(parents=True, exist_ok=True)
    hist = []
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.eval()
        order = np.random.default_rng(args.seed * 1000 + ep).permutation(len(tr))
        losses = []
        for i in range(0, len(tr) - args.bs + 1, args.bs):
            idx = order[i : i + args.bs]
            xs, ys = [], []
            for j in idx:
                d = load_patch(CACHE, tr[j])
                xs.append(make_input(d))
                ys.append(d["target"].astype(np.float32))
            x = to_batch(xs)
            y = torch.from_numpy(np.stack(ys, 0)).cuda()
            pred = model(x)[:, 0, USABLE_IDX] / 10.0        # (B,10,H,W)
            loss = F.l1_loss(pred, y[:, USABLE_IDX])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss))
        sched.step()
        ps = evaluate(model, va)
        hist.append({"epoch": ep, "loss": float(np.mean(losses)), "val_psnr": ps})
        print(f"  epoch {ep}/{args.epochs}  L1 {np.mean(losses):.4f}  "
              f"val {ps:.2f} dB  用时 {(time.time()-t0)/60:.1f} 分钟", flush=True)
        if ps > best["psnr"]:
            best = {"psnr": ps, "epoch": ep}
            torch.save({"state_dict_G": model.state_dict(), "epoch": ep, "val_psnr": ps},
                       OUT / f"{args.tag}.pth.tar")

    print("\n" + "=" * 66)
    print(f"最佳 epoch {best['epoch']}，验证 {best['psnr']:.2f} dB（起点 {base:.2f}，"
          f"提升 {best['psnr']-base:+.2f}）")

    # 用最佳权重在测试集上评估一次
    if (OUT / f"{args.tag}.pth.tar").exists():
        ck = torch.load(OUT / f"{args.tag}.pth.tar", map_location="cuda")
        model.load_state_dict(ck["state_dict_G"])
    print(f"\n测试集（{len(te)} patch，只评估一次）：")
    test_psnr = evaluate(model, te)

    print("\n判定：")
    if test_psnr >= 30.5:
        print("  ✓ 收敛良好 → 数据管线正确，原 26.5 dB 是域偏移，可作为「域内微调」基线")
    elif test_psnr >= 28.5:
        print("  ~ 有提升但不足 → 管线基本正确、适配不充分，需要更多 epoch 或更大 lr")
    else:
        print("  ✗ 学不动 → 接入很可能仍有问题，建议停止投入")
    (OUT / f"{args.tag}_history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    print(f"\n权重 {OUT/f'{args.tag}.pth.tar'}")
    print("=" * 66)


if __name__ == "__main__":
    main()
