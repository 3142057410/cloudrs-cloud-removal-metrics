"""
eval_seeds.py —— 种子稳定性评估（投稿 checklist：报 3 个种子的均值±标准差）

覆盖全部 6 个学习型基线：
    unet / dsen2cr   × opt/sar × seed 0/1/2
    diffusion        × seed 0/1/2
    uncrtaints_ft    × seed 0/11/22

同一个测试集、同一掩膜、同一反演器下重算 PSNR / SAM / ΔLAI。

输出：
    results/tables/seed_stability.csv
    results/tables/seed_stability_summary.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import USABLE_IDX, load_patch
from downstream import DownstreamRetrieval, retrieval_delta
from metrics import sam
from models import CKPT_DIR, load_diffusion, load_model, predict, predict_diffusion

FT_DIR = Path("third_party/weights/uncrtaints_ft")

COMBOS = []
for arch in ("unet", "dsen2cr"):
    for var in ("opt", "sar"):
        for seed in (0, 1, 2):
            name = f"{arch}_{var}" if seed == 0 else f"{arch}_{var}_s{seed}"
            COMBOS.append((arch, var, seed, name))
for seed in (0, 1, 2):
    COMBOS.append(("diffusion", "cond", seed,
                   "diffusion" if seed == 0 else f"diffusion_s{seed}"))
for seed, tag in ((0, "ft10"), (11, "ft10_s11"), (22, "ft10_s22")):
    COMBOS.append(("uncrtaints", "sar", seed, tag))


def ckpt_path(arch, name):
    if arch == "uncrtaints":
        return FT_DIR / f"{name}.pth.tar"
    return CKPT_DIR / f"{name}.pt"


def infer(arch, name, d, device):
    """按模型族分派推理，返回 (13,H,W) 预测。"""
    if arch == "diffusion":
        key = f"diff_{name}"
        if key not in _CACHE:
            net, sched, steps = load_diffusion(device, name=name)
            _CACHE[key] = (net, sched, steps)
        net, sched, steps = _CACHE[key]
        return predict_diffusion((net, sched, steps), d["optical"], d["sar"], d["mask"], device)
    if arch == "uncrtaints":
        from uncrtaints_adapter import ensure_weights, predict as upred
        ensure_weights(ckpt_path(arch, name), device)
        return upred(d["optical"], d["sar"], d["mask"])
    net, use_sar = load_model(name, device)
    return predict(net, use_sar, d["optical"], d["sar"], d["mask"], device)


_CACHE: dict = {}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", default="unet,dsen2cr,diffusion,uncrtaints",
                    help="按模型族分批跑，避免一次加载太多模型导致显存耗尽")
    ap.add_argument("--append", action="store_true", help="与已有结果合并")
    args = ap.parse_args()
    keep = set(args.archs.split(","))

    cfg = yaml.safe_load(open("configs/p1.yaml", encoding="utf-8"))
    cache = Path(cfg["data"]["cache_dir"])
    names = json.load(open(cache / "split.json", encoding="utf-8"))["test"]
    ds = DownstreamRetrieval(cfg["downstream"]["lut_path"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    for arch, var, seed, name in COMBOS:
        if arch not in keep:
            continue
        ck = ckpt_path(arch, name)
        if not ck.exists():
            print(f"  跳过 {name}（无权重）", flush=True)
            continue
        ps, sa, dl = [], [], []
        for n in names:
            d = load_patch(cache, n)
            m = d["mask"].astype(bool)
            if m.sum() == 0:
                continue
            pred = infer(arch, name, d, device)
            p, t = pred[USABLE_IDX], d["target"][USABLE_IDX]
            mse = ((p[:, m] - t[:, m]) ** 2).mean()
            ps.append(10 * np.log10(1 / max(mse, 1e-8)))
            sa.append(sam(p, t, mask=d["mask"]))
            dl.append(retrieval_delta(ds, pred, d["target"], d["mask"])["dLAI_mean"])
        rows.append({"arch": arch, "variant": var, "seed": seed, "ckpt": name,
                     "n_patch": len(ps),
                     "PSNR": float(np.mean(ps)), "SAM": float(np.mean(sa)),
                     "dLAI": float(np.mean(dl))})
        print(f"  {name:16s} PSNR {rows[-1]['PSNR']:.2f}  SAM {rows[-1]['SAM']:.2f}  "
              f"ΔLAI {rows[-1]['dLAI']:.4f}", flush=True)
        # 每族跑完立即释放显存，否则连续加载十几个模型会静默退出
        _CACHE.clear()
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    out = Path(cfg["output"]["tables"])
    out.mkdir(parents=True, exist_ok=True)
    if args.append and (out / "seed_stability.csv").exists():
        old = pd.read_csv(out / "seed_stability.csv")
        df = pd.concat([old, df], ignore_index=True).drop_duplicates(
            subset=["ckpt"], keep="last")
    df.to_csv(out / "seed_stability.csv", index=False)

    agg = df.groupby(["arch", "variant"])[["PSNR", "SAM", "dLAI"]].agg(["mean", "std"]).round(4)
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(out / "seed_stability_summary.csv", index=False)
    print("\n种子稳定性汇总（mean ± std）")
    print(agg.to_string(index=False))
    print(f"\n-> {out/'seed_stability.csv'}")


if __name__ == "__main__":
    main()
