"""
run_uncrtaints.py —— 把 UnCRtainTS 系列权重接入主评估表

两种推理模式：
  --from-cache   用缓存里的 128 块直接推理。适用于**域内微调过**的权重
                 （微调就在 128 上做的，口径一致）。
  默认（native） 回到 parquet 的原始 256 推理，再降采样到 128 评价。
                 适用于官方原始权重：它在 256 上训练，喂 128 会严重过平滑。

产出（追加 / 覆盖同名方法）：
  results/tables/per_patch_metrics.csv        图像指标
  results/tables/per_patch_downstream.csv     反演误差 ΔLAI / ΔCab
  results/tables/ndvi_task.csv                NDVI 分类误差 Δclass

用法：
  python src/run_uncrtaints.py --name uncrtaints_raw --from-cache      # 官方权重，128 口径
  python src/run_uncrtaints.py --name uncrtaints_ft --from-cache \
         --weights third_party/weights/uncrtaints_ft/ft10.pth.tar      # 微调权重
  python src/run_uncrtaints.py --name uncrtaints_r256                  # 官方权重，原生 256
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import USABLE_IDX, iter_samples, load_patch, make_cloud_mask, warp_resize
from downstream import DownstreamRetrieval, retrieval_delta
from metrics import evaluate_all
from ndvi_task import classify
from uncrtaints_adapter import get_model, predict, predict_native


def build_index(cfg, need):
    """扫一遍 parquet，只留下需要的 patch（按 season_scene_idx 命名）。"""
    raw = Path(cfg["data"]["raw_dir"])
    idx = {}
    for s in iter_samples(sorted(raw.glob("*.parquet"))):
        n = f"{s['season']}_{s['scene']}_{s['idx']}"
        if n in need:
            idx[n] = s
        if len(idx) == len(need):
            break
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p1.yaml")
    ap.add_argument("--name", required=True, help="写进结果表的方法名")
    ap.add_argument("--weights", default=None, help="可选，加载指定 .pth.tar（微调权重）")
    ap.add_argument("--from-cache", action="store_true", help="用 128 缓存块推理")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    tab = Path(cfg["output"]["tables"])
    cache = Path(cfg["data"]["cache_dir"])
    ps = cfg["data"]["patch_size"]
    names = json.load(open(cache / "split.json", encoding="utf-8"))["test"]
    if args.limit:
        names = names[: args.limit]

    raw = None
    if not args.from_cache:
        raw = build_index(cfg, set(names))
        miss = [n for n in names if n not in raw]
        if miss:
            print(f"  警告：{len(miss)} 个 patch 在 parquet 里找不到，将跳过（如 {miss[0]}）")

    ds = DownstreamRetrieval(cfg["downstream"]["lut_path"])
    model = get_model()
    if args.weights:
        import torch
        ck = torch.load(args.weights, map_location="cuda")
        model.load_state_dict(ck["state_dict_G"])
        print(f"  已加载微调权重 {args.weights}"
              f"（epoch {ck.get('epoch','?')}，val {ck.get('val_psnr', float('nan')):.2f} dB）")

    mode = "128 缓存块" if args.from_cache else "原生 256 → 128 评价"
    print(f"测试 patch {len(names)} 个 | 方法 {args.name} | {mode}")

    rows_img, rows_dn, rows_nd = [], [], []
    t0 = time.time()
    for i, n in enumerate(names, 1):
        if args.from_cache:
            d = load_patch(cache, n)
            pred = predict(d["optical"], d["sar"], d["mask"])
            tgt = d["target"]
            mask = d["mask"]
        else:
            s = raw.get(n)
            if s is None:
                continue
            opt = s["optical"] / cfg["data"]["scale"]
            tgt = warp_resize(s["target"] / cfg["data"]["scale"], ps)
            pred = warp_resize(predict_native(opt, s["sar"]), ps)
            mask = make_cloud_mask(warp_resize(opt, ps), cfg["data"]["cloud_threshold"])

        if mask.sum() < 50:
            continue

        res = evaluate_all(pred[USABLE_IDX], tgt[USABLE_IDX], mask=mask,
                           data_range=cfg["metrics"]["data_range"])
        res.update({"method": args.name, "patch": n})
        rows_img.append(res)

        dd = retrieval_delta(ds, pred, tgt, mask)
        dd.update({"method": args.name, "patch": n})
        rows_dn.append(dd)

        mb = mask.astype(bool)
        def ndvi(img):
            nir, red = img[7][mb], img[3][mb]
            return (nir - red) / (nir + red + 1e-6)
        cp, ct = classify(ndvi(pred)), classify(ndvi(tgt))
        rows_nd.append({"method": args.name, "patch": n, "n_px": int(mb.sum()),
                        "OA": float((cp == ct).mean()),
                        "dclass": float(1.0 - (cp == ct).mean()),
                        "dNDVI": float(np.mean(np.abs(ndvi(pred) - ndvi(tgt))))})

        if i % 100 == 0:
            print(f"  {i}/{len(names)}  PSNR "
                  f"{np.mean([r['PSNR'] for r in rows_img]):.2f} dB  "
                  f"用时 {(time.time()-t0)/60:.1f} 分钟", flush=True)

    def upsert(df, out):
        if out.exists():
            old = pd.read_csv(out)
            old = old[old.method != args.name]
            df = pd.concat([old, df], ignore_index=True)
        df.to_csv(out, index=False)
        return df

    img = upsert(pd.DataFrame(rows_img), tab / "per_patch_metrics.csv")
    upsert(pd.DataFrame(rows_dn), tab / "per_patch_downstream.csv")
    upsert(pd.DataFrame(rows_nd), tab / "ndvi_task.csv")

    # method_summary.csv 原本只由 run_baselines 生成，追加方法后必须同步重建，否则论文读到旧表
    cols = ["PSNR", "SSIM", "SAM", "ERGAS", "MAE", "RMSE"]
    summ = img.groupby("method")[cols].agg(["mean", "std"]).round(4)
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(tab / "method_summary.csv", index=False)
    print(f"  已重建 method_summary.csv（{len(summ)} 个方法）")

    m = pd.DataFrame(rows_img)
    print("\n" + "=" * 60)
    print(f"{args.name} 汇总（{len(m)} patch，用时 {(time.time()-t0)/60:.1f} 分钟）")
    print(f"  PSNR {m.PSNR.mean():.2f} dB  SSIM {m.SSIM.mean():.3f}  "
          f"SAM {m.SAM.mean():.2f}°  ERGAS {m.ERGAS.mean():.2f}")
    if rows_dn:
        print(f"  ΔLAI {pd.DataFrame(rows_dn).dLAI_mean.mean():.3f}  "
              f"ΔCab {pd.DataFrame(rows_dn).dCab_mean.mean():.3f}  "
              f"Δclass {pd.DataFrame(rows_nd).dclass.mean():.3f}")
    print(f"  表内现有 {img.method.nunique()} 个方法 | {len(img)} 行")
    print("=" * 60)


if __name__ == "__main__":
    main()
