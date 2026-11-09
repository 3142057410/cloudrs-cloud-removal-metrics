"""Revision analyses for P0-1 (PSNR range restriction) and P0-3 (pseudoreplication).

Outputs a JSON summary used to write the revised paragraphs of Section 4.2/4.9.
"""
import csv, json, math, os, random, statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
# the script sits either in <project>/_repo/code or in <project>/_review
for _cand in (os.path.join(HERE, os.pardir, "tables", "main"),
              os.path.join(HERE, os.pardir, "_repo", "tables", "main")):
    if os.path.isdir(_cand):
        TM = os.path.abspath(_cand)
        break
else:
    raise SystemExit("cannot locate tables/main next to this script")
OUT = os.path.join(os.getcwd(), "truncation_stats.json")

METRICS = ["PSNR", "SSIM", "SAM", "ERGAS", "MAE", "RMSE"]


def spearman(x, y):
    n = len(x)
    rx = _rank(x)
    ry = _rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy)


def _rank(v):
    idx = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and v[idx[j + 1]] == v[idx[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[idx[k]] = avg
        i = j + 1
    return r


# ---------- method-level table ----------
ms = {r["method"]: r for r in csv.DictReader(open(os.path.join(TM, "method_summary.csv")))}
dla = defaultdict(list)
for r in csv.DictReader(open(os.path.join(TM, "per_patch_downstream.csv"))):
    dla[r["method"]].append(float(r["dLAI_mean"]))
dLAI = {m: st.mean(v) for m, v in dla.items()}

methods = sorted(ms)
print("methods:", len(methods))
psnr = {m: float(ms[m]["PSNR_mean"]) for m in methods}
full = [m for m in methods]
qual = [m for m in methods if psnr[m] >= 25.0]
print("qualified (PSNR>=25):", len(qual), sorted(qual))

res = {"n_full": len(full), "n_qual": len(qual)}


def rho_pool(pool, metric):
    return spearman([float(ms[m][f"{metric}_mean"]) for m in pool], [dLAI[m] for m in pool])


# ---------- 1. range restriction on the selection variable ----------
sd_psnr_full = st.pstdev([psnr[m] for m in full])
sd_psnr_qual = st.pstdev([psnr[m] for m in qual])
sd_sam_full = st.pstdev([float(ms[m]["SAM_mean"]) for m in full])
sd_sam_qual = st.pstdev([float(ms[m]["SAM_mean"]) for m in qual])
r_psnr_qual = rho_pool(qual, "PSNR")
r_sam_qual = rho_pool(qual, "SAM")

# Thorndike Case II: true r in the unrestricted population
def thorndike(r, sd_r, sd_f):
    u = sd_f / sd_r
    return r * u / math.sqrt(1 + r * r * (u * u - 1))


res["range_restriction"] = {
    "sd_PSNR_full": round(sd_psnr_full, 3),
    "sd_PSNR_qual": round(sd_psnr_qual, 3),
    "sd_SAM_full": round(sd_sam_full, 3),
    "sd_SAM_qual": round(sd_sam_qual, 3),
    "ratio_PSNR": round(sd_psnr_qual / sd_psnr_full, 3),
    "ratio_SAM": round(sd_sam_qual / sd_sam_full, 3),
    "rho_PSNR_qual": round(r_psnr_qual, 3),
    "rho_SAM_qual": round(r_sam_qual, 3),
    "rho_PSNR_corrected": round(thorndike(r_psnr_qual, sd_psnr_qual, sd_psnr_full), 3),
    "rho_SAM_corrected": round(thorndike(r_sam_qual, sd_sam_qual, sd_sam_full), 3),
}

# ---------- 2. pools defined without reference to any image metric ----------
pools = {
    "full15": full,
    "dLAI_le_0.6": [m for m in full if dLAI[m] <= 0.60],
    "dLAI_le_0.5": [m for m in full if dLAI[m] <= 0.50],
    "learning_only": [m for m in full if m.startswith(("unet", "dsen2cr", "uncrtaints", "diffusion"))],
    "nonlearning_only": [m for m in full if not m.startswith(("unet", "dsen2cr", "uncrtaints", "diffusion"))],
}
res["pools"] = {}
for name, pool in pools.items():
    if len(pool) < 4:
        continue
    d = {"n": len(pool), "members": sorted(pool)}
    for mt in METRICS:
        d[mt] = round(rho_pool(pool, mt), 3)
    res["pools"][name] = d
    print(f"{name:16s} n={len(pool):2d} " + " ".join(f"{mt}={d[mt]:+.3f}" for mt in METRICS))

# ---------- 3. pseudoreplication: scene-wise replication ----------
rows = list(csv.DictReader(open(os.path.join(TM, "per_patch_metrics.csv"))))
by_scene = defaultdict(list)
import re
for r in rows:
    scene = re.sub(r"_p\d+$", "", r["patch"])
    by_scene[scene].append(r)

# per-patch dLAI lookup
dpatch = {}
for r in csv.DictReader(open(os.path.join(TM, "per_patch_downstream.csv"))):
    dpatch[(r["method"], r["patch"])] = float(r["dLAI_mean"])

res["scenes"] = {}
for scene, rr in sorted(by_scene.items()):
    methods_s = sorted(set(x["method"] for x in rr))
    d = {"n_patches": len(rr) // len(methods_s), "n_methods": len(methods_s)}
    # method means within this scene
    agg = defaultdict(lambda: defaultdict(list))
    dagg = defaultdict(list)
    for x in rr:
        for mt in METRICS:
            agg[x["method"]][mt].append(float(x[mt]))
        dagg[x["method"]].append(dpatch[(x["method"], x["patch"])])
    ml = {m: st.mean(dagg[m]) for m in methods_s}
    for mt in METRICS:
        d[mt] = round(spearman([st.mean(agg[m][mt]) for m in methods_s], [ml[m] for m in methods_s]), 3)
    # patch level
    for mt in METRICS:
        d[mt + "_patch"] = round(spearman([float(x[mt]) for x in rr], [dpatch[(x["method"], x["patch"])] for x in rr]), 3)
    res["scenes"][scene] = d
    print(scene, d["n_patches"], "patches | method-level",
          " ".join(f"{mt}={d[mt]:+.3f}" for mt in METRICS))

# ---------- 4. scene-clustered bootstrap of the method-level rho ----------
random.seed(7)
def scene_cluster_boot(metric, n_boot=2000):
    scenes = sorted(by_scene)
    out = []
    for _ in range(n_boot):
        pick = [random.choice(scenes) for _ in scenes]  # resample scenes with replacement
        agg = defaultdict(lambda: defaultdict(list))
        dag = defaultdict(list)
        for sc in pick:
            for x in by_scene[sc]:
                agg[x["method"]][metric].append(float(x[metric]))
                dag[x["method"]].append(dpatch[(x["method"], x["patch"])])
        ms_ = sorted(agg)
        if len(ms_) < 5:
            continue
        out.append(spearman([st.mean(agg[m][metric]) for m in ms_], [st.mean(dag[m]) for m in ms_]))
    out.sort()
    lo = out[int(0.025 * len(out))]
    hi = out[int(0.975 * len(out))]
    return round(lo, 3), round(hi, 3), len(out)


res["scene_cluster_boot"] = {mt: scene_cluster_boot(mt, 500) for mt in ["PSNR", "SAM", "SSIM", "ERGAS"]}
print("scene-cluster bootstrap (2 clusters => degenerates to two-point support):",
      res["scene_cluster_boot"])

# ---------- 5. patch-level rho per scene, for the qualified pool ----------
res["qual_by_scene"] = {}
for scene, rr in sorted(by_scene.items()):
    methods_s = [m for m in sorted(set(x["method"] for x in rr)) if psnr[m] >= 25.0]
    agg = defaultdict(lambda: defaultdict(list))
    dag = defaultdict(list)
    for x in rr:
        if x["method"] not in methods_s:
            continue
        for mt in METRICS:
            agg[x["method"]][mt].append(float(x[mt]))
        dag[x["method"]].append(dpatch[(x["method"], x["patch"])])
    dl = {m: st.mean(dag[m]) for m in methods_s}
    res["qual_by_scene"][scene] = {"n_methods": len(methods_s)}
    for mt in METRICS:
        res["qual_by_scene"][scene][mt] = round(
            spearman([st.mean(agg[m][mt]) for m in methods_s], [dl[m] for m in methods_s]), 3)
    print("qual @" + scene, res["qual_by_scene"][scene])

json.dump(res, open(OUT, "w"), indent=2)
print("\nwritten", OUT)
