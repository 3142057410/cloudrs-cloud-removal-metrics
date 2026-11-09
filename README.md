# Image-domain metrics misrank cloud-removal models: reproducibility package

Code, derived tables and figures for the manuscript

> **Image-domain metrics misrank cloud-removal models for Sentinel-2 LAI, Cab and
> NDVI retrieval**
> submitted to *International Journal of Applied Earth Observation and Geoinformation*

The study asks whether the four metrics normally used to accept a cloud-removal model
(PSNR, SSIM, SAM, ERGAS) rank models in the same order as the accuracy of the
biophysical parameters retrieved from their output. It answers with 15 methods,
428 cloudy patches, two datasets, three downstream targets, and one operational
LAI product (MCD15A3H).

---

## Contents

| Directory | What it holds | Size |
|---|---|---|
| `code/` | All processing, training and analysis scripts (49 files). Entry points are listed below. | 532 KB |
| `tables/main/` | Per-method and per-patch results on the main dataset (SEN12MS-CR), 37 CSV files. | 8.5 MB |
| `tables/repro/` | Results on the independently built Sentinel-2 L2A dataset, the MODIS LAI product validation, and the two revision analyses (noise floor, range restriction), 9 CSV + 3 JSON. | 836 KB |
| `figures/` | `fig1.png` … `fig10.png`, the files actually printed as Figures 1–10. | 2.6 MB |
| `meta/` | Patch geolocation, Sentinel-2 pair inventory, MODIS granule index. | 52 KB |
| `configs/` | `p1.yaml`, the single configuration file every script reads. | 2 KB |
| `paper/` | `main.tex` (the submitted LaTeX source, snapshot), `main.aux`, and the scripts that build the Word version from it. Run `pdflatex` first, then `prep_tex.py` and `style_docx.py`; `pandoc` must be on PATH. | 164 KB |

**Table-to-file guide.** Each CSV header states its own columns, so any table in the
paper can be traced from the file name:

| Paper | File |
|---|---|
| Table 2 — metrics and downstream errors of the 15 methods | `tables/main/method_summary.csv`, `per_patch_downstream.csv` |
| Table 3 — rank correlation under three aggregation schemes | `tables/main/corr_method_level.csv`, `corr_sample_level.csv` |
| Table 4 — candidate-pool composition | `tables/main/subset_comparison.csv`, `highq_relative.csv` |
| Table 5 — band-level decomposition | `tables/main/band_mechanism.csv`, `per_band_psnr.csv` |
| Table 6 — stratification by cloud type / thickness / land cover | `tables/main/corr_by_cloudtype.csv`, `corr_by_cloudthick.csv`, `corr_by_landcover.csv` |
| Table 7 — three downstream tasks | `tables/main/corr_ndvi_task.csv`, `ndvi_task.csv` |
| Table 8 — task-consistency weights | `tables/main/tcws_weights.csv`, `tcws_weights_cab.csv` |
| Table 9 — mask boundary off by one pixel | `tables/main/mask_sensitivity_summary.csv` |
| Table 10 — seed stability | `tables/main/seed_stability_summary.csv` |
| Table 11 — error-mode composition | `tables/main/failure_decomposition.csv`, `failure_summary.csv` |
| Table 12 — main vs. reproduction dataset | `tables/main/corr_method_level.csv` + `tables/repro/corr_s2_crossregion.csv` |
| Table 13 — metrics and product error against MODIS LAI | `tables/repro/product_validation_method.csv`, `product_validation_corr.csv`, `product_validation_corr_boot.csv` |
| Table 14, Table 1 | compiled from the above; no separate source file |
| Section 4.1 noise floor (split-half, radiometric) | `tables/repro/noise_floor.csv`, `noise_floor.json`, `code/noise_floor.py` |
| Section 4.2 range restriction and per-scene replication | `tables/repro/truncation_stats.json`, `code/analysis_truncation.py` |

---

## Revision analyses

Two analyses were added during revision and can be recomputed from the archived tables:

* **Noise floor** (`code/noise_floor.py`, Section 4.1). Two null comparisons on the
  cloud-free reference alone: a split-half test (two disjoint 300-pixel samples of the
  same patch, mean |dLAI| = 0.63, median 0.44) and a 2% multiplicative-noise test
  (median 0.036). The second is eleven times smaller than the smallest between-method
  difference, which is why the paired difference is not retrieval-noise limited. Needs
  the processed patch cubes, so it runs only in a full working copy.
* **Range restriction** (`code/analysis_truncation.py`, Section 4.2). The qualified pool
  is defined on PSNR, so the pool halves the standard deviation of the variable it is
  used to correlate (4.94 dB over 15 methods, 2.44 dB over 9). The Case II correction
  moves the restricted coefficient from -0.867 to -0.962. The same script reports pools
  defined without any image metric (dLAI <= 0.6, learning only, non-learning only) and
  the per-scene replication that replaces a cluster bootstrap, since the 428 patches
  come from only two scenes.

## Requirements

Python 3.9. The code was developed and run under Linux and Windows with:

```
numpy  pandas  scipy  pyyaml  matplotlib  pyarrow
rasterio  opencv-python  scikit-image
torch  torchvision  diffusers
python-docx            # only for paper/style_docx.py
```

`rasterio` is needed only by the Sentinel-2 and MODIS readers; the main-dataset
analysis runs with numpy/pandas/scipy alone. A CUDA device is required to retrain the
learned baselines, but **not** to reproduce any table or figure — every number in the
paper can be recomputed from the archived tables.

### Directory convention

Every script takes its paths from `configs/p1.yaml` or from a command-line default, and
those defaults are **relative**. Run everything from the repository root:

```
repro-package/
├── configs/p1.yaml
├── code/
├── meta/
├── tables/
├── figures/
├── paper/
├── data/            # created locally, not shipped
└── results/         # created locally, not shipped
```

`prodval_bootstrap.py` and `replot_prodval.py` have no command-line options; they read
and write `results/tables_boa/` by fixed path, so keep the `--out` value of the
previous step as shown above if you want them to run unmodified.

---

## Reproduction

### 1. Environment

```bash
conda create -n cloudrs python=3.9
conda activate cloudrs
pip install numpy pandas scipy pyyaml matplotlib pyarrow rasterio opencv-python scikit-image
pip install torch torchvision diffusers
```

### 2. PROSAIL lookup table (not shipped, 19 MB)

The retrieval operator is an 8-nearest-neighbour lookup over a PROSAIL LUT. The LUT is
regenerated rather than archived:

```bash
pip install prosail
python code/gen_lut.py --config configs/p1.yaml
# -> data/processed/prosail_lut.npz
```

All retrieval numbers in the paper were produced from this file. If you swap PROSAIL
versions, expect small shifts in absolute LAI and note them; the *ranking* of methods,
which is what the paper tests, is insensitive to it.

### 3. Main dataset (SEN12MS-CR)

```bash
python code/build_cache.py --config configs/p1.yaml   # unpack and cache patches
python code/run_baselines.py --config configs/p1.yaml # nine non-learning operators
python code/run_downstream.py --config configs/p1.yaml # LAI / Cab / NDVI retrieval
python code/analyze.py --config configs/p1.yaml --bootstrap 1000
```

The nine learned baselines are retrained with `train_unet.py`, `train_diffusion.py`
and `finetune_uncrtaints.py`; their checkpoints are not shipped, but the metrics they
produced are in `tables/main/`.

### 4. Sentinel-2 reproduction dataset

```bash
python code/search_s2_pairs.py                                  # -> meta/s2_pairs.csv
python code/build_s2_all.py --out data/s2_processed_boa --boa-offset-dn -1000
python code/patch_geolocation.py --processed data/s2_processed_boa
python code/eval_s2_crossregion.py --processed data/s2_processed_boa --out results/tables_boa
```

`--boa-offset-dn` defaults to `-1000`, the value used throughout the paper; it is
stated explicitly above because that is the parameter that matters (see the note on
the BOA offset below). `patch_geolocation.py` only needs to be re-run if the patch
centres are rebuilt — the result is already shipped as `meta/patch_geo.csv`.

`build_s2_all.py` processes two date pairs per subprocess on purpose: loading a full
Scene Classification Layer product needs several GB, and batching all pairs in one
process runs out of memory on a 32 GB machine.

### 5. Product-level validation against MODIS LAI

```bash
python code/fetch_modis_lai.py        # MCD15A3H, 500 m, 4-day, via Planetary Computer
python code/fill_modis_std.py         # fill LaiStdDev for composites that straddle a date
python code/lai_product_validation.py --processed data/s2_processed_boa \
       --out results/tables_boa --figdir results/figures \
       --metrics results/tables_boa/per_patch_metrics_s2.csv
python code/prodval_bootstrap.py      # 1000x patch bootstrap for the 95% CI of rho
python code/replot_prodval.py         # final 400 dpi versions of Figures 9 and 10
```

`--metrics` carries the source of the image-domain columns in Table 13; it is spelled
out because the script's default points at the pre-BOA directory of Section 4 and
would silently mix the two datasets. `lai_product_validation.py` reads the per-patch
metrics written by `eval_s2_crossregion.py`, so that step must come first.

`fetch_modis_lai.py` needs no account: it resolves MCD15A3H COGs through the Microsoft
Planetary Computer STAC API and signs each URL separately. The container-level SAS
token endpoint of that API does not survive some HTTP proxies; signing one URL at a
time does. Both are implemented, the per-URL route is the default.

### 6. Manuscript

```bash
cd paper
python reframe.py        # applies the text revisions to main.tex
python prep_tex.py       # LaTeX -> pandoc input (resolves \ref, \cite, column specs)
python style_docx.py     # applies the Elsevier single-column layout
```

---

## One implementation detail that matters: the BOA offset

Sentinel-2 L2A surface reflectance is **not** `DN / 10000`. For processing baselines
04.00 and later, the relation is

```
rho = (DN + BOA_ADD_OFFSET) / 10000     with BOA_ADD_OFFSET = -1000
```

A plain division by 10000 adds +0.100 to every band. The effect is not cosmetic: in
this study it pushed retrieved LAI down to 0.06–0.15 against a true range of 1–2, and it
made PSNR plateau below 25 dB for reasons that had nothing to do with the models.
`build_s2_dataset.py` takes the offset as a parameter (`--boa-offset-dn`, default
`-1000`) for exactly this reason. If you rebuild the dataset from another source,
verify one clear-sky patch against the L2A metadata before trusting any downstream
number.

A second trap, in the retrieval code: `LookupTable.select()` expects an array shaped
`(n_samples, n_bands)`. Passing the transpose silently returns the mean of misaligned
samples — plausible-looking numbers that mean nothing. `lai_product_validation.py`
documents this at the call site.

---

## Data sources

No imagery is redistributed here; all three inputs are public and must be obtained from
their archives.

| Product | Access |
|---|---|
| SEN12MS-CR | Ebel et al. (2022), doi:10.1109/TGRS.2022.3146246 |
| Sentinel-2 L2A | Copernicus Data Space Ecosystem, https://dataspace.copernicus.eu |
| MCD15A3H v061 | NASA LP DAAC / Microsoft Planetary Computer, collection `modis-15A3H-061` |

`meta/patch_geo.csv` gives the centre coordinate of every patch in the reproduction
dataset, so the Sentinel-2 and MODIS windows can be re-extracted without the raw
archive.

---

## What is not included

- **PROSAIL LUT** (19 MB) — regenerated by `code/gen_lut.py`.
- **Sentinel-2 patch cubes** (~128 MB of `.npz`) — rebuilt by `build_s2_all.py` from
  public imagery plus `meta/s2_pairs.csv`.
- **Model checkpoints** — the learned baselines are retrained by the training scripts;
  the metrics they produced are archived.
- **SEN12MS-CR imagery** — subject to its own licence.

---

## License and citation

Code under the MIT License (see `LICENSE`). The derived tables in `tables/` are released
under CC BY 4.0.

If you use this package, please cite the manuscript above and the archived version:

```bibtex
@article{ma2026imagemetrics,
  author  = {Ma, Zhaoyang and Liu, Dawei and Yang, Jinghan},
  title   = {Image-domain metrics misrank cloud-removal models for {Sentinel-2}
             {LAI}, {Cab} and {NDVI} retrieval},
  journal = {International Journal of Applied Earth Observation and Geoinformation},
  year    = {2026},
  note    = {Code and derived tables: https://doi.org/10.5281/zenodo.24070784}
}
```

The archived package (all versions) is at https://doi.org/10.5281/zenodo.24070784
(Concept DOI, always resolving to the latest version). Version DOIs:
v1.0.0 https://doi.org/10.5281/zenodo.24070785,
v1.1.0 https://doi.org/10.5281/zenodo.24071316,
v1.2.0 https://doi.org/10.5281/zenodo.24071597 (first version containing the
revision analyses of Sections 4.1 and 4.2).

Corresponding author: Dawei Liu (wjmicheal@163.com),
Engineering University of the Chinese People's Armed Police Force, Xi'an, Shaanxi, China.
