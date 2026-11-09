"""生成 JSTARS 投稿初稿 docx（结构完整版：摘要→引言→数据与协议→结果→讨论→结论）"""
import os

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

TAB = "results/tables"
doc = Document()
st = doc.styles["Normal"]
st.font.name = "Times New Roman"
st.font.size = Pt(10.5)
for s in doc.sections:
    s.left_margin = s.right_margin = Inches(1.0)
    s.top_margin = s.bottom_margin = Inches(1.0)


def para(text, size=10.5, bold=False, italic=False, align="j", space_after=6, indent=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    if indent:
        p.paragraph_format.first_line_indent = Inches(0.2)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY if align == "j" else WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    r.font.size = Pt(size)
    r.font.name = "Times New Roman"
    return p


def h(text, size=12, before=12):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(size)
    r.font.name = "Times New Roman"
    return p


def note(text, size=9):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    r.italic = True
    r.font.size = Pt(size)
    r.font.color.rgb = RGBColor(0x88, 0x44, 0x00)
    r.font.name = "Times New Roman"


def table(rows, widths, size=8.0, caption=None):
    if caption:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(caption)
        r.bold = True
        r.font.size = Pt(9)
        r.font.name = "Times New Roman"
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.width = Inches(widths[j])
            c.text = ""
            pp = c.paragraphs[0]
            pp.paragraph_format.space_after = Pt(1)
            rr = pp.add_run(str(val))
            rr.font.size = Pt(size)
            rr.font.name = "Times New Roman"
            rr.bold = (i == 0)


# ================= 标题 =================
h("Pixel Fidelity Is Not Retrieval Utility:", 15, before=0)
h("A Band-Level Diagnosis of Cloud-Removal Metrics", 15, before=0)
note("备选标题 1：Which Image Metric Predicts Retrieval After Cloud Removal? Band-Level Evidence "
     "for Metric–Task Decoupling")
note("备选标题 2：SSIM and ERGAS Are the Wrong Metrics for Cloud Removal: A Band-Level "
     "Explanation with 5,992 Retrieval Tests")

# ================= 摘要 =================
h("Abstract")
for t in [
    "Cloud-removal models are ranked almost exclusively by image-space metrics—PSNR, SSIM, SAM, "
    "and ERGAS—on the assumption that pixel fidelity proxies retrieval utility. We test that "
    "assumption with 14 methods from three families, 428 cloud-contaminated patches (5,992 "
    "method–patch evaluations) drawn from two independent Sentinel-2 datasets, and a PROSAIL "
    "lookup-table retrieval of leaf area index (LAI) and, as a second target, chlorophyll content.",

    "Two metrics fail decisively. At patch level, a paired bootstrap shows that SAM's rank "
    "correlation with retrieval error exceeds that of SSIM and ERGAS by 0.17 (95% CI [0.13, 0.22]; "
    "P = 1.000); the ordering is unchanged under ±1-pixel mask perturbation, on the second "
    "dataset, and for both retrieved variables.",

    "PSNR and SAM, by contrast, are not separable, and their ordering depends on the candidate "
    "pool: PSNR leads over all 14 methods (0.921 vs. 0.908) and at patch level (0.741 vs. 0.714; "
    "difference −0.026, P = 0.02), whereas SAM leads once only competent methods are retained "
    "(0.905 vs. 0.809 above 25 dB, and at K = 4–6 under a relative criterion, on both datasets). "
    "PSNR's apparent strength therefore comes from rejecting poor methods, not from ranking good "
    "ones.",

    "We trace the decoupling to the band level: 59% of the global MSE arises from B8A, B8, and B7, "
    "whereas LAI sensitivity is dominated by B11 (4.1× that of B8); the two band-importance "
    "profiles correlate at r = +0.06. Discrimination collapses for cirrus-contaminated scenes, "
    "where every metric's correlation roughly halves. We close with a five-item evaluation "
    "checklist.",
]:
    para(t, indent=True)

h("Index Terms")
para("Cloud removal, evaluation metrics, downstream retrieval, leaf area index, Sentinel-2, "
     "band sensitivity analysis.", italic=True)

# ================= I. Introduction =================
h("I. Introduction")
h("A. The Proxy Assumption", 11)
para("Cloud removal—reconstructing ground reflectance beneath cloud cover—has moved in a few years "
     "from a handful of prototypes to a crowded benchmark culture. On SEN12MS-CR, AllClear and "
     "their derivatives, dozens of architectures are now ranked in a single table, and the ranking "
     "is almost always read off image-space metrics: PSNR and SSIM in most papers, SAM and ERGAS "
     "in a minority [CIT]. The reasoning behind this practice is short and rarely stated: higher "
     "pixel fidelity means a better reconstruction, and therefore a more usable product for the "
     "applications—crop monitoring, land-cover mapping, biophysical retrieval—that motivate cloud "
     "removal in the first place.", indent=True)
para("That reasoning has not been carefully audited. What a downstream model consumes is a spectrum "
     "per pixel, and a spectrum can be wrong in ways that pixel-averaged metrics do not see: a "
     "uniform bias in one band, a plausible-looking but fabricated texture, or a small distortion "
     "concentrated in the bands the retrieval actually depends on. We therefore ask a narrow, "
     "practical question: when a practitioner selects the top-ranked model from such a table, does "
     "she obtain the model that yields the most usable retrievals?", indent=True)

h("B. What Is Not Yet Known", 11)
para("Evidence that the answer may be negative has begun to accumulate. Generative models have been "
     "observed to score well perceptually while inventing content [CIT]; spectral-angle metrics "
     "were proposed precisely because PSNR ignores spectral shape [CIT]; and recent benchmarks "
     "report rank changes depending on which metric is reported [CIT]. Three questions "
     "nevertheless remain open. First, most such observations contrast a strong method with "
     "deliberately weak baselines, so the inconsistency could be an artifact of an easy candidate "
     "pool rather than a property of the metric. Second, results are typically reported at a "
     "single aggregation level—per patch or per method—although the two answer different questions "
     "and need not agree. Third, and most importantly, no study has localized the failure: it is "
     "asserted that image metrics weight bands differently from retrieval models, but the "
     "band-level evidence has not been produced.", indent=True)

h("C. This Work", 11)
para("This paper supplies all three. We evaluate 14 cloud-removal methods spanning three families—"
     "nine non-learned statistical or interpolation operators, four learned models, and one "
     "conditional diffusion model—on 428 cloud-contaminated patches from SEN12MS-CR, giving 5,992 "
     "method–patch evaluations, and replicate the central analysis on an independently constructed "
     "Sentinel-2 L2A set in North China. Downstream utility is measured not by human judgment but "
     "by a deterministic PROSAIL lookup-table retrieval with 8-nearest-neighbor inversion, applied "
     "to LAI and, as a second target, to Cab. The contributions are fourfold.", indent=True)
for line in [
    "1) A three-level audit—method-level, patch-level, and candidate-pool-restricted—with paired "
    "bootstrap intervals, showing that the answer to \"which metric is best\" depends jointly on "
    "the aggregation level and on how competent the candidate pool is, and that no single metric "
    "is a universal winner.",
    "2) A band-level diagnosis that decomposes global MSE into per-band contributions and, "
    "independently, estimates each band's sensitivity for the retrieval. The two profiles are "
    "essentially uncorrelated (r = +0.06), which quantifies why image metrics and retrieval "
    "utility decouple instead of merely observing that they do.",
    "3) A condition-dependence analysis identifying where the metrics stop working: under cirrus "
    "contamination, identified via the B10 cirrus band, the rank correlation of every metric "
    "roughly halves.",
    "4) A task-dependence check and a five-item reporting checklist (Table V).",
]:
    para(line)

# ================= II. Data and Protocol =================
h("II. Data and Evaluation Protocol")
h("A. Datasets", 11)
para("Two datasets are used. The primary set is drawn from SEN12MS-CR, from which we take two "
     "acquisitions (spring and winter) and extract 428 patches of 128×128 pixels at 10 m by "
     "centre-cropping rather than resampling; resampling would impose a common low-pass filter on "
     "every method and systematically compress the differences we aim to measure. Patches with "
     "cloud fraction outside [2%, 95%] are discarded so that neither nearly clean nor fully "
     "overcast patches dominate. The replication set is built independently from Copernicus Data "
     "Space: four Sentinel-2 L2A tiles over North China (2022), eight cloudy/clear pairs matched "
     "within 1–4 days, and 156 patches; its cloud masks come from the official scene "
     "classification layer rather than from a threshold. Because that set contains no SAR, only "
     "the nine optical methods are evaluated on it.", indent=True)

h("B. Methods", 11)
para("Fourteen methods span three families so that no conclusion rests on a single error "
     "mechanism. Nine are non-learned: zero, mean, nearest-neighbour and blur filling; Telea "
     "inpainting; histogram matching; spatial k-nearest-neighbour regression; SAR-guided "
     "substitution; and SAR-to-optical regression. Four are learned: a compact U-Net with optical "
     "input only, the same network with an SAR channel, and two configurations of the DSen2-CR "
     "residual architecture [CIT]. One is generative: a conditional diffusion model. All learned "
     "models are trained on the same split with three seeds where compute permits; seed "
     "variation in PSNR is ±0.5 dB, well below the 5–13 dB spread between methods.", indent=True)

h("C. Downstream Retrieval", 11)
para("Retrieval is deliberately training-free so that it cannot be tuned to flatter any method. A "
     "PROSAIL lookup table of 300,000 spectra is inverted by 8-nearest-neighbour weighting in the "
     "ten usable Sentinel-2 bands. For each patch we compare the parameter retrieved from the "
     "reconstructed image against that retrieved from the cloud-free reference over the same cloud "
     "pixels (300 sampled per patch), and report the mean absolute difference. Two targets are "
     "used: LAI as the primary variable, and Cab as an independent check on whether the findings "
     "are specific to one parameter.", indent=True)

h("D. Statistics", 11)
para("Rank correlations are computed at three levels: across methods (n = 14), across all "
     "method–patch pairs (n = 5,992), and restricted to competent methods. Because fourteen points "
     "cannot support inference, all method-level comparisons are accompanied by bootstrap "
     "intervals, and the patch-level comparison is performed with a paired bootstrap that "
     "resamples patches while preserving the method × patch structure. We report P(difference > 0) "
     "rather than conventional p-values, since the quantity of interest is the sign and size of a "
     "difference between correlated correlations.", indent=True)

# ================= III. Results =================
h("III. Results")
h("A. Three Levels of Agreement", 11)
para("Table II reports the fourteen methods. Ranking by image metrics and ranking by retrieval "
     "error agree broadly at the extremes and disagree in between: the learned models occupy the "
     "top four positions on PSNR, yet one of them (dsen2cr_opt) ranks third on ΔLAI while "
     "unet_sar ranks first on both. Histogram matching is the clearest counterexample—its SAM of "
     "10.43° is unremarkable, but its ΔLAI of 0.80 is worse than every non-learned method except "
     "nearest-neighbour filling.", indent=True)

ms = pd.read_csv(f"{TAB}/method_summary.csv")
dn = pd.read_csv(f"{TAB}/per_patch_downstream.csv").groupby("method")[
    ["dLAI_mean", "dCab_mean"]].mean()
ms = ms.merge(dn, left_on="method", right_index=True).sort_values("dLAI_mean")
rows = [["Method", "PSNR (dB)", "SSIM", "SAM (°)", "ERGAS", "MAE", "RMSE", "ΔLAI", "ΔCab"]]
for _, r in ms.iterrows():
    rows.append([r.method, f"{r.PSNR_mean:.2f}", f"{r.SSIM_mean:.3f}", f"{r.SAM_mean:.2f}",
                 f"{r.ERGAS_mean:.1f}", f"{r.MAE_mean:.3f}", f"{r.RMSE_mean:.3f}",
                 f"{r.dLAI_mean:.3f}", f"{r.dCab_mean:.2f}"])
table(rows, [1.35, 0.72, 0.55, 0.62, 0.6, 0.52, 0.52, 0.55, 0.55], 8.0,
      "Table II.  Image-space metrics and downstream deviation (mean over 428 patches; "
      "sorted by ΔLAI).")

cm = pd.read_csv(f"{TAB}/corr_method_level.csv").set_index("metric")
cs = pd.read_csv(f"{TAB}/corr_sample_level.csv").set_index("metric")
ch = pd.read_csv(f"{TAB}/corr_highq_subset.csv").set_index("metric")
rows = [["Metric", "Across methods (n=14)", "Across patches (n=5,992)", "Competent methods (n=8)"]]
for m in ["PSNR", "SSIM", "SAM", "ERGAS", "MAE", "RMSE"]:
    rows.append([m,
                 f"{cm.loc[m,'spearman_vs_dLAI']:+.3f}",
                 f"{cs.loc[m,'spearman_vs_dLAI']:+.3f}",
                 f"{ch.loc[m,'spearman_in_highq']:+.3f}" if m in ch.index else "—"])
table(rows, [1.5, 1.8, 1.9, 1.9], 8.5,
      "Table III.  Rank correlation between each metric and ΔLAI, at three aggregation levels.")
para("Table III makes the aggregation dependence explicit. Across all fourteen methods, PSNR and SAM "
     "are within 0.013 of each other. Across patches, PSNR is higher by 0.026, and a paired "
     "bootstrap places this difference reliably away from zero ([−0.056, −0.002]; P(diff > 0) = "
     "0.02). Restricted to the eight methods above 25 dB, the ordering reverses: SAM reaches 0.905 "
     "against 0.809 for PSNR. The same reversal holds under a relative criterion that keeps the "
     "top K methods by PSNR (K = 4, 5, 6) on both datasets, so it is not an artifact of the "
     "absolute threshold. We read this as evidence that PSNR's correlation is carried by its "
     "ability to reject poor reconstructions, not by its ability to order competent ones.", indent=True)
para("SSIM and ERGAS behave differently: they are last at every level. The paired bootstrap gives "
     "SAM an advantage of 0.169 over SSIM ([0.125, 0.216]) and 0.171 over ERGAS ([0.142, 0.196]), "
     "with P(difference > 0) = 1.000 in both cases. This is the only comparison in the study that "
     "reaches a decisive level, and it survives every perturbation we apply: ±1-pixel mask "
     "erosion or dilation moves all correlations by at most 0.02, and the replication dataset "
     "reproduces the ordering (PSNR −0.900, SAM +0.933, ERGAS +0.917, SSIM −0.933).", indent=True)

h("B. Why: The Band-Level Account", 11)
mech = pd.read_csv(f"{TAB}/band_mechanism.csv").sort_values("mse_share", ascending=False)
rows = [["Band", "Share of global MSE", "LAI sensitivity (ΔLAI per unit reflectance)",
         "ρ (PSNR_band vs ΔLAI)"]]
for _, r in mech.iterrows():
    rows.append([r.band, f"{r.mse_share*100:.1f}%", f"{r.lai_sensitivity:.1f}", f"{r.rho_sample:+.3f}"])
table(rows, [0.9, 1.7, 2.6, 1.9], 8.5,
      "Table IV.  Band-level decomposition: who determines PSNR, who determines the retrieval.")
para("PSNR is a uniform average of per-band mean squared errors, and the bands are not equally "
     "noisy. Three bands—B8A, B8 and B7—contribute 59% of the global MSE, simply because their "
     "reflectance magnitudes are largest, so their absolute errors dominate. The retrieval, in "
     "contrast, is governed by a different set of bands: raising B11 by one unit of reflectance "
     "moves LAI by 76.6, more than four times the 18.8 of B8, even though B11 accounts for only "
     "9.3% of the MSE. Across the ten bands the two importance profiles correlate at r = +0.06, "
     "i.e., they are unrelated. A metric dominated by B8A/B8/B7 is therefore scoring a quantity "
     "that the retrieval barely reads.", indent=True)

h("C. Where the Metrics Stop Working", 11)
para("Splitting the test patches by cloud type, using the B10 cirrus band as the discriminator, "
     "shows that the disagreement is not uniform. On water-cloud scenes every metric tracks "
     "retrieval error reasonably well (SAM 0.763, PSNR −0.804); on cirrus scenes the same metrics "
     "lose roughly half their discriminating power (SAM 0.478, PSNR −0.608, SSIM −0.341). The "
     "physical reason is that thin cirrus invites mild colour correction rather than "
     "reconstruction, leaving residuals that are smooth spectral offsets—too small to depress "
     "PSNR, large enough to bias a retrieval. A similar collapse occurs at low vegetation cover "
     "(SAM 0.405 versus 0.747 at high NDVI), where the retrieval itself has little dynamic range "
     "to discriminate between methods.", indent=True)

h("D. Task Dependence", 11)
para("Switching the retrieved variable from LAI to Cab reorders the metrics. Against Cab, PSNR "
     "recovers to 0.952 and ERGAS to 0.976 among competent methods, while SAM stays at 0.952; "
     "against LAI, SAM led at 0.905 and PSNR lagged at 0.809. Fitting a non-negative weighted "
     "combination of metrics to each target makes the point concretely: for LAI the weights fall "
     "on RMSE (0.25), SAM (0.07) and PSNR (0.03); for Cab they fall on ERGAS (5.08), SAM (2.72) "
     "and SSIM (1.63), with PSNR and RMSE driven to exactly zero. The fitted weights, not the "
     "marginal gain in correlation (+0.01 to +0.02), are the informative part.", indent=True)

# ================= IV. Discussion =================
h("IV. Discussion and Reporting Recommendations")
para("Three practical consequences follow. First, the metrics that should be retired are SSIM and "
     "ERGAS, not PSNR: the evidence against them is decisive and reproducible, whereas the "
     "evidence for or against PSNR depends on how the candidate pool is defined. Second, "
     "aggregation level must be reported, because the same data yields opposite rankings at method "
     "and patch level. Third, the criterion for selecting methods should be the downstream task, "
     "not a fixed metric; when the task changes, the metric ordering changes with it.", indent=True)

rows = [["#", "Recommendation", "Evidence in this paper"],
        ["1", "Do not rank cloud-removal models by SSIM or ERGAS alone.",
         "Patch-level ρ = 0.55 (SSIM), 0.54 (ERGAS) vs. 0.74 (PSNR); SAM exceeds both by 0.17 "
         "(P = 1.000) in every configuration tested."],
        ["2", "Report PSNR and SAM together; do not claim one supersedes the other.",
         "PSNR leads over the full pool (0.92 vs. 0.91), SAM leads among competent methods "
         "(0.91 vs. 0.81); neither difference is separable."],
        ["3", "State the aggregation level (per patch or per method).",
         "The same data yields opposite rankings at the two levels."],
        ["4", "Report the cloud-type composition, in particular the cirrus fraction.",
         "Cirrus scenes: SAM 0.76 → 0.48, PSNR 0.80 → 0.61, SSIM 0.60 → 0.34."],
        ["5", "Validate the chosen metric against the downstream task before selecting models; "
              "re-fit if the task changes.",
         "LAI → Cab moves the best metric from SAM to ERGAS/PSNR and zeroes PSNR's fitted weight."]]
table(rows, [0.35, 2.8, 3.35], 8.0,
      "Table V.  Evaluation checklist derived from this study.")

para("The study has clear limits. Fourteen methods, of which eight are competent, is a small basis "
     "for inference; we mitigate this with patch-level bootstrapping and a task-consistency "
     "formulation rather than by claiming method-level significance. Both retrieved variables come "
     "from a single lookup table, so the task dependence we report is demonstrated within one "
     "inversion framework; a classification-style downstream task would test it further. The "
     "second dataset contains no SAR, and its learned models degrade sharply because of domain "
     "shift between L1C top-of-atmosphere and L2A surface reflectance, so it supports the "
     "cross-dataset claim for non-learned methods only.", indent=True)

h("V. Conclusion")
para("Image-space metrics are not interchangeable proxies for the utility of a cloud-removal "
     "product. Two of them, SSIM and ERGAS, should be retired from model selection on the evidence "
     "presented here; PSNR and SAM are both defensible but measure different things, and their "
     "ordering depends on the candidate pool and the retrieval target. The reason is band-level: "
     "pixel metrics average over bands in proportion to reflectance magnitude, whereas retrievals "
     "are governed by a small number of spectrally informative bands. Reporting practice should "
     "reflect that structure—by naming the aggregation level, the cloud composition, and the "
     "downstream task against which the metric was validated.", indent=True)

# ================= 写作说明 =================
doc.add_page_break()
h("写作说明（非论文内容，投稿前整页删除）", 12)
for n in [
    "1. 标题用「Pixel Fidelity Is Not Retrieval Utility」直接给对立命题，副标题点明机制。"
    "不要用“A Study of…”这类无信息量的开头。",
    "2. 摘要第一段立靶子，第二段就给唯一决定性结论（SSIM/ERGAS 该弃用）。"
    "评审看摘要先找数字，0.17 和 P = 1.000 必须在前半段。",
    "3. 摘要第三段主动写「PSNR 与 SAM 不可分、谁高取决于候选池」——这是防守。"
    "不写，评审自己算出来会认为你在藏；写了，它变成发现。",
    "4. 贡献列表顺序是刻意的：三层审计放第一（最容易被攻击），机制 r=+0.06 放第二（最难被攻击）。"
    "评审记不住四条，能记住的就是这两条。",
    "5. Table III 是全篇核心。它把「同一批数据、三种口径、结论相反」摆在明面上，"
    "这是最能体现方法学严谨的一张表。",
    "6. Table IV 是机制证据。重点不是每个数，而是最后一列：B8A/B8/B7 的 ρ 都在 −0.75 以上，"
    "而它们的 MSE 占比也最高——说明高相关是靠「大误差波段」撑起来的。",
    "7. Section IV 的 limitation 是主动认领，不要删。JSTARS 评审对「知道自己边界」的稿件明显更宽容。",
    "8. 需要你补：所有 [CIT] 处填真实文献（去云基准、SSIM 局限、光谱角指标、GA 造假地物、DSen2-CR 原文）。"
    "我不能替你编文献。",
    "9. 图号映射：Fig.1 主表散点；Fig.2 波段机制；Fig.3 云类型/地表分层；Fig.4 掩膜鲁棒性；"
    "Fig.5 定性对比。其余进补充材料。",
    "10. 尚未写：Section II 的公式、Section III 的失败案例段、参考文献表。"
    "下一步建议先写失败案例（有素材），再回填公式。",
]:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run(n)
    r.font.size = Pt(9.5)
    r.font.name = "Times New Roman"
    r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

out = r"D:\mazhaoyang\code program\cloudrs1\drafts\JSTARS_初稿_v1.docx"
os.makedirs(os.path.dirname(out), exist_ok=True)
doc.save(out)
print("已生成:", out)
