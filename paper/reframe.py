# -*- coding: utf-8 -*-
"""Phase-1 text repositioning for the JAG version of main.tex."""
import io, re

p = 'main.tex'
s = io.open(p, encoding='utf-8').read()


def sub1(pat, rep):
    global s
    n = len(re.findall(pat, s, re.S))
    assert n == 1, ("count", n, pat[:60])
    m = re.search(pat, s, re.S)
    s = s[:m.start()] + rep + s[m.end():]


# ---------------- E1: title ----------------
sub1(r"\\shorttitle\{[^}]*\}", r"\shorttitle{Image Metrics Misrank Cloud-Removal Models}")
sub1(r"\\title\[mode=title\]\{[^}]*\}",
     r"\title[mode=title]{From image fidelity to product quality: image-domain metrics"
     r" misrank cloud-removal models for Sentinel-2 LAI, Cab and NDVI retrieval}")

# ---------------- E2: abstract (<=250 words) ----------------
ABSTRACT = r"""Cloud-removed Sentinel-2 reflectance is an input to operational leaf area index (LAI), canopy chlorophyll (Cab) and NDVI products, yet the models that produce it are accepted almost entirely on image-domain metrics---peak signal-to-noise ratio (PSNR), structural similarity (SSIM), spectral angle metric (SAM) and relative dimensionless global error in synthesis (ERGAS)---on the assumption that pixel fidelity proxies product quality. We test it on 15 methods from non-learning, learning and generative families, 428 cloudy patches (6,420 method--patch evaluations), two datasets and three targets: LAI, Cab and NDVI classification. SSIM and ERGAS lack transferable discriminative power: at patch level, SAM tracks retrieval error 0.162 and 0.152 more closely (95\% CI [0.120, 0.208] and [0.126, 0.175]), a gap that survives masks, datasets and targets. PSNR and SAM are not separable, and their order depends on the candidate pool and target: PSNR leads over all 15 methods (0.936 vs. 0.925) and at patch level (0.756 vs. 0.726), yet SAM overtakes it once only the nine qualified methods are kept (0.933 vs. 0.867) and leads by 0.16 for NDVI; PSNR therefore screens out poor methods rather than ranking good ones. The decoupling is rooted in the spectrum: B8A/B8/B7 contribute 60\% of the global MSE while retrieval is governed by B11, with 4.1$\times$ the sensitivity of B8, and the two importance profiles correlate at only r = +0.06. Discriminative power collapses under cirrus (SAM 0.779 $\rightarrow$ 0.490). We restate these results as acceptance criteria for the cloud-removal step of a product chain."""
wc = len(re.sub(r'\\[a-zA-Z]+|[{}\\]|\$', ' ', ABSTRACT).split())
print("abstract words:", wc)
assert wc <= 246, wc
sub1(r"\\begin\{abstract\}.*?\\end\{abstract\}",
     "\\begin{abstract}\n" + ABSTRACT + "\n\\end{abstract}")

# ---------------- E3: introduction I-A ----------------
IA = r"""\subsection{An Untested Assumption Behind Cloud-Removal Acceptance}

Sentinel-2 Level-2A surface reflectance is the working input of operational biophysical products: leaf area index (LAI), canopy chlorophyll content (Cab) and NDVI composites that feed cropland, forest and grassland monitoring. None of these products can be generated where clouds obscure the surface, so cloud removal is not an optional refinement but a routine step of the preprocessing chain, and the quality of the reconstruction propagates directly into the parameter finally delivered to a user.

That step is nevertheless accepted on image-domain evidence. Cloud removal has grown in a few years from a handful of prototype algorithms into a highly competitive field, and dozens of architectures are now ranked in a single table on SEN12MS-CR, AllClear and their derivatives. The ranking is almost always read from image-domain metrics: mostly the peak signal-to-noise ratio (PSNR) and structural similarity (SSIM), sometimes the spectral angle metric (SAM) and the relative dimensionless global error in synthesis (ERGAS). The reasoning chain is short and rarely stated explicitly: higher pixel fidelity means a better reconstruction, and a better reconstruction means a more usable product.

Cloud occlusion of optical observations is systematic. Roughly half of spaceborne optical observations are affected by clouds or haze \cite{ref1}. This makes cloud removal an unavoidable step in the preprocessing chain for land-surface monitoring. After deep learning entered remote sensing in the last decade, the task moved quickly from hand-crafted operators to end-to-end networks \cite{ref2}. Yet the transfer from pixel fidelity to product quality has never been tested on the terms a product actually imposes. Downstream models consume per-pixel spectra, and a spectrum can fail in ways that image metrics cannot see. A single band may carry a systematic offset. The texture may look plausible while the content is fabricated. Or the error may be small yet concentrate exactly on the few bands the target parameter depends on. This paper therefore asks a practical question. When a product team selects the top-ranked model from such a table, does it obtain the model that yields the most reliable LAI, Cab and NDVI estimates?"""
sub1(r"\\subsection\{An Untested Proxy Assumption\}.*?most usable for their downstream results\?", IA)

# ---------------- E4: contributions ----------------
sub1(r"Downstream usability is provided by a deterministic pipeline: PROSAIL lookup tables"
     r" with 8-nearest-neighbor retrieval \(LAI, Cab\), plus NDVI threshold classification\.",
     r"Downstream usability is measured through a deterministic retrieval pipeline of the kind"
     r" used to generate LAI and Cab products---PROSAIL lookup tables with 8-nearest-neighbor"
     r" retrieval (LAI, Cab), plus NDVI threshold classification---although it is driven by"
     r" simulated rather than operational retrievals (Section 5).")
sub1(r"The paper makes three main contributions\.",
     r"The paper makes four main contributions.")

C4 = r"""

(4) Acceptance criteria for a product chain. The three findings are restated as criteria for accepting the cloud-removal step of a biophysical product: which metrics may be used to accept a model, what must be reported alongside them, and how both change with the target parameter and the cloud regime (Table~\ref{tab:13})."""
m = re.search(r"\n\\section\{Related Work\}", s)
assert m
s = s[:m.start()] + C4 + s[m.start():]

# ---------------- E5: Table 13 -> acceptance checklist ----------------
sub1(r"\\subsection\{Recommendations for Evaluation Practice\}",
     r"\subsection{From Evaluation Practice to Acceptance Criteria}")

sub1(r"\\caption\{A Checklist of Cloud-Removal Evaluation Practices Derived from Our Results\\label\{tab:13\}\}",
     r"\caption{A Checklist for Accepting the Cloud-Removal Step of a Biophysical Product"
     r" Chain, Derived from Our Results\label{tab:13}}")

T13_NEW = r"""1 & Do not accept a cloud-removal model on SSIM or ERGAS alone. & At the patch level, $\rho$ is 0.56 and 0.57, respectively, and 0.76 for PSNR. SAM is higher than the two by 0.16 and 0.15 with probability 1.000, unchanged across the three tasks. \\
2 & Report PSNR and SAM together in the quality assessment of the cloud-removal step; neither replaces the other. & PSNR is slightly higher on the full set (0.94 vs. 0.93), while SAM is higher on qualified methods (0.93 vs. 0.87). SAM leads by 0.16 in NDVI classification. \\
3 & State the aggregation scheme (per patch or per method) in the product's quality documentation. & The same data give opposite rankings under the two schemes. \\
4 & Report the cloud-type composition, especially the cirrus fraction, because acceptance thresholds are regime dependent. & In cirrus scenes, SAM 0.78$\rightarrow$0.49, PSNR 0.82$\rightarrow$0.62, SSIM 0.61$\rightarrow$0.35. \\
5 & Derive the acceptance metric from the target product, and re-derive it when the product changes. & LAI$\rightarrow$Cab changes the best metric from SAM to ERGAS/PSNR, and the fitted PSNR weight goes to zero. \\
6 & State the sensitivity of the acceptance decision to candidate-pool composition. & Restricting the pool from all 15 methods to the nine qualified ones changes method-level SAM from 0.925 to 0.964; the direction is unchanged but the values move noticeably. \\
7 & Report SAM and the cirrus fraction jointly, so that the loss of discrimination under cirrus stays traceable for a downstream user. & SAM falls from 0.78 to 0.49 in cirrus scenes; a user who receives only a pooled score cannot know which regime it came from. \\
\hline"""

old13 = re.search(r"1 & Do not use SSIM or ERGAS alone.*?direction is unchanged but the values move noticeably\. \\\\\n\\hline", s, re.S)
assert old13
s = s[:old13.start()] + T13_NEW + s[old13.end():]
# now there are two \hline in a row; collapse one
s = s.replace("direction is unchanged but the values move noticeably. \\\\\n\\hline\\hline",
              "direction is unchanged but the values move noticeably. \\\\\n\\hline")

# surrounding paragraph
sub1(r"This checklist can be applied directly to two scenarios: paper writing and product acceptance\. For paper writing, items 2, 3, and 6 imply that a method table should not list only PSNR, and one should not claim a metric is more reliable without stating the scheme and the candidate pool\. For product acceptance, item 5 implies that the acceptance metric should be derived from the business parameter, rather than following image-quality convention\.",
     r"The checklist addresses a product chain first and a paper second. For product acceptance,"
     r" the decision rule is not interchangeable across targets (item 5), the accept/reject"
     r" threshold depends on the cloud regime (item 4), and a pooled image-domain score cannot"
     r" tell a user how much of it came from cirrus (item 7); an acceptance protocol that reports"
     r" SAM together with the cirrus fraction is therefore the minimum that keeps the decision"
     r" auditable. For paper writing, items 2, 3, and 6 still apply: a method table should not"
     r" list only PSNR, and no metric should be called more reliable without stating the scheme"
     r" and the candidate pool.")

# ---------------- E6: limitations ----------------
sub1(r"This paper has five limitations\.", r"This paper has six limitations.")
sub1(r"Fifth, the generative family contains only one small model, so our results are insufficient to generalize about generative cloud-removal methods\.",
     r"Fifth, the generative family contains only one small model, so our results are"
     r" insufficient to generalize about generative cloud-removal methods. Sixth, the downstream"
     r" targets are PROSAIL-based retrievals from simulated spectra, not operational LAI"
     r" products. No comparison against an external product such as Copernicus or MODIS LAI is"
     r" made here, so the link between metric choice and product-level accuracy is inferred from"
     r" the retrieval mechanism rather than measured end to end; closing that gap is the next"
     r" step of this work.")

# ---------------- E7: conclusion ----------------
sub1(r"Image-domain metrics cannot serve interchangeably as proxies for the usability of cloud-removal products\.",
     r"Image-domain metrics cannot serve as interchangeable acceptance criteria for the"
     r" cloud-removal step that feeds a biophysical product.")

# ---------------- E8: keywords (max 7) ----------------
sub1(r"\\begin\{keywords\}.*?\\end\{keywords\}",
     r"""\begin{keywords}
Cloud removal \sep data quality \sep biophysical parameter retrieval \sep leaf area index \sep
NDVI classification \sep product validation \sep Sentinel-2
\end{keywords}""")

# ---------------- E9: header comment ----------------
sub1(r"%% Converted from the IEEEtran \(JSTARS\) version\. Body text is unchanged except\n%% that hard-coded 'Table I'/'Figure 1' references now use \\ref so that they\n%% follow Elsevier numbering \(Table 1, Fig\. 1\)\.",
     r"%% Converted from the IEEEtran (JSTARS) version and repositioned for the IJAEOG scope:\n%% the paper is framed as a quality-assurance problem for the cloud-removal step of a\n%% biophysical product chain rather than as a survey of metric consistency. Experiments,\n%% numbers and conclusions are unchanged. Hard-coded 'Table I'/'Figure 1' references now\n%% use \ref so that they follow Elsevier numbering (Table 1, Fig. 1).")

io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print("written:", p, " ascii:", s.isascii())
