# -*- coding: utf-8 -*-
"""Apply Elsevier/JAG manuscript formatting to the pandoc output."""
import copy
import re

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.table import _Cell
from docx.text.paragraph import Paragraph
from docx.shared import Cm, Pt

PATH = "Elsevier_JAG_Manuscript.docx"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"

doc = docx.Document(PATH)
TEXT_W = Cm(16.5)


def set_style_font(style, name="Times New Roman", size=None, bold=None):
    f = style.font
    f.name = name
    if size is not None:
        f.size = Pt(size)
    if bold is not None:
        f.bold = bold
    rpr = style.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.append(rf)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rf.set(qn(a), name)


def fmt(p, size=12, bold=False, align=None, before=0, after=0, spacing=2.0,
        keep_next=False, italic=False):
    pf = p.paragraph_format
    if align is not None:
        pf.alignment = align
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = spacing
    pf.keep_with_next = keep_next
    for r in p.runs:
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.name = "Times New Roman"


# ---------------------------------------------------------------- page setup
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.25)
sec.top_margin = sec.bottom_margin = Cm(2.5)
ln = OxmlElement("w:lnNumType")
ln.set(qn("w:countBy"), "1")
ln.set(qn("w:restart"), "continuous")
ln.set(qn("w:distance"), "360")
sec._sectPr.append(ln)

fp = sec.footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
fld = OxmlElement("w:fldSimple")
fld.set(qn("w:instr"), "PAGE")
r_el = OxmlElement("w:r")
t_el = OxmlElement("w:t")
t_el.text = "1"
r_el.append(t_el)
fld.append(r_el)
fp._p.append(fld)

# ---------------------------------------------------------------- styles
def get_style(name):
    try:
        return doc.styles[name]
    except KeyError:
        pass
    for cand in doc.styles:
        if cand.name and cand.name.lower() == name.lower():
            return cand
    raise KeyError(name)


for name in ("Normal", "Body Text", "First Paragraph", "Compact"):
    try:
        s = get_style(name)
    except KeyError:
        continue
    set_style_font(s, size=12)
    s.paragraph_format.line_spacing = 2.0
    s.paragraph_format.space_after = Pt(0)
    s.paragraph_format.space_before = Pt(0)

for name in ("Heading 1", "Heading 2"):
    s = get_style(name)
    set_style_font(s, size=12, bold=True)
    s.font.italic = False
    rPr = s.element.get_or_add_rPr()
    col = rPr.find(qn("w:color"))
    if col is None:
        col = OxmlElement("w:color")
        rPr.append(col)
    for a in ("w:themeColor", "w:themeShade", "w:themeTint"):
        if col.get(qn(a)) is not None:
            del col.attrib[qn(a)]
    col.set(qn("w:val"), "000000")
    s.paragraph_format.space_before = Pt(18 if name == "Heading 1" else 12)
    s.paragraph_format.space_after = Pt(6)
    s.paragraph_format.line_spacing = 1.5
    s.paragraph_format.keep_with_next = True

# ---------------------------------------------------------------- section numbers
UNNUM = {"References", "Declaration of competing interest", "Data availability",
         "Credit authorship contribution statement"}
h1 = h2 = 0
for p in doc.paragraphs:
    if p.style.name == "Heading 1":
        if p.text.strip() in UNNUM:
            continue
        h1 += 1
        h2 = 0
        if p.runs:
            p.runs[0].text = "%d.  %s" % (h1, p.runs[0].text)
    elif p.style.name == "Heading 2":
        h2 += 1
        if p.runs:
            p.runs[0].text = "%d.%d  %s" % (h1, h2, p.runs[0].text)

# ---------------------------------------------------------------- front matter
MARK = {
    "@@TITLE@@": dict(size=16, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
                      before=0, after=12, spacing=1.5),
    "@@AUTH@@": dict(size=12, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER,
                     before=0, after=6, spacing=1.5),
    "@@AFF@@": dict(size=10, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER,
                    before=0, after=4, spacing=1.5),
    "@@CORR@@": dict(size=10, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER,
                     before=0, after=18, spacing=1.5),
    "@@ABS@@": dict(size=12, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT,
                    before=12, after=6, spacing=1.5),
    "@@KWD@@": dict(size=11, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
                    before=8, after=12, spacing=1.5),
}
for p in doc.paragraphs:
    hit = False
    for mk, opt in MARK.items():
        if p.text.startswith(mk):
            p.runs[0].text = p.runs[0].text.replace(mk, "").lstrip()
            fmt(p, **opt)
            hit = True
            break
    if hit:
        continue
    if p.text.startswith("Cloud-removed Sentinel-2 reflectance is an input"):
        fmt(p, size=11, align=WD_ALIGN_PARAGRAPH.JUSTIFY, before=0, after=8, spacing=1.5)

# ---------------------------------------------------------------- captions
CAP = re.compile(r"^(Table \d+\.|Fig\. \d+\.)")
for p in doc.paragraphs:
    t = p.text.strip()
    if not CAP.match(t):
        continue
    is_tab = t.startswith("Table")
    fmt(p, size=10, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
        before=12 if is_tab else 8, after=4, spacing=1.0, keep_next=True, italic=False)

# ---------------------------------------------------------------- references
in_refs = False
for p in doc.paragraphs:
    if p.style.name == "Heading 1" and p.text.strip() == "References":
        in_refs = True
        continue
    if in_refs and p.text.strip():
        pf = p.paragraph_format
        pf.line_spacing = 1.5
        pf.space_after = Pt(4)
        pf.left_indent = Cm(0.9)
        pf.first_line_indent = Cm(-0.9)
        pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
        for r in p.runs:
            r.font.size = Pt(10)

# ---------------------------------------------------------------- tables
def tbl_borders(tbl):
    tblPr = tbl._tbl.tblPr
    for old in tblPr.findall(qn("w:tblBorders")):
        tblPr.remove(old)
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement("w:" + edge)
        if edge in ("top", "bottom", "insideH"):
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), "6")
            el.set(qn("w:color"), "000000")
        else:
            el.set(qn("w:val"), "none")
            el.set(qn("w:sz"), "0")
            el.set(qn("w:color"), "auto")
        el.set(qn("w:space"), "0")
        b.append(el)
    tblPr.append(b)


for tbl in doc.tables:
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_borders(tbl)
    tblPr = tbl._tbl.tblPr
    for old in tblPr.findall(qn("w:tblW")):
        tblPr.remove(old)
    tw = OxmlElement("w:tblW")
    tw.set(qn("w:type"), "pct")
    tw.set(qn("w:w"), "5000")
    tblPr.append(tw)
    for old in tblPr.findall(qn("w:tblLayout")):
        tblPr.remove(old)
    lay = OxmlElement("w:tblLayout")
    lay.set(qn("w:type"), "autofit")
    tblPr.append(lay)

    for ri, tr in enumerate(tbl._tbl.findall(qn("w:tr"))):
        trPr = tr.find(qn("w:trPr"))
        if trPr is None:
            trPr = OxmlElement("w:trPr")
            tr.insert(0, trPr)
        if ri == 0:
            trPr.append(OxmlElement("w:tblHeader"))
        for tc in tr.findall(qn("w:tc")):
            cell = _Cell(tc, tbl)
            for p in cell.paragraphs:
                pf = p.paragraph_format
                pf.space_before = Pt(1)
                pf.space_after = Pt(1)
                pf.line_spacing = 1.0
                for r in p.runs:
                    r.font.size = Pt(9)
                    r.font.name = "Times New Roman"
                    if ri == 0:
                        r.font.bold = True

# keep short tables on a single page (long ones are allowed to break)
from docx.text.paragraph import Paragraph as _P

for tbl in doc.tables:
    trs = tbl._tbl.findall(qn("w:tr"))
    if len(trs) > 9:
        continue
    for tr in trs[:-1]:
        for tc in tr.findall(qn("w:tc")):
            for pel in tc.findall(qn("w:p")):
                _P(pel, tbl).paragraph_format.keep_with_next = True

# ---------------------------------------------------------------- figures
for shape in doc.inline_shapes:
    if shape.width > TEXT_W:
        ratio = TEXT_W / shape.width
        shape.width = TEXT_W
        shape.height = int(shape.height * ratio)

for p in doc.paragraphs:
    if p._p.findall(".//" + qn("w:drawing")):
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.0

# ---------------------------------------------------------------- equations
num = 0
for p in doc.paragraphs:
    mp = p._p.find("{%s}oMathPara" % M)
    if mp is None:
        continue
    oms = mp.findall("{%s}oMath" % M)
    mtext = "".join(t.text or "" for t in mp.findall(".//{%s}t" % M))
    idx = list(p._p).index(mp)
    p._p.remove(mp)
    for i, om in enumerate(oms):
        p._p.insert(idx + i, om)
    pf = p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf.line_spacing = 1.5
    pf.space_before = Pt(6)
    pf.space_after = Pt(6)
    if mtext.replace(" ", "").startswith("PSNR=10log"):
        continue
    num += 1
    pf.tab_stops.add_tab_stop(TEXT_W, WD_TAB_ALIGNMENT.RIGHT)
    r = p.add_run("\t(%d)" % num)
    r.font.size = Pt(12)
    r.font.name = "Times New Roman"
print("equations numbered:", num)

# ---------------------------------------------------------------- CRediT block
ref_p = None
for p in doc.paragraphs:
    if p.style.name == "Heading 1" and p.text.strip() == "Declaration of competing interest":
        ref_p = p
        break
assert ref_p is not None

head_text = "CRediT authorship contribution statement"
body_text = ("Zhaoyang Ma: Methodology, Software, Validation, Formal analysis, "
             "Writing - original draft, Visualization. Dawei Liu: Conceptualization, "
             "Writing - review & editing, Supervision. "
             "Jinghan Yang: Investigation, Data curation, Validation.")
for text, is_head in ((head_text, True), (body_text, False)):
    newp = copy.deepcopy(ref_p._p)
    for child in list(newp):
        if child.tag != qn("w:pPr"):
            newp.remove(child)
    ref_p._p.addprevious(newp)
    para = Paragraph(newp, ref_p._parent)
    para.add_run(text)
    if is_head:
        para.style = get_style("Heading 1")
        para.runs[0].text = text
        fmt(para, size=12, bold=True, before=18, after=6, spacing=1.5, keep_next=True)
    else:
        para.style = get_style("Normal")
        fmt(para, size=12, align=WD_ALIGN_PARAGRAPH.JUSTIFY, before=0, after=12, spacing=2.0)

# ---------------------------------------------------------------- properties
doc.core_properties.title = ("From image fidelity to product quality: image-domain "
                             "metrics misrank cloud-removal models for Sentinel-2 LAI, "
                             "Cab and NDVI retrieval")
doc.core_properties.author = "Zhaoyang Ma; Dawei Liu; Jinghan Yang"
doc.core_properties.subject = ("International Journal of Applied Earth Observation "
                               "and Geoinformation")

doc.save(PATH)
print("saved", PATH)
