# -*- coding: utf-8 -*-
"""Regenerate highlights.docx from highlights.txt (Elsevier: 3-5 bullets, each <=85 chars)."""
import io
from pathlib import Path

from docx import Document
from docx.shared import Pt

ROOT = Path(r"D:/mazhaoyang/code program/cloudrs1/last/_jag")
lines = [l.strip() for l in io.open(ROOT / "highlights.txt", encoding="utf-8").read().splitlines() if l.strip()]
assert 3 <= len(lines) <= 5, len(lines)
for l in lines:
    assert len(l) <= 85, (len(l), l)

doc = Document()
st = doc.styles["Normal"]
st.font.name = "Times New Roman"
st.font.size = Pt(12)
st.paragraph_format.space_after = Pt(0)

h = doc.add_paragraph()
r = h.add_run("Highlights")
r.bold = True
r.font.size = Pt(14)
h.paragraph_format.space_after = Pt(12)

for l in lines:
    p = doc.add_paragraph(l, style="List Bullet")
    p.paragraph_format.space_after = Pt(8)

out = ROOT / "highlights.docx"
doc.save(out)
print("wrote", out, "items:", len(lines))
