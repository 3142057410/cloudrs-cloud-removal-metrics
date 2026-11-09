# -*- coding: utf-8 -*-
"""Build a pandoc-friendly LaTeX source from main.tex (cas-dc) for the Word version."""
import io, os, re, subprocess, sys

# Paths are relative to this directory so that the archived copy runs as-is:
#   pandoc must be on PATH; run from paper/ after a successful pdflatex pass.
HERE = __import__("pathlib").Path(__file__).resolve().parent
SRC = str(HERE / "main.tex")
AUX = str(HERE / "main.aux")
OUT_TEX = str(HERE / "word.tex")
OUT_DOCX = str(HERE / "Elsevier_JAG_Manuscript.docx")

if not (HERE / "main.aux").exists():
    raise SystemExit(
        "main.aux is missing: run `pdflatex main.tex` twice in paper/ first "
        "(the build needs the label numbers from the .aux file).")
src = io.open(SRC, encoding="utf-8").read()
aux = io.open(AUX, encoding="utf-8").read()

# ---- label -> number ----
labels = {}
for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", aux):
    labels[m.group(1)] = m.group(2)
print("labels:", len(labels))

# ---- bibliography order ----
body = src[src.index(r"\begin{document}") + len(r"\begin{document}"): src.index(r"\end{document}")]
bib_part = body[body.index(r"\begin{thebibliography}"):]
keys = re.findall(r"\\bibitem\{([^}]+)\}", bib_part)
keymap = {k: i + 1 for i, k in enumerate(keys)}
print("bibitems:", len(keys))
assert 40 <= len(keys) <= 80, len(keys)  # sanity only; the count follows the manuscript

main_body = body[:body.index(r"\begin{thebibliography}")]

# ---- abstract / keywords ----
abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", src, re.S).group(1).strip()
kw = re.search(r"\\begin\{keywords\}(.*?)\\end\{keywords\}", src, re.S).group(1)
kws = [k.strip() for k in re.split(r"\\sep", kw) if k.strip()]
print("keywords:", len(kws), kws)

# ---- citations ----
def cite_repl(m):
    ks = [k.strip() for k in m.group(1).split(",")]
    nums = sorted({keymap[k] for k in ks})
    return "[" + ", ".join(str(n) for n in nums) + "]"

main_body = re.sub(r"\\cite\{([^}]+)\}", cite_repl, main_body)
left = re.findall(r"\\cite\{", main_body)
assert not left, left

# ---- cross references ----
def ref_repl(m):
    k = m.group(1)
    assert k in labels, k
    return labels[k]

main_body = re.sub(r"\\ref\{([^}]+)\}", ref_repl, main_body)

# ---- keep only from Introduction on ----
main_body = main_body[main_body.index(r"\section{Introduction}"):]

# ---- float environments: starred -> plain ----
main_body = main_body.replace(r"\begin{table*}", r"\begin{table}").replace(r"\end{table*}", r"\end{table}")
main_body = main_body.replace(r"\begin{figure*}", r"\begin{figure}").replace(r"\end{figure*}", r"\end{figure}")
main_body = re.sub(r"\\begin\{(table|figure)\}\[[^\]]*\]", r"\\begin{\1}", main_body)

# ---- drop layout-only commands inside floats ----
for pat in (r"\\renewcommand\{\\arraystretch\}\{[^}]*\}\n?",
            r"\\setlength\{\\tabcolsep\}\{[^}]*\}\n?",
            r"\\scriptsize\n?", r"\\centering\n?",
            r"\\footnotesize\n?"):
    main_body = re.sub(pat, "", main_body)

# ---- expand *{n}{spec} ----
def expand_star(spec):
    while True:
        m = re.search(r"\*\{(\d+)\}\{", spec)
        if not m:
            return spec
        n = int(m.group(1))
        depth, i = 1, m.end()
        while depth and i < len(spec):
            if spec[i] == "{":
                depth += 1
            elif spec[i] == "}":
                depth -= 1
            i += 1
        inner = spec[m.end():i - 1]
        spec = spec[:m.start()] + inner * n + spec[i:]


def strip_decor(spec):
    """Remove >{...}/<{...} column decorators and @{} spacing tokens."""
    out, i = [], 0
    while i < len(spec):
        if spec[i] in "><" and i + 1 < len(spec) and spec[i + 1] == "{":
            depth, j = 0, i + 1
            while True:
                if spec[j] == "{":
                    depth += 1
                elif spec[j] == "}":
                    depth -= 1
                if depth == 0:
                    break
                j += 1
            i = j + 1
            continue
        if spec[i:i + 3] == "@{}":
            i += 3
            continue
        if spec[i] in "|":
            i += 1
            continue
        out.append(spec[i])
        i += 1
    return "".join(out)


def fix_tabular(s):
    key = r"\begin{tabular}"
    out, i = [], 0
    while True:
        j = s.find(key, i)
        if j < 0:
            out.append(s[i:])
            break
        out.append(s[i:j])
        k = j + len(key)
        assert s[k] == "{", s[k:k + 20]
        depth, m = 0, k
        while True:
            if s[m] == "{":
                depth += 1
            elif s[m] == "}":
                depth -= 1
            if depth == 0:
                break
            m += 1
        spec = strip_decor(expand_star(s[k + 1:m]))
        out.append(key + "{" + spec + "}")
        i = m + 1
    return "".join(out)


main_body = fix_tabular(main_body)
main_body = main_body.replace(r"$^{\circ}$", "\u00b0")

# ---- captions: internal label -> printed number ----
def cap_repl(m):
    text, key = m.group(1), m.group(2)
    kind = key.split(":")[0]
    num = labels[key]
    prefix = "Table %s." % num if kind == "tab" else "Fig. %s." % num
    return r"\caption{" + prefix + " " + text.strip() + "}"

main_body = re.sub(r"\\caption\{((?:(?!\\caption).)*?)\\label\{((?:tab|fig):[^}]+)\}\}",
                   cap_repl, main_body, flags=re.S)
main_body = re.sub(r"\\label\{[^}]*\}", "", main_body)

# ---- CRediT: cas-dc prints it via \printcredits; Word builds its own block ----
main_body = re.sub(r"(?m)^\s*%.*printcredits.*\n?", "", main_body)
main_body = main_body.replace(r"\printcredits", "")

# ---- graphics: drop width options ----
main_body = re.sub(r"\\includegraphics\[[^\]]*\]", r"\\includegraphics", main_body)

# ---- equations: keep, numbering added later ----
main_body = re.sub(r"\\label\{eq:[^}]*\}\n?", "", main_body)

# ---- bibliography ----
entries = re.findall(r"\\bibitem\{([^}]+)\}\s*(.*?)(?=\\bibitem|\Z)",
                     bib_part[:bib_part.index(r"\end{thebibliography}")], re.S)
refs = []
for k, txt in entries:
    txt = " ".join(txt.split())
    refs.append("[%d] %s" % (keymap[k], txt))
print("refs built:", len(refs))

TITLE = ("Image-domain metrics misrank cloud-removal models for Sentinel-2 LAI, "
         "Cab and NDVI retrieval")

front = []
front.append("@@TITLE@@ " + TITLE)
front.append("")
front.append("@@AUTH@@ Zhaoyang Ma\\textsuperscript{1}, Dawei Liu\\textsuperscript{1,*}, "
             "Jinghan Yang\\textsuperscript{1}")
front.append("")
front.append("@@AFF@@ \\textsuperscript{1}Engineering University of the Chinese People's "
             "Armed Police Force, Xi'an, Shaanxi, China")
front.append("")
front.append("@@CORR@@ \\textsuperscript{*}Corresponding author. E-mail addresses: "
             "3142057410@qq.com (Z. Ma), wjmicheal@163.com (D. Liu), "
             "19031816221@163.com (J. Yang). ORCID: 0009-0007-8875-3515 (Z. Ma); "
             "0000-0003-2471-4713 (D. Liu); 0009-0007-0198-4098 (J. Yang)")
front.append("")
front.append("@@ABS@@ Abstract")
front.append("")
front.append(abstract)
front.append("")
front.append("@@KWD@@ Keywords: " + "; ".join(kws))
front.append("")

doc = ["\\documentclass{article}", "\\begin{document}", ""]
doc += front
doc.append(main_body.strip())
doc.append("")
doc.append("\\section*{References}")
doc.append("")
for r in refs:
    doc.append(r)
    doc.append("")
doc.append("\\end{document}")

io.open(OUT_TEX, "w", encoding="utf-8", newline="\n").write("\n".join(doc))
print("wrote", OUT_TEX)

r = subprocess.run(["pandoc", "word.tex", "-f", "latex", "-t", "docx",
                    "--resource-path=" + os.pathsep.join([str(HERE.parent), str(HERE.parent / "figures")]),
                    "-o", "Elsevier_JAG_Manuscript.docx"],
                   cwd=str(HERE), capture_output=True, text=True)
print("pandoc rc:", r.returncode, r.stderr[:800])
