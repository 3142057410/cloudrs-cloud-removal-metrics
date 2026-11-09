# -*- coding: utf-8 -*-
"""把中文完整稿 v4 重排为 IEEE Transactions 模板格式（正文仍为中文）。

做法：读取 v4 的块（段落/表/图），按模板规范重新排版，而不是重打一遍内容——
这样数字、表、图与 v4 完全一致，不会在重排中引入数值漂移。

排版要点（取自模板实测）：
  纸张 US Letter 21.59×27.94 cm；边距 上/下 1.78 cm、左/右 1.65 cm
  题名区单栏；正文双栏，栏间距 720 twips（0.5 in）
  Title 24 pt；作者 11 pt；摘要/关键词 9 pt 粗；一级标题居中（段前 12 pt、段后 4 pt）；
  二级标题斜体（段前 6 pt、段后 3 pt）；正文 10 pt 两端对齐；图表题注 8 pt；
  参考文献 8 pt 悬挂缩进 13.5 pt；公式居中、编号右对齐
  全幅表格与图用「连续分节 + 单栏」跨栏
"""
import io
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "drafts/论文_中文完整稿_v4.docx")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "drafts/论文_IEEE格式_中文_v3.docx")
REFMD = Path("drafts/参考文献_42条已核实.md")

# ---------------------------------------------------------------- 版面常量
PAGE_W, PAGE_H = Cm(21.59), Cm(27.94)
MG_TB, MG_LR = Cm(1.78), Cm(1.65)
GUTTER_TW = 720
COL_W_IN = 3.35          # 单栏宽
FULL_W_IN = 7.16         # 双栏总宽
BODY_PT = 10.0

CN, EN, CN_B = "宋体", "Times New Roman", "黑体"

# ---------------------------------------------------------------- 基础工具


def set_run(run, size=BODY_PT, bold=False, italic=False, cn=CN, en=EN):
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.name = en
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.insert(0, rf)
    rf.set(qn("w:ascii"), en)
    rf.set(qn("w:hAnsi"), en)
    rf.set(qn("w:eastAsia"), cn)


def add_rich(p, text, size=BODY_PT, bold=False, italic=False, cn=CN, en=EN):
    """支持 **粗体** 行内标记——原稿里用作强调，重排时要还原成真粗体。"""
    for k, seg in enumerate(text.split("**")):
        if not seg:
            continue
        set_run(p.add_run(seg), size=size, bold=(bold or k % 2 == 1),
                italic=italic, cn=cn, en=en)


def para(doc, text="", size=BODY_PT, bold=False, italic=False, align="just",
         indent_first=0.0, space_before=0, space_after=0, line=1.0,
         left_indent=0.0, cn=CN, en=EN, keep_next=False):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.line_spacing = line
    pf.left_indent = Pt(left_indent)
    if indent_first:
        pf.first_line_indent = Pt(indent_first)
    pf.keep_with_next = keep_next
    pf.alignment = {"just": WD_ALIGN_PARAGRAPH.JUSTIFY,
                    "center": WD_ALIGN_PARAGRAPH.CENTER,
                    "left": WD_ALIGN_PARAGRAPH.LEFT,
                    "right": WD_ALIGN_PARAGRAPH.RIGHT}[align]
    if text:
        add_rich(p, text, size, bold, italic, cn, en)
    return p


def set_cols(section, num):
    sectPr = section._sectPr
    cols = sectPr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sectPr.append(cols)
    cols.set(qn("w:num"), str(num))
    cols.set(qn("w:space"), str(GUTTER_TW))
    cols.set(qn("w:equalWidth"), "1")


def page_setup(section):
    section.page_width, section.page_height = PAGE_W, PAGE_H
    section.top_margin = section.bottom_margin = MG_TB
    section.left_margin = section.right_margin = MG_LR
    section.header_distance = Cm(0.9)
    section.footer_distance = Cm(0.9)


def span_begin(doc):
    """开始一个跨栏（单栏）区块。"""
    s = doc.add_section(WD_SECTION.CONTINUOUS)
    page_setup(s)
    set_cols(s, 1)
    return s


def span_end(doc):
    s = doc.add_section(WD_SECTION.CONTINUOUS)
    page_setup(s)
    set_cols(s, 2)
    return s


def ieee_table_borders(table, header_rows=1):
    """IEEE 表：仅上下边线与表头下线，无竖线。"""
    tbl = table._tbl
    tblPr = tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge, sz in (("top", 6), ("bottom", 6), ("left", 0), ("right", 0),
                     ("insideH", 0), ("insideV", 0)):
        el = OxmlElement(f"w:{edge}")
        if sz:
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), str(sz))
            el.set(qn("w:color"), "000000")
        else:
            el.set(qn("w:val"), "none")
            el.set(qn("w:sz"), "0")
        borders.append(el)
    tblPr.append(borders)
    for cell in table.rows[header_rows - 1].cells:
        tcPr = cell._tc.get_or_add_tcPr()
        b = OxmlElement("w:tcBorders")
        bot = OxmlElement("w:bottom")
        bot.set(qn("w:val"), "single")
        bot.set(qn("w:sz"), "4")
        bot.set(qn("w:color"), "000000")
        b.append(bot)
        tcPr.append(b)


def add_table_block(doc, data, cap_no_roman, caption, note, size=7.0, widths=None,
                    width_in=FULL_W_IN):
    """表题在上（TABLE + 说明），随后是表体，最后是注。"""
    para(doc, f"TABLE {cap_no_roman}", size=8, align="center",
         space_before=8, space_after=1, cn=CN_B, keep_next=True)
    para(doc, caption, size=8, align="center", space_after=3, cn=CN_B, keep_next=True)
    nrow, ncol = len(data), len(data[0])
    t = doc.add_table(rows=nrow, cols=ncol)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    # 沿用原表列宽比例，按目标宽度缩放
    if widths and len(widths) == ncol and sum(widths) > 0:
        scale = width_in / sum(widths)
        cw = [Inches(max(0.4, w * scale)) for w in widths]
    else:
        cw = [Inches(width_in / ncol)] * ncol
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.width = cw[j]
            c.text = ""
            pp = c.paragraphs[0]
            pp.paragraph_format.space_before = Pt(1.5)
            pp.paragraph_format.space_after = Pt(1.5)
            pp.paragraph_format.line_spacing = 1.0
            pp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_run(pp.add_run(str(val)), size=size, bold=(i == 0))
    ieee_table_borders(t)
    if note:
        para(doc, note, size=7.5, align="just", space_before=2, space_after=6, cn=CN)


def add_fig_block(doc, blob, cap_no, caption, width_in=6.4):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(io.BytesIO(blob), width=Inches(width_in))
    para(doc, f"Fig. {cap_no}. {caption}", size=8, align="just",
         space_after=8, cn=CN)


def add_equation(doc, text, no):
    """公式居中、编号右对齐；过长公式按逗号折行，每行独立居中，编号只标在末行。"""
    parts = [x.strip() for x in re.split(r",\s{3,}", text) if x.strip()]
    for k, part in enumerate(parts):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.space_before = Pt(4 if k == 0 else 0)
        pf.space_after = Pt(4 if k == len(parts) - 1 else 0)
        pf.line_spacing = 1.0
        pf.first_line_indent = Pt(0)
        pf.left_indent = Pt(0)
        pf.tab_stops.add_tab_stop(Inches(COL_W_IN / 2 - 0.12), WD_TAB_ALIGNMENT.CENTER)
        pf.tab_stops.add_tab_stop(Inches(COL_W_IN - 0.02), WD_TAB_ALIGNMENT.RIGHT)
        set_run(p.add_run("\t" + part), size=8.5, cn=CN)
        if k == len(parts) - 1:
            set_run(p.add_run(f"\t({no})"), size=8.5, cn=CN)



def add_page_number(section):
    """页脚居中页码（IEEE 评审稿需要页号）。"""
    from docx.oxml import OxmlElement as OE
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    set_run(run, size=9)
    fld = OE("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    r = OE("w:r")
    t = OE("w:t")
    t.text = "1"
    r.append(t)
    fld.append(r)
    p._p.append(fld)

# ---------------------------------------------------------------- 读 v4 块
def read_blocks(path):
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = Document(str(path))
    blocks = []
    for child in d.element.body.iterchildren():
        if child.tag == qn("w:p"):
            p = Paragraph(child, d)
            blips = child.findall(".//" + qn("a:blip"))
            text = p.text.strip()
            if blips:
                rid = blips[0].get(qn("r:embed"))
                blocks.append({"t": "img", "blob": d.part.related_parts[rid].blob})
            elif text:
                sz = None
                for r in p.runs:
                    if r.font.size:
                        sz = r.font.size.pt
                        break
                blocks.append({
                    "t": "p", "text": text, "size": sz,
                    "bold": bool(p.runs[0].bold) if p.runs else False,
                    "italic": bool(p.runs[0].italic) if p.runs else False,
                    "center": p.alignment == WD_ALIGN_PARAGRAPH.CENTER,
                })
        elif child.tag == qn("w:tbl"):
            t = Table(child, d)
            widths = []
            for c in t.columns:
                try:
                    widths.append(c.width.inches if c.width else 0)
                except Exception:
                    widths.append(0)
            blocks.append({"t": "tbl",
                           "data": [[c.text.strip() for c in row.cells] for row in t.rows],
                           "widths": widths})
    return blocks


# ---------------------------------------------------------------- 文本改写
# 旧编号 -> 新编号（工作编号，之后按正文首现重排）
FIXES = [
    # §II-A 生成族：原来错挂 Fmask/Foga
    ("其优势在于能合成纹理，风险在于可能编造云下并不存在的地物 [9], [10]。",
     "其优势在于能合成纹理，风险在于可能编造云下并不存在的地物 [4], [5], [6], [8], [16]。"),
    # §II-A 指标质疑：原来错挂 Cloud-GAN/GUANet
    ("因而主张引入光谱角指标；也有基准报告方法排序会随报告指标而变 [4], [12]。",
     "因而主张引入光谱角指标 [19]；也有基准报告方法排序会随报告指标而变 [31], [32]。"),
    # §II-B 波段依赖：原来错挂 DiffCR
    ("LAI 主要依赖红边与短波红外，Cab 主要依赖可见光的蓝红吸收 [8]。",
     "LAI 主要依赖红边与短波红外，Cab 主要依赖可见光的蓝红吸收 [33], [35], [36]。"),
    # §III-D PROSAIL：原来错挂 DiffCR
    ("使用规模为 30 万条的 PROSAIL 查找表 [8]",
     "使用规模为 30 万条的 PROSAIL 查找表 [34], [35]"),
    # §III-A 云掩膜来源：补云检测综述
    ("云掩膜取自官方场景分类波段（SCL），而非阈值法。",
     "云掩膜取自官方场景分类波段（SCL），而非阈值法——云检测算法的选择本身会显著影响"
     "掩膜的完备性与边缘精度 [13]。"),
    # §I-B 证据累积：原句无引用，就地补上，避免另起一段重复同一内容
    ("认为答案可能是否定的证据正在累积：生成式模型被观察到在感知指标上表现良好却编造云下地物；"
     "光谱角指标之所以被提出，正是因为 PSNR 忽略光谱形状；近年的若干基准也报告排序会随所选指标而变化。",
     "认为答案可能是否定的证据正在累积：生成式模型被观察到在感知指标上表现良好却编造云下地物 "
     "[8], [16]；光谱角指标之所以被提出，正是因为 PSNR 忽略光谱形状 [19]；"
     "近年的若干基准也报告排序会随所选指标而变化 [31], [32]。"),
    # §III-C 指标定义：出处必须标
    ("其中 SAM 逐像素计算后取云区均值，",
     "其中 SSIM 的定义见 [18]，SAM 见 [19]，ERGAS 的定义与用法见 [29]、[30]；"
     "SAM 逐像素计算后取云区均值，"),
    # §III-B 方法来源
    ("微调的 UnCRtainTS（0.52 M 参数）。",
     "微调的 UnCRtainTS（0.52 M 参数）[41]。"),
    ("DSen2-CR 残差结构的两种配置，",
     "DSen2-CR 残差结构的两种配置 [3]，"),
    # §III-E 统计方法：非参数检验先例
    ("即“a 优于 b”的重采样频率，而非传统 p 值；",
     "即“a 优于 b”的重采样频率，而非传统 p 值（同类比较中亦有工作改用符号检验 [32]）；"),
    # §III-A 数据集本身必须给出处
    ("主数据集取自 SEN12MS-CR，",
     "主数据集取自 SEN12MS-CR [2]，"),

    # ===== 按模板「缩写首次出现即定义」补全（摘要与正文各一次） =====
    ("（PSNR、SSIM、SAM、ERGAS），其隐含假设",
     "（峰值信噪比 PSNR、结构相似性 SSIM、光谱角 SAM、相对全局综合误差 ERGAS），其隐含假设"),
    ("与 NDVI 分类——系统检验该假设",
     "与归一化差异植被指数（NDVI）分类——系统检验该假设"),
    ("多数论文用 PSNR 与 SSIM，少数用 SAM 与 ERGAS。",
     "多数论文用峰值信噪比（PSNR）与结构相似性（SSIM），少数用光谱角（SAM）与相对全局综合误差（ERGAS）。"),
    ("叶面积指数与叶绿素含量的反演对波段敏感区间不同",
     "叶面积指数（LAI）与叶绿素含量（Cab）的反演对波段敏感区间不同"),
    ("图像域指标取 PSNR、SSIM、SAM、ERGAS、MAE、RMSE 六项",
     "图像域指标取 PSNR、SSIM、SAM、ERGAS、平均绝对误差（MAE）、均方根误差（RMSE）六项"),
    # 模板：符号须在公式出现前或紧随其后定义
    ("SSIM 在逐波段、逐窗口计算后取平均。",
     "SSIM 在逐波段、逐窗口计算后取平均。式(2) 中 ⟨·,·⟩ 表示两向量的内积，‖·‖ 为其 L2 范数；"
     "式(3) 中 RMSE_b 为第 b 波段的均方根误差。"),
    ("（3）NDVI 分类：在云区像素上",
     "（3）归一化差异植被指数（NDVI）分类：在云区像素上"),

    # ===== 表号统一为罗马数字（模板：Table 用罗马数字，且不缩写） =====
    # 原稿「表 3/4/5」与实际表序错位一位（主结果表是第 2 张表），按实际表序修正
    ("表 3 给出 15 个方法", "表 II 给出 15 个方法"),
    ("表 4 使聚合口径的依赖性变得明确", "表 III 使聚合口径的依赖性变得明确"),
    ("减到 13、11、9（表 5）", "减到 13、11、9（表 IV）"),
    ("表 7 是本文最具实践意义的一张表", "表 VII 是本文最具实践意义的一张表"),
]

# 段落末尾补「图表引用」：原稿 8 图 9 表从未在正文被提及
APPEND = {
    "该数据不含 SAR，故仅评估 9 个纯光学方法。":
        "两套数据的基本配置见表 I。",
    "复现数据集上顺序同样保持。":
        "剔除明显差的方法后，同一关系的重建见图 2。",
    "而 UnCRtainTS 在两者上均列第一。":
        "各指标与 ΔLAI 的散点关系见图 1。",
    "这就是脱耦的物理来源。":
        "各波段的 MSE 占比与反演敏感度见表 V，两类重要性剖面的对比见图 3。",
    "符合值域受限效应。":
        "三个方向的分层秩相关见表 VI，对应的判别力对比见图 4。",
    "对每个任务分别用非负最小二乘拟合指标权重":
        "拟合权重见表 VIII。",
    "前述结果均假设云掩膜准确":
        "三种掩膜变体下的判别力见表 IX，其变化幅度见图 5。",
    "换一个训练种子，PSNR 可以移动 3 dB 而下游排序几乎不动。":
        "六个学习基线的种子稳定性见表 X。",
    "这两项与下游的关联弱得多。":
        "各方法的误差形态构成见表 XI，三个分量的构成与判别力对比见图 6。",
    "提示自训练模型不能跨数据集直接比较。":
        "两套数据上的判别力对照见表 XII；图 7 给出一例失败案例的误差分布，"
        "图 8 为典型重建结果的目视对比。",
    "这份清单可直接落到论文写作与产品验收两个场景。":
        "清单见表 XIII。",
}

# 段落级插入：锚点句 -> 追加到该段之后的新段
AFTER = {
    "去云——重建云覆盖之下的地表反射率——在短短几年内从少数原型发展为拥挤的基准竞赛。":
        "云对光学观测的遮挡是系统性的：约半数星载光学观测受云或霾影响 [2]，这使去云成为地表监测"
        "预处理链条上无法绕过的一步；近十年深度学习进入遥感后，这一任务迅速从手工算子转向端到端网络 [42]。",
    "这条链条从未被仔细审查。":
        "这类质疑并非始于本文。信号保真度是否等同于感知质量，在图像处理领域已争论二十余年 [21], [26]；"
        "在遥感图像融合领域，“指标能否代表产品可用性”同样长期没有定论，评价标准缺失本身就是该领域"
        "反复讨论的核心问题 [28], [30]。",
    "本文补上这三点。":
        "与本文最接近的是 Wang 等 [31]：他们同样指出图像相似度指标对去云结果“可能无效甚至误导”，"
        "并以甚高分辨率影像上的场景分类验证了这一点。但该工作以提出新模型为目标，其批评依据是地表变化"
        "与视角差异；它既未固定候选池，也未讨论聚合口径的影响，更未定位失效的波段来源。Zhang 等 [32] "
        "则指出以均值为唯一汇总会掩盖方法间的细节差异。本文因此不与之竞争“更好的模型”，"
        "而是回答一个它们未涉及的问题：在固定候选池上，指标排序与连续参数反演的精度是否一致，"
        "以及不一致从哪几个波段来。",
    "第三类是生成式方法":
        "近三年这一路线进一步分化：厚云场景下陆续出现多尺度多分支结构 [17] 与全局—局部时序特征 [14]，"
        "合成孔径雷达（SAR）引导的条件扩散与双分支扩散也被用于同一问题 [8], [11], [16]；多时相序列重建则把缺失像元"
        "当作时序插值求解 [7]；非配对设定与风格自适应进一步放松了对训练数据的要求 [5]；"
        "光学与 SAR 的生成式融合亦被用于高分辨率场景 [15]。这些工作共同把 PSNR 推高，"
        "但正如后文所示，PSNR 的提升并不等价于反演可用性的提升。",
}

# 章节末尾追加（用于 §IV、§V 这些原文没有引用的节）
SECTION_TAIL = {
    "B.  聚合口径：同一批数据、相反的结论":
        "以单一均值汇总会掩盖方法间的细节差异，这一判断此前已被指出 [32]；本文进一步表明，"
        "换一个聚合口径不仅改变细节，还会改变结论的方向。",
    "D.  机制：波段级的解释":
        "这一结果与反演理论一致：辐射传输反演本身是病态的，观测误差会被放大到参数空间 [37]，"
        "而不同反演方法对波段的依赖并不相同 [36]，因此误差落在哪个波段，决定了它是否进入反演。",
    "G.  鲁棒性：掩膜边界与随机种子":
        "掩膜定义本身即是一个受质疑的环节：已有工作指出静态二值掩膜无法刻画云厚变化 [12]，"
        "而重建可靠性的另一条刻画路径是显式给出不确定性 [41]。本文的扰动实验说明，"
        "在本文的数据与候选池下，此类差异不足以改变排序结论。",
    "H.  失败形态：同一个 PSNR，不同的错法":
        "把失真分解为若干正交分量并非新做法，图像质量指标早已把失真拆为相关损失、亮度失真与"
        "对比度失真三项 [20]；本文的分解只是把这一思路移到“偏移—对比度—结构”三个与重建误差"
        "直接对应的分量上，并用它区分不同族的失败方式 [8], [16]。",
    "A.  为什么 PSNR 会被高估":
        "需要强调的是，文献 [21] 并不主张弃用 MSE，而是反对不加论证地替换；本文结论与之相容——"
        "PSNR 在筛除差方法上有效，问题出在用它区分好方法。而失真与感知质量此消彼长的关系"
        "已被证明 [25]，光谱角指标正是为弥补 PSNR 忽略光谱形状而提出 [19]。",
    "B.  为什么 SSIM 尤其不适合":
        "SSIM 家族本身也在演进：从单尺度 [18] 到多尺度 [22]，以及信息保真度 [23] 与特征相似度 [24] "
        "两条替代路线，均以全参考保真度为前提。深度特征度量进一步表明，PSNR 与 SSIM 只是浅层函数，"
        "抓不住感知细节 [26]；而在生成式重建中，失真与感知的最优并不同时取得 [25]。",
    "C.  三个任务的共同启示":
        "这与近期“下游任务感知”的评价主张方向一致 [31]，区别在于本文给出的是跨 15 个方法、"
        "三种聚合口径与两类任务机制的定量证据，而非单一任务上的个例。",
    "D.  对评价实践的建议":
        "在指标选择上，融合质量评价领域关于“无真值即无可靠评价”的讨论可直接借鉴 [27]："
        "当缺少高分辨率参考时，任何全参考指标的排序都应被谨慎解读 [29], [30]。"
        "在报告规范上，建议同时给出口径、候选池与不确定性，并公开代码与权重以保证可复现 [40]。",
    "E.  局限":
        "此外，云掩膜来自单一算法，而不同云检测算法给出的掩膜本就有系统差异 [9], [10]；"
        "已有基准指出参考数据缺乏共识会妨碍比较 [38]，去云领域也长期缺少统一基准 [39]。"
        "掩膜假设的影响已在 §IV-G 量化，但它与云厚分布的关系仍未展开 [12]。",
}


def apply_text_edits(blocks):
    text_hits = 0
    for b in blocks:
        if b["t"] != "p":
            continue
        for old, new in FIXES:
            if old in b["text"]:
                b["text"] = b["text"].replace(old, new)
                text_hits += 1
    # 段末补图表引用
    used, app = set(), 0
    for b in blocks:
        if b["t"] != "p":
            continue
        for key, extra in APPEND.items():
            if key in b["text"] and key not in used:
                b["text"] = b["text"].rstrip() + extra
                used.add(key)
                app += 1
    miss_app = [k for k in APPEND if k not in used]
    # 段后插入
    out, ins = [], 0
    for b in blocks:
        out.append(b)
        if b["t"] == "p":
            for anchor, extra in AFTER.items():
                if anchor in b["text"]:
                    out.append({"t": "p", "text": extra, "size": BODY_PT,
                                "bold": False, "italic": False, "center": False})
                    ins += 1
    print(f"  文本替换命中 {text_hits}/{len(FIXES)}；段后插入 {ins}/{len(AFTER)}；"
          f"图表引用补入 {app}/{len(APPEND)}")
    for k in miss_app:
        print(f"    !! 未命中: {k[:40]}")
    return out


def heading_level(b):
    """按 v4 的字号规则判定标题层级：13 pt 为一级，11.5 pt 为二级。"""
    if b["t"] != "p":
        return 0
    sz = b.get("size") or 0
    tx = b["text"].strip()
    if sz >= 12 and re.match(r"^(I{1,3}|IV|V|VI)\.\s", tx):
        return 1
    if 10.5 <= sz < 12 and re.match(r"^[A-Z]\.\s", tx):
        return 2
    return 0


def _norm(s):
    return re.sub(r"[\s\u3000]+", "", s)


def inject_section_tails(blocks):
    """在每个目标二级节的最后一个正文段之后插入收束段（带引用）。"""
    n = 0
    for title, extra in SECTION_TAIL.items():
        tn = _norm(title)[:14]
        idx = None
        for i, b in enumerate(blocks):
            if b["t"] == "p" and _norm(b["text"]).startswith(tn):
                idx = i
                break
        if idx is None:
            print(f"  !! 未找到小节：{title[:20]}")
            continue
        j, last = idx + 1, idx
        while j < len(blocks):
            b = blocks[j]
            if b["t"] == "p" and heading_level(b):
                break
            if b["t"] == "p" and b["text"].strip():
                last = j
            j += 1
        blocks.insert(last + 1, {"t": "p", "text": extra, "size": BODY_PT,
                                 "bold": False, "italic": False, "center": False})
        n += 1
    print(f"  章节收束段插入 {n}/{len(SECTION_TAIL)}")
    return blocks


# ---------------------------------------------------------------- 参考文献
def load_refs():
    txt = REFMD.read_text(encoding="utf-8")
    refs = {}
    for m in re.finditer(r"^\[(\d+)\]\s+(.+?)(?=\n\n|\n>|\n\[|\Z)", txt, re.M | re.S):
        num = int(m.group(1))
        body = re.sub(r"\s+", " ", m.group(2)).strip()
        body = re.sub(r"\s*\*([^*]+)\*", r" \1", body).rstrip(".")
        if num not in refs:
            refs[num] = body
    return refs


ROMAN_TAB = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
             "XI", "XII", "XIII", "XIV", "XV", "XVI"]


def main():
    blocks = read_blocks(SRC)
    print(f"读入 {len(blocks)} 块：段落 {sum(1 for b in blocks if b['t']=='p')}、"
          f"表 {sum(1 for b in blocks if b['t']=='tbl')}、图 {sum(1 for b in blocks if b['t']=='img')}")

    blocks = apply_text_edits(blocks)
    blocks = inject_section_tails(blocks)

    # 截断到「附：结构与投稿说明」之前；去掉旧的参考文献列表
    cut = len(blocks)
    for i, b in enumerate(blocks):
        if b["t"] == "p" and b["text"].startswith("附：结构与投稿说明"):
            cut = i
            break
    blocks = blocks[:cut]
    ref_start = None
    for i, b in enumerate(blocks):
        if b["t"] == "p" and b["text"].strip() in ("参考文献", "参 考 文 献"):
            ref_start = i
            break
    if ref_start is not None:
        blocks = blocks[:ref_start]

    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = EN
    st.font.size = Pt(BODY_PT)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), CN)
    sec0 = doc.sections[0]
    page_setup(sec0)
    set_cols(sec0, 1)
    add_page_number(sec0)

    # ---------------- 题名区（单栏）
    title = blocks[0]["text"]
    para(doc, title, size=24, bold=False, align="center",
         space_before=6, space_after=10, cn=CN_B, line=1.0)
    para(doc, "作者姓名（待填）", size=11, align="center", space_after=2)
    para(doc, "作者单位、城市、国家（待填）", size=9, align="center",
         space_after=14, italic=True)

    i = 1
    # 跳过英文题名建议
    while i < len(blocks) and blocks[i]["t"] == "p" and blocks[i]["text"].startswith("（英文题名建议"):
        i += 1
    # 跳过「摘  要」小标题
    if blocks[i]["t"] == "p" and blocks[i]["text"].replace(" ", "") in ("摘要",):
        i += 1
    first_abs = True
    abs_paras = []
    while i < len(blocks) and blocks[i]["t"] == "p" and \
            not blocks[i]["text"].startswith(("关键词", "关键字")):
        abs_paras.append(blocks[i]["text"])
        i += 1
    # IEEE 摘要为单段，把原稿的 4 段合并
    if abs_paras:
        p = para(doc, "", size=9, bold=True, align="just",
                 indent_first=10.1, space_after=2, line=1.0)
        set_run(p.add_run("摘要——"), size=9, bold=True, cn=CN_B)
        add_rich(p, " ".join(abs_paras), size=9, bold=True)
    if i < len(blocks):
        kw = blocks[i]["text"].split("：", 1)
        kw_txt = kw[1] if len(kw) > 1 else kw[0]
        para(doc, "关键词——" + kw_txt, size=9, bold=True, align="just",
             indent_first=10.1, space_before=6, space_after=10)
        i += 1

    # ---------------- 正文（双栏）
    body_sec = doc.add_section(WD_SECTION.CONTINUOUS)
    page_setup(body_sec)
    set_cols(body_sec, 2)

    fig_no = tab_no = eq_no = 0
    pending_caption = None
    j = i
    while j < len(blocks):
        b = blocks[j]
        if b["t"] == "tbl":
            data = b["data"]
            j += 1
            continue

        if b["t"] == "img":
            # 连续出现的图放在同一个跨栏块里，避免中间夹出空的单栏节
            group = []
            while j < len(blocks) and blocks[j]["t"] == "img":
                blob = blocks[j]["blob"]
                cap = ""
                j += 1
                if j < len(blocks) and blocks[j]["t"] == "p" and \
                        re.match(r"^图\s*\d+\s*", blocks[j]["text"]):
                    cap = re.sub(r"^图\s*\d+\s*", "", blocks[j]["text"])
                    j += 1
                group.append((blob, cap))
            span_begin(doc)
            for blob, cap in group:
                fig_no += 1
                w = 6.4 if len(group) == 1 else 5.2
                add_fig_block(doc, blob, fig_no, cap, width_in=w)
            span_end(doc)
            continue

        tx = b["text"]
        # 表题
        m = re.match(r"^表\s*(\d+)\s+(.*)$", tx)
        if m and j + 1 < len(blocks) and blocks[j + 1]["t"] == "tbl":
            tab_no += 1
            data = blocks[j + 1]["data"]
            note = ""
            k = j + 2
            if k < len(blocks) and blocks[k]["t"] == "p" and blocks[k]["text"].startswith(("注", "说明：")):
                note = blocks[k]["text"]
                k += 1
            ncol = len(data[0])
            if ncol <= 4:
                # 窄表直接放进单栏，避免无意义的跨栏留白
                add_table_block(doc, data, ROMAN_TAB[tab_no - 1], m.group(2), note,
                                widths=blocks[j + 1].get("widths"),
                                width_in=COL_W_IN, size=6.5)
            else:
                span_begin(doc)
                add_table_block(doc, data, ROMAN_TAB[tab_no - 1], m.group(2), note,
                                widths=blocks[j + 1].get("widths"),
                                width_in=FULL_W_IN, size=7.0)
                span_end(doc)
            j = k
            continue

        # 一级标题
        if heading_level(b) == 1:
            para(doc, tx, size=10, bold=True, align="center", space_before=12,
                 space_after=4, cn=CN_B, keep_next=True)
            j += 1
            continue
        # 二级标题
        if heading_level(b) == 2:
            para(doc, tx, size=10, bold=True, align="left", space_before=6,
                 space_after=3, cn=CN_B, keep_next=True)
            j += 1
            continue
        # 公式
        me = re.search(r"\((\d+)\)\s*$", tx)
        if me and b.get("center"):
            eq_no += 1
            add_equation(doc, re.sub(r"\s*\(\d+\)\s*$", "", tx), me.group(1))
            j += 1
            continue
        # 正文
        para(doc, tx, size=BODY_PT, align="just", indent_first=20,
             space_after=0, line=1.12)
        j += 1

    # ---------------- 参考文献
    refs = load_refs()
    # 按正文首现顺序重排
    full_text = "\n".join(p.text for p in doc.paragraphs)
    order = []
    for m in re.finditer(r"\[(\d+)\]", full_text):
        n = int(m.group(1))
        if n not in order and 1 <= n <= 42:
            order.append(n)
    missing = [n for n in range(1, 43) if n not in order]
    print(f"  正文引用到 {len(order)}/42 条；未出现：{missing}")
    rest = [n for n in range(1, 43) if n not in order]
    order += rest
    remap = {old: new for new, old in enumerate(order, 1)}
    # 工作编号 -> 成稿编号 的对照，便于与核实表交叉检查
    with open("drafts/_ref_map.csv", "w", encoding="utf-8") as f:
        f.write("成稿编号,工作编号,题名\n")
        for new_no, old in enumerate(order, 1):
            t = refs.get(old, "")[:70].replace(",", "，")
            f.write(f"{new_no},{old},{t}\n")

    def renum(m):
        n = int(m.group(1))
        return f"[{remap.get(n, n)}]" if n in remap else m.group(0)
    for p in doc.paragraphs:
        for r in p.runs:
            if "[" in r.text:
                r.text = re.sub(r"\[(\d+)\]", renum, r.text)

    para(doc, "参考文献", size=10, bold=True, align="center", space_before=12,
         space_after=4, cn=CN_B)
    for new_no, old in enumerate(order, 1):
        body = refs.get(old, "")
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.left_indent = Pt(13.5)
        pf.first_line_indent = Pt(-13.5)
        pf.space_after = Pt(1.5)
        pf.line_spacing = 1.0
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        set_run(p.add_run(f"[{new_no}] {body}"), size=8)

    try:
        doc.save(str(OUT))
    except PermissionError:
        import time as _t
        alt = OUT.with_name(OUT.stem + _t.strftime("_%H%M") + ".docx")
        doc.save(str(alt))
        print(f"  !! 原文件被占用，已改存 -> {alt}")
        globals()["OUT"] = alt
    print(f"\n已生成 {OUT}")
    print(f"  表 {tab_no} 个 / 图 {fig_no} 个 / 公式 {eq_no} 个 / 文献 {len(order)} 条")


if __name__ == "__main__":
    main()
