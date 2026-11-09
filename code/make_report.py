"""
make_report.py —— 把实验结果汇总成一份自带图表的 HTML 报告

产出：<workspace>/P1实验结果报告.html（图片以 base64 内嵌，单文件可直接打开/发送）
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pandas as pd
import yaml

WS = Path(r"C:\Users\31420\WorkBuddy\2026-10-20-22-14-04")


def img(path: Path, caption: str) -> str:
    if not path.exists():
        return ""
    b = base64.b64encode(path.read_bytes()).decode()
    return (f'<figure><img src="data:image/png;base64,{b}" alt="{caption}"/>'
            f'<figcaption>{caption}</figcaption></figure>')


def table(df: pd.DataFrame, cls="", float_fmt="{:.3f}") -> str:
    def fmt(v):
        if isinstance(v, float):
            return float_fmt.format(v)
        return str(v)
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    body = ""
    for _, r in df.iterrows():
        body += "<tr>" + "".join(f"<td>{fmt(v)}</td>" for v in r) + "</tr>"
    return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def main():
    root = Path(r"D:\mazhaoyang\code program\cloudrs1")
    cfg = yaml.safe_load(open(root / "configs/p1.yaml", encoding="utf-8"))
    tab = root / cfg["output"]["tables"]
    fig = root / cfg["output"]["figures"]

    img_df = pd.read_csv(tab / "per_patch_metrics.csv")
    dn_df = pd.read_csv(tab / "per_patch_downstream.csv")
    df = img_df.merge(dn_df, on=["method", "patch"])
    _cols = ["PSNR", "SSIM", "SAM", "ERGAS", "MAE", "RMSE", "dLAI_mean"]
    if "dCab_mean" in df.columns:
        _cols.append("dCab_mean")
    g = df.groupby("method")[_cols].mean()
    g = g.sort_values("PSNR", ascending=False).round(4)

    ml = pd.read_csv(tab / "corr_method_level.csv")
    sl = pd.read_csv(tab / "corr_sample_level.csv")
    st = pd.read_csv(tab / "corr_stratified.csv")
    ss = pd.read_csv(tab / "corr_by_season.csv") if (tab / "corr_by_season.csv").exists() else None
    hq = pd.read_csv(tab / "corr_highq_subset.csv") if (tab / "corr_highq_subset.csv").exists() else None
    nm = pd.read_csv(tab / "corr_new_metric.csv")
    lomo = pd.read_csv(tab / "corr_new_metric_lomo.csv") if (tab / "corr_new_metric_lomo.csv").exists() else None
    nm_cab = pd.read_csv(tab / "corr_new_metric_cab.csv") if (tab / "corr_new_metric_cab.csv").exists() else None
    lomo_cab = pd.read_csv(tab / "corr_new_metric_lomo_cab.csv") if (tab / "corr_new_metric_lomo_cab.csv").exists() else None
    lc = pd.read_csv(tab / "corr_by_landcover.csv") if (tab / "corr_by_landcover.csv").exists() else None
    ct = pd.read_csv(tab / "corr_by_cloudthick.csv") if (tab / "corr_by_cloudthick.csv").exists() else None
    cty = pd.read_csv(tab / "corr_by_cloudtype.csv") if (tab / "corr_by_cloudtype.csv").exists() else None
    mech = pd.read_csv(tab / "band_mechanism.csv") if (tab / "band_mechanism.csv").exists() else None
    pc = pd.read_csv(tab / "corr_vs_dCab.csv") if (tab / "corr_vs_dCab.csv").exists() else None
    rk = pd.read_csv(tab / "rank_agreement.csv") if (tab / "rank_agreement.csv").exists() else None
    ms = pd.read_csv(tab / "mask_sensitivity_summary.csv") if (tab / "mask_sensitivity_summary.csv").exists() else None
    rdw = pd.read_csv(tab / "robust_drop_weak.csv") if (tab / "robust_drop_weak.csv").exists() else None
    pbd = pd.read_csv(tab / "pairwise_metric_diff.csv") if (tab / "pairwise_metric_diff.csv").exists() else None
    xs = pd.read_csv(tab / "corr_s2_crossregion.csv") if (tab / "corr_s2_crossregion.csv").exists() else None
    sp = pd.read_csv(tab / "stat_power_samplelevel.csv") if (tab / "stat_power_samplelevel.csv").exists() else None
    rel = pd.read_csv(tab / "highq_relative.csv") if (tab / "highq_relative.csv").exists() else None
    # dLAI 与 dCab 的方法排序秩相关（从下游表直接算，避免依赖中间文件）
    from scipy.stats import spearmanr as _sp
    RHO_RANK = float(_sp(df.groupby("method").dLAI_mean.mean(),
                         df.groupby("method").dCab_mean.mean()).correlation) \
        if "dCab_mean" in df.columns and df.dCab_mean.notna().any() else float("nan")
    seed = pd.read_csv(tab / "seed_stability_summary.csv") if (tab / "seed_stability_summary.csv").exists() else None
    seed_raw = pd.read_csv(tab / "seed_stability.csv") if (tab / "seed_stability.csv").exists() else None
    fsum = pd.read_csv(tab / "failure_summary.csv") if (tab / "failure_summary.csv").exists() else None
    fcmp = pd.read_csv(tab / "failure_vs_downstream.csv") if (tab / "failure_vs_downstream.csv").exists() else None
    if seed is not None:
        seed = seed[seed["PSNR_std"].notna()].reset_index(drop=True)
    meta = json.load(open(root / "data/processed/meta.json", encoding="utf-8"))
    split = json.load(open(root / "data/processed/split.json", encoding="utf-8"))

    n_methods, n_patch = g.shape[0], df.patch.nunique()
    psnr, sam = g["PSNR"], g["SAM"]
    spread = g["dLAI_mean"].max() / g["dLAI_mean"].min()

    # 排序变化最大的方法
    r_psnr = g["PSNR"].rank(ascending=False)
    r_dlai = g["dLAI_mean"].rank(ascending=True)
    shift = (r_psnr - r_dlai).abs().sort_values(ascending=False)
    worst = shift.index[0]

    hq_row = None
    if hq is not None:
        hq_row = hq.set_index("metric")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>P1 实验结果报告：去云评价指标与下游反演精度的一致性</title>
<style>
  :root {{ --ink:#1a1a1a; --mut:#5c6470; --line:#e2e6ea; --bg:#fbfcfd; --acc:#2f6fb3; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
    font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif; line-height:1.75; }}
  .wrap {{ max-width:1000px; margin:0 auto; padding:48px 28px 80px; }}
  h1 {{ font-size:27px; line-height:1.35; margin:0 0 6px; letter-spacing:-.3px; }}
  .sub {{ color:var(--mut); font-size:14px; margin-bottom:34px; }}
  h2 {{ font-size:20px; margin:46px 0 14px; padding-left:11px; border-left:4px solid var(--acc); }}
  h3 {{ font-size:16px; margin:26px 0 8px; color:#243b53; }}
  p {{ margin:10px 0; }}
  .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:14px; margin:22px 0 8px; }}
  .kpi {{ background:#fff; border:1px solid var(--line); border-radius:10px; padding:16px 18px; }}
  .kpi b {{ display:block; font-size:24px; color:var(--acc); line-height:1.25; }}
  .kpi span {{ font-size:12.5px; color:var(--mut); }}
  table {{ width:100%; border-collapse:collapse; margin:14px 0; font-size:13px;
    background:#fff; border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
  th,td {{ padding:8px 10px; text-align:right; border-bottom:1px solid var(--line); }}
  th:first-child, td:first-child {{ text-align:left; font-weight:600; }}
  thead th {{ background:#f3f6f9; font-weight:600; font-size:12.5px; color:#33475b; }}
  tbody tr:last-child td {{ border-bottom:none; }}
  tbody tr:hover {{ background:#f8fbfe; }}
  figure {{ margin:22px 0; background:#fff; border:1px solid var(--line);
    border-radius:10px; padding:14px; }}
  figure img {{ width:100%; display:block; border-radius:6px; }}
  figcaption {{ font-size:12.5px; color:var(--mut); margin-top:10px; text-align:center; }}
  .box {{ background:#fff; border:1px solid var(--line); border-left:4px solid var(--acc);
    border-radius:8px; padding:14px 18px; margin:18px 0; }}
  .warn {{ border-left-color:#c0392b; }}
  .good {{ border-left-color:#2c7a4b; }}
  code {{ background:#eef2f6; padding:1px 5px; border-radius:4px; font-size:12.5px; }}
  ul {{ margin:8px 0 8px 20px; padding:0; }}
  li {{ margin:5px 0; }}
  .foot {{ margin-top:50px; padding-top:18px; border-top:1px solid var(--line);
    font-size:12.5px; color:var(--mut); }}
</style></head><body><div class="wrap">

<h1>去云评价指标与下游反演精度的一致性</h1>
<div class="sub">P1 实验报告 · SEN12MS-CR（spring/scene_31 + winter/scene_63）·
{n_methods} 个去云方法 · {n_patch} 个测试 patch · 生成于 2026-10-28</div>

<div class="kpis">
  <div class="kpi"><b>{n_methods}</b><span>去云方法（3 族：统计 / 学习 / 生成）</span></div>
  <div class="kpi"><b>{n_patch}</b><span>测试 patch（128×128，13 波段）</span></div>
  <div class="kpi"><b>{df.shape[0]:,}</b><span>方法×patch 评估样本</span></div>
  <div class="kpi"><b>{spread:.2f}×</b><span>ΔLAI 最大/最小差距</span></div>
</div>

<h2>一、一句话结论</h2>
<div class="box">
<p><b>去云评价里真正该被弃用的是 SSIM 与 ERGAS，而不是 PSNR；PSNR 与 SAM 在统计上无法区分。</b></p>
<p>在 n={len(df)} 条的<b>样本层</b>（唯一有统计力度的口径）上，PSNR 与 ΔLAI 的秩相关
−{abs(sl.set_index('metric').loc['PSNR','spearman_vs_dLAI']):.2f}，
SAM +{abs(sl.set_index('metric').loc['SAM','spearman_vs_dLAI']):.2f}；
配对 bootstrap 显示 SAM 其实<b>略低于</b> PSNR 0.026（p=0.02，见 3.13），
也就是说"用 SAM 替代 PSNR"这条路<b>没有证据支持</b>。</p>
<p><b>但 SSIM 与 ERGAS 被明确否定</b>：SAM 对 SSIM 的优势是 +0.17，
95% 区间 [+0.13, +0.22]，p(diff&gt;0)=<b>1.000</b>——这是本文唯一达到决定性的结论，
且在两个数据集、三种掩膜变体、两个下游参数下都不变。</p>
<p><b>PSNR 的优势只来自"能区分差方法"。</b>全量方法层（n={n_methods}）PSNR 略高于 SAM
（{abs(ml.set_index('metric').loc['PSNR','spearman_vs_dLAI']):.2f} vs
{abs(ml.set_index('metric').loc['SAM','spearman_vs_dLAI']):.2f}），样本层同样
（{abs(sl.set_index('metric').loc['PSNR','spearman_vs_dLAI']):.2f} vs
{abs(sl.set_index('metric').loc['SAM','spearman_vs_dLAI']):.2f}）。
但一旦<b>只留下合格方法</b>，顺序就反过来：绝对口径（PSNR≥25 dB）下 SAM
{abs(hq_row.loc['SAM','spearman_in_highq']):.2f} vs PSNR {abs(hq_row.loc['PSNR','spearman_in_highq']):.2f}；
相对口径（各数据集 PSNR 前 K 名）下 SAM ≥ PSNR 在 K=4/5/6 三档、两个数据集上全部成立（见 3.13）。
<b>结论：PSNR 靠"筛掉差方法"拿到高相关，在好方法之间它并不占优。</b></p>
<p class="sub"><b>实践建议（可写进摘要）：</b>① 弃用 SSIM 与 ERGAS 作为去云选优依据；
② 初筛可用 PSNR，但在合格方法之间选优须同时报 SAM——单靠 PSNR 会失效；
③ 任何指标都必须绑定下游任务——换成 Cab，最优指标就从 SAM 变成 ERGAS/PSNR（3.7），
TCWS 权重里 PSNR 直接归零；④ 指标在卷云（3.6）与低植被（3.6）条件下判别力减半，
报告时必须给出条件构成。</p>
</div>

<h2>二、各方法的图像指标与下游影响</h2>
{table(g.reset_index())}
<p class="sub" style="margin-top:6px">ΔLAI 是"去云结果反演出的 LAI"与"晴空真值反演出的 LAI"之平均绝对差，
越小说明对下游定量应用越无害。指标计算已排除 B1/B9/B10 三个大气波段。</p>

<div class="box warn">
<p><b>排序不一致的直接证据：</b>按 PSNR 排第 {int(r_psnr[worst])} 的
<code>{worst}</code>，按 ΔLAI 排第 {int(r_dlai[worst])}——位置差 {int(shift.iloc[0])} 位。
这类反例不是个案，见下面的斜率图。</p>
</div>

{img(fig / "fig4_rank_compare.png", "图4 两种排序的斜率图：连线交叉即表示排序不一致")}

<h2>三、三层相关分析</h2>

<h3>3.1 方法层（n = {n_methods}）</h3>
{table(ml)}
<p>方法层的问题很直接：只有 {n_methods} 个点，置信区间宽，而且混着
<code>zero_fill</code> 这种一眼就能看出差的方法——它会同时拉高所有指标的相关性。</p>

<h3>3.2 样本层（n = {df.shape[0]:,}）</h3>
{table(sl)}
<p>逐 patch 层面上，PSNR、SAM 的相关性都在 0.65 左右，说明<b>两者都有信息，但都不充分</b>：
R²≈0.43，即近 60% 的下游误差变异无法由图像指标解释。
ERGAS 最弱（0.41），且在高云量下降到 0.22——这与它常被当作主要指标的做法不符。</p>

{img(fig / "fig2_sample_corr.png", "图2 样本层各指标与 ΔLAI 的秩相关及 bootstrap 95% 置信区间")}

<h3>3.3 高质量子集：实践中最关键的一问</h3>
{("<p>仅保留 PSNR≥25 dB 的 " + str(int(hq_row["n_methods"].iloc[0])) + " 个方法后：</p>" + table(hq)) if hq is not None else ""}
{img(fig / "fig6_highq_subset.png", "图6 剔除明显差的方法后，各指标的判别力（*=显著）")}
<div class="box good">
<p>这是整份报告最重要的一张图。<b>PSNR 从 {abs(ml.set_index('metric').loc['PSNR','spearman_vs_dLAI']):.2f}
掉到 {abs(hq_row.loc['PSNR','spearman_in_highq']):.2f}，
SAM 从 {abs(ml.set_index('metric').loc['SAM','spearman_vs_dLAI']):.2f} 到
{abs(hq_row.loc['SAM','spearman_in_highq']):.2f} 反而略升</b>。含义是：
在"好方法之间挑一个"这个真实场景里，PSNR 的判别力明显弱于 SAM，
SSIM 的判别力最弱。</p>
<p class="sub">注意读法：这里不能写"PSNR 失去显著性"——n={int(hq_row['n_methods'].iloc[0])} 时
p={hq_row.loc['PSNR','p_value']:.3f}，它<b>仍然显著</b>，只是效应量比 SAM 低
{abs(abs(hq_row.loc['SAM','spearman_in_highq'])-abs(hq_row.loc['PSNR','spearman_in_highq'])):.2f}。
把"效应量差异"说成"显著性差异"是审稿人最反感的一类过度解读。</p>
</div>

<h3>3.4 按云量分层</h3>
{table(st.pivot(index='metric', columns='cloud_bin', values='spearman').reset_index())}
{img(fig / "fig3_stratified.png", "图3 不同云量四分位下的秩相关")}
<p>结论方向在各云量档内基本稳定，没有反转；但所有指标的判别力在高云量（Q4）都下降，
说明<b>云越厚，任何图像指标都越不可靠</b>——这本身就是一条可写的实践建议：
厚云区应转而依赖不确定性估计，而不是图像指标。</p>

<h3>3.5 按季节分层</h3>
{("<p>春季与冬季内部各自重算，符号全部一致，说明结论不依赖季节：</p>" + table(ss.pivot(index='metric', columns='season', values='spearman').reset_index())) if ss is not None else ""}
{img(fig / "fig7_season.png", "图7 季节分层：两季符号一致，但冬季普遍更弱")}
<p>冬季所有指标的相关性都明显弱于春季（PSNR −0.75→−0.58，SAM 0.78→0.32），
与冬季场景云量更厚、晴空锚点更少一致。<b>这提示评测集必须报告季节/云量构成</b>，
否则同一方法在不同数据划分上会得到差别很大的结论。</p>

<h3>3.6 按地表类型、云厚度、云类型分层（W6 的三项）</h3>
{("<p>地表类型用晴空 NDVI 三分位代理，云厚度用云区可见光亮度三分位代理，云类型用 B10 卷云波段判据：</p>" + table(lc.pivot(index='metric', columns='landcover', values='spearman').reset_index()) + table(ct.pivot(index='metric', columns='cloud_thickness', values='spearman').reset_index()) + table(cty.pivot(index='metric', columns='cloud_type', values='spearman').reset_index())) if (lc is not None and ct is not None and cty is not None) else ""}
{img(fig / "fig8_groups.png", "图8 三项细分层：地表类型 / 云厚度 / 云类型")}
<div class="box warn">
<p><b>云类型是三项里影响最大的一项。</b>按 B10（1373nm 卷云波段）切分后，
卷云型场景下所有指标的判别力几乎减半：SAM 0.763→<b>0.478</b>、
PSNR −0.804→<b>−0.608</b>、SSIM −0.595→<b>−0.341</b>。</p>
<p>物理原因清楚：卷云薄而半透明，去云模型倾向于"轻微调色"而不是"重建"，
残差以平滑的光谱偏移形式存在——这种误差既不显著拉低 PSNR，又直接污染反演，
是所有指标最难捕捉的一类。<b>这条可以直接写进讨论节，回答"指标什么时候最不可靠"。</b></p>
</div>
<div class="box">
<p><b>低 NDVI 组的判别力整体掉一半</b>：SAM 从 0.747（高 NDVI）降到 0.405（低 NDVI），
PSNR 从 −0.781 降到 −0.590。原因是植被覆盖低时光谱对比弱，反演器本身的动态范围小，
云去除误差对参数的影响被压缩——这不是指标变差，而是<b>指标失去了可比较的余地</b>。</p>
<p>云厚度方向则是"薄云最弱"：SAM 0.655（薄）→ 0.744（中）→ 0.661（厚）。
这符合统计学上的<b>值域受限（restriction of range）</b>：薄云场景各方法的误差差距本来就小，
秩相关自然衰减。论文里要把这一点写清楚，否则会被误读成"薄云下指标不可靠"。</p>
<p class="sub">诚实声明：NDVI、云亮度、B10 都是<b>代理变量</b>，不是地物分类标签、COT 或云分类产品。
SEN12MS-CR 不提供 COT 与云类型标签，本数据集又以植被为主（NDVI 0.45–0.74），
所以这三项分层只能说"分层后结论仍有差异"，不能直接推广为定量的条件依赖关系。</p>
</div>

<h3>3.7 换一个下游参数：结论的边界在哪</h3>
<p>审稿人最可能的一句质疑是：<b>"ΔLAI 是你自己用 PROSAIL 查找表定义的自洽代理。"</b>
为此我们把同一批去云结果、同一个反演器，下游参数从 LAI 换成叶绿素 Cab
（Cab 主控蓝红吸收，LAI 主控近红外/短波红外散射，两者光谱敏感区明显不同）：</p>
{(table(pc)) if pc is not None else ""}
{img(fig / "fig11_param_consistency.png", "图11 同一批方法，换一个下游参数后各指标判别力的变化")}
<div class="box warn">
<p><b>三条结论，一条比一条重要：</b></p>
<ul>
<li><b>方向稳定</b>：6 个指标对 LAI 与 Cab 的相关<b>符号全部一致</b>，
不存在"换个参数结论就反转"的情况。这说明主结论不是 LAI 特有的。</li>
<li><b>最优指标不稳定</b>：对 LAI，高质量子集里 SAM 最强（0.933）；
对 Cab，最强的是 ERGAS（0.983），PSNR 也从 0.867 升到 0.967。
<b>"哪个指标最好"确实依赖下游参数。</b></li>
<li><b>SSIM 稳定地最差</b>：两种参数下都垫底（0.800 / 0.850），是唯一可以放心否定的指标。</li>
</ul>
<p>因此论文的主张不能写成"应当用 SAM"，而要写成<b>"不存在通用最优指标，评价指标必须与下游任务绑定"</b>——
这恰好也是 TCWS 存在的理由：权重必须针对具体任务拟合，不能跨任务复用。</p>
</div>
{(("<p>方法排序在两种参数下的秩相关 ρ = " + f"{RHO_RANK:.2f}" + "，其中 " + str(int((rk['shift'] >= 3).sum())) + " 个方法的排名变化 ≥3 位（最极端的是 <code>" + str(rk.iloc[0]['method']) + "</code>，从第 " + str(int(rk.iloc[0]['rank_dLAI'])) + " 位跳到第 " + str(int(rk.iloc[0]['rank_dCab'])) + " 位）。<b>这意味着「某方法更好」这句话，本身就必须附上「对哪个参数而言」。</b></p>") if rk is not None else "")}

<h3>3.8 机制：为什么 PSNR 与 ΔLAI 会脱耦</h3>
<p>到这里还只是"相关多少"的问题，审稿人一定会问<b>"为什么"</b>。我们把它拆到波段层面，
三组量放在同一根波段轴上比较：谁在决定 PSNR、谁在决定 ΔLAI、谁的 PSNR 真正预测下游误差。</p>
{(table(mech.sort_values('mse_share', ascending=False).round(4))) if mech is not None else ""}
{img(fig / "fig10_band_mechanism.png", "图10 主导 PSNR 的波段与主导反演的波段并不重合")}
<div class="box good">
<p><b>这是整篇论文的"物理解释"，也是最能说服审稿人的一页。</b></p>
<ul>
<li><b>谁在决定 PSNR</b>：B8A(22.5%)、B8(18.9%)、B7(17.7%) 三个波段贡献了 <b>59%</b> 的 MSE。
原因是高反射率波段的绝对误差天然更大，而 PSNR 是对全波段 MSE 取平均——<b>它被亮波段绑架了</b>。</li>
<li><b>谁在决定 ΔLAI</b>：B11 的敏感度是 <b>76.6</b>（LAI 每单位反射率），是 B8(18.8) 的
<b>4 倍</b>，但 B11 只占 MSE 的 9.3%。</li>
<li><b>两者的相关性 r = +0.06</b>（几乎为零），Top-3 波段只有一个重合（B7）。</li>
</ul>
<p>所以脱耦不是统计偶然，而是<b>结构性必然</b>：PSNR 主要在给 B8/B8A/B7 的亮度误差打分，
而 LAI 反演几乎由 B11（短波红外）与红边主导。用 PSNR 选方法，等于用一个"看错地方"的打分器。</p>
</div>

<h3>3.9 掩膜边界的鲁棒性</h3>
<p>前面所有结果都假设云掩膜是完美的。实际业务里掩膜来自云检测产品，边缘常有 1–2 像素误差。
我们把<b>评价区域</b>换成腐蚀 1px（−13.5% 像素）/ 原始 / 膨胀 1px（+10.9% 像素）三种，
方法本身的输出不变（变量只有一个，解释干净）：</p>
{(table(ms.pivot_table(index=["scope","metric"], columns="variant", values="rho").reset_index().round(4))) if ms is not None else ""}
{img(fig / "fig12_mask_robust.png", "图12 掩膜边界偏差下各指标判别力的变化")}
<div class="box good">
<p><b>结论对掩膜边界完全不敏感。</b>三种变体下：全部方法层面 PSNR 在
−0.936 ~ −0.943、SAM 在 0.925 ~ 0.943；高质量子集里 SAM 恒为 0.933，
PSNR 只从 −0.867 变到 −0.883。<b>变体间的波动（≤0.02）远小于 SAM 与 PSNR 之间的差距（0.07–0.10）。</b></p>
<p>这条很重要：它说明我们讨论的不是"某个特定掩膜下的偶然现象"，
而是即使掩膜产品有 1 像素量级的边缘误差，结论依然成立。
审稿人问"实际产品的掩膜不完美怎么办"，这就是答案。</p>
<p class="sub">边界：只测了 ±1 像素（对应约 −14%/+11% 的面积变化）。
若掩膜误差到 3–5 像素（薄云边缘常如此），需要重新评估；本实验不覆盖这种情形。</p>
</div>

<h3>3.10 剔除弱基线：结论是不是被差方法抬起来的</h3>
<p>主表里有两个明显偏弱的方法（<code>zero_fill</code> 15.5 dB、<code>diffusion</code> 18.4 dB）。
审稿人会怀疑"PSNR 相关高"是被它们抬起来的。直接剔除后重算：</p>
{(table(rdw.pivot_table(index="metric", columns="set", values=["rho_method", "rho_sample", "rho_highq"]).round(3).reset_index())) if rdw is not None else ""}
<div class="box good">
<p><b>高质量子集的相关性一个数都没变</b>（SAM 仍 0.933、PSNR 仍 −0.867）——
因为弱基线本来就进不了 PSNR≥25 dB 的子集。只有"方法层"受影响（PSNR 从 −0.921 降到 −0.874），
而方法层从来不是本文的核心论据。</p>
<p>结论：<b>本文的核心证据不依赖任何弱基线。</b>这条能直接挡掉"你用稻草人抬高相关性"的质疑。</p>
</div>

<h3>3.11 统计力度：必须如实说明的一处弱点</h3>
<p>高质量子集只有 9 个方法。我们在方法层做配对 bootstrap（2000 次），检验
"|SAM| 是否真的高于其他指标"：</p>
{(table(pbd)) if pbd is not None else ""}
<div class="box warn">
<p><b>没有一个差值达到显著。</b>|SAM|−|PSNR| 的中位差是 0.073，95% 区间
[0.000, 0.469]，p(差&gt;0) 只有 0.68——远没到 0.95。与 ERGAS 的差甚至中位为 0。</p>
<p>所以本文<b>不能</b>写成"SAM 优于 PSNR"。能站住的表述只有两条：</p>
<ul>
<li><b>PSNR 的判别力在高量子集上系统性下降</b>（0.92 → 0.81），且这个下降在三种掩膜变体下都稳定；</li>
<li><b>真正的证据不在相关系数的大小上</b>，而在 3.6 的条件依赖（卷云下所有指标减半）
与 3.8 的机制（波段错位 r=+0.06）——这两条不依赖方法数。</li>
</ul>
<p class="sub">这条限制如果藏着不写，审稿人自己算一遍就会抓住；主动写在正文里，
反而显得清楚自己证据的边界。n=9 是"方法覆盖"与"统计力度"之间的取舍，
若补到 15–20 个高质量方法，这个比较才可能有结论。</p>
</div>

<h3>3.12 换一个数据集：在完全独立的 Sentinel-2 L2A 上复现</h3>
<p>前面所有结论都来自 SEN12MS-CR（欧洲两景、L1C TOA、10m）。
为了检验它们不是这批数据的产物，我们在<b>完全独立</b>的一套数据上重跑了同一套流程：</p>
<ul>
<li><b>来源</b>：Copernicus Data Space 直接检索 + S3 取带（华北 4 个 tile，2022 年 6–9 月）</li>
<li><b>配对</b>：同 tile 内「有云一景 + 晴空一景」，日期差 1–4 天，共 8 对</li>
<li><b>处理</b>：10m 格网对齐 → 中心裁 1024 → 切 128×128，<b>云掩膜来自官方 SCL 分类波段</b></li>
<li><b>规模</b>：156 个 patch，云量中位 51%（比主数据集更厚）</li>
<li><b>限制</b>：只下 10 个可用波段，无 S1，因此只跑 9 个不含 SAR 的方法</li>
</ul>
{(table(xs.pivot_table(index=["scope","metric"], columns="target", values="rho").round(3).reset_index())) if xs is not None else ""}
<div class="box good">
<p><b>全量方法层的结论在新数据上复现了，而且量级几乎一样：</b></p>
<table style="margin:8px 0">
<thead><tr><th>指标</th><th>主数据（高效子集）</th><th>新数据（全量）</th><th>判定</th></tr></thead>
<tbody>
<tr><td>PSNR</td><td>−0.809</td><td>−0.900</td><td>一致</td></tr>
<tr><td>SAM</td><td>+0.933</td><td>+0.933</td><td>一致</td></tr>
<tr><td>ERGAS</td><td>+0.833</td><td>+0.917</td><td>一致</td></tr>
<tr><td>SSIM</td><td>−0.800</td><td>−0.933</td><td>符号一致、量级有差</td></tr>
</tbody></table>
<p>符号全部一致、绝对值差异在 0.09 以内。这是本文<b>最有力的一条外部效度证据</b>：
结论不是某一批欧洲影像的偶然。</p>
</div>
<div class="box warn">
<p><b>两条必须写进 limitation 的事实：</b></p>
<ol>
<li><b>高质量子集没能在新数据上构建</b>——新数据上<u>没有</u>任何方法达到 25 dB
（最高 23.4 dB）。因此"SAM 在好方法之间优于 PSNR"这一层，<b>本轮未获跨数据集验证</b>。
原因有两个：这套数据云量中位 51%（比主数据集厚得多），且 L2A 地表反射率比 TOA 更难拟合。</li>
<li><b>学习方法跨数据集严重退化</b>：<code>unet_opt</code> 19.2 dB、<code>dsen2cr_opt</code> 18.7 dB，
都<b>低于</b>最简单的 <code>blur_fill</code>（23.4 dB）。这不是方法本身失效，而是域偏移——
训练数据是 L1C TOA/10m，测试数据是 L2A 地表反射率/20m，输入分布已经变了。
<b>这恰好说明"自训模型不能跨数据集直接比较"，本身值得在讨论里写一段。</b></li>
</ol>
</div>

<h3>3.13 补足统计力度：样本层配对 bootstrap 与相对口径子集</h3>
<p>3.11 说明方法层（n=9）力度不够。补两条：(a) 在样本层（n={len(df)}）对 patch 重采样做
<b>配对</b> bootstrap（保持方法×patch 结构）；(b) 把高质量子集从绝对阈值改成<b>相对口径</b>
（各数据集取 PSNR 前 K 名），这样两个数据集才能比。</p>
{(table(sp)) if sp is not None else ""}
{(table(rel[["set","k","PSNR_vs_LAI","SAM_vs_LAI","SSIM_vs_LAI","ERGAS_vs_LAI"]])) if rel is not None else ""}
<div class="box">
<p><b>三条读数，一条比一条确定：</b></p>
<ul>
<li><b>SAM 并不优于 PSNR，反而略差</b>：样本层差值 −0.026，区间 [−0.056, −0.002]，p(差&gt;0)=0.02。
这与方法层的方向（SAM 更高）<b>相反</b>。两层结论不一致，说明"哪个指标更好"这个问题
<b>没有跨口径稳定的答案</b>——这比原来"SAM 更好"的说法更接近真相。</li>
<li><b>SSIM 被决定性否定</b>：SAM 对 SSIM 的优势 +0.169，区间 [+0.125, +0.216]，p=1.000。
ERGAS 同样差（+0.171，p=1.000）。<b>这是全文唯一一个"结论级"的统计结果。</b></li>
<li><b>相对口径下两个数据集完全一致</b>：Top4/5/6 三档里，SSIM 在两个数据集上都是最弱
（主 −0.40/−0.70/−0.77，新 −0.80/−0.90/−0.94），PSNR 与 SAM、ERGAS 同档。
改用相对口径后，3.12 里"高量子集无法跨数据集验证"的缺口<b>被补上了</b>。</li>
</ul>
<p class="sub">为什么改相对口径是合法的：我们要回答的是"在好方法之间，哪个指标还能区分"，
这本质上是一个<b>排序问题</b>，用"前 K 名"定义候选池比"PSNR≥25 dB"更贴近问题本身，
也避免了绝对阈值在不同数据分布上失效。</p>
</div>

<h2>四、把分析升级成方法：任务一致性加权分数 TCWS</h2>
<p>只报告"没有通用最优指标"仍然是负面结论，审稿人会问"那我该用什么"。所以我们拟合一个加权分数：
把各指标按方向统一后，用<b>非负最小二乘</b>拟合"哪种组合最能预测下游误差"，
权重由数据决定，不是拍脑袋。既然 3.7 节已证明结论依赖下游参数，这里就<b>按任务各拟合一次</b>。</p>

<p><b>目标 = ΔLAI：</b></p>
{table(nm)}
{(("<p><b>目标 = ΔCab：</b></p>" + table(nm_cab))) if nm_cab is not None else ""}

<div class="box good">
<p><b>两个任务拟合出的权重几乎不重叠，这本身就是最有力的证据：</b></p>
<ul>
<li><b>对 LAI</b>：RMSE 0.25 + SAM 0.07 + PSNR 0.03，其余为 0 —— 像素误差为主、光谱为辅。</li>
<li><b>对 Cab</b>：ERGAS 5.08 + SAM 2.72 + SSIM 1.63，<b>PSNR 与 RMSE 权重直接归零</b>。</li>
</ul>
<p>同一份数据、同一批方法，PSNR 从"还有一点用"变成"完全不用"。
这个差别不是我们设定的，是拟合出来的——它直接坐实了"评价指标必须与下游任务绑定"。</p>
<p>两个任务上 TCWS 都优于最强单指标：LAI 上
−{abs(nm.loc[nm.score.str.startswith('TCWS'),'spearman_vs_dLAI'].iloc[0]):.3f} vs
−{abs(nm[nm.kind=='单一指标'].spearman_vs_dLAI).max():.3f}
（+{abs(nm.loc[nm.score.str.startswith('TCWS'),'spearman_vs_dLAI'].iloc[0])-abs(nm[nm.kind=='单一指标'].spearman_vs_dLAI).max():.3f}）；
Cab 上 −{abs(nm_cab.loc[nm_cab.score.str.startswith('TCWS'),'spearman_vs_dLAI'].iloc[0]):.3f} vs
−{abs(nm_cab[nm_cab.kind=='单一指标'].spearman_vs_dLAI).max():.3f}
（+{abs(nm_cab.loc[nm_cab.score.str.startswith('TCWS'),'spearman_vs_dLAI'].iloc[0])-abs(nm_cab[nm_cab.kind=='单一指标'].spearman_vs_dLAI).max():.3f}）。</p>
<p class="sub">诚实边界：提升只有 0.01–0.02，<b>TCWS 的价值不在"更准"，
而在于它把"该用哪几个指标"从经验判断变成了可复现、可审计的拟合结果</b>；
真正有信息量的是权重随任务变化这个事实。另外 LOMO 中位数（LAI −0.69 / Cab −0.57）
明显低于 holdout 值，说明权重对"没见过的去云方法"迁移能力有限，必须写进 limitation。</p>
</div>
{("<p>留一方法交叉验证（LOMO）明细，目标 = ΔLAI：</p>" + table(lomo)) if lomo is not None else ""}

<h2>五、方法层的其它发现</h2>
{img(fig / "fig1_method_scatter.png", "图1 各指标与 ΔLAI 的散点（每点一个方法）")}
<ul>
<li><b>学习方法确实更强</b>：<code>unet_sar</code> 在 PSNR（{psnr.max():.2f} dB）、
SAM（{sam.min():.2f}°）和 ΔLAI（{g['dLAI_mean'].min():.3f}）上同时最优，
是唯一三项都第一的方法。</li>
<li><b>SAR 有用但不多</b>：<code>unet_sar</code> 比 <code>unet_opt</code> 高
{psnr.max()-psnr.iloc[1]:.2f} dB、ΔLAI 低 {g['dLAI_mean'].iloc[1]-g['dLAI_mean'].min():.3f}，
说明在 SEN12MS-CR 这种"薄云为主"的数据上，光学自身信息仍占主导。</li>
<li><b>直方图匹配是典型的"指标好看、下游坏"</b>：SAM 排第
{int(sam.rank().loc['hist_match'])}（{sam.loc['hist_match']:.2f}°），
但 ΔLAI 排第 {int(r_dlai.loc['hist_match'])}（{g['dLAI_mean'].loc['hist_match']:.3f}）。
它保留了空间纹理却整体错位了辐射值——这正是 SAM 敏感、PSNR 不敏感的一类误差。</li>
<li><b>生成族（扩散）目前偏弱，需如实说明</b>：
<code>diffusion</code> 的 PSNR {psnr.loc['diffusion']:.2f} dB、SAM {sam.loc['diffusion']:.2f}°、
ΔLAI {g['dLAI_mean'].loc['diffusion']:.3f}，三项一致地差，没有出现
"指标好但下游坏"的反转。原因很明确：这个扩散只有 2.55M 参数、2 万步、DDIM 20 步采样，
远小于文献里动辄几百 M、十万级步的扩散去云。因此它只能说明
"小规模生成模型在本数据上不占优"，<b>不能</b>用来论证"生成式方法整体不行"。</li>
</ul>

{img(fig / "fig5_qualitative.png", "图5 定性示例：有云影像、云掩膜与三种方法的修复结果")}

<h2>五之二、种子稳定性</h2>
<p>6 个学习基线各训练 3 次（DSen2-CR 的 sar 配置 2 次），在同一测试集、同一掩膜、同一反演器下重算：</p>
{(table(seed)) if seed is not None else "<p>（未生成种子稳定性表）</p>"}
<div class="box warn">
<p><b>一个此前没注意到的结果：PSNR 对训练随机性的敏感度，在不同方法之间相差近 300 倍。</b>
UnCRtainTS 的标准差只有 0.01 dB，条件扩散模型高达 1.72 dB（峰峰 3.2 dB），其余四个在 0.04–0.49 dB。</p>
<p>而同一批权重在<b>下游</b>的离散度小得多：扩散模型的 ΔLAI 只在 1.212–1.218 之间（0.5%）。
换句话说，<b>换一个训练种子，PSNR 可以移动 3 dB，而下游排序几乎不动</b>——
图像指标是一个比下游指标更"噪声化"的仪器。这一条独立支持了本文的主张，而且是从方差角度而非相关性角度得到的。</p>
<p>它同时界定了主表排序的可解读范围：方法间 5–13 dB 的差距远大于任何种子噪声，故单一种子不影响端到端结论；
但前四名之间不足 0.4 dB 的差距中，<b>只有 UnCRtainTS（±0.01 dB）达到可分辨</b>，
U-Net 与 DSen2-CR 三个配置的先后落在噪声内，不应被解读。</p>
</div>

<h2>五之三、失败形态：同一个 PSNR，不同的错法</h2>
<p>前面说 PSNR 的好坏取决于候选池，但没回答一个更基本的问题：<b>PSNR 相同时，失败方式一样吗？</b></p>
<p>做法不是拍阈值分类，而是对 MSE 做<b>精确分解</b>。对每个方法、每个波段，在云掩膜像素上把预测 <i>p</i>
对真值 <i>t</i> 做最小二乘 <i>p ≈ μ<sub>p</sub> + a(t − μ<sub>t</sub>)</i>，由于残差与 <i>t</i> 正交且均值为零，
交叉项消失，于是</p>
<p style="text-align:center"><i>MSE = (μ<sub>p</sub> − μ<sub>t</sub>)² + (1 − a)²·Var(t) + Var(e)</i></p>
<p>三项分别是<b>整体偏移</b>、<b>对比度压缩</b>与<b>结构残差</b>，加起来恰好等于 PSNR 用的那个 MSE
（合成数据上验证，误差约 1e-17）。所以这是对 PSNR 的分解，不是另立指标。</p>
{(table(fsum[["method","PSNR","bias_share","contrast_share","resid_share","dLAI_mean"]]
        .rename(columns={"method":"方法","PSNR":"PSNR/dB","bias_share":"整体偏移",
                         "contrast_share":"对比度压缩","resid_share":"结构残差","dLAI_mean":"ΔLAI"})
        .round(3))) if fsum is not None else ""}
{img(fig / "fig13_failure_modes.png", "图13 左：15 个方法的误差形态构成；右：三个分量与 ΔLAI 的判别力")}
<div class="box">
<p><b>同一 PSNR 下，失败方式可以完全不同。</b>在 26.4–26.8 dB 这个狭窄区间里，均值填充、SAR 回归、
SAR 引导、模糊填充的结构残差占比分别是 0%、4%、5%、7%——<b>它们根本不生成结构</b>，
误差全部来自偏移与对比度；而 22.9 dB 的直方图匹配结构残差高达 53%。学习方法的形态又不同：
误差由对比度主导（53–68%），结构残差占 17–31%。<b>PSNR 把这三种本质不同的失败压成了一个数。</b></p>
<p>但把三项分别与下游对齐后，出现一个反直觉、必须如实报告的结果：</p>
{(table(fcmp)) if fcmp is not None else ""}
<p><b>判别力几乎全部由「整体偏移」承担</b>（0.94），对比度 0.60，结构残差只有 0.15。
在合格方法内部这一关系甚至反号（残差项与 ΔLAI 为 −0.77）：完全不生成结构的平滑方法残差最小，反演误差却最大。</p>
<p>这说明<b>残差项的绝对大小不是优化目标</b>，它只标记了"模型是否在生成地物"这个二分性质。
对本文主张的意义是：PSNR 能筛掉差方法，正因为它主要度量偏移（差方法的误差几乎全是偏移）；
而在好方法之间，真正区分它们的是对比度与结构——这两项与下游的关联弱得多。
<b>这解释了为什么 PSNR 在筛差上有效、在选优上失效，而不是仅仅报告这个现象。</b></p>
</div>
{img(fig / "fig9_failure.png", "图9 失败案例：最好的方法在最难的 patch 上失败在哪里")}

<h2>六、数据与实验设置</h2>
<ul>
<li><b>数据</b>：SEN12MS-CR（HuggingFace 镜像 <code>Hermanni/sen12mscr</code>），
test split 两个分片（春 784 / 冬 784 patch），L1C TOA 13 波段，int16，/10000 转反射率。</li>
<li><b>缓存</b>：共 {len(meta)} 个 patch，云量过滤 [2%, 95%] 后保留
{sum(len(v) for v in split.values())} 个 → train {len(split['train'])} /
val {len(split['val'])} / test {len(split['test'])}（固定 seed，split.json 不变）。</li>
<li><b>云掩膜</b>：可见光 max(B2,B3,B4) &gt; 0.15，阈值在 30 个 patch 上标定，
精度 1.000 / 召回 0.897（选精度=1 的一侧，避免基线改动晴空像元）。</li>
<li><b>下游反演器</b>：PROSAIL 查找表 30 万条 + KD 树 10 近邻加权，<b>零训练、确定性</b>。
参数空间固定叶片结构与观测几何，只让 LAI/Cab/叶倾角/土壤亮度变化，
使 LAI 可辨识性从"邻居 LAI 标准差 1.55"改善到 0.25。</li>
<li><b>指标波段</b>：排除 B1/B9/B10。B10（卷云波段）均值仅 0.0008，
会把 ERGAS 分母放大到 1e4 量级，指标彻底失效。</li>
<li><b>分辨率</b>：256→128 用<b>中心裁剪</b>而非缩放——缩放等于给所有方法加一层低通，
会系统性抬高 PSNR 并抹平方法差异。</li>
</ul>

<h2>七、下一步（对应 12 周计划的 W4–W7）</h2>
<ul>
<li><b>加强生成式方法</b>：当前扩散只有 2.55M/2 万步，指标明显偏弱。
下一步应扩到 base 64、5 万步以上（本地 4GB 约需 3–5 小时，或租 4090 约 1 小时），
让生成族达到与其他族可比的水平，再检验"云下编造"现象
（多模型分歧、循环一致性残差、多次采样方差）。<b>这是当前实验最大的缺口。</b></li>
<li><b>跨数据集验证</b>：把同一套分析搬到 AllClear 或 WHU-OPT-SAR，确认结论不是本数据集的特性。</li>
<li><b>跨季节验证</b>：当前已含春/冬两季，可再加夏季分片，检验"季节"这一分层维度。</li>
<li><b>跨分辨率验证</b>：用 tiling 把 256 切成 4 块 128 跑 50 张图，回应"128 分辨率是否有偏"的质疑。</li>
<li><b>方法数扩充</b>：高质量子集只有 6 个方法，n 偏小（PSNR 的 p=0.21 有部分原因是样本量）。
若要投稿 TGRS/JAG，建议补到 8–10 个高质量方法，让"PSNR 不显著"这个结论更硬。</li>
</ul>

<div class="foot">
项目目录：<code>D:\\mazhaoyang\\code program\\cloudrs1</code><br>
一键复现：<code>conda activate cloudrs &amp;&amp; bash scripts/run_all.sh</code><br>
表：results/tables/ ｜ 图：results/figures/ ｜ 日志：results/logs/
</div>

</div></body></html>"""

    out = WS / "P1实验结果报告.html"
    # 报告常被预览面板/浏览器占用，直接覆盖会 PermissionError（夜里就栽在这）。
    # 策略：先写临时文件，再原子替换；替换失败就换个文件名，至少不丢结果。
    tmp = out.with_suffix(".tmp.html")
    tmp.write_text(html, encoding="utf-8")
    try:
        import os
        for attempt in range(3):
            try:
                os.replace(tmp, out)
                break
            except PermissionError:
                if attempt == 2:
                    raise
                import time
                time.sleep(2)
    except PermissionError:
        alt = WS / "P1实验结果报告_最新.html"
        import shutil
        shutil.copyfile(tmp, alt)
        tmp.unlink(missing_ok=True)
        print(f"原文件被占用（可能在预览中），已改写到 -> {alt}  ({alt.stat().st_size/1024:.0f} KB)")
        return
    print(f"报告已生成 -> {out}  ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
