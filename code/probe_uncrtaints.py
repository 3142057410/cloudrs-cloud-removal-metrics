"""probe: 在现有 torch 1.12 环境里构建 UNCRTAINTS 并加载权重，跑一次前向"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path("third_party/uncrtaints")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from model.src.backbones import uncrtaints as U  # noqa: E402

conf = json.load(open("third_party/weights/uncrtaints/conf.json", encoding="utf-8"))
# conf.json 缺 block_type（训练时用 parse_args 的默认值），补上默认 'mbconv'
conf.setdefault("block_type", "mbconv")
conf.setdefault("var_nonLinearity", "softplus")
cfg = SimpleNamespace(**conf)

S1_BANDS, S2_BANDS = 2, 13
input_dim = S1_BANDS * int(cfg.use_sar) + S2_BANDS
print(f"torch {torch.__version__} | use_sar={cfg.use_sar} | input_t={cfg.input_t} | "
      f"is_mono={cfg.pretrain} | input_dim={input_dim}")

model = U.UNCRTAINTS(
    input_dim=input_dim,
    encoder_widths=cfg.encoder_widths,
    decoder_widths=cfg.decoder_widths,
    out_conv=cfg.out_conv,
    out_nonlin_mean=cfg.mean_nonLinearity,
    out_nonlin_var=cfg.var_nonLinearity,
    agg_mode=cfg.agg_mode,
    encoder_norm=cfg.encoder_norm,
    decoder_norm=cfg.decoder_norm,
    n_head=cfg.n_head,
    d_model=cfg.d_model,
    d_k=cfg.d_k,
    pad_value=cfg.pad_value,
    padding_mode=cfg.padding_mode,
    positional_encoding=cfg.positional_encoding,
    covmode=cfg.covmode,
    scale_by=cfg.scale_by,
    separate_out=cfg.separate_out,
    use_v=cfg.use_v,
    block_type=cfg.block_type,
    is_mono=cfg.pretrain,
)
n = sum(p.numel() for p in model.parameters()) / 1e6
print(f"模型构建成功，参数 {n:.2f}M")

ck = torch.load("third_party/weights/uncrtaints/model.pth.tar", map_location="cpu")
# 权重文件结构：{'state_dict', 'state_dict_G', 'optimizer_G', ...}
# 两处命名差异都要处理：
#   ① 键名带 'netG.' 前缀
#   ② 权重里是 in_block1/out_block1..5（1 起编号），当前代码是 in_block.0/out_block.0..（0 起，ModuleList）
import re as _re


def _remap(k):
    k = k.replace("netG.", "").replace("module.", "")
    k = _re.sub(r"\bin_block(\d+)", lambda m: f"in_block.{int(m.group(1)) - 1}", k)
    k = _re.sub(r"\bout_block(\d+)", lambda m: f"out_block.{int(m.group(1)) - 1}", k)
    return k


sd = ck.get("state_dict_G") or ck.get("state_dict") or ck
sd = {_remap(k): v for k, v in sd.items()}
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"权重加载：缺失 {len(missing)} 项，多余 {len(unexpected)} 项")
if missing[:3]:
    print("  缺失示例:", missing[:3])
if unexpected[:3]:
    print("  多余示例:", unexpected[:3])

model.eval()
for hw in (128, 256):
    x = torch.rand(1, 1, input_dim, hw, hw)
    try:
        with torch.no_grad():
            y = model(x)
        out = y[0] if isinstance(y, (tuple, list)) else y
        print(f"  前向 {hw}×{hw}: 输出 {tuple(out.shape)} 范围 [{out.min():.2f}, {out.max():.2f}]")
    except Exception as e:
        print(f"  前向 {hw}×{hw} 失败: {type(e).__name__} {str(e)[:160]}")
