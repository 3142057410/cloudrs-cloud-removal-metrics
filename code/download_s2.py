"""
download_s2.py —— 从 Copernicus Data Space 的 S3 接口只取需要的波段（不下载整个 SAFE）

为什么走 S3 而不是 OData：
  一个 S2 L2A 的 SAFE 约 1.1 GB，8 对景（16 景）就是 16.7 GB。
  但我们只需要 10 个波段 + 1 个 SCL，S3 可以按文件单独取，总量约 2 GB。
  且 S3 支持断点、并发，比 OData 的整包下载稳。

前置（只需做一次）：
  1. 注册 https://dataspace.copernicus.eu
  2. 登录后进 Dashboard → Sentinel Hub / S3 credentials → 生成 Access Key + Secret Key
  3. 把两个值写进项目根目录的 .cdse_s3（两行，第一行 key，第二行 secret），
     或设环境变量 CDSE_S3_KEY / CDSE_S3_SECRET

用法：
  python src/download_s2.py --pairs results_meta/s2_pairs.csv --out data/s2_raw
  python src/download_s2.py --pairs results_meta/s2_pairs.csv --bands B4,B5,B8,B11,SCL   # 只取几个

产物：
  data/s2_raw/<product_name>/<band>.jp2      只含需要的波段
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

# Sentinel-2 L2A 各波段的原生分辨率（米）——决定 S3 路径里的 R10m/R20m/R60m
BAND_RES = {
    "B2": 10, "B3": 10, "B4": 10, "B8": 10,
    "B5": 20, "B6": 20, "B7": 20, "B8A": 20, "B11": 20, "B12": 20,
    "SCL": 20, "AOT": 10, "WVP": 10, "TCI": 10,
}
DEFAULT_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "SCL"]
ENDPOINT = "https://eodata.dataspace.copernicus.eu"
BUCKET = "eodata"


def read_creds():
    key = os.environ.get("CDSE_S3_KEY")
    sec = os.environ.get("CDSE_S3_SECRET")
    if key and sec:
        return key, sec
    f = Path(".cdse_s3")
    if f.exists():
        lines = [l.strip() for l in f.read_text().splitlines() if l.strip()]
        if len(lines) >= 2:
            return lines[0], lines[1]
    raise SystemExit(
        "缺少 S3 凭据。二选一：\n"
        "  · 项目根目录建 .cdse_s3，两行分别写 Access Key 和 Secret Key\n"
        "  · 或设置环境变量 CDSE_S3_KEY / CDSE_S3_SECRET"
    )


def product_prefix(product_name: str) -> str:
    """
    产品在 S3 上的根前缀。

    坑 1：根目录取决于处理基线（产品名里的 N 段）。
          N0400 及更老 → Sentinel-2/MSI/L2A/
          N0500 及更新 → Sentinel-2/MSI/L2A_N0500/
          实测：N0510 的产品在 L2A/ 下一条都查不到，只存在于 L2A_N0500/。
    坑 2：GRANULE 目录名含绝对轨道号（如 A028129），无法从产品名推导，需运行时列目录。
    """
    p = product_name.replace(".SAFE", "")
    parts = p.split("_")
    datatake, baseline = parts[2], parts[3]
    root = "L2A_N0500" if int(baseline[1:]) >= 500 else "L2A"
    return (f"Sentinel-2/MSI/{root}/{datatake[:4]}/{datatake[4:6]}/{datatake[6:8]}/{p}.SAFE")


def find_granule(s3, product_name: str) -> str:
    """
    定位 GRANULE 目录名。它长这样：L2A_T50SLJ_A036246_20220726T030529
    其中的 A036246 是**绝对轨道号**，无法从产品名推导，只能列 S3 目录取。
    这是官方文档没写清、但踩了就必然 404 的一个坑。
    """
    prefix = product_prefix(product_name) + "/GRANULE/"
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, Delimiter="/")
    cps = resp.get("CommonPrefixes", [])
    if not cps:
        raise FileNotFoundError(f"在 S3 上找不到 GRANULE：{prefix}")
    return cps[0]["Prefix"].rstrip("/").split("/")[-1]


def list_band_index(s3, product_name: str, granule: str, res_list=(10, 20, 60)) -> dict:
    """
    列出该产品 IMG_DATA 下所有 jp2，建成 {'B04':key, 'SCL':key, ...} 索引。

    为什么不直接拼文件名：不同产品（S2A/S2B、不同基线、不同分片）的文件前缀里
    tile 与 datatake 时间可能与产品名不完全一致，手拼会 404。
    实测 T50TLK 那一景就是拼出来的名字对不上。列目录一次（约 30 个 key），
    后面按后缀匹配，稳且不慢。
    """
    base = f"{product_prefix(product_name)}/GRANULE/{granule}/IMG_DATA/"
    idx = {}
    order = {10: 0, 20: 1, 60: 2}
    hits = {}
    for res in res_list:
        pre = f"{base}R{res}m/"
        token = None
        while True:
            kw = {"Bucket": BUCKET, "Prefix": pre, "MaxKeys": 100}
            if token:
                kw["ContinuationToken"] = token
            r = s3.list_objects_v2(**kw)
            for o in r.get("Contents", []):
                fn = o["Key"].split("/")[-1]
                if not fn.endswith(".jp2"):
                    continue
                stem = fn[:-4].split("_")
                for tok in stem:
                    if tok in ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
                               "B8A", "B09", "B10", "B11", "B12", "SCL", "AOT",
                               "WVP", "TCI", "MSK_CLDPRB", "MSK_SNWPRB"):
                        hits.setdefault(tok, []).append((order[res], o["Key"]))
                        break
            if r.get("IsTruncated"):
                token = r.get("NextContinuationToken")
            else:
                break
    # 同一波段可能在多个分辨率目录下都有（如 R20m 里也有 B04 的重采样版），
    # 取可用的最低分辨率，文件更小。
    for b, lst in hits.items():
        idx[b] = sorted(lst)[0][1]
    return idx


def download_one(s3, product_name, bands, out_root, verbose=True):
    """把一个产品需要的波段文件从 S3 取到本地目录。已存在则跳过。"""
    p = product_name.replace(".SAFE", "")
    parts = p.split("_")
    tile, datatake = parts[5], parts[2]
    dest_dir = Path(out_root) / p
    dest_dir.mkdir(parents=True, exist_ok=True)
    # 若该产品已完整（需要的波段都在且有大小），直接跳过
    if all((dest_dir / f"{b}.jp2").exists() and (dest_dir / f"{b}.jp2").stat().st_size > 0
           for b in bands):
        if verbose:
            print(f"  {tile}_{datatake}  已存在，跳过", flush=True)
        return len(bands), []

    try:
        granule = find_granule(s3, product_name)
        idx = list_band_index(s3, product_name, granule)
    except Exception as e:
        if verbose:
            print(f"  {tile}_{datatake}  跳过：{type(e).__name__} {e}", flush=True)
        return 0, list(bands)

    ok, miss = 0, []
    for b in bands:
        # 波段名对齐：我们习惯写 B4/B8，S2 文件里是 B04/B08（两位）。B8A/B11/B12 本来就是三位。
        want = b if len(b) >= 3 else b[0] + b[1:].zfill(2)
        key = idx.get(want)
        out = dest_dir / f"{b}.jp2"
        if out.exists() and out.stat().st_size > 0:
            ok += 1
            continue
        if not key:
            miss.append(f"{b}(目录里没有)")
            continue
        try:
            s3.download_file(BUCKET, key, str(out))
            ok += 1
        except Exception as e:
            miss.append(f"{b}({type(e).__name__})")
            if out.exists():
                out.unlink(missing_ok=True)
    if verbose:
        flag = "" if not miss else f"  缺失 {miss}"
        print(f"  {tile}_{datatake}  {ok}/{len(bands)} 个波段{flag}", flush=True)
    return ok, miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="results_meta/s2_pairs.csv")
    ap.add_argument("--out", default="data/s2_raw")
    ap.add_argument("--bands", default=",".join(DEFAULT_BANDS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="先试前 N 景")
    args = ap.parse_args()

    import boto3
    from botocore.config import Config

    key, sec = read_creds()
    s3 = boto3.client(
        "s3", endpoint_url=ENDPOINT,
        aws_access_key_id=key, aws_secret_access_key=sec,
        config=Config(signature_version="s3v4", retries={"max_attempts": 5}),
    )

    pairs = pd.read_csv(args.pairs)
    names = sorted(set(pairs.cloudy_name) | set(pairs.clear_name))
    if args.limit:
        names = names[: args.limit]
    bands = [b.strip() for b in args.bands.split(",")]

    print(f"待下载 {len(names)} 景 × {len(bands)} 波段 -> {args.out}")
    print(f"波段：{bands}")
    total_ok, total_miss = 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(download_one, s3, n, bands, args.out) for n in names]
        for f in futs:
            try:
                ok, miss = f.result()
                total_ok += ok
                total_miss += miss
            except Exception as e:
                print("  异常：", type(e).__name__, e)
    print(f"\n完成：成功 {total_ok} 个波段文件，缺失 {len(total_miss)}")
    if total_miss:
        print("  缺失示例：", total_miss[:10])
        print("  若全部 403：检查 Access Key 是否属于 Sentinel Hub S3，以及账号是否已激活 S3 访问。")


if __name__ == "__main__":
    main()
