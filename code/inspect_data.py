"""
inspect_data.py —— 探查 SEN12MS-CR parquet 分片的结构与数值范围

用法：
    python src/inspect_data.py data/raw/test/scene_31.parquet

目的：写 Dataset 之前先搞清楚三件事
  1. 有哪些字段、每个字段的 shape 与 dtype
  2. 数值范围（判断是 DN 值、TOA 反射率还是地表反射率，是否需要除以 10000）
  3. 云量分布（判断能否做"按云厚度分层"分析）
"""
import sys
import numpy as np
import pyarrow.parquet as pq


def decode(col_name, buf, shape):
    """把二进制 buffer 还原成 ndarray，按列名猜 dtype。"""
    dt = np.float32 if col_name.startswith("sar") else np.int16
    arr = np.frombuffer(buf, dtype=dt)
    if shape is not None:
        arr = arr.reshape(tuple(int(s) for s in shape))
    return arr


def main(path):
    print("=" * 70)
    print(f"文件：{path}")
    print("=" * 70)

    pf = pq.ParquetFile(path)
    md = pf.metadata
    print(f"行数 {md.num_rows} | 列数 {md.num_columns} | "
          f"row groups {md.num_row_groups}\n")

    print("列结构：")
    for i, name in enumerate(pf.schema_arrow.names):
        print(f"  {i:2d}  {name:<18} {pf.schema_arrow.field(i).type}")
    print()

    tbl = pf.read_row_group(0)
    df = tbl.to_pandas()

    bin_cols = [c for c in df.columns
                if df[c].dtype == object and isinstance(df[c].iloc[0], (bytes, bytearray))]

    print(f"二进制列（共 {len(bin_cols)} 个）：{bin_cols}\n")

    for c in bin_cols:
        shape_col = f"{c}_shape"
        shape = None
        if shape_col in df.columns:
            v = df[shape_col].iloc[0]
            shape = v if isinstance(v, (list, tuple, np.ndarray)) else None
        try:
            a = decode(c, df[c].iloc[0], shape)
        except Exception as e:
            print(f"  {c}: 解码失败 {type(e).__name__} {e}")
            continue
        print(f"  {c:<12} shape={a.shape} dtype={a.dtype}")
        print(f"  {'':12} min={a.min():.4f}  max={a.max():.4f}  mean={a.mean():.4f}  "
              f"非零占比={np.mean(a != 0):.3f}")

    print("\n其他元数据列（首行取值）：")
    for c in df.columns:
        if c not in bin_cols and not c.endswith("_shape"):
            print(f"  {c:<18} = {df[c].iloc[0]!r}")

    print("\n" + "=" * 70)
    print("判读提示：")
    print("  · cloudy/target 的 max 在 10000 附近 → 放大 10000 倍的反射率，需除以 10000")
    print("  · max 在 1 附近 → 已是 [0,1] 反射率")
    print("  · sar 为 dB 或线性功率皆可，但必须统一归一化")
    print("=" * 70)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/test/scene_31.parquet")
