"""从 pCloud 公开链接下载 UnCRtainTS 权重（monotemporalL2）"""
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

CODE = "kZsdbk0Z5Y2Y2UEm48XLwOvwSVlL8R2L3daV"
API = "https://api.pcloud.com"
OUT = Path("third_party/weights/uncrtaints")
UA = {"User-Agent": "Mozilla/5.0"}


def api(path, **params):
    params["code"] = CODE
    url = f"{API}/{path}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r)


def main():
    root = api("showpublink")["metadata"]
    targets = {}
    for sub in root.get("contents", []):
        if sub.get("name") != "monotemporalL2":
            continue
        for f in sub.get("contents", []):
            targets[f["name"]] = f["fileid"]
    if not targets:
        raise SystemExit("没找到 monotemporalL2")

    OUT.mkdir(parents=True, exist_ok=True)
    for name, fid in targets.items():
        dest = OUT / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  已存在，跳过 {name}")
            continue
        d = api("getpublinkdownload", fileid=fid)
        if d.get("result") != 0:
            print(f"  取下载地址失败 {name}: {d.get('error')}")
            continue
        # pCloud 返回 hosts + path，拼出真实 URL
        hosts = d.get("hosts") or []
        path = d.get("path")
        ok = False
        for h in hosts:
            url = f"https://{h}{path}"
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
                    data = r.read()
                dest.write_bytes(data)
                print(f"  下载完成 {name}  {len(data)/1e6:.2f} MB  <- {h}")
                ok = True
                break
            except Exception as e:
                print(f"    {h} 失败: {type(e).__name__}")
        if not ok:
            print(f"  !! {name} 全部 host 失败")

    print("\n目录内容：")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name:20s} {p.stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
