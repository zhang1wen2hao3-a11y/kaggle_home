#!/usr/bin/env python3
"""**稳一点的 Kaggle 提交**：绕开 Google 上传域名被墙/黑洞的问题。

2026-11-18 现场：`kaggle competitions submit` 卡在 0%（20.5 MB 传 10 分钟一个字节没动），
`ss -tnp` 显示进程卡在 `SYN-SENT … 172.217.116.4:443`——上传域名解析到了不通的 Google IP，
而同一域名下另一些 IP（142.250.x / 108.177.x）是通的。sudo 不可用 ⇒ 不能改 /etc/hosts。

做法：先探测 `storage.googleapis.com` 的若干 A 记录，挑一个 3 秒内能建连的 IP，
再把 `socket.getaddrinfo` 打补丁，让 requests 只解析到这个 IP（TLS 的 SNI/Host 仍是原域名，
签名 URL 校验不受影响），然后走官方 SDK 的 `competition_submit`。

    python3 tools/kaggle_submit.py dist/ptcg_sub_p9d6e.tar.gz "p9d6e anchor"
"""
from __future__ import annotations
import os
import socket
import sys
import time

COMP = "the-pokemon-company-ptcg-ai-battle-challenge-playground"
UPLOAD_HOSTS = ("storage.googleapis.com", "www.googleapis.com", "storage.cloud.google.com")


def _reachable(ip: str, port: int = 443, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False


def _good_ips(host: str):
    ips = []
    try:
        for fam, _, _, _, sa in socket.getaddrinfo(host, 443, socket.AF_INET):
            if sa[0] not in ips:
                ips.append(sa[0])
    except OSError:
        return []
    return ips


def main() -> int:
    path = sys.argv[1]
    msg = sys.argv[2] if len(sys.argv) > 2 else ""

    pin = {}
    # 候选 IP 池：上传域名自身解析 + 已知可用的 Google 前端 IP。
    # TLS 由 SNI 决定，把 www.googleapis.com 钉到别的 Google 前端 IP 是安全的
    # （curl --resolve 实测回 HTTP 400，说明 TLS 与 Host 校验都通过）。
    FALLBACK = ["142.250.69.187", "108.177.98.207", "142.251.34.219"]
    pool = []
    for host in UPLOAD_HOSTS:
        for ip in _good_ips(host):
            if ip not in pool:
                pool.append(ip)
    for ip in FALLBACK:
        if ip not in pool:
            pool.append(ip)

    def first_ok(ips):
        for ip in ips:
            if _reachable(ip):
                return ip
        return None

    for host in UPLOAD_HOSTS:
        own = first_ok(_good_ips(host))
        if own:
            print(f"  pin {host} → {own}", flush=True)
            pin[host] = own
            continue
        shared = first_ok(pool)  # 本域名 IP 全被黑洞 ⇒ 借一个可达的 Google 前端 IP
        if shared:
            print(f"  pin {host} → {shared}（原解析全不可达，借用）", flush=True)
            pin[host] = shared
        else:
            print(f"  ⚠️ {host} 无可用 IP", flush=True)

    orig = socket.getaddrinfo

    def patched(host, port, *a, **kw):  # noqa: ANN001
        if host in pin:
            return orig(pin[host], port, *a, **kw)
        if os.environ.get("KAGGLE_SUBMIT_TRACE"):
            print(f"  resolve {host}:{port}", flush=True)
        return orig(host, port, *a, **kw)

    socket.getaddrinfo = patched

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    t0 = time.time()
    resp = api.competition_submit(file_name=path, message=msg, competition=COMP, quiet=False)
    print(f"  elapsed {time.time()-t0:.0f}s", flush=True)
    print(json_safe(resp), flush=True)
    return 0


def json_safe(resp):
    try:
        import json

        return json.dumps(resp.to_dict(), ensure_ascii=False, default=str)[:600]
    except Exception:
        return str(resp)[:600]


if __name__ == "__main__":
    raise SystemExit(main())
