#!/usr/bin/env python3
"""**按队伍抓天梯回放**：排行榜 → teamId → 该队最近的 submission → 该 submission 的 episode → 下载回放。

为什么需要：`fetch_episodes.py` 只能按 **submission ref** 抓；而我们想知道
"1300+/1500+ 的高手到底在打什么牌"时手上只有**队名**。这条链路（用户 2026-11-18 演示）是：

    kaggle competitions leaderboard <slug> -d            # 榜单（CSV，含 teamId/teamName/score）
    kaggle competitions team-submissions <teamId>        # 该队的提交（id/日期/publicScore）
    kaggle competitions episodes <subId>                 # 该提交的对局（跳过 VALIDATION）
    kaggle competitions replay <episodeId>               # 下载单局回放（约 10 MB）

    ./tools/py tools/fetch_team_replays.py --teams "knshnb,gurume" --per-team 20
    ./tools/py tools/fetch_team_replays.py --top 8 --per-team 15 --out epsoide_top18
"""
from __future__ import annotations
import argparse, csv, io, os, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KAGGLE = "/home/zhang/miniforge3/bin/kaggle"
SLUG = "the-pokemon-company-ptcg-ai-battle-challenge-playground"


def run(args, timeout=180):
    try:
        r = subprocess.run([KAGGLE] + args, capture_output=True, text=True, timeout=timeout, cwd=ROOT)
        return r.stdout
    except Exception as e:
        print(f"  ! kaggle {' '.join(args[:2])} 失败: {e}")
        return ""


def leaderboard():
    """榜单：`kaggle competitions leaderboard <slug> -d` 下的是 **zip**（文件名 = slug.zip，
    里面一张 `<slug>-publicleaderboard-<时间>.csv`，列 Rank,TeamId,TeamName,LastSubmissionDate,Score…）。
    ⚠️ 文件名里**没有** "leaderboard" 字样，早先按关键字找 zip 会找不到（已修）。
    """
    import zipfile, time as _t
    tmp = os.path.join(ROOT, "data", "leaderboard")
    os.makedirs(tmp, exist_ok=True)
    # 先用已解出的 CSV（新鲜度 1 小时）
    csvs = sorted((os.path.join(tmp, f) for f in os.listdir(tmp) if f.endswith(".csv")),
                  key=lambda p: -os.path.getmtime(p))
    if csvs and _t.time() - os.path.getmtime(csvs[0]) < 3600:
        return list(csv.DictReader(open(csvs[0], encoding="utf-8-sig", errors="replace"))), [csvs[0]]
    before = _t.time()
    run(["competitions", "leaderboard", SLUG, "-d"])
    zips = [os.path.join(ROOT, f) for f in os.listdir(ROOT)
            if f.endswith(".zip") and os.path.getmtime(os.path.join(ROOT, f)) >= before - 5]
    if not zips:
        return [], []
    with zipfile.ZipFile(sorted(zips, key=os.path.getmtime)[-1]) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        if not names:
            return [], []
        z.extract(names[0], tmp)
        path = os.path.join(tmp, names[0])
    return list(csv.DictReader(open(path, encoding="utf-8-sig", errors="replace"))), [path]


def team_submissions(team_id):
    out = run(["competitions", "team-submissions", str(team_id)])
    subs = []
    for line in out.splitlines():
        m = re.match(r"\s*(\d+)\s+(\d{4}-\d{2}-\d{2} [\d:.]+)\s+([\d.]+)?", line)
        if m:
            subs.append(dict(id=int(m.group(1)), date=m.group(2), score=float(m.group(3) or 0)))
    return subs


def episodes(sub_id):
    out = run(["competitions", "episodes", str(sub_id)])
    eps = []
    for line in out.splitlines():
        m = re.match(r"\s*(\d+)\s+(\d{4}-\d{2}-\d{2} [\d:.]+).*?(EpisodeState\.\w+)\s+(EpisodeType\.\w+)", line)
        if m and "VALIDATION" not in m.group(4):
            eps.append(dict(id=int(m.group(1)), time=m.group(2)))
    return eps


def download(ep_id, outdir):
    dst = os.path.join(ROOT, outdir, f"episode-{ep_id}-replay.json")
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        return True
    os.makedirs(os.path.join(ROOT, outdir), exist_ok=True)
    out = run(["competitions", "replay", str(ep_id)])
    # CLI 把文件落在 ROOT/ 下，搬进目标目录
    src = os.path.join(ROOT, f"episode-{ep_id}-replay.json")
    if os.path.exists(src):
        os.replace(src, dst)
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teams", default="", help="逗号分隔的队名（留空则用 --top）")
    ap.add_argument("--top", type=int, default=0, help="取榜单前 N 名")
    ap.add_argument("--per-team", type=int, default=15, help="每个队最多抓多少局")
    ap.add_argument("--out", default="epsoide_top18")
    ap.add_argument("--max-teams", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=1.2, help="每次下载之间等待秒数（防 429）")
    a = ap.parse_args()
    rows, _ = leaderboard()
    if not rows:
        print("✗ 拿不到榜单"); return 2
    key_tid = next((k for k in rows[0] if k.lower() in ("teamid", "team_id")), "TeamId")
    key_name = next((k for k in rows[0] if k.lower() in ("teamname", "team_name")), "TeamName")
    key_score = next((k for k in rows[0] if k.lower() == "score"), "Score")
    want = {t.strip().lower() for t in a.teams.split(",") if t.strip()}
    picked = []
    for r in rows:
        nm = (r.get(key_name) or "").strip()
        if want and nm.lower() not in want:
            continue
        picked.append(dict(tid=r.get(key_tid), name=nm, score=float(r.get(key_score) or 0)))
        if a.top and len(picked) >= a.top:
            break
        if a.max_teams and len(picked) >= a.max_teams:
            break
    print(f"目标队伍 {len(picked)} 个：", ", ".join(f"{p['name']}({p['score']:.0f})" for p in picked))
    total = 0
    for p in picked:
        subs = team_submissions(p["tid"])
        subs.sort(key=lambda s: s["date"], reverse=True)          # 最新提交优先
        n_team = 0
        for s in subs[:2]:
            eps = episodes(s["id"])
            eps.sort(key=lambda e: e["time"], reverse=True)
            for e in eps[: max(0, a.per_team - n_team)]:
                if download(e["id"], a.out):
                    n_team += 1; total += 1
                    time.sleep(a.sleep)
            if n_team >= a.per_team:
                break
        print(f"  {p['name']:28s} 抓 {n_team} 局（提交 {len(subs)} 个）")
    print(f"\n合计 {total} 局 → {a.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
