"""**"首次高置信命中才切一次"这个策略安不安全**：量首次命中的**精确率**（不是识别率）。

    ./tools/py tools/measure_first_hit.py

2026-11-15 实测（50 局真实回放，双方卡组已知）：

    阈值                 首次命中局数  正确  精确率  中位回合
    cov≥0.80 m≥0.15          14      14   100%     2
    cov≥0.70 m≥0.10          24      23    96%     2
    cov≥0.60 m≥0.08          40      39    98%     3

⇒ 首次命中的精确率很高（96~100%），**"切一次不回头"是安全的**；
   代价是命中率（覆盖率）低 —— 只有 28%~80% 的局会触发切换。
   建议用 cov≥0.70 m≥0.10：一半的局能切、精确率 96%。
"""
import json, collections, os, sys, itertools
sys.path.insert(0,'.'); sys.path.insert(0,'tools')
from router_core import load_families
from framework.replay import iter_decisions
fams=load_families()
H2FAM={d["hash"]:fam for fam,decks in fams.items() for d in decks}
cases=[]
for line in itertools.islice(open("data/ladder_games.tsv", encoding="utf-8", errors="replace"), 1, 600):
    r=line.rstrip("\r\n").split("\t")
    if len(r)<9: continue
    f,d0,d1,mir,w=r[1],r[2][:8],r[4][:8],r[6],r[8]
    if mir=="1" or w not in ("0","1") or not os.path.exists(f): continue
    if H2FAM.get(d0) and H2FAM.get(d1): cases.append((f,H2FAM[d0],H2FAM[d1]))
    if len(cases)>=50: break
def vis(obs,idx):
    cur=obs.get("current") or {}; ps=cur.get("players") or []; v=set()
    if idx<len(ps):
        p=ps[idx]
        for z in ("active","bench","discard"):
            for c in (p.get(z) or []):
                if c and c.get("id"): v.add(int(c["id"]))
    for c in (cur.get("stadium") or []):
        if c and c.get("id"): v.add(int(c["id"]))
    return v
print(f"{'阈值':22}{'首次命中局数':>10}{'其中正确':>8}{'精确率':>8}{'中位回合':>9}")
for cov0,marg in ((0.80,0.15),(0.70,0.10),(0.65,0.10),(0.60,0.08)):
    nfire=ncorrect=0; turns=[]
    for p,f0,f1 in cases:
        steps=json.load(open(p)).get("steps") or []
        v=set(); fired=None
        for t,j,obs,act in iter_decisions(steps):
            opp=f1 if j==0 else f0
            v|=vis(obs,1-j)
            if len(v)<3: continue
            scored=[]
            for fam,decks in fams.items():
                best=0.0
                for dd in decks: best=max(best,len(v&dd["cards"])/len(v))
                scored.append((best,fam))
            scored.sort(reverse=True)
            c,fam=scored[0]; m=c-(scored[1][0] if len(scored)>1 else 0)
            if c>=cov0 and m>=marg:
                fired=(fam==opp,(obs.get("current") or {}).get("turn")); break
        if fired: 
            nfire+=1; ncorrect+= 1 if fired[0] else 0; turns.append(fired[1])
    med=sorted(turns)[len(turns)//2] if turns else None
    print(f"cov≥{cov0:.2f} m≥{marg:.2f}      {nfire:>10}{ncorrect:>8}{(ncorrect/max(1,nfire)*100):7.0f}%{str(med):>9}")
