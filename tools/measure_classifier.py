"""**量路由器能不能用**：原型族识别率 vs 阈值（路由器没识别率就没价值）。

    ./tools/py tools/measure_classifier.py

输出每种阈值下的"识别率（该族被正确认出的局数占比）"和"误判帧/局"，用于选 DECIDE_MIN_*
（router_core.py / router_main.py 里的两个阈值）。2026-11-15 实测结论：
  · 默认 cov≥0.80 m≥0.15：slowking 23% / abra 12% ⇒ **路由器几乎不会触发**
  · cov≥0.60 m≥0.08：slowking 77% / abra 88%，但误判帧 18.9/局
⇒ 路由包能用，但**分类器是瓶颈**；要真上路由，先把识别做准（或改成"首次高置信命中才切一次"）。
"""
import json, collections, os, sys, itertools
sys.path.insert(0,'.'); sys.path.insert(0,'tools')
from router_core import load_families
from framework.replay import iter_decisions
fams=load_families()
H2FAM={d["hash"]:fam for fam,decks in fams.items() for d in decks}
# 卡面里"基本能量"的 id（1-9 之类）应该从可见集里剔除，否则覆盖率被稀释
BASIC_E = set()
try:
    import csv
    for r in csv.DictReader(open("build/port_plain/EN_Card_Data.csv", encoding="utf-8-sig")):
        if (r.get("Stage (Pokémon)/Type (Energy and Trainer)") or "")=="Basic Energy":
            BASIC_E.add(int(r["Card ID"]))
except Exception as e:
    print("能量表读取失败", e)
print("基本能量 id 数:", len(BASIC_E))

def visible_of(obs, opp_index, drop_energy=True):
    cur=obs.get("current") or {}; ps=cur.get("players") or []
    v=set()
    if opp_index < len(ps):
        p=ps[opp_index]
        for z in ("active","bench","discard"):
            for c in (p.get(z) or []):
                if c and c.get("id"): v.add(int(c["id"]))
    for c in (cur.get("stadium") or []):
        if c and c.get("id"): v.add(int(c["id"]))
    return v - BASIC_E if drop_energy else v

cases=[]
for line in itertools.islice(open("data/ladder_games.tsv", encoding="utf-8", errors="replace"), 1, 500):
    r=line.rstrip("\r\n").split("\t")
    if len(r)<9: continue
    f,d0,d1,mir,w = r[1], r[2][:8], r[4][:8], r[6], r[8]
    if mir=="1" or w not in ("0","1") or not os.path.exists(f): continue
    f0,f1=H2FAM.get(d0),H2FAM.get(d1)
    if f0 and f1: cases.append((f,f0,f1))
    if len(cases)>=45: break
print("回放", len(cases))

def eval_cfg(min_cov, min_margin, min_cards=3, drop_energy=True):
    hit=collections.Counter(); tot=collections.Counter(); wrong=0; frames=0
    for p,f0,f1 in cases:
        steps=json.load(open(p)).get("steps") or []
        v=set(); first=None
        for t,j,obs,act in iter_decisions(steps):
            opp = f1 if j==0 else f0
            v |= visible_of(obs, 1-j, drop_energy)
            frames+=1
            if len(v) < min_cards: continue
            scored=[]
            for fam,decks in fams.items():
                best=0.0
                for dd in decks:
                    best=max(best, len(v & dd["cards"])/len(v))
                scored.append((best,fam))
            scored.sort(reverse=True)
            cov,fam=scored[0]; margin=cov-(scored[1][0] if len(scored)>1 else 0)
            if cov>=min_cov and margin>=min_margin:
                if fam==opp and first is None: first=(obs.get("current") or {}).get("turn")
                elif fam!=opp: wrong+=1
        tot[f1]+=1
        if first is not None: hit[f1]+=1
    return hit,tot,wrong/max(1,len(cases))
for cov,marg in ((0.80,0.15),(0.70,0.10),(0.60,0.08),(0.55,0.06),(0.50,0.05)):
    for de in (True,False):
        hit,tot,wrong=eval_cfg(cov,marg,3,de)
        line=f"cov≥{cov:.2f} m≥{marg:.2f} 去能量={de}: "
        for fam in sorted(tot, key=lambda f:-tot[f]):
            line+=f"{fam.split('_')[0][:6]}={hit[fam]/tot[fam]*100:.0f}%({tot[fam]}) "
        line+=f" | 误判帧/局={wrong:.1f}"
        print(line)
