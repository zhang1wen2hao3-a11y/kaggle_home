#!/usr/bin/env python3
"""**任务队列**（`ptcg_league/state/jobs.jsonl`）：训练—评估—晋级轮转的状态机底座。

为什么需要：现在的 `run_league.py next` 是**单进程顺序**跑一代 —— GPU 训完才评测，
评测完才计划下一代。GPU 在评测的 30 分钟里空着，CPU 在训练的 15 分钟里空着。
改成"每个 job 一个状态、由调度器按状态分派"，训练 worker 和评估 worker 就能解耦：

    PLANNED → TRAINING → STAGED → SCREENING → TACTICAL_EVAL
           → CONFIRMING → ANCHOR → DECIDED → {PROMOTED|SPECIALIST|COUNTER|ARCHIVED}

⚠️ 这一层**只管理状态**，不碰引擎/训练/评测；晋级判定仍由 `rules.json` + `league.decide()`
决定（那套是反复验证过的，不许动）。断点恢复靠"每步落盘、按状态续跑"，与 run_league 一致。
"""
import json
import os
import time

HERE = os.environ.get("PTCG_LEAGUE_HOME") or os.path.dirname(os.path.abspath(__file__))

STATES = ("PLANNED", "TRAINING", "STAGED", "SCREENING", "TACTICAL_EVAL",
          "CONFIRMING", "ANCHOR", "DECIDED")
TERMINAL = {"PROMOTED", "SPECIALIST", "COUNTER", "ARCHIVED"}
# 每个中间态 → 下一个中间态（终态不在这里）
NEXT = dict(zip(STATES, STATES[1:]))


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class JobQueue:
    """追加式 job 日志。同一 job_id 只取最新一条（后写覆盖索引）。"""

    def __init__(self, path=None):
        self.path = path or os.path.join(HERE, "state", "jobs.jsonl")
        self.rows = []
        self.latest = {}
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    self.rows.append(r)
                    self.latest[r["job_id"]] = r

    def get(self, job_id):
        return self.latest.get(job_id)

    def all(self):
        return [self.latest[j] for j in self.latest]

    def add(self, job_id, spec):
        if job_id in self.latest:
            raise ValueError(f"job_id 已存在：{job_id}")
        r = dict(job_id=job_id, state="PLANNED", spec=spec,
                 created=now(), updated=now(), gen=None, events=[])
        self._append(r)

    def advance(self, job_id, state, **meta):
        old = self.latest.get(job_id)
        if old is None:
            raise KeyError(f"没有这个 job：{job_id}")
        if state not in STATES and state not in TERMINAL:
            raise ValueError(f"非法状态：{state}")
        r = dict(old)
        r["state"] = state
        r["updated"] = now()
        r["events"] = (old.get("events") or []) + [
            dict(state=state, at=now(), **meta)]
        for k, v in meta.items():
            r[k] = v
        self._append(r)

    def _append(self, r):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        self.rows.append(r)
        self.latest[r["job_id"]] = r

    def actionable(self):
        """下一个还没进终态、且没有卡在等待的 job（先到先得）。"""
        for r in self.all():
            if r["state"] in TERMINAL:
                continue
            if r["state"] in ("TRAINING",) and r.get("awaiting_train"):
                continue      # 等 GPU 训练产出，跳过
            return r
        return None


if __name__ == "__main__":
    import sys
    q = JobQueue(os.path.join(HERE, "state", "_selftest_jobs.jsonl"))
    print("states:", STATES)
    print("terminal:", sorted(TERMINAL))
    print("NEXT:", NEXT)
