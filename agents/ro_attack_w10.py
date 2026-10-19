"""rollout 变体：MODE=attack, WORLDS=10, MARGIN=1.5, CANDS=all。

为什么不用环境变量：`evaluate.py` 的 worker 进程里同时加载两侧 agent，
共享 os.environ —— 两个 rollout agent 会互相污染。所以这里把配置**烘进模块**。
`main_rollout.agent` 读的是模块级全局，import 之后再覆盖即可。
"""
import importlib.util as _ilu
import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_spec = _ilu.spec_from_file_location("_mr_attack_w10",
                                     _os.path.join(_HERE, "..", "main_rollout.py"))
_m = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_m)

_m.MODE = "attack"        # attack | all | off
_m.WORLDS = 10           # 每个局面的 determinization 采样数
_m.MARGIN = 1.5            # 偏离规则选择所需的 z 阈值
_m.CANDS = "all"           # 候选集：全部选项

agent = _m.agent
