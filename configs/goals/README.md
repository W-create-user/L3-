# 目标配置草案

本目录保留 SYNTHETIC_CONTRACT_ONLY 夹具：contract_fixture.json 和 contract_fixture_registry.json。所有数值和单位都用于校验测试，没有工艺含义，不得提交给真实执行器。没有默认生产目标，不会被 main 自动加载。

```python
import json
from pathlib import Path
from l3_system.planning.goals import validate_goal_spec

root = Path('configs/goals')
goal = json.loads((root / 'contract_fixture.json').read_text())
registry = json.loads((root / 'contract_fixture_registry.json').read_text())
validated = validate_goal_spec(goal, registry)
```

返回值只是校验后的目标副本，不是执行许可或完成结果。已新增 B 固定版本的 atropine_model_registry.json 与来源说明，重建命令为 `python -m experiments.build_a_registry`；其数值域不是安全围栏，动作单位保留来源推断限定。评价器与有限 HTN 的调用、样本格式和限制见 [11月16日交付](../../docs/A_20261116_接口与后续.md)。生产目标配置仍需 B/D 确认及真实目标实验；Atropine 和 mAb 不得共用字典。完整草案见 ../../docs/contracts_v2.md。
