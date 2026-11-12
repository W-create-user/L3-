# 目标配置草案

本目录仅含 SYNTHETIC_CONTRACT_ONLY 夹具：contract_fixture.json 和 contract_fixture_registry.json。所有数值和单位都用于校验测试，没有工艺含义，不得提交给真实执行器。没有默认生产目标，不会被 main 自动加载。

```python
import json
from pathlib import Path
from l3_system.planning.goals import validate_goal_spec

root = Path('configs/goals')
goal = json.loads((root / 'contract_fixture.json').read_text())
registry = json.loads((root / 'contract_fixture_registry.json').read_text())
validated = validate_goal_spec(goal, registry)
```

返回值只是校验后的目标副本，不是执行许可或完成结果。生产配置要等 B03 变量映射、D02 围栏依据、A02 评价器及真实实验；Atropine 和 mAb 不得共用字典。完整草案见 ../../docs/contracts_v2.md。
