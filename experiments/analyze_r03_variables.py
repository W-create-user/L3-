
import json
from pathlib import Path

# 自动寻找B提供的资料目录
folders = list(Path("references").glob("B*2026-11-04"))

if not folders:
    raise FileNotFoundError("找不到B的资料目录")

schema_file = folders[0] / "observation_schema.json"

with open(schema_file, "r", encoding="utf-8") as f:
    schema = json.load(f)

fields = schema["fields"]

print("=" * 100)
print("AtropineEnv 39维 Observation 完整变量清单")
print("=" * 100)

for item in fields:
    print(
        f"\n索引：{item['index']}"
        f"\n变量名称：{item['name']}"
        f"\n上游名称：{item['upstream_name']}"
        f"\n物理含义：{item['meaning']}"
        f"\n单位：{item['unit']}"
        f"\n所属类别：{item['group']}"
        f"\nReset样本：{item['reset_example']}"
        f"\nStep样本：{item['step_example']}"
        f"\nBox下界：{item['model_box_low']}"
        f"\nBox上界：{item['model_box_high']}"
    )
    print("-" * 80)

print("\n变量总数：", len(fields))
print("检查结果：", "正确" if len(fields) == 39 else "数量异常")
