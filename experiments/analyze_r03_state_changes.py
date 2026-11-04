
import json
import numpy as np
from pathlib import Path

folders = list(Path("references").glob("B*2026-11-04"))

if not folders:
    raise FileNotFoundError("找不到B的资料目录")

data_file = folders[0] / "observation_samples.json"
schema_file = folders[0] / "observation_schema.json"

with open(data_file, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(schema_file, "r", encoding="utf-8") as f:
    schema = json.load(f)

before = np.asarray(data["reset"], dtype=float)
after = np.asarray(data["after_one_step"], dtype=float)

assert before.shape == (39,)
assert after.shape == (39,)

results = []

for item in schema["fields"]:
    i = item["index"]
    delta = after[i] - before[i]

    if before[i] != 0:
        rate = delta / abs(before[i]) * 100
    else:
        rate = None

    results.append({
        "index": i,
        "name": item["name"],
        "unit": item["unit"],
        "before": before[i],
        "after": after[i],
        "delta": delta,
        "rate": rate
    })

# 按绝对变化量排序，仅用于初步筛选
results.sort(key=lambda x: abs(x["delta"]), reverse=True)

print("=" * 75)
print("R-03 真实状态变化分析")
print("=" * 75)

for item in results[:10]:
    print(
        f"Index {item['index']:2d} | "
        f"{item['name']:25s} | "
        f"变化量={item['delta']:.10g} | "
        f"变化率={item['rate'] if item['rate'] is not None else 'N/A'}"
    )

print("\n分析样本：", data["seed"])
print("执行步数：", data["step_count"])
print("分析变量数：", len(results))
print("\n注意：变化幅度不等于危险程度。")
