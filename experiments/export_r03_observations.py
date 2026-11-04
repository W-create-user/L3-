
import csv
import json
from pathlib import Path

folders = list(Path("references").glob("B*2026-11-04"))

if not folders:
    raise FileNotFoundError("找不到B的资料目录")

folder = folders[0]

with open(folder / "observation_samples.json", "r", encoding="utf-8") as f:
    sample = json.load(f)

with open(folder / "observation_schema.json", "r", encoding="utf-8") as f:
    schema = json.load(f)

fields = schema["fields"]

output = Path("experiments/r03_observation_samples.csv")

with output.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)

    writer.writerow(["sample_type"] + [item["name"] for item in fields])

    writer.writerow(["RESET"] + sample["reset"])
    writer.writerow(["AFTER_ONE_STEP"] + sample["after_one_step"])

print("CSV导出成功：", output)
print("样本数量：2")
print("变量数量：39")
