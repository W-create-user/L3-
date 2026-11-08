
import json
import numpy as np
from pathlib import Path


def analyze_timeseries(observations):
    data = np.asarray(observations, dtype=float)

    if data.ndim != 2 or data.shape[1] != 39:
        raise ValueError("数据必须是 N×39 数组")

    if len(data) < 2:
        raise ValueError("至少需要两个观测样本")

    if not np.all(np.isfinite(data)):
        raise ValueError("数据包含 NaN 或 Inf")

    delta = np.diff(data, axis=0)

    selected = [6, 7, 8, 30, 37, 38]

    print("=" * 60)
    print("R-03 Observation 连续变化分析")
    print("=" * 60)

    for index in selected:
        changes = delta[:, index]

        print(f"\n变量索引：{index}")
        print(f"最大变化：{np.max(changes):.10g}")
        print(f"最小变化：{np.min(changes):.10g}")
        print(f"平均变化：{np.mean(changes):.10g}")
        print(f"最大绝对变化：{np.max(np.abs(changes)):.10g}")

    print("\n样本数量：", len(data))
    print("变化区间：", len(data) - 1)


def main():
    folders = list(Path("references").glob("B*2026-11-04"))

    if not folders:
        raise FileNotFoundError("找不到B提供的资料目录")

    file_path = folders[0] / "observation_samples.json"

    with open(file_path, "r", encoding="utf-8") as f:
        sample = json.load(f)

    # 当前只有两个真实观测样本
    observations = [
        sample["reset"],
        sample["after_one_step"]
    ]

    print("数据来源：B提供的真实 observation 样本")
    print("Seed：", sample["seed"])
    print("Step count：", sample["step_count"])

    analyze_timeseries(observations)


if __name__ == "__main__":
    main()
