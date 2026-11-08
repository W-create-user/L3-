
import json
import numpy as np
from pathlib import Path


def load_observations():
    """读取 B 提供的 observation 样本"""

    folders = list(Path("references").glob("B*2026-11-04"))

    if not folders:
        raise FileNotFoundError("找不到 B 的资料目录")

    file_path = folders[0] / "observation_samples.json"

    with open(file_path, "r", encoding="utf-8") as f:
        sample = json.load(f)

    observations = np.asarray(
        [sample["reset"], sample["after_one_step"]],
        dtype=float
    )

    return observations


def analyze_correlation(data):
    """计算重点变量的相关性"""

    if data.ndim != 2 or data.shape[1] != 39:
        raise ValueError("数据必须是 N×39 数组")

    if len(data) < 3:
        print("样本不足3组，不能计算可靠的相关系数。")
        print("当前仅用于检查数据结构。")
        return

    if not np.all(np.isfinite(data)):
        raise ValueError("数据包含 NaN 或 Inf")

    selected = [6, 7, 8, 30, 37, 38]

    selected_data = data[:, selected]

    # 删除没有变化的变量，避免相关系数出现 NaN
    std = np.std(selected_data, axis=0)
    valid_mask = std > 0

    valid_indices = [
        selected[i]
        for i in range(len(selected))
        if valid_mask[i]
    ]

    if len(valid_indices) < 2:
        print("有效变化变量不足，暂时不能计算相关性。")
        return

    valid_data = data[:, valid_indices]

    correlation = np.corrcoef(
        valid_data,
        rowvar=False
    )

    print("\n有效变量：", valid_indices)
    print("\n相关系数矩阵：")
    print(np.round(correlation, 4))


def main():
    print("=" * 60)
    print("R-03 多变量关联分析")
    print("=" * 60)

    data = load_observations()

    print("数据维度：", data.shape)
    print("样本数量：", len(data))

    analyze_correlation(data)

    print("\n分析结束。")


if __name__ == "__main__":
    main()
