
"""
D模块 R-03 双基线统计分析
数据来源：
  baseline_seed1101.csv
  baseline_seed1102.csv

分析内容：
  1. 数据完整性检查
  2. 39维 observation 提取
  3. 每个变量的均值、标准差、最大值、最小值
  4. 两个随机种子的差异比较
  5. 输出 CSV 统计表和 Markdown 报告

注意：统计范围属于模型经验数据，不是工业安全阈值。
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ==============================
# 1. 路径配置
# ==============================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    ROOT / "references"
    / "B_300步真实数据"
    / "发给D_300步原始数据"
)

FILE_1101 = DATA_DIR / "baseline_seed1101.csv"
FILE_1102 = DATA_DIR / "baseline_seed1102.csv"

OUTPUT_CSV = ROOT / "experiments" / "r03_baseline_statistics.csv"
OUTPUT_REPORT = ROOT / "docs" / "D_R03双基线统计分析报告_v0.1.md"

OBS_COLUMNS = [f"obs_{i:03d}" for i in range(39)]


# ==============================
# 2. 读取数据
# ==============================

def load_data(file_path, expected_seed):
    if not file_path.exists():
        raise FileNotFoundError(f"找不到数据文件：{file_path}")

    df = pd.read_csv(file_path)

    missing = [
        col for col in OBS_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"缺少 observation 字段：{missing}")

    if len(df) != 100:
        raise ValueError(
            f"{file_path.name} 应有100行，实际为{len(df)}行"
        )

    if not (df["seed"] == expected_seed).all():
        raise ValueError(
            f"{file_path.name} 的 seed 与预期不一致"
        )

    if df["transition_index"].duplicated().any():
        raise ValueError("发现重复 transition_index")

    if not df["transition_index"].is_monotonic_increasing:
        raise ValueError("transition_index 顺序异常")

    obs = df[OBS_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )

    if not np.isfinite(obs.to_numpy()).all():
        raise ValueError("observation 存在 NaN 或 Inf")

    print(f"[PASS] {file_path.name}")
    print(f"       数据行数：{len(df)}")
    print(f"       observation维度：{len(OBS_COLUMNS)}")
    print(f"       Seed：{expected_seed}")

    return df, obs


# ==============================
# 3. 计算统计指标
# ==============================

def calculate_statistics(obs, seed_name):
    result = pd.DataFrame({
        "variable": OBS_COLUMNS,
        "mean": obs.mean().values,
        "std": obs.std(ddof=1).values,
        "min": obs.min().values,
        "max": obs.max().values,
        "range": (
            obs.max() - obs.min()
        ).values,
        "median": obs.median().values,
    })

    result["seed"] = seed_name

    return result


# ==============================
# 4. 主程序
# ==============================

def main():

    print("=" * 60)
    print("D模块 R-03 双基线统计分析")
    print("=" * 60)

    # 读取两条真实基线
    df1, obs1 = load_data(FILE_1101, 1101)
    df2, obs2 = load_data(FILE_1102, 1102)

    # 检查两条轨迹步号是否一致
    if not np.array_equal(
        df1["transition_index"].to_numpy(),
        df2["transition_index"].to_numpy()
    ):
        raise ValueError("两条基线的步号不一致")

    # 计算统计结果
    stats1 = calculate_statistics(
        obs1, "seed1101"
    )

    stats2 = calculate_statistics(
        obs2, "seed1102"
    )

    # 合并两条基线的统计结果
    all_stats = pd.concat(
        [stats1, stats2],
        ignore_index=True
    )

    # 比较两个种子的均值
    comparison = pd.DataFrame({
        "variable": OBS_COLUMNS,
        "mean_seed1101": obs1.mean().values,
        "mean_seed1102": obs2.mean().values,
        "mean_difference": (
            obs2.mean() - obs1.mean()
        ).values,
        "std_seed1101": obs1.std(ddof=1).values,
        "std_seed1102": obs2.std(ddof=1).values,
    })

    # 保存 CSV
    OUTPUT_CSV.parent.mkdir(
        parents=True, exist_ok=True
    )

    all_stats.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    comparison.to_csv(
        OUTPUT_CSV.with_name(
            "r03_baseline_comparison.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # ==============================
    # 5. 数据运行情况
    # ==============================

    execution_summary = []

    for name, df in [
        ("seed1101", df1),
        ("seed1102", df2)
    ]:
        execution_summary.append({
            "scenario": name,
            "rows": len(df),
            "approved": int(df["approved"].astype(bool).sum()),
            "executed": int(df["executed"].astype(bool).sum()),
            "success": int(df["success"].astype(bool).sum()),
            "terminated": int(df["terminated"].astype(bool).sum()),
            "truncated": int(df["truncated"].astype(bool).sum()),
        })

    # ==============================
    # 6. 生成 Markdown 报告
    # ==============================

    lines = [
        "# D模块 R-03 双基线统计分析报告 v0.1",
        "",
        "## 1. 数据来源",
        "",
        "- baseline_seed1101.csv",
        "- baseline_seed1102.csv",
        "- 每条轨迹100步",
        "- 每步39维 observation",
        "",
        "## 2. 数据完整性",
        "",
        f"- Seed 1101：{len(df1)}步",
        f"- Seed 1102：{len(df2)}步",
        "- observation 非有限值检查：通过",
        "- transition_index 顺序检查：通过",
        "",
        "## 3. 执行情况",
        "",
        "| 轨迹 | 步数 | PASS | 执行 | 成功 | terminated | truncated |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for item in execution_summary:
        lines.append(
            f"| {item['scenario']} "
            f"| {item['rows']} "
            f"| {item['approved']} "
            f"| {item['executed']} "
            f"| {item['success']} "
            f"| {item['terminated']} "
            f"| {item['truncated']} |"
        )

    lines.extend([
        "",
        "## 4. 39维变量统计",
        "",
        "| 变量 | Seed | 均值 | 标准差 | 最小值 | 最大值 | 极差 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])

    for _, row in all_stats.iterrows():
        lines.append(
            f"| {row['variable']} "
            f"| {row['seed']} "
            f"| {row['mean']:.8g} "
            f"| {row['std']:.8g} "
            f"| {row['min']:.8g} "
            f"| {row['max']:.8g} "
            f"| {row['range']:.8g} |"
        )

    lines.extend([
        "",
        "## 5. 两个随机种子的均值差异",
        "",
        "| 变量 | Seed1101均值 | Seed1102均值 | 均值差 |",
        "|---|---:|---:|---:|",
    ])

    for _, row in comparison.iterrows():
        lines.append(
            f"| {row['variable']} "
            f"| {row['mean_seed1101']:.8g} "
            f"| {row['mean_seed1102']:.8g} "
            f"| {row['mean_difference']:.8g} |"
        )

    lines.extend([
        "",
        "## 6. 结果解释与限制",
        "",
        "本报告描述两个随机种子下的模型运行经验统计特征。",
        "两条轨迹不能代表全部正常工况。",
        "统计极值不是经过工艺验证的安全阈值。",
        "不能仅根据变量波动大小判断危险程度。",
        "",
        "R-03正式阈值仍需结合物理含义、工艺依据、",
        "更多工况数据及后续验证确定。",
        "",
        "## 7. 输出文件",
        "",
        "- experiments/r03_baseline_statistics.csv",
        "- experiments/r03_baseline_comparison.csv",
        "- docs/D_R03双基线统计分析报告_v0.1.md",
    ])

    OUTPUT_REPORT.parent.mkdir(
        parents=True, exist_ok=True
    )

    OUTPUT_REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("双基线分析完成！")
    print("=" * 60)

    print(f"统计表：{OUTPUT_CSV}")
    print(
        "对比表：",
        OUTPUT_CSV.with_name(
            "r03_baseline_comparison.csv"
        )
    )
    print(f"分析报告：{OUTPUT_REPORT}")

    print("\n前10个变量的均值对比：")
    print(
        comparison[
            ["variable", "mean_seed1101",
             "mean_seed1102", "mean_difference"]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
