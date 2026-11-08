
"""
D模块 R-03 合法输入变化响应分析 v0.1

数据：
1. baseline_seed1101.csv
2. baseline_seed1102.csv
3. legal_input_change.csv

分析：
1. 建立双基线参考统计
2. 按 input_condition 分析输入变化阶段
3. 计算39维 observation 的均值和标准差
4. 计算相对基线的均值差
5. 计算标准化偏移（基线标准差非零时）
6. 分析输入变化记录及动作输入
7. 输出 CSV 和 Markdown 报告

注意：
本程序只分析状态响应，不直接判定工业危险。
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    ROOT / "references"
    / "B_300步真实数据"
    / "发给D_300步原始数据"
)

BASE_FILES = [
    DATA_DIR / "baseline_seed1101.csv",
    DATA_DIR / "baseline_seed1102.csv",
]

LEGAL_FILE = DATA_DIR / "legal_input_change.csv"

OUT_DIR = ROOT / "experiments"
REPORT = ROOT / "docs" / "D_R03合法输入变化响应分析报告_v0.1.md"

OBS = [f"obs_{i:03d}" for i in range(39)]
INPUTS = [f"actual_input_mL_min_{i:02d}" for i in range(4)]


def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")

    df = pd.read_csv(path)

    required = OBS + [
        "transition_index",
        "input_condition",
        "input_changed",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"{path.name} 缺少字段：{missing}")

    if len(df) != 100:
        raise ValueError(
            f"{path.name} 应有100行，实际{len(df)}行"
        )

    values = df[OBS].apply(pd.to_numeric, errors="coerce")

    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"{path.name} 的 observation 存在 NaN/Inf")

    if df["transition_index"].duplicated().any():
        raise ValueError(f"{path.name} 存在重复步号")

    return df, values


def main():
    print("=" * 60)
    print("D模块 R-03 合法输入变化响应分析")
    print("=" * 60)

    # 1. 读取两条基线
    base_frames = []
    base_observations = []

    for path in BASE_FILES:
        df, obs = load_csv(path)
        base_frames.append(df)
        base_observations.append(obs)

        print(f"[PASS] {path.name}: {len(df)}步")

    # 两条基线合并，建立经验参考
    baseline = pd.concat(
        base_observations,
        ignore_index=True
    )

    baseline_mean = baseline.mean()
    baseline_std = baseline.std(ddof=1)

    # 2. 读取合法输入变化轨迹
    legal_df, legal_obs = load_csv(LEGAL_FILE)

    print(f"[PASS] {LEGAL_FILE.name}: {len(legal_df)}步")

    # 3. 按输入条件分组
    legal_df = legal_df.copy()
    legal_df["_row_id"] = np.arange(len(legal_df))

    results = []
    condition_summaries = []

    for condition, group in legal_df.groupby(
        "input_condition", sort=False, dropna=False
    ):
        indices = group.index
        obs_group = legal_obs.loc[indices]

        condition_name = (
            "UNKNOWN" if pd.isna(condition) else str(condition)
        )

        changed = group["input_changed"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )

        condition_summaries.append({
            "input_condition": condition_name,
            "rows": len(group),
            "first_transition": int(group["transition_index"].min()),
            "last_transition": int(group["transition_index"].max()),
            "input_changed_rows": int(changed.sum()),
        })

        for col in OBS:
            mean_value = float(obs_group[col].mean())
            std_value = float(obs_group[col].std(ddof=1))
            base_mean = float(baseline_mean[col])
            base_std = float(baseline_std[col])

            difference = mean_value - base_mean

            if base_std > 0:
                standardized = difference / base_std
            else:
                standardized = np.nan

            results.append({
                "input_condition": condition_name,
                "variable": col,
                "sample_count": len(group),
                "mean": mean_value,
                "std": std_value,
                "baseline_mean": base_mean,
                "baseline_std": base_std,
                "mean_difference": difference,
                "standardized_difference": standardized,
                "min": float(obs_group[col].min()),
                "max": float(obs_group[col].max()),
            })

    response = pd.DataFrame(results)
    conditions = pd.DataFrame(condition_summaries)

    # 4. 输入流量统计
    input_results = []

    available_inputs = [
        col for col in INPUTS if col in legal_df.columns
    ]

    for col in available_inputs:
        values = pd.to_numeric(
            legal_df[col], errors="coerce"
        )

        input_results.append({
            "input_variable": col,
            "mean": values.mean(),
            "std": values.std(ddof=1),
            "min": values.min(),
            "max": values.max(),
        })

    input_stats = pd.DataFrame(input_results)

    # 5. 保存 CSV
    response.to_csv(
        OUT_DIR / "r03_input_response_statistics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    conditions.to_csv(
        OUT_DIR / "r03_input_condition_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    input_stats.to_csv(
        OUT_DIR / "r03_input_flow_statistics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 6. 生成报告
    lines = [
        "# D模块 R-03 合法输入变化响应分析报告 v0.1",
        "",
        "## 1. 数据来源",
        "",
        "- Seed 1101 基线：100步",
        "- Seed 1102 基线：100步",
        "- 合法输入变化轨迹：100步",
        "- 基线参考：两条基线合并后的200条 observation",
        "",
        "## 2. 数据完整性",
        "",
        "- 三份 CSV 均完成读取",
        "- 每条轨迹100行",
        "- observation 均为39维",
        "- NaN/Inf 检查通过",
        "",
        "## 3. 输入条件分组",
        "",
        "| 输入条件 | 样本数 | 首步 | 末步 | input_changed记录数 |",
        "|---|---:|---:|---:|---:|",
    ]

    for _, row in conditions.iterrows():
        lines.append(
            f"| {row['input_condition']} "
            f"| {row['rows']} "
            f"| {row['first_transition']} "
            f"| {row['last_transition']} "
            f"| {row['input_changed_rows']} |"
        )

    lines.extend([
        "",
        "## 4. 输入流量统计",
        "",
        "| 输入变量 | 均值 | 标准差 | 最小值 | 最大值 |",
        "|---|---:|---:|---:|---:|",
    ])

    for _, row in input_stats.iterrows():
        lines.append(
            f"| {row['input_variable']} "
            f"| {row['mean']:.8g} "
            f"| {row['std']:.8g} "
            f"| {row['min']:.8g} "
            f"| {row['max']:.8g} |"
        )

    lines.extend([
        "",
        "## 5. 状态响应统计",
        "",
        "| 输入条件 | 变量 | 均值 | 基线均值 | 均值差 | 标准化偏移 |",
        "|---|---|---:|---:|---:|---:|",
    ])

    for _, row in response.iterrows():
        z = row["standardized_difference"]
        z_text = "NA" if pd.isna(z) else f"{z:.6g}"

        lines.append(
            f"| {row['input_condition']} "
            f"| {row['variable']} "
            f"| {row['mean']:.8g} "
            f"| {row['baseline_mean']:.8g} "
            f"| {row['mean_difference']:.8g} "
            f"| {z_text} |"
        )

    lines.extend([
        "",
        "## 6. 结果解释与限制",
        "",
        "本分析比较合法输入变化轨迹与两条基线的合并经验统计。",
        "标准化偏移仅表示相对于当前基线波动尺度的差异，",
        "不代表异常概率或危险等级。",
        "",
        "不同输入阶段的样本可能具有时间相关性，",
        "不能将分组均值差直接解释为因果效应。",
        "",
        "合法输入变化不等于危险事件。",
        "当前数据没有提供经过确认的状态异常正样本，",
        "因此本报告不建立危险标签，也不确定工业安全阈值。",
        "",
        "## 7. 输出文件",
        "",
        "- experiments/r03_input_response_statistics.csv",
        "- experiments/r03_input_condition_summary.csv",
        "- experiments/r03_input_flow_statistics.csv",
    ])

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 60)
    print("输入变化响应分析完成！")
    print("=" * 60)

    print("状态响应表：experiments/r03_input_response_statistics.csv")
    print("输入阶段表：experiments/r03_input_condition_summary.csv")
    print("输入流量表：experiments/r03_input_flow_statistics.csv")
    print(f"分析报告：{REPORT}")

    print("\n输入条件分组：")
    print(conditions.to_string(index=False))

    # 输出每个输入阶段偏离基线最大的前5个变量
    print("\n各输入阶段标准化偏移较大的变量：")

    for condition, group in response.groupby("input_condition"):
        ranked = group.dropna(
            subset=["standardized_difference"]
        ).copy()

        ranked["abs_z"] = ranked[
            "standardized_difference"
        ].abs()

        print(f"\n[{condition}]")
        print(
            ranked.nlargest(5, "abs_z")[
                ["variable", "mean_difference",
                 "standardized_difference"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
