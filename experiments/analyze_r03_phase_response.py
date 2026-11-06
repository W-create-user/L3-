
"""
D模块 R-03 四阶段状态响应分析 v0.1

P1: transition_index 1-30   初始基线
P2: transition_index 31-60  正向输入变化
P3: transition_index 61-90  反向输入变化
P4: transition_index 91-100 输入恢复

分析对象：39维 observation
参考数据：两条独立基线 + 合法输入变化轨迹

注意：
统计偏移不等于危险程度。
本程序不生成工业安全阈值。
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

BASE_FILES = {
    "seed1101": DATA_DIR / "baseline_seed1101.csv",
    "seed1102": DATA_DIR / "baseline_seed1102.csv",
}

LEGAL_FILE = DATA_DIR / "legal_input_change.csv"

EXPERIMENT_DIR = ROOT / "experiments"
DOCS_DIR = ROOT / "docs"

OBS = [f"obs_{i:03d}" for i in range(39)]

PHASES = {
    "P1_initial_baseline": (1, 30),
    "P2_positive_input": (31, 60),
    "P3_negative_input": (61, 90),
    "P4_recovery": (91, 100),
}

# 仅用于数值计算中识别近零标准差，不是安全阈值
STD_RELATIVE_TOLERANCE = 1e-8
STD_ABSOLUTE_TOLERANCE = 1e-12


def load_observation(path, expected_rows=100):
    """读取并检查39维 observation。"""

    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")

    df = pd.read_csv(path)

    missing = [c for c in OBS if c not in df.columns]

    if missing:
        raise ValueError(f"{path.name} 缺少字段：{missing}")

    if "transition_index" not in df.columns:
        raise ValueError("缺少 transition_index 字段")

    if len(df) != expected_rows:
        raise ValueError(
            f"{path.name} 应有{expected_rows}行，实际{len(df)}行"
        )

    steps = pd.to_numeric(
        df["transition_index"], errors="coerce"
    )

    if steps.isna().any():
        raise ValueError("transition_index 存在无效值")

    if steps.duplicated().any():
        raise ValueError("transition_index 存在重复值")

    if not steps.is_monotonic_increasing:
        raise ValueError("transition_index 顺序异常")

    df = df.copy()
    df["transition_index"] = steps.astype(int)

    values = df[OBS].apply(
        pd.to_numeric, errors="coerce"
    )

    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"{path.name} 的 observation 存在 NaN/Inf")

    return df, values


def load_baseline_reference():
    """合并两条基线，计算39维变量的参考统计量。"""

    frames = []

    for seed_name, path in BASE_FILES.items():
        df, obs = load_observation(path)

        frames.append(obs)

        print(
            f"[PASS] {path.name}: "
            f"{len(df)}步，39维"
        )

    pooled = pd.concat(frames, ignore_index=True)

    reference = pd.DataFrame({
        "variable": OBS,
        "baseline_mean": pooled.mean().values,
        "baseline_std": pooled.std(ddof=1).values,
        "baseline_min": pooled.min().values,
        "baseline_max": pooled.max().values,
    })

    return reference


def identify_low_variance(mean_value, std_value):
    """
    识别数值上接近零的基线标准差。

    仅用于避免对极小分母作过度解释，
    不代表变量安全或危险。
    """

    tolerance = max(
        STD_ABSOLUTE_TOLERANCE,
        abs(mean_value) * STD_RELATIVE_TOLERANCE
    )

    return std_value <= tolerance


def main():

    print("=" * 65)
    print("D模块 R-03 四阶段状态响应分析")
    print("=" * 65)

    # 1. 建立双基线参考
    reference = load_baseline_reference()

    # 2. 读取合法输入变化轨迹
    legal_df, legal_obs = load_observation(LEGAL_FILE)

    print(
        f"[PASS] {LEGAL_FILE.name}: "
        f"{len(legal_df)}步，39维"
    )

    # 3. 检查步号覆盖
    expected_steps = np.arange(1, 101)

    if not np.array_equal(
        legal_df["transition_index"].to_numpy(),
        expected_steps
    ):
        raise ValueError("合法输入轨迹步号不是连续1-100")

    # 4. 四阶段统计
    phase_records = []
    phase_details = []

    # P1 均值作为同一实验内部的变化前参考
    p1_start, p1_end = PHASES["P1_initial_baseline"]

    p1_mask = legal_df["transition_index"].between(
        p1_start, p1_end
    )

    p1_obs = legal_obs.loc[p1_mask.to_numpy()]

    p1_mean = p1_obs.mean()

    for phase_name, (start, end) in PHASES.items():

        mask = legal_df["transition_index"].between(
            start, end
        )

        phase_df = legal_df.loc[mask].copy()
        phase_obs = legal_obs.loc[mask.to_numpy()]

        if len(phase_df) != end - start + 1:
            raise ValueError(
                f"{phase_name} 样本数异常"
            )

        print(
            f"[PASS] {phase_name}: "
            f"步号{start}-{end}，"
            f"{len(phase_df)}条"
        )

        for i, variable in enumerate(OBS):

            values = phase_obs[variable]

            phase_mean = float(values.mean())
            phase_std = float(values.std(ddof=1))
            phase_min = float(values.min())
            phase_max = float(values.max())

            base_mean = float(
                reference.loc[i, "baseline_mean"]
            )

            base_std = float(
                reference.loc[i, "baseline_std"]
            )

            initial_mean = float(p1_mean[variable])

            # 相对双基线的绝对差
            baseline_difference = phase_mean - base_mean

            # 相对实验自身初始阶段的变化
            initial_difference = phase_mean - initial_mean

            low_variance = identify_low_variance(
                base_mean, base_std
            )

            if low_variance:
                standardized = np.nan
            else:
                standardized = (
                    baseline_difference / base_std
                )

            record = {
                "phase": phase_name,
                "start_step": start,
                "end_step": end,
                "sample_count": len(phase_df),
                "variable": variable,
                "phase_mean": phase_mean,
                "phase_std": phase_std,
                "phase_min": phase_min,
                "phase_max": phase_max,
                "baseline_mean": base_mean,
                "baseline_std": base_std,
                "baseline_difference": baseline_difference,
                "initial_phase_mean": initial_mean,
                "difference_from_P1": initial_difference,
                "standardized_baseline_difference": standardized,
                "baseline_low_variance": low_variance,
            }

            phase_records.append(record)

            # 便于按阶段查看的简化结果
            phase_details.append({
                "phase": phase_name,
                "variable": variable,
                "phase_mean": phase_mean,
                "phase_std": phase_std,
                "difference_from_P1": initial_difference,
                "baseline_std": base_std,
                "standardized_baseline_difference": standardized,
                "baseline_low_variance": low_variance,
            })

    # 5. 检查输入变化记录
    input_change_records = []

    if "input_changed" in legal_df.columns:

        for _, row in legal_df.iterrows():

            changed = str(
                row["input_changed"]
            ).lower() in ["true", "1", "yes"]

            if changed:
                input_change_records.append({
                    "transition_index": int(
                        row["transition_index"]
                    ),
                    "input_condition": row.get(
                        "input_condition", ""
                    ),
                    "action_00": row.get("action_00", np.nan),
                    "action_01": row.get("action_01", np.nan),
                    "action_02": row.get("action_02", np.nan),
                    "action_03": row.get("action_03", np.nan),
                })

    input_changes = pd.DataFrame(input_change_records)

    # 6. 转换结果
    result = pd.DataFrame(phase_records)

    # 7. 输出 CSV
    EXPERIMENT_DIR.mkdir(
        parents=True, exist_ok=True
    )

    result.to_csv(
        EXPERIMENT_DIR / "r03_phase_response_statistics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    input_changes.to_csv(
        EXPERIMENT_DIR / "r03_phase_input_changes.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 8. 生成每个变量的四阶段宽表
    wide = result.pivot(
        index="variable",
        columns="phase",
        values="phase_mean"
    ).reset_index()

    wide.to_csv(
        EXPERIMENT_DIR / "r03_phase_mean_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 9. 生成报告
    lines = [
        "# D模块 R-03 四阶段状态响应分析报告 v0.1",
        "",
        "## 1. 数据来源",
        "",
        "- baseline_seed1101.csv：100步",
        "- baseline_seed1102.csv：100步",
        "- legal_input_change.csv：100步",
        "- observation：39维",
        "- 双基线合计200条记录",
        "",
        "## 2. 四阶段划分",
        "",
        "| 阶段 | 步号 | 样本数 |",
        "|---|---:|---:|",
    ]

    for name, (start, end) in PHASES.items():
        lines.append(
            f"| {name} | {start}-{end} | {end-start+1} |"
        )

    lines.extend([
        "",
        "## 3. 输入变化事件",
        "",
        "| 步号 | 输入条件 | action_00 | action_01 | action_02 | action_03 |",
        "|---:|---|---:|---:|---:|---:|",
    ])

    if input_changes.empty:
        lines.append("| 未检测到 input_changed=True 记录 | | | | | |")
    else:
        for _, row in input_changes.iterrows():
            lines.append(
                f"| {row['transition_index']} "
                f"| {row['input_condition']} "
                f"| {row['action_00']} "
                f"| {row['action_01']} "
                f"| {row['action_02']} "
                f"| {row['action_03']} |"
            )

    lines.extend([
        "",
        "## 4. 各阶段重点变量变化",
        "",
        "以下按相对P1初始阶段的绝对均值变化排序。",
        "",
        "| 阶段 | 变量 | 阶段均值 | 相对P1变化 | 基线标准差 | 近零方差标记 |",
        "|---|---|---:|---:|---:|---|",
    ])

    for phase_name in PHASES:

        subset = result[
            result["phase"] == phase_name
        ].copy()

        subset["abs_change"] = subset[
            "difference_from_P1"
        ].abs()

        top = subset.nlargest(10, "abs_change")

        for _, row in top.iterrows():

            lines.append(
                f"| {phase_name} "
                f"| {row['variable']} "
                f"| {row['phase_mean']:.8g} "
                f"| {row['difference_from_P1']:.8g} "
                f"| {row['baseline_std']:.8g} "
                f"| {'是' if row['baseline_low_variance'] else '否'} |"
            )

    lines.extend([
        "",
        "## 5. 分析限制",
        "",
        "1. P1作为同一实验的初始阶段参考。",
        "2. 双基线标准差用于描述当前基线波动。",
        "3. 基线标准差接近零时不计算标准化偏移。",
        "4. 数值上的近零方差标记不是安全阈值。",
        "5. 阶段均值差不能单独证明输入与状态之间的因果关系。",
        "6. 恢复阶段只有10步，不能据此证明系统已完全恢复。",
        "7. 当前数据不包含经工艺确认的危险状态正样本。",
        "8. 本报告不建立工业安全阈值或危险标签。",
        "",
        "## 6. 输出文件",
        "",
        "- experiments/r03_phase_response_statistics.csv",
        "- experiments/r03_phase_input_changes.csv",
        "- experiments/r03_phase_mean_comparison.csv",
        "- docs/D_R03四阶段状态响应分析报告_v0.1.md",
    ])

    DOCS_DIR.mkdir(
        parents=True, exist_ok=True
    )

    report_path = (
        DOCS_DIR / "D_R03四阶段状态响应分析报告_v0.1.md"
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    # 10. 控制台结果
    print("\n" + "=" * 65)
    print("四阶段状态响应分析完成！")
    print("=" * 65)

    print(
        "状态统计表："
        "experiments/r03_phase_response_statistics.csv"
    )

    print(
        "输入变化事件："
        "experiments/r03_phase_input_changes.csv"
    )

    print(
        "阶段均值对比："
        "experiments/r03_phase_mean_comparison.csv"
    )

    print(f"分析报告：{report_path}")

    print("\n输入变化事件记录：")

    if input_changes.empty:
        print("没有检测到 input_changed=True")
    else:
        print(input_changes.to_string(index=False))

    print("\n各阶段相对P1变化最大的前5个变量：")

    for phase_name in PHASES:

        subset = result[
            result["phase"] == phase_name
        ].copy()

        subset["abs_change"] = subset[
            "difference_from_P1"
        ].abs()

        print(f"\n[{phase_name}]")

        print(
            subset.nlargest(5, "abs_change")[
                [
                    "variable",
                    "difference_from_P1",
                    "baseline_std",
                    "baseline_low_variance",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
