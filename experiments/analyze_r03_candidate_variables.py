
"""
D模块 R-03 候选监测变量筛选 v0.1

输入：
1. experiments/r03_baseline_statistics.csv
2. experiments/r03_phase_response_statistics.csv

筛选依据：
1. 双基线波动情况
2. 正向输入响应
3. 反向输入响应
4. 输入恢复特征

注意：
- obs_000 是步数索引，不作为物理状态变量。
- 标准化偏移不等于危险程度。
- 候选排名不等于安全等级。
- 不在本程序中设定工业安全阈值。
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

BASELINE_FILE = (
    ROOT / "experiments"
    / "r03_baseline_statistics.csv"
)

PHASE_FILE = (
    ROOT / "experiments"
    / "r03_phase_response_statistics.csv"
)

OUT_DIR = ROOT / "experiments"
REPORT = (
    ROOT / "docs"
    / "D_R03候选监测变量筛选报告_v0.1.md"
)

EXCLUDED_VARIABLES = {"obs_000"}

PHASE_NAMES = {
    "P1_initial_baseline",
    "P2_positive_input",
    "P3_negative_input",
    "P4_recovery",
}


def load_inputs():
    """读取之前已经生成的分析结果。"""

    for path in [BASELINE_FILE, PHASE_FILE]:
        if not path.exists():
            raise FileNotFoundError(
                f"找不到输入文件：{path}"
            )

    baseline = pd.read_csv(BASELINE_FILE)
    phase = pd.read_csv(PHASE_FILE)

    baseline_required = {
        "variable", "seed", "mean", "std"
    }

    phase_required = {
        "phase", "variable", "phase_mean",
        "phase_std", "baseline_mean",
        "baseline_std", "baseline_difference",
        "difference_from_P1",
        "baseline_low_variance",
    }

    if not baseline_required.issubset(baseline.columns):
        raise ValueError(
            "双基线统计表缺少必要字段"
        )

    if not phase_required.issubset(phase.columns):
        raise ValueError(
            "四阶段统计表缺少必要字段"
        )

    return baseline, phase


def build_baseline_summary(baseline):
    """计算两个种子之间的基线差异。"""

    summary = []

    for variable, group in baseline.groupby("variable"):

        if len(group) != 2:
            raise ValueError(
                f"{variable} 应有两个种子的统计记录"
            )

        means = group["mean"].astype(float)
        stds = group["std"].astype(float)

        summary.append({
            "variable": variable,
            "seed1101_mean": float(
                group.loc[
                    group["seed"] == "seed1101", "mean"
                ].iloc[0]
            ),
            "seed1102_mean": float(
                group.loc[
                    group["seed"] == "seed1102", "mean"
                ].iloc[0]
            ),
            "baseline_mean_difference": float(
                means.iloc[1] - means.iloc[0]
            ),
            "mean_within_seed_std": float(
                stds.mean()
            ),
            "max_within_seed_std": float(
                stds.max()
            ),
        })

    return pd.DataFrame(summary)


def get_phase_values(phase, phase_name, column):
    """提取指定阶段的变量数据。"""

    subset = phase[
        phase["phase"] == phase_name
    ][["variable", column]].copy()

    if subset["variable"].duplicated().any():
        raise ValueError(
            f"{phase_name} 存在重复变量记录"
        )

    return subset.set_index("variable")[column]


def build_candidate_table(baseline, phase):
    """整合基线、响应和恢复指标。"""

    baseline_summary = build_baseline_summary(
        baseline
    )

    required_phases = PHASE_NAMES

    available_phases = set(phase["phase"].unique())

    missing_phases = required_phases - available_phases

    if missing_phases:
        raise ValueError(
            f"四阶段数据缺失：{missing_phases}"
        )

    # 提取各阶段数据
    p1_mean = get_phase_values(
        phase, "P1_initial_baseline", "phase_mean"
    )

    p2_mean = get_phase_values(
        phase, "P2_positive_input", "phase_mean"
    )

    p3_mean = get_phase_values(
        phase, "P3_negative_input", "phase_mean"
    )

    p4_mean = get_phase_values(
        phase, "P4_recovery", "phase_mean"
    )

    p2_change = get_phase_values(
        phase, "P2_positive_input", "difference_from_P1"
    )

    p3_change = get_phase_values(
        phase, "P3_negative_input", "difference_from_P1"
    )

    p4_change = get_phase_values(
        phase, "P4_recovery", "difference_from_P1"
    )

    p2_z = get_phase_values(
        phase, "P2_positive_input",
        "standardized_baseline_difference"
    )

    p3_z = get_phase_values(
        phase, "P3_negative_input",
        "standardized_baseline_difference"
    )

    low_variance = get_phase_values(
        phase, "P2_positive_input",
        "baseline_low_variance"
    )

    # 以变量名称作为统一索引
    result = baseline_summary.set_index(
        "variable"
    )

    result["P1_mean"] = p1_mean
    result["P2_mean"] = p2_mean
    result["P3_mean"] = p3_mean
    result["P4_mean"] = p4_mean

    result["P2_change_from_P1"] = p2_change
    result["P3_change_from_P1"] = p3_change
    result["P4_change_from_P1"] = p4_change

    result["P2_standardized_offset"] = p2_z
    result["P3_standardized_offset"] = p3_z

    result["baseline_low_variance"] = (
        low_variance.astype(str).str.lower().isin(
            ["true", "1"]
        )
    )

    # 输入响应幅度：两个输入阶段中较大的绝对均值变化
    result["max_input_response"] = np.maximum(
        result["P2_change_from_P1"].abs(),
        result["P3_change_from_P1"].abs()
    )

    # 标准化响应仅用于描述当前基线尺度下的相对偏移
    result["max_abs_standardized_offset"] = np.maximum(
        result["P2_standardized_offset"].abs(),
        result["P3_standardized_offset"].abs()
    )

    # 恢复阶段相对初始阶段的残余偏差
    result["recovery_residual"] = (
        result["P4_change_from_P1"].abs()
    )

    # 恢复比例：越低表示恢复阶段均值越接近初始阶段
    denominator = result["max_input_response"]

    result["recovery_residual_ratio"] = np.where(
        denominator > 0,
        result["recovery_residual"] / denominator,
        np.nan
    )

    result["recovery_direction"] = np.select(
        [
            result["recovery_residual"]
            < result["max_input_response"],
            result["recovery_residual"]
            >= result["max_input_response"],
        ],
        [
            "恢复阶段偏差小于输入阶段最大偏差",
            "恢复阶段偏差未小于输入阶段最大偏差",
        ],
        default="无法判断"
    )

    # 排除时间索引
    result["is_physical_candidate"] = (
        ~result.index.isin(EXCLUDED_VARIABLES)
    )

    # 排名只使用非近零方差变量
    result["eligible_for_descriptive_ranking"] = (
        result["is_physical_candidate"]
        & ~result["baseline_low_variance"]
        & result["max_abs_standardized_offset"].notna()
    )

    # 不合成任意的危险分数，只按描述性响应幅度排序
    eligible = result[
        result["eligible_for_descriptive_ranking"]
    ].copy()

    eligible["response_rank"] = (
        eligible["max_abs_standardized_offset"]
        .rank(method="min", ascending=False)
    )

    result["response_rank"] = eligible[
        "response_rank"
    ]

    result = result.reset_index()

    return result


def generate_report(result):
    """生成 Markdown 研究报告。"""

    lines = [
        "# D模块 R-03 候选监测变量筛选报告 v0.1",
        "",
        "## 1. 数据来源",
        "",
        "- 双基线统计结果：200条 observation",
        "- 合法输入变化轨迹：100步",
        "- 状态维度：39维",
        "",
        "## 2. 筛选原则",
        "",
        "本报告综合考察基线波动、正向输入响应、",
        "反向输入响应和恢复阶段表现。",
        "",
        "obs_000 是步数索引，不作为物理状态候选变量。",
        "基线标准差接近零的变量不参与标准化响应排名。",
        "",
        "响应排名仅用于描述性研究，不代表危险程度。",
        "",
        "## 3. 候选变量排名",
        "",
        "| 排名 | 变量 | P2变化 | P3变化 | 最大标准化偏移 | 恢复残余偏差 | 恢复比例 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]

    ranked = result.dropna(
        subset=["response_rank"]
    ).sort_values("response_rank")

    for _, row in ranked.iterrows():

        ratio = row["recovery_residual_ratio"]

        ratio_text = (
            "NA" if pd.isna(ratio)
            else f"{ratio:.6g}"
        )

        lines.append(
            f"| {int(row['response_rank'])} "
            f"| {row['variable']} "
            f"| {row['P2_change_from_P1']:.8g} "
            f"| {row['P3_change_from_P1']:.8g} "
            f"| {row['max_abs_standardized_offset']:.6g} "
            f"| {row['recovery_residual']:.8g} "
            f"| {ratio_text} |"
        )

    lines.extend([
        "",
        "## 4. 近零方差变量",
        "",
        "以下变量不参与标准化响应排名，",
        "但保留在完整结果中供后续研究。",
        "",
        "| 变量 | 基线标准差 | P2变化 | P3变化 |",
        "|---|---:|---:|---:|",
    ])

    low_var = result[
        result["baseline_low_variance"]
    ]

    for _, row in low_var.iterrows():
        lines.append(
            f"| {row['variable']} "
            f"| {row['mean_within_seed_std']:.8g} "
            f"| {row['P2_change_from_P1']:.8g} "
            f"| {row['P3_change_from_P1']:.8g} |"
        )

    lines.extend([
        "",
        "## 5. 时间索引排除",
        "",
        "obs_000 表示步数索引，不属于物理状态变量，",
        "因此不参与候选监测变量排名。",
        "",
        "## 6. 结果解释限制",
        "",
        "1. 当前排名是描述性筛选，不是危险等级。",
        "2. 基线只有两个随机种子，代表性有限。",
        "3. 合法输入变化只有一条100步轨迹。",
        "4. P4恢复阶段只有10步，恢复结论仍需更多数据验证。",
        "5. 标准化偏移受到基线方差大小影响。",
        "6. 变量物理含义仍需结合 observation_schema.json 核实。",
        "7. 当前没有经确认的危险状态正样本。",
        "8. 本报告不确定工业安全阈值。",
        "",
        "## 7. 下一步建议",
        "",
        "对排名靠前的变量进一步核对物理含义、单位、",
        "时间响应和多变量关系，再研究候选异常指标。",
        "",
        "## 8. 输出文件",
        "",
        "- experiments/r03_candidate_variables.csv",
        "- docs/D_R03候选监测变量筛选报告_v0.1.md",
    ])

    REPORT.parent.mkdir(
        parents=True, exist_ok=True
    )

    REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


def main():

    print("=" * 65)
    print("D模块 R-03 候选监测变量筛选")
    print("=" * 65)

    baseline, phase = load_inputs()

    print("[PASS] 双基线统计文件读取成功")
    print("[PASS] 四阶段响应文件读取成功")

    result = build_candidate_table(
        baseline, phase
    )

    output = OUT_DIR / "r03_candidate_variables.csv"

    result.to_csv(
        output,
        index=False,
        encoding="utf-8-sig"
    )

    generate_report(result)

    ranked = result.dropna(
        subset=["response_rank"]
    ).sort_values("response_rank")

    print("\n候选变量排名前10：")

    display_columns = [
        "response_rank",
        "variable",
        "P2_change_from_P1",
        "P3_change_from_P1",
        "max_abs_standardized_offset",
        "recovery_residual",
        "recovery_residual_ratio",
    ]

    print(
        ranked[display_columns]
        .head(10)
        .to_string(index=False)
    )

    print("\n近零方差变量数量：")
    print(
        int(result["baseline_low_variance"].sum())
    )

    print("\n物理状态候选变量数量：")
    print(
        int(result["is_physical_candidate"].sum())
    )

    print("\n参与描述性排名的变量数量：")
    print(
        int(
            result[
                "eligible_for_descriptive_ranking"
            ].sum()
        )
    )

    print("\n" + "=" * 65)
    print("候选变量筛选完成！")
    print("=" * 65)

    print(f"候选变量表：{output}")
    print(f"分析报告：{REPORT}")


if __name__ == "__main__":
    main()
