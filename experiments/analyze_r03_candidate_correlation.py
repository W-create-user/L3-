
"""
D模块 R-03 候选变量多阶段关联分析 v0.1

分析内容：
1. 两个基线种子分别分析
2. 合法输入变化轨迹按P1-P4分阶段分析
3. Pearson相关系数
4. Spearman秩相关系数
5. 重点变量组关联分析
6. 不同阶段相关系数变化比较

注意：
相关性不代表因果关系。
P4只有10个样本，结果仅作探索性参考。
不建立工业安全阈值。
"""

from pathlib import Path
import itertools
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    ROOT / "references"
    / "B_300步真实数据"
    / "发给D_300步原始数据"
)

BASE_FILES = {
    "baseline_seed1101": DATA_DIR / "baseline_seed1101.csv",
    "baseline_seed1102": DATA_DIR / "baseline_seed1102.csv",
}

LEGAL_FILE = DATA_DIR / "legal_input_change.csv"

OUT_DIR = ROOT / "experiments"
REPORT = (
    ROOT / "docs"
    / "D_R03候选变量多阶段关联分析报告_v0.1.md"
)

# 排除obs_000，因为它是步数索引
CANDIDATES = [
    "obs_025",
    "obs_023",
    "obs_007",
    "obs_029",
    "obs_026",
    "obs_038",
    "obs_008",
    "obs_037",
    "obs_034",
    "obs_033",
]

CANDIDATE_GROUPS = {
    "aqueous_molar_flow": [
        "obs_025",
        "obs_023",
        "obs_029",
        "obs_026",
        "obs_034",
        "obs_033",
    ],
    "estimator": [
        "obs_007",
        "obs_008",
    ],
    "phase_volume_flow": [
        "obs_037",
        "obs_038",
    ],
}

PHASES = {
    "P1_initial_baseline": (1, 30),
    "P2_positive_input": (31, 60),
    "P3_negative_input": (61, 90),
    "P4_recovery": (91, 100),
}


def load_data(path, expected_rows=100):
    """读取并检查数据。"""

    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")

    df = pd.read_csv(path)

    required = CANDIDATES + ["transition_index"]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path.name} 缺少字段：{missing}"
        )

    if len(df) != expected_rows:
        raise ValueError(
            f"{path.name} 应有{expected_rows}行，"
            f"实际{len(df)}行"
        )

    steps = pd.to_numeric(
        df["transition_index"],
        errors="coerce"
    )

    if steps.isna().any():
        raise ValueError("transition_index存在无效值")

    if steps.duplicated().any():
        raise ValueError("transition_index存在重复值")

    df = df.copy()
    df["transition_index"] = steps.astype(int)

    for col in CANDIDATES:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    values = df[CANDIDATES].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            f"{path.name}候选变量存在NaN或Inf"
        )

    return df


def calculate_correlations(df, dataset_name, phase_name):
    """计算Pearson和Spearman相关系数。"""

    data = df[CANDIDATES]
    sample_count = len(data)

    pearson = data.corr(method="pearson")
    spearman = data.corr(method="spearman")

    records = []

    for var_a, var_b in itertools.combinations(
        CANDIDATES, 2
    ):
        p = pearson.loc[var_a, var_b]
        s = spearman.loc[var_a, var_b]

        records.append({
            "dataset": dataset_name,
            "phase": phase_name,
            "variable_a": var_a,
            "variable_b": var_b,
            "sample_count": sample_count,
            "pearson_r": p,
            "spearman_rho": s,
            "small_sample_warning": sample_count < 20,
            "pearson_abs": abs(p),
            "spearman_abs": abs(s),
        })

    return records


def calculate_group_correlations(records):
    """给每对变量标注所属研究类别。"""

    group_map = {}

    for group_name, variables in CANDIDATE_GROUPS.items():
        for variable in variables:
            group_map[variable] = group_name

    result = []

    for row in records:
        a = row["variable_a"]
        b = row["variable_b"]

        group_a = group_map.get(a, "other")
        group_b = group_map.get(b, "other")

        if group_a == group_b:
            pair_type = "组内关联"
        else:
            pair_type = "跨组关联"

        result.append({
            **row,
            "group_a": group_a,
            "group_b": group_b,
            "pair_type": pair_type,
        })

    return result


def main():

    print("=" * 68)
    print("D模块 R-03 候选变量多阶段关联分析")
    print("=" * 68)

    all_records = []

    # 1. 分别分析两个基线
    baseline_frames = {}

    for dataset_name, path in BASE_FILES.items():

        df = load_data(path)

        baseline_frames[dataset_name] = df

        print(
            f"[PASS] {dataset_name}: "
            f"{len(df)}步，"
            f"{len(CANDIDATES)}个候选变量"
        )

        records = calculate_correlations(
            df,
            dataset_name,
            "full_baseline"
        )

        all_records.extend(records)

    # 2. 分析合法输入变化轨迹
    legal_df = load_data(LEGAL_FILE)

    expected_steps = np.arange(1, 101)

    if not np.array_equal(
        legal_df["transition_index"].to_numpy(),
        expected_steps
    ):
        raise ValueError(
            "合法输入轨迹步号不是连续1-100"
        )

    print(
        f"[PASS] legal_input_change.csv: "
        f"{len(legal_df)}步"
    )

    phase_frames = {}

    for phase_name, (start, end) in PHASES.items():

        mask = legal_df["transition_index"].between(
            start, end
        )

        phase_df = legal_df.loc[mask].copy()

        expected_count = end - start + 1

        if len(phase_df) != expected_count:
            raise ValueError(
                f"{phase_name}样本数异常"
            )

        phase_frames[phase_name] = phase_df

        print(
            f"[PASS] {phase_name}: "
            f"{len(phase_df)}条"
        )

        records = calculate_correlations(
            phase_df,
            "legal_input_change",
            phase_name
        )

        all_records.extend(records)

    # 3. 生成完整相关系数表
    all_records = calculate_group_correlations(
        all_records
    )

    result = pd.DataFrame(all_records)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pearson
    pearson_output = (
        OUT_DIR / "r03_candidate_pearson.csv"
    )

    result[[
        "dataset",
        "phase",
        "variable_a",
        "variable_b",
        "group_a",
        "group_b",
        "pair_type",
        "sample_count",
        "pearson_r",
        "small_sample_warning",
    ]].to_csv(
        pearson_output,
        index=False,
        encoding="utf-8-sig"
    )

    # Spearman
    spearman_output = (
        OUT_DIR / "r03_candidate_spearman.csv"
    )

    result[[
        "dataset",
        "phase",
        "variable_a",
        "variable_b",
        "group_a",
        "group_b",
        "pair_type",
        "sample_count",
        "spearman_rho",
        "small_sample_warning",
    ]].to_csv(
        spearman_output,
        index=False,
        encoding="utf-8-sig"
    )

    # 4. 生成跨阶段比较表
    comparison = result.pivot_table(
        index=["variable_a", "variable_b"],
        columns=["dataset", "phase"],
        values="pearson_r"
    ).reset_index()

    comparison_output = (
        OUT_DIR
        / "r03_candidate_correlation_comparison.csv"
    )

    comparison.to_csv(
        comparison_output,
        index=False,
        encoding="utf-8-sig"
    )

    # 5. 重点变量组分析
    group_records = []

    for dataset_name, phase_name, df in [
        (row["dataset"], row["phase"], frame)
        for row, frame in [
            *[
                (
                    {
                        "dataset": name,
                        "phase": "full_baseline"
                    },
                    frame
                )
                for name, frame in baseline_frames.items()
            ],
            *[
                (
                    {
                        "dataset": "legal_input_change",
                        "phase": phase_name
                    },
                    frame
                )
                for phase_name, frame in phase_frames.items()
            ],
        ]
    ]:

        for group_name, variables in CANDIDATE_GROUPS.items():

            for var_a, var_b in itertools.combinations(
                variables, 2
            ):

                pearson = df[[
                    var_a, var_b
                ]].corr(method="pearson").iloc[0, 1]

                spearman = df[[
                    var_a, var_b
                ]].corr(method="spearman").iloc[0, 1]

                group_records.append({
                    "dataset": dataset_name,
                    "phase": phase_name,
                    "group": group_name,
                    "variable_a": var_a,
                    "variable_b": var_b,
                    "sample_count": len(df),
                    "pearson_r": pearson,
                    "spearman_rho": spearman,
                })

    group_result = pd.DataFrame(group_records)

    group_output = (
        OUT_DIR / "r03_candidate_group_correlations.csv"
    )

    group_result.to_csv(
        group_output,
        index=False,
        encoding="utf-8-sig"
    )

    # 6. 生成报告
    lines = [
        "# D模块 R-03 候选变量多阶段关联分析报告 v0.1",
        "",
        "## 1. 数据来源",
        "",
        "- Seed 1101 基线：100步",
        "- Seed 1102 基线：100步",
        "- 合法输入变化轨迹：100步",
        "- 候选变量：10个",
        "- 排除obs_000时间索引",
        "",
        "## 2. 分析方法",
        "",
        "- Pearson线性相关系数",
        "- Spearman秩相关系数",
        "- 两条基线分别分析",
        "- 合法输入变化按P1-P4分阶段分析",
        "",
        "## 3. 样本数量与限制",
        "",
        "| 数据集 | 阶段 | 样本数 |",
        "|---|---|---:|",
    ]

    for dataset_name, df in baseline_frames.items():
        lines.append(
            f"| {dataset_name} | full_baseline | {len(df)} |"
        )

    for phase_name, df in phase_frames.items():
        lines.append(
            f"| legal_input_change | {phase_name} | {len(df)} |"
        )

    lines.extend([
        "",
        "## 4. 重点变量组",
        "",
        "| 变量组 | 变量 |",
        "|---|---|",
    ])

    for group_name, variables in CANDIDATE_GROUPS.items():
        lines.append(
            f"| {group_name} | {', '.join(variables)} |"
        )

    lines.extend([
        "",
        "## 5. 关联程度较高的变量对",
        "",
        "以下按照Pearson相关系数绝对值排序，",
        "仅作为描述性探索，不代表因果关系。",
        "",
        "| 数据集 | 阶段 | 变量A | 变量B | Pearson r | Spearman rho | 样本数 |",
        "|---|---|---|---|---:|---:|---:|",
    ])

    for (dataset_name, phase_name), subset in result.groupby(
        ["dataset", "phase"]
    ):

        top = subset.nlargest(5, "pearson_abs")

        for _, row in top.iterrows():
            lines.append(
                f"| {dataset_name} "
                f"| {phase_name} "
                f"| {row['variable_a']} "
                f"| {row['variable_b']} "
                f"| {row['pearson_r']:.6f} "
                f"| {row['spearman_rho']:.6f} "
                f"| {row['sample_count']} |"
            )

    lines.extend([
        "",
        "## 6. 特别关注的变量关系",
        "",
        "1. obs_007 与 obs_008：状态估计器坐标关系。",
        "2. obs_037 与 obs_038：有机相与水相体积流量关系。",
        "3. 水相组分摩尔流量之间的关联。",
        "",
        "## 7. 结果解释限制",
        "",
        "1. 相关性不代表因果关系。",
        "2. 每条基线只包含一个随机种子的轨迹。",
        "3. P4只有10个样本，相关系数不稳定。",
        "4. 阶段内的时间趋势可能影响相关系数。",
        "5. 不同阶段相关系数变化不能直接证明工艺机制变化。",
        "6. 本分析不建立工业安全阈值或危险等级。",
        "",
        "## 8. 输出文件",
        "",
        "- experiments/r03_candidate_pearson.csv",
        "- experiments/r03_candidate_spearman.csv",
        "- experiments/r03_candidate_correlation_comparison.csv",
        "- experiments/r03_candidate_group_correlations.csv",
    ])

    REPORT.parent.mkdir(parents=True, exist_ok=True)

    REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    # 7. 控制台摘要
    print("\n" + "=" * 68)
    print("重点变量组关联结果")
    print("=" * 68)

    selected = group_result[
        (
            (group_result["variable_a"] == "obs_007")
            & (group_result["variable_b"] == "obs_008")
        )
        |
        (
            (group_result["variable_a"] == "obs_037")
            & (group_result["variable_b"] == "obs_038")
        )
    ]

    print(
        selected.to_string(index=False)
    )

    print("\n各数据阶段Pearson绝对值最高的前5组：")

    for (dataset, phase), subset in result.groupby(
        ["dataset", "phase"]
    ):
        print(f"\n[{dataset} / {phase}]")

        print(
            subset.nlargest(5, "pearson_abs")[
                [
                    "variable_a",
                    "variable_b",
                    "pearson_r",
                    "spearman_rho",
                    "sample_count",
                ]
            ].to_string(index=False)
        )

    print("\n" + "=" * 68)
    print("多阶段关联分析完成！")
    print("=" * 68)

    print(f"Pearson结果：{pearson_output}")
    print(f"Spearman结果：{spearman_output}")
    print(f"阶段比较表：{comparison_output}")
    print(f"分组关联表：{group_output}")
    print(f"分析报告：{REPORT}")


if __name__ == "__main__":
    main()
