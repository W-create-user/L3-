
"""
D模块 R-03 时间趋势校正与关联稳健性验证 v0.2

分析内容：
1. 原始 Pearson / Spearman 相关性
2. 线性时间去趋势 Pearson / Spearman
3. 一阶差分 Pearson / Spearman
4. 双基线分别分析
5. 合法输入变化 P1-P4 分阶段分析
6. 重点变量关系分析
7. 输出完整关联结果和分析报告

修订：
- 增加 obs_037 与 obs_038 重点关联分析
- 使用 NumPy 实现 Spearman，避免 SciPy DLL 问题
- 保留原始数据和之前的分析逻辑

注意：
相关性不代表因果关系。
P4只有10个样本，差分后只有9个样本。
本程序不建立工业安全阈值。
"""

from pathlib import Path
import itertools
import numpy as np
import pandas as pd


# ============================================================
# 1. 路径配置
# ============================================================

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
    / "D_R03时间趋势校正与关联稳健性验证报告_v0.2.md"
)


# ============================================================
# 2. 候选变量
# ============================================================

VARIABLES = [
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


# 重点变量关系
# itertools.combinations按照VARIABLES顺序生成变量组合
# 因此obs_038与obs_037在这里能够正确纳入

FOCUS_PAIRS = [
    ("obs_007", "obs_008"),
    ("obs_038", "obs_037"),
    ("obs_025", "obs_033"),
    ("obs_025", "obs_023"),
    ("obs_023", "obs_033"),
    ("obs_029", "obs_026"),
]


PHASES = {
    "P1_initial_baseline": (1, 30),
    "P2_positive_input": (31, 60),
    "P3_negative_input": (61, 90),
    "P4_recovery": (91, 100),
}


# ============================================================
# 3. 读取数据
# ============================================================

def load_data(path):

    if not path.exists():
        raise FileNotFoundError(
            f"数据文件不存在：{path}"
        )

    df = pd.read_csv(path)

    required = VARIABLES + ["transition_index"]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path.name} 缺少字段：{missing}"
        )

    if len(df) != 100:
        raise ValueError(
            f"{path.name} 应有100行，实际为{len(df)}行"
        )

    df = df.copy()

    df["transition_index"] = pd.to_numeric(
        df["transition_index"],
        errors="raise"
    )

    for col in VARIABLES:
        df[col] = pd.to_numeric(
            df[col],
            errors="raise"
        )

    values = df[VARIABLES].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            f"{path.name}存在NaN或Inf"
        )

    if df["transition_index"].duplicated().any():
        raise ValueError(
            f"{path.name}存在重复时间步"
        )

    return df


# ============================================================
# 4. 相关系数计算
# ============================================================

def correlation_value(x, y, method):
    """
    使用NumPy计算Pearson和Spearman。

    不依赖SciPy。
    Spearman通过平均秩转换后计算Pearson。
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    if method == "spearman":

        x = pd.Series(x).rank(
            method="average"
        ).to_numpy()

        y = pd.Series(y).rank(
            method="average"
        ).to_numpy()

    elif method != "pearson":

        raise ValueError(
            f"不支持的相关系数方法：{method}"
        )

    result = np.corrcoef(x, y)[0, 1]

    return float(result)


# ============================================================
# 5. 线性时间去趋势
# ============================================================

def detrend_dataframe(df):

    steps = df["transition_index"].to_numpy(
        dtype=float
    )

    x = steps - steps.mean()

    residuals = pd.DataFrame(
        index=df.index
    )

    for variable in VARIABLES:

        values = df[variable].to_numpy(
            dtype=float
        )

        if np.std(values) == 0:

            residuals[variable] = np.zeros(
                len(values)
            )

            continue

        slope, intercept = np.polyfit(
            x,
            values,
            1
        )

        fitted = slope * x + intercept

        residuals[variable] = values - fitted

    return residuals


# ============================================================
# 6. 一阶差分
# ============================================================

def first_difference(df):

    return (
        df[VARIABLES]
        .diff()
        .dropna()
    )


# ============================================================
# 7. 单个阶段分析
# ============================================================

def analyze_segment(df, dataset, phase):

    df = df.reset_index(drop=True)

    raw = df[VARIABLES]

    detrended = detrend_dataframe(df)

    differenced = first_difference(df)

    records = []

    for a, b in itertools.combinations(
        VARIABLES, 2
    ):

        for method in [
            "pearson",
            "spearman"
        ]:

            raw_r = correlation_value(
                raw[a],
                raw[b],
                method
            )

            detrended_r = correlation_value(
                detrended[a],
                detrended[b],
                method
            )

            difference_r = correlation_value(
                differenced[a],
                differenced[b],
                method
            )

            records.append({

                "dataset": dataset,

                "phase": phase,

                "variable_a": a,

                "variable_b": b,

                "method": method,

                "raw_correlation": raw_r,

                "detrended_correlation": detrended_r,

                "difference_correlation": difference_r,

                "raw_sample_count": len(raw),

                "detrended_sample_count": len(detrended),

                "difference_sample_count": len(differenced),

            })

    return records


# ============================================================
# 8. 筛选重点变量关系
# ============================================================

def select_focus_pairs(result):

    conditions = []

    for a, b in FOCUS_PAIRS:

        condition = (
            (result["variable_a"] == a)
            & (result["variable_b"] == b)
        )

        conditions.append(condition)

    mask = conditions[0]

    for condition in conditions[1:]:
        mask = mask | condition

    return result.loc[mask].copy()


# ============================================================
# 9. 生成Markdown报告
# ============================================================

def generate_report(result, focus):

    lines = [

        "# D模块 R-03 时间趋势校正与关联稳健性验证报告 v0.2",

        "",

        "## 1. 数据来源",

        "",

        "- Seed 1101基线：100步",

        "- Seed 1102基线：100步",

        "- 合法输入变化轨迹：100步",

        "- 候选变量：10个",

        "- 不包含obs_000时间索引",

        "",

        "## 2. 分析方法",

        "",

        "1. 原始Pearson相关系数",

        "2. 原始Spearman秩相关系数",

        "3. 线性时间去趋势Pearson相关系数",

        "4. 线性时间去趋势Spearman相关系数",

        "5. 一阶差分Pearson相关系数",

        "6. 一阶差分Spearman相关系数",

        "",

        "## 3. 重点变量关联结果",

        "",

        "| 数据集 | 阶段 | 变量A | 变量B | 方法 | 原始相关 | 去趋势相关 | 差分相关 | 样本数 |",

        "|---|---|---|---|---|---:|---:|---:|---:|",

    ]

    for _, row in focus.iterrows():

        lines.append(

            f"| {row['dataset']} "

            f"| {row['phase']} "

            f"| {row['variable_a']} "

            f"| {row['variable_b']} "

            f"| {row['method']} "

            f"| {row['raw_correlation']:.6f} "

            f"| {row['detrended_correlation']:.6f} "

            f"| {row['difference_correlation']:.6f} "

            f"| {row['raw_sample_count']} |"

        )

    lines.extend([

        "",

        "## 4. 重点研究变量",

        "",

        "| 变量组合 | 研究内容 |",

        "|---|---|",

        "| obs_007 / obs_008 | 状态估计器坐标关联 |",

        "| obs_037 / obs_038 | 有机相与水相体积流量关联 |",

        "| obs_025 / obs_033 | 苯乙酰氯与托品酯水相流量关联 |",

        "| obs_025 / obs_023 | 苯乙酰氯与托品水相流量关联 |",

        "| obs_023 / obs_033 | 托品与托品酯水相流量关联 |",

        "| obs_029 / obs_026 | 氢氧化钠与中间产物水相流量关联 |",

        "",

        "## 5. 结果解释限制",

        "",

        "1. 相关性不代表因果关系。",

        "2. 线性去趋势不能消除全部时间依赖。",

        "3. 一阶差分可能放大噪声。",

        "4. 两条基线分别只有一个种子轨迹。",

        "5. P4只有10个观测点，差分后只有9个点。",

        "6. 不能仅凭高相关性确定工艺机制。",

        "7. 本分析不建立工业安全阈值。",

        "",

        "## 6. 输出文件",

        "",

        "- experiments/r03_detrended_correlation_all.csv",

        "- experiments/r03_detrended_correlation_focus.csv",

        "- experiments/r03_detrended_correlation_comparison.csv",

        "",

        "## 7. 研究结论",

        "",

        "本报告提供候选变量关联的统计描述，",

        "后续需要结合工艺知识、更多随机种子及异常样本",

        "验证关联的稳定性和实际监管价值。",

    ])

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


# ============================================================
# 10. 主程序
# ============================================================

def main():

    print("=" * 68)

    print(
        "D模块 R-03 时间趋势校正与关联稳健性验证 v0.2"
    )

    print("=" * 68)

    all_records = []

    # --------------------------------------------------------
    # 双基线分析
    # --------------------------------------------------------

    for dataset_name, path in BASE_FILES.items():

        df = load_data(path)

        print(
            f"[PASS] {dataset_name}: {len(df)}步"
        )

        records = analyze_segment(
            df,
            dataset_name,
            "full_baseline"
        )

        all_records.extend(records)

    # --------------------------------------------------------
    # 合法输入变化轨迹
    # --------------------------------------------------------

    legal = load_data(LEGAL_FILE)

    expected_steps = np.arange(1, 101)

    if not np.array_equal(
        legal["transition_index"].to_numpy(),
        expected_steps
    ):

        raise ValueError(
            "合法输入变化轨迹步号不是连续1-100"
        )

    print(
        "[PASS] legal_input_change.csv: 100步"
    )

    # --------------------------------------------------------
    # P1-P4阶段分析
    # --------------------------------------------------------

    for phase_name, (start, end) in PHASES.items():

        mask = legal[
            "transition_index"
        ].between(start, end)

        segment = legal.loc[mask].copy()

        expected_count = end - start + 1

        if len(segment) != expected_count:

            raise ValueError(
                f"{phase_name}样本数异常"
            )

        print(
            f"[PASS] {phase_name}: {len(segment)}步"
        )

        records = analyze_segment(
            segment,
            "legal_input_change",
            phase_name
        )

        all_records.extend(records)

    # --------------------------------------------------------
    # 汇总结果
    # --------------------------------------------------------

    result = pd.DataFrame(all_records)

    result["detrending_change"] = (
        result["detrended_correlation"]
        - result["raw_correlation"]
    )

    result["difference_change"] = (
        result["difference_correlation"]
        - result["raw_correlation"]
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # 全部结果
    all_output = (
        OUT_DIR
        / "r03_detrended_correlation_all.csv"
    )

    result.to_csv(
        all_output,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 重点变量关系
    # --------------------------------------------------------

    focus = select_focus_pairs(result)

    focus_output = (
        OUT_DIR
        / "r03_detrended_correlation_focus.csv"
    )

    focus.to_csv(
        focus_output,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 比较结果
    # --------------------------------------------------------

    comparison_output = (
        OUT_DIR
        / "r03_detrended_correlation_comparison.csv"
    )

    result.to_csv(
        comparison_output,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 生成报告
    # --------------------------------------------------------

    generate_report(
        result,
        focus
    )

    # --------------------------------------------------------
    # 输出重点结果
    # --------------------------------------------------------

    print("\n" + "=" * 68)

    print("重点变量去趋势前后对比")

    print("=" * 68)

    pearson_focus = focus[
        focus["method"] == "pearson"
    ]

    print(

        pearson_focus[[

            "dataset",

            "phase",

            "variable_a",

            "variable_b",

            "raw_correlation",

            "detrended_correlation",

            "difference_correlation",

        ]].to_string(index=False)

    )

    # --------------------------------------------------------
    # 检查两相流量组合
    # --------------------------------------------------------

    phase_volume = focus[
        (
            (focus["variable_a"] == "obs_038")
            &
            (focus["variable_b"] == "obs_037")
        )
        |
        (
            (focus["variable_a"] == "obs_037")
            &
            (focus["variable_b"] == "obs_038")
        )
    ]

    print("\n" + "=" * 68)

    print("两相流量关联专项检查")

    print("=" * 68)

    if phase_volume.empty:

        print(
            "[WARNING] obs_037与obs_038没有生成关联记录"
        )

    else:

        print(

            phase_volume[[

                "dataset",

                "phase",

                "method",

                "raw_correlation",

                "detrended_correlation",

                "difference_correlation",

            ]].to_string(index=False)

        )

    # --------------------------------------------------------
    # 完成
    # --------------------------------------------------------

    print("\n" + "=" * 68)

    print("时间趋势校正分析完成！")

    print("=" * 68)

    print(f"全部结果：{all_output}")

    print(f"重点变量：{focus_output}")

    print(f"对比结果：{comparison_output}")

    print(f"分析报告：{REPORT}")


if __name__ == "__main__":

    main()
