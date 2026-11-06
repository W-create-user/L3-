
"""
D模块 R-03 候选变量关联稳健性分类 v0.2

修订内容：
1. 修复证据不足型的错误分类逻辑
2. 区分数据不完整和未满足预设分类条件
3. 保留双基线和四阶段分析
4. 保留五种多标签分类
5. 增加未满足预设分类条件类别
6. 自动生成CSV和Markdown报告

注意：
所有分类阈值均为探索性统计判据，
不是工业安全阈值。
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 1. 路径配置
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "experiments"
    / "r03_detrended_correlation_all.csv"
)

OUTPUT_DIR = ROOT / "experiments"

OUTPUT_CSV = (
    OUTPUT_DIR
    / "r03_correlation_stability.csv"
)

REPORT_FILE = (
    ROOT
    / "docs"
    / "D_R03关联稳健性分类报告_v0.2.md"
)


# ============================================================
# 2. 研究对象和阶段
# ============================================================

BASELINE_NAMES = [
    "baseline_seed1101",
    "baseline_seed1102",
]

PHASE_NAMES = [
    "P1_initial_baseline",
    "P2_positive_input",
    "P3_negative_input",
    "P4_recovery",
]

FOCUS_PAIRS = [
    ("obs_007", "obs_008"),
    ("obs_038", "obs_037"),
    ("obs_025", "obs_033"),
    ("obs_025", "obs_023"),
    ("obs_023", "obs_033"),
    ("obs_029", "obs_026"),
]


# ============================================================
# 3. 探索性分类参数
# ============================================================

STRONG_CORRELATION = 0.80

CHANGE_THRESHOLD = 0.30

TREND_THRESHOLD = 0.20

SEED_THRESHOLD = 0.30

MIN_SAMPLE_COUNT = 10

STABLE_PHASE_SPREAD = 0.25

STABLE_TREND_CHANGE = 0.20


# ============================================================
# 4. 数据读取
# ============================================================

def load_data():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"找不到输入文件：{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "dataset",
        "phase",
        "variable_a",
        "variable_b",
        "method",
        "raw_correlation",
        "detrended_correlation",
        "difference_correlation",
        "raw_sample_count",
        "detrended_sample_count",
        "difference_sample_count",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"输入文件缺少字段：{missing}"
        )

    # 仅使用Pearson进行主要分类
    df = df[
        df["method"] == "pearson"
    ].copy()

    numeric_columns = [
        "raw_correlation",
        "detrended_correlation",
        "difference_correlation",
        "raw_sample_count",
        "detrended_sample_count",
        "difference_sample_count",
    ]

    for col in numeric_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # 统一变量组合顺序
    df["pair"] = df.apply(
        lambda row: tuple(
            sorted([
                row["variable_a"],
                row["variable_b"]
            ])
        ),
        axis=1
    )

    if df.empty:

        raise ValueError(
            "没有找到Pearson分析结果"
        )

    return df


# ============================================================
# 5. 数据提取
# ============================================================

def get_value(group, dataset, phase, column):

    selected = group[
        (group["dataset"] == dataset)
        &
        (group["phase"] == phase)
    ]

    if selected.empty:

        return np.nan

    if len(selected) > 1:

        raise ValueError(
            f"发现重复记录：{dataset}/{phase}/{column}"
        )

    return float(
        selected.iloc[0][column]
    )


def get_sample_count(group, dataset, phase):

    selected = group[
        (group["dataset"] == dataset)
        &
        (group["phase"] == phase)
    ]

    if selected.empty:

        return 0

    if len(selected) > 1:

        raise ValueError(
            f"发现重复样本记录：{dataset}/{phase}"
        )

    return int(
        selected.iloc[0]["raw_sample_count"]
    )


def sign_of(value):

    if not np.isfinite(value):

        return 0

    if value > 0:

        return 1

    if value < 0:

        return -1

    return 0


def safe_float(value):

    if np.isfinite(value):

        return float(value)

    return np.nan


# ============================================================
# 6. 单个变量组合分析
# ============================================================

def analyze_pair(group):

    pair = group["pair"].iloc[0]

    variable_a, variable_b = pair

    labels = []

    evidence_notes = []

    # --------------------------------------------------------
    # 双基线数据
    # --------------------------------------------------------

    baseline_raw = []

    baseline_detrended = []

    baseline_difference = []

    baseline_counts = []

    for dataset in BASELINE_NAMES:

        raw = get_value(
            group,
            dataset,
            "full_baseline",
            "raw_correlation"
        )

        detrended = get_value(
            group,
            dataset,
            "full_baseline",
            "detrended_correlation"
        )

        difference = get_value(
            group,
            dataset,
            "full_baseline",
            "difference_correlation"
        )

        count = get_sample_count(
            group,
            dataset,
            "full_baseline"
        )

        baseline_raw.append(raw)

        baseline_detrended.append(detrended)

        baseline_difference.append(difference)

        baseline_counts.append(count)

    # --------------------------------------------------------
    # 四阶段数据
    # --------------------------------------------------------

    phase_raw = []

    phase_detrended = []

    phase_difference = []

    phase_counts = []

    for phase in PHASE_NAMES:

        raw = get_value(
            group,
            "legal_input_change",
            phase,
            "raw_correlation"
        )

        detrended = get_value(
            group,
            "legal_input_change",
            phase,
            "detrended_correlation"
        )

        difference = get_value(
            group,
            "legal_input_change",
            phase,
            "difference_correlation"
        )

        count = get_sample_count(
            group,
            "legal_input_change",
            phase
        )

        phase_raw.append(raw)

        phase_detrended.append(detrended)

        phase_difference.append(difference)

        phase_counts.append(count)

    baseline_raw = np.array(
        baseline_raw,
        dtype=float
    )

    baseline_detrended = np.array(
        baseline_detrended,
        dtype=float
    )

    baseline_difference = np.array(
        baseline_difference,
        dtype=float
    )

    phase_raw = np.array(
        phase_raw,
        dtype=float
    )

    phase_detrended = np.array(
        phase_detrended,
        dtype=float
    )

    phase_difference = np.array(
        phase_difference,
        dtype=float
    )

    # --------------------------------------------------------
    # 数据完整性检查
    # --------------------------------------------------------

    complete_baseline = (
        np.isfinite(baseline_detrended).all()
        and np.isfinite(baseline_raw).all()
        and np.isfinite(baseline_difference).all()
    )

    complete_phases = (
        np.isfinite(phase_detrended).all()
        and np.isfinite(phase_raw).all()
        and np.isfinite(phase_difference).all()
    )

    min_count = min(
        baseline_counts + phase_counts
    )

    evidence_insufficient = (
        not complete_baseline
        or not complete_phases
        or min_count < MIN_SAMPLE_COUNT
    )

    if evidence_insufficient:

        labels.append("证据不足型")

        evidence_notes.append(
            "部分数据缺失、统计值无效或样本量不足"
        )

    # --------------------------------------------------------
    # 种子敏感性
    # --------------------------------------------------------

    seed_signs = [
        sign_of(x)
        for x in baseline_detrended
    ]

    seed_gap = np.nan

    seed_sensitive = False

    if complete_baseline:

        seed_gap = abs(
            baseline_detrended[0]
            - baseline_detrended[1]
        )

        seed_sensitive = (
            seed_signs[0] != seed_signs[1]
            or seed_gap >= SEED_THRESHOLD
        )

        if seed_sensitive:

            labels.append("种子敏感型")

            evidence_notes.append(
                "两个随机种子之间关联方向或强度存在明显差异"
            )

    # --------------------------------------------------------
    # 阶段敏感性
    # --------------------------------------------------------

    valid_phase = phase_detrended[
        np.isfinite(phase_detrended)
    ]

    phase_spread = np.nan

    phase_sensitive = False

    phase_sign_change = False

    if len(valid_phase) >= 2:

        phase_spread = (
            np.max(valid_phase)
            - np.min(valid_phase)
        )

        phase_signs = [
            sign_of(x)
            for x in valid_phase
        ]

        nonzero_signs = set(
            x for x in phase_signs
            if x != 0
        )

        phase_sign_change = (
            len(nonzero_signs) > 1
        )

        phase_sensitive = (
            phase_spread >= CHANGE_THRESHOLD
            or phase_sign_change
        )

        if phase_sensitive:

            labels.append("阶段敏感型")

            evidence_notes.append(
                "不同输入阶段的关联强度或方向存在明显变化"
            )

    # --------------------------------------------------------
    # 时间趋势影响
    # --------------------------------------------------------

    trend_changes = []

    for dataset in BASELINE_NAMES:

        raw = get_value(
            group,
            dataset,
            "full_baseline",
            "raw_correlation"
        )

        detrended = get_value(
            group,
            dataset,
            "full_baseline",
            "detrended_correlation"
        )

        if (
            np.isfinite(raw)
            and np.isfinite(detrended)
        ):

            trend_changes.append(
                abs(detrended - raw)
            )

    for phase in PHASE_NAMES:

        raw = get_value(
            group,
            "legal_input_change",
            phase,
            "raw_correlation"
        )

        detrended = get_value(
            group,
            "legal_input_change",
            phase,
            "detrended_correlation"
        )

        if (
            np.isfinite(raw)
            and np.isfinite(detrended)
        ):

            trend_changes.append(
                abs(detrended - raw)
            )

    max_trend_change = (
        max(trend_changes)
        if trend_changes
        else np.nan
    )

    trend_sensitive = (
        np.isfinite(max_trend_change)
        and max_trend_change >= TREND_THRESHOLD
    )

    if trend_sensitive:

        labels.append("时间趋势影响型")

        evidence_notes.append(
            "去趋势前后相关系数变化达到探索性判据"
        )

    # --------------------------------------------------------
    # 稳定关联候选
    # --------------------------------------------------------

    all_detrended = np.concatenate([
        baseline_detrended,
        phase_detrended
    ])

    valid_all = all_detrended[
        np.isfinite(all_detrended)
    ]

    same_direction = False

    strong_all = False

    stable_candidate = False

    if (
        complete_baseline
        and complete_phases
        and len(valid_all) == 6
    ):

        signs = [
            sign_of(x)
            for x in valid_all
        ]

        same_direction = (
            len(set(signs)) == 1
            and signs[0] != 0
        )

        strong_all = (
            np.min(
                np.abs(valid_all)
            ) >= STRONG_CORRELATION
        )

        stable_candidate = (

            same_direction

            and strong_all

            and seed_gap < SEED_THRESHOLD

            and phase_spread < STABLE_PHASE_SPREAD

            and max_trend_change < STABLE_TREND_CHANGE

        )

        if stable_candidate:

            labels.append("稳定关联候选")

            evidence_notes.append(
                "双基线和四阶段方向一致，"
                "关联强度及去趋势结果相对稳定"
            )

    # --------------------------------------------------------
    # v0.2核心修复：
    # 区分证据不足与未满足预设分类条件
    # --------------------------------------------------------

    if not labels:

        if evidence_insufficient:

            labels.append("证据不足型")

            evidence_notes.append(
                "数据缺失或样本量不足，无法完成可靠分类"
            )

        else:

            labels.append("未满足预设分类条件")

            evidence_notes.append(
                "数据完整，但未达到当前设定的"
                "稳定关联、阶段敏感、趋势影响或种子敏感判据"
            )

    # --------------------------------------------------------
    # 差分关联补充指标
    # --------------------------------------------------------

    valid_diff = phase_difference[
        np.isfinite(phase_difference)
    ]

    diff_mean_abs = (
        float(np.mean(np.abs(valid_diff)))
        if len(valid_diff) > 0
        else np.nan
    )

    # --------------------------------------------------------
    # 返回结果
    # --------------------------------------------------------

    return {

        "variable_a": variable_a,

        "variable_b": variable_b,

        "classification": "；".join(labels),

        "classification_count": len(labels),

        "baseline_seed1101_detrended":
            safe_float(baseline_detrended[0]),

        "baseline_seed1102_detrended":
            safe_float(baseline_detrended[1]),

        "baseline_seed_gap":
            safe_float(seed_gap),

        "P1_detrended":
            safe_float(phase_detrended[0]),

        "P2_detrended":
            safe_float(phase_detrended[1]),

        "P3_detrended":
            safe_float(phase_detrended[2]),

        "P4_detrended":
            safe_float(phase_detrended[3]),

        "phase_spread":
            safe_float(phase_spread),

        "max_trend_change":
            safe_float(max_trend_change),

        "mean_abs_phase_difference_correlation":
            safe_float(diff_mean_abs),

        "same_direction":
            same_direction,

        "strong_all":
            strong_all,

        "stable_candidate":
            stable_candidate,

        "evidence_insufficient":
            evidence_insufficient,

        "minimum_sample_count":
            min_count,

        "evidence_notes":
            "；".join(evidence_notes),

    }


# ============================================================
# 7. 生成Markdown报告
# ============================================================

def format_value(value):

    if pd.isna(value):

        return "N/A"

    return f"{value:.6f}"


def generate_report(result):

    lines = [

        "# D模块 R-03 关联稳健性分类报告 v0.2",

        "",

        "## 一、分析目的",

        "",

        "综合双基线、四阶段、时间趋势校正及一阶差分结果，",

        "对候选变量组合进行探索性关联稳健性分类。",

        "",

        "## 二、分类方法",

        "",

        "| 分类 | 判断依据 |",

        "|---|---|",

        "| 稳定关联候选 | 双基线和四阶段方向一致，强度较高且变化较小 |",

        "| 阶段敏感型 | 阶段间相关强度变化或方向变化 |",

        "| 时间趋势影响型 | 原始和去趋势相关系数差异达到判据 |",

        "| 种子敏感型 | 两个基线种子结果方向或强度差异明显 |",

        "| 证据不足型 | 数据缺失、统计值无效或样本量不足 |",

        "| 未满足预设分类条件 | 数据完整但没有达到其他分类条件 |",

        "",

        "## 三、候选变量分类结果",

        "",

        "| 变量组合 | 分类 | 基线差异 | 阶段波动 | 最大趋势影响 |",

        "|---|---|---:|---:|---:|",

    ]

    for _, row in result.iterrows():

        lines.append(

            f"| {row['variable_a']} / {row['variable_b']} "

            f"| {row['classification']} "

            f"| {format_value(row['baseline_seed_gap'])} "

            f"| {format_value(row['phase_spread'])} "

            f"| {format_value(row['max_trend_change'])} |"

        )

    lines.extend([

        "",

        "## 四、变量组合详细结果",

        "",

    ])

    for _, row in result.iterrows():

        lines.extend([

            f"### {row['variable_a']} / {row['variable_b']}",

            "",

            f"- 分类：{row['classification']}",

            f"- Seed 1101：{format_value(row['baseline_seed1101_detrended'])}",

            f"- Seed 1102：{format_value(row['baseline_seed1102_detrended'])}",

            f"- P1：{format_value(row['P1_detrended'])}",

            f"- P2：{format_value(row['P2_detrended'])}",

            f"- P3：{format_value(row['P3_detrended'])}",

            f"- P4：{format_value(row['P4_detrended'])}",

            f"- 证据说明：{row['evidence_notes']}",

            "",

        ])

    lines.extend([

        "## 五、分类数量统计",

        "",

    ])

    categories = [

        "稳定关联候选",

        "阶段敏感型",

        "时间趋势影响型",

        "种子敏感型",

        "证据不足型",

        "未满足预设分类条件",

    ]

    for category in categories:

        count = result[
            result["classification"].str.contains(
                category,
                regex=False
            )
        ].shape[0]

        lines.append(
            f"- {category}：{count}组"
        )

    lines.extend([

        "",

        "注意：分类采用多标签机制，各类别数量可能重叠。",

        "",

        "## 六、方法限制",

        "",

        "1. 分类阈值是探索性统计判据，不是工业安全阈值。",

        "2. 两个随机种子不足以证明统计上的跨种子稳健性。",

        "3. P4只有10个样本，一阶差分只有9个样本。",

        "4. 相关关系不代表因果关系。",

        "5. obs_007和obs_008尚未确认对应的现场物理量。",

        "6. 当前数据没有提供真实异常状态正样本。",

        "7. 本报告不能作为工业安全监管规则的直接依据。",

        "",

        "## 七、后续工作",

        "",

        "1. 增加不同随机种子的基线实验。",

        "2. 补充更多合法输入组合。",

        "3. 结合工艺机理核对变量关系。",

        "4. 设计候选异常指标。",

        "5. 获取真实异常样本验证指标敏感性。",

        "6. 在验证前不将探索性相关判据作为安全阈值。",

        "",

        "## 八、输出文件",

        "",

        "- experiments/r03_correlation_stability.csv",

        "- docs/D_R03关联稳健性分类报告_v0.2.md",

    ])

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


# ============================================================
# 8. 主程序
# ============================================================

def main():

    print("=" * 68)

    print(
        "D模块 R-03 候选变量关联稳健性分类 v0.2"
    )

    print("=" * 68)

    df = load_data()

    print(
        f"[PASS] Pearson数据读取成功：{len(df)}条"
    )

    results = []

    for pair in FOCUS_PAIRS:

        selected = df[
            df["pair"] == tuple(sorted(pair))
        ].copy()

        if selected.empty:

            print(
                f"[WARNING] 缺少变量组合：{pair}"
            )

            continue

        result = analyze_pair(selected)

        results.append(result)

        print(
            f"[PASS] {pair[0]} / {pair[1]} "
            f"分类完成：{result['classification']}"
        )

    if not results:

        raise RuntimeError(
            "没有成功分析任何变量组合"
        )

    result_df = pd.DataFrame(results)

    result_df = result_df.sort_values(
        by=[
            "evidence_insufficient",
            "classification_count"
        ],
        ascending=[True, False]
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    result_df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    generate_report(result_df)

    # --------------------------------------------------------
    # 输出分类结果
    # --------------------------------------------------------

    print("\n" + "=" * 68)

    print("关联稳健性分类结果")

    print("=" * 68)

    for _, row in result_df.iterrows():

        print(
            f"\n{row['variable_a']} / "
            f"{row['variable_b']}"
        )

        print(
            f"分类：{row['classification']}"
        )

        print(
            f"基线差异：{format_value(row['baseline_seed_gap'])}"
        )

        print(
            f"阶段波动：{format_value(row['phase_spread'])}"
        )

        print(
            f"最大趋势影响：{format_value(row['max_trend_change'])}"
        )

    # --------------------------------------------------------
    # 分类统计
    # --------------------------------------------------------

    print("\n" + "=" * 68)

    print("分类统计")

    print("=" * 68)

    categories = [

        "稳定关联候选",

        "阶段敏感型",

        "时间趋势影响型",

        "种子敏感型",

        "证据不足型",

        "未满足预设分类条件",

    ]

    for category in categories:

        count = result_df[
            result_df["classification"].str.contains(
                category,
                regex=False
            )
        ].shape[0]

        print(
            f"{category}：{count}组"
        )

    print("\n" + "=" * 68)

    print("关联稳健性分类完成！")

    print("=" * 68)

    print(f"分类表：{OUTPUT_CSV}")

    print(f"分析报告：{REPORT_FILE}")


if __name__ == "__main__":

    main()
