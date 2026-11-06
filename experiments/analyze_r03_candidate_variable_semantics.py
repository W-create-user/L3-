
"""
D模块 R-03 候选变量物理含义核对 v0.1

读取：
1. B提供的 observation_schema.json
2. D生成的 r03_candidate_variables.csv

输出：
1. 候选变量物理含义对应表
2. 完整39维变量映射表
3. Markdown核对报告

不自行推断物理含义，不将模型Box边界解释为工业安全阈值。
"""

from pathlib import Path
import json
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SCHEMA_FILE = (
    ROOT / "references"
    / "B_300步真实数据"
    / "发给D_300步原始数据"
    / "observation_schema.json"
)

CANDIDATE_FILE = (
    ROOT / "experiments"
    / "r03_candidate_variables.csv"
)

OUT_DIR = ROOT / "experiments"
REPORT = (
    ROOT / "docs"
    / "D_R03候选变量物理含义核对报告_v0.1.md"
)


def load_schema():
    if not SCHEMA_FILE.exists():
        raise FileNotFoundError(
            f"找不到Schema文件：{SCHEMA_FILE}"
        )

    with open(SCHEMA_FILE, "r", encoding="utf-8-sig") as f:
        schema = json.load(f)

    if isinstance(schema, list):
        fields = schema
    elif isinstance(schema, dict):
        fields = schema.get("fields")

        if fields is None:
            raise ValueError(
                "Schema中没有找到fields字段，请检查JSON结构"
            )
    else:
        raise ValueError("Schema顶层结构不是dict或list")

    if not isinstance(fields, list) or len(fields) != 39:
        raise ValueError(
            f"预期39个变量定义，实际为"
            f"{len(fields) if isinstance(fields, list) else '非列表'}"
        )

    return schema, fields


def load_candidates():
    if not CANDIDATE_FILE.exists():
        raise FileNotFoundError(
            f"找不到候选变量表：{CANDIDATE_FILE}"
        )

    df = pd.read_csv(CANDIDATE_FILE)

    if "variable" not in df.columns:
        raise ValueError("候选表缺少variable字段")

    return df


def extract_index(variable):
    match = re.fullmatch(r"obs_(\d{3})", str(variable))

    if not match:
        raise ValueError(f"无法识别变量编号：{variable}")

    return int(match.group(1))


def get_field(field, key):
    """读取字段，保留Schema原始信息。"""
    value = field.get(key, "")

    if value is None:
        return ""

    return value


def build_mapping(fields):
    records = []

    for field in fields:

        if not isinstance(field, dict):
            raise ValueError("Schema中的变量定义不是dict")

        index = int(field["index"])

        records.append({
            "observation_index": index,
            "variable": f"obs_{index:03d}",
            "upstream_name": get_field(
                field, "upstream_name"
            ),
            "name": get_field(
                field, "name"
            ),
            "meaning": get_field(
                field, "meaning"
            ),
            "unit": get_field(
                field, "unit"
            ),
            "group": get_field(
                field, "group"
            ),
            "model_box_low": get_field(
                field, "model_box_low"
            ),
            "model_box_high": get_field(
                field, "model_box_high"
            ),
        })

    mapping = pd.DataFrame(records)

    if mapping["observation_index"].duplicated().any():
        raise ValueError("Schema中存在重复index")

    if set(mapping["observation_index"]) != set(range(39)):
        raise ValueError("Schema索引不是完整的0-38")

    return mapping


def generate_report(merged, schema):
    lines = [
        "# D模块 R-03 候选变量物理含义核对报告 v0.1",
        "",
        "## 1. 数据来源",
        "",
        "- B提供的 observation_schema.json",
        "- D生成的 r03_candidate_variables.csv",
        "- observation维度：39",
        "",
        "## 2. 候选变量物理含义",
        "",
        "| 排名 | 变量 | 模型名称 | 物理含义 | 单位 | 类别 |",
        "|---:|---|---|---|---|---|",
    ]

    ranked = merged.dropna(
        subset=["response_rank"]
    ).sort_values("response_rank")

    for _, row in ranked.iterrows():

        rank = int(row["response_rank"])

        lines.append(
            f"| {rank} "
            f"| {row['variable']} "
            f"| {row['name']} "
            f"| {row['meaning']} "
            f"| {row['unit']} "
            f"| {row['group']} |"
        )

    lines.extend([
        "",
        "## 3. 近零方差变量",
        "",
        "以下变量没有参与标准化响应排名，但仍保留在完整变量映射表中。",
        "",
        "| 变量 | 模型名称 | 物理含义 | 单位 | 类别 |",
        "|---|---|---|---|---|",
    ])

    low_variance = merged[
        merged["baseline_low_variance"].astype(str).str.lower()
        .isin(["true", "1"])
    ]

    for _, row in low_variance.iterrows():
        lines.append(
            f"| {row['variable']} "
            f"| {row['name']} "
            f"| {row['meaning']} "
            f"| {row['unit']} "
            f"| {row['group']} |"
        )

    lines.extend([
        "",
        "## 4. 模型观测空间边界",
        "",
        "| 变量 | 下界 | 上界 |",
        "|---|---:|---:|",
    ])

    for _, row in merged.iterrows():
        lines.append(
            f"| {row['variable']} "
            f"| {row['model_box_low']} "
            f"| {row['model_box_high']} |"
        )

    lines.extend([
        "",
        "## 5. 重要说明",
        "",
        "1. 变量名称、物理含义、单位和类别直接来自B提供的Schema。",
        "2. 本报告不自行推断未在Schema中定义的物理意义。",
        "3. model_box_low/high是模型观测空间边界。",
        "4. 模型观测空间边界不等于工业安全阈值。",
        "5. 候选排名是统计响应排名，不是危险等级。",
        "6. 变量是否适合用于监管，需要结合工艺知识和更多真实数据验证。",
        "",
        "## 6. 完整39维变量映射",
        "",
        "完整变量映射请查看：",
        "",
        "experiments/r03_all_variable_mapping.csv",
        "",
        "## 7. 输出文件",
        "",
        "- experiments/r03_candidate_variable_semantics.csv",
        "- experiments/r03_all_variable_mapping.csv",
        "- docs/D_R03候选变量物理含义核对报告_v0.1.md",
    ])

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():

    print("=" * 65)
    print("D模块 R-03 候选变量物理含义核对")
    print("=" * 65)

    schema, fields = load_schema()
    candidates = load_candidates()

    print(f"[PASS] Schema读取成功：{len(fields)}个变量")
    print(f"[PASS] 候选表读取成功：{len(candidates)}条记录")

    mapping = build_mapping(fields)

    candidates["observation_index"] = (
        candidates["variable"].apply(extract_index)
    )

    # 合并候选统计结果和Schema原始信息
    merged = candidates.merge(
        mapping,
        on=["variable", "observation_index"],
        how="left",
        validate="one_to_one"
    )

    if merged["name"].isna().any():
        raise ValueError("存在未匹配的变量定义")

    # 保存候选变量物理含义表
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    candidate_output = (
        OUT_DIR / "r03_candidate_variable_semantics.csv"
    )

    merged.to_csv(
        candidate_output,
        index=False,
        encoding="utf-8-sig"
    )

    # 保存完整39维变量映射
    all_output = (
        OUT_DIR / "r03_all_variable_mapping.csv"
    )

    mapping.to_csv(
        all_output,
        index=False,
        encoding="utf-8-sig"
    )

    # 生成报告
    generate_report(merged, schema)

    # 控制台输出
    print("\n候选变量物理含义：")

    ranked = merged.dropna(
        subset=["response_rank"]
    ).sort_values("response_rank")

    columns = [
        "response_rank",
        "variable",
        "name",
        "meaning",
        "unit",
        "group",
    ]

    print(
        ranked[columns].head(10).to_string(index=False)
    )

    print("\n" + "=" * 65)
    print("候选变量物理含义核对完成！")
    print("=" * 65)

    print(f"候选变量对应表：{candidate_output}")
    print(f"完整39维映射表：{all_output}")
    print(f"分析报告：{REPORT}")


if __name__ == "__main__":
    main()
