
import json
from pathlib import Path

folders = list(Path("references").glob("B*2026-11-04"))

if not folders:
    raise FileNotFoundError("找不到B的资料目录")

schema_file = folders[0] / "observation_schema.json"

with open(schema_file, "r", encoding="utf-8") as f:
    schema = json.load(f)

fields = schema["fields"]

lines = [
    "# D模块 R-03关键状态变量分析表 v0.1",
    "",
    "## 一、数据基本信息",
    "",
    "- 数据来源：AtropineEnv observation_schema.json",
    "- Observation维度：39",
    "- 数据格式：39维扁平数组",
    "- 采样间隔：10分钟",
    "- Normalize：False",
    "",
    "## 二、39维变量清单",
    "",
    "| 索引 | 变量名称 | 物理含义 | 单位 | 类别 |",
    "|---:|---|---|---|---|"
]

for item in fields:
    meaning = str(item["meaning"]).replace("|", "/")
    lines.append(
        f"| {item['index']} | {item['name']} | "
        f"{meaning} | {item['unit']} | {item['group']} |"
    )

lines += [
    "",
    "## 三、重点研究变量",
    "",
    "| 变量 | 索引 | 研究方向 |",
    "|---|---:|---|",
    "| E_factor_current | 6 | 当前E-factor变化 |",
    "| Q_organic | 37 | 有机相体积流量变化 |",
    "| Q_aqueous | 38 | 水相体积流量变化 |",
    "| USS_1至USS_4 | 1-4 | 基准输入流量变化 |",
    "| F_organic及F_aqueous | 9-36 | 两相组分流量变化 |",
    "| estimator_x_1及estimator_x_2 | 7-8 | 状态估计器变化 |",
    "",
    "## 四、注意事项",
    "",
    "1. 第0维是步号，不是实际物理时间。",
    "2. 第7、8维允许为负数。",
    "3. Box上下界不等于工业安全阈值。",
    "4. Step观测经过Box裁剪。",
    "5. 当前只有初始和一步执行样本，不能确定安全阈值。",
    "",
    "## 五、当前开发状态",
    "",
    "- R-01：已完成。",
    "- R-02：真实环境测试通过。",
    "- R-03：变量清单已整理，规则待开发。",
    "- 阈值：等待更多实验数据及依据。"
]

output = Path("docs") / "D_R03关键状态变量分析表_v0.1.md"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("\n".join(lines), encoding="utf-8")

print("分析表生成成功！")
print("变量数量：", len(fields))
print("保存位置：", output)
