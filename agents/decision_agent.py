from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Union
from uuid import uuid4


# ==================================================
# 1. 定义子任务数据结构
# ==================================================

@dataclass
class SubTask:

    subtask_id: str
    name: str
    description: str

    # 子任务执行顺序
    order: int

    # 依赖的前置子任务ID
    dependencies: List[str]

    # 所需能力标签，供调度Agent参考
    required_capability: str

    # 子任务状态
    status: str = "PENDING"


# ==================================================
# 2. 定义总任务数据结构
# ==================================================

@dataclass
class Task:

    task_id: str
    goal_type: str
    description: str

    priority: int

    constraints: Dict[str, Any]

    subtasks: List[SubTask]

    status: str = "CREATED"

    source: str = "rule_mapping"


# ==================================================
# 3. DecisionAgent基础类
# ==================================================

class DecisionAgent:

    def __init__(self):

        # 规则映射表
        self.goal_mapping = {

            "检查当前仿真状态": "inspect_state",

            "检查状态": "inspect_state",

            "执行一次仿真控制任务": "control_cycle",

            "执行一次控制任务": "control_cycle",

        }


    # ------------------------------------------------
    # 目标识别
    # ------------------------------------------------

    def recognize_goal(self, goal: Union[str, Dict]):

        """
        支持字符串和结构化字典两种输入。
        返回统一的目标信息。
        """

        # 情况一：字符串输入
        if isinstance(goal, str):

            goal = goal.strip()

            if not goal:
                raise ValueError("目标字符串不能为空")

            if goal not in self.goal_mapping:

                raise ValueError(
                    f"暂不支持该目标：{goal}"
                )

            return {
                "goal_type": self.goal_mapping[goal],
                "description": goal,
                "priority": 3,
                "constraints": {}
            }


        # 情况二：结构化字典输入
        elif isinstance(goal, dict):

            goal_type = goal.get("goal_type")

            description = goal.get(
                "description",
                goal_type
            )

            priority = goal.get("priority", 3)

            constraints = goal.get(
                "constraints",
                {}
            )

            if goal_type not in [
                "inspect_state",
                "control_cycle"
            ]:

                raise ValueError(
                    f"不支持的任务类型：{goal_type}"
                )

            if not isinstance(description, str) or not description.strip():
                raise ValueError("description必须是非空字符串")

            if isinstance(priority, bool) or not isinstance(priority, int) or not 1 <= priority <= 5:
                raise ValueError("priority必须是1到5之间的整数")

            if not isinstance(constraints, dict):
                raise ValueError("constraints必须是字典")

            return {
                "goal_type": goal_type,
                "description": description.strip(),
                "priority": priority,
                "constraints": constraints
            }

        else:

            raise TypeError(
                "输入必须是字符串或字典"
            )


    # ------------------------------------------------
    # 任务模板选择
    # ------------------------------------------------

    def select_template(self, goal_type):

        """
        根据目标类型选择预定义任务模板。
        """

        if goal_type == "inspect_state":

            return [

                {
                    "name": "collect_observation",
                    "description": "获取当前仿真环境状态信息",
                    "capability": "state_read"
                },

                {
                    "name": "evaluate_state",
                    "description": "根据任务要求检查当前状态信息",
                    "capability": "decision"
                },

                {
                    "name": "return_state_result",
                    "description": "整理并返回状态检查结果",
                    "capability": "decision"
                }

            ]


        elif goal_type == "control_cycle":

            return [

                {
                    "name": "collect_observation",
                    "description": "获取当前仿真环境状态信息",
                    "capability": "state_read"
                },

                {
                    "name": "evaluate_goal",
                    "description": "根据任务目标分析当前状态",
                    "capability": "decision"
                },

                {
                    "name": "generate_action_intent",
                    "description": "生成待审核的抽象控制动作意图",
                    "capability": "decision"
                },

                {
                    "name": "safety_review",
                    "description": "将候选动作提交监管模块进行安全检查",
                    "capability": "monitoring"
                },

                {
                    "name": "execute_approved_action",
                    "description": "执行经过监管放行的动作",
                    "capability": "environment_execution"
                },

                {
                    "name": "verify_result",
                    "description": "获取执行结果并检查任务完成情况",
                    "capability": "decision"
                }

            ]

        else:

            raise ValueError(
                f"没有对应的任务模板：{goal_type}"
            )


    # ------------------------------------------------
    # 子任务分解
    # ------------------------------------------------

    def decompose_task(self, template):

        """
        将任务模板转换成具有ID和依赖关系的子任务。
        """

        subtasks = []

        previous_id = None

        for index, item in enumerate(template):

            subtask_id = f"SUB-{uuid4().hex[:8]}"

            if previous_id is None:

                dependencies = []

            else:

                dependencies = [previous_id]


            subtask = SubTask(

                subtask_id=subtask_id,

                name=item["name"],

                description=item["description"],

                order=index + 1,

                dependencies=dependencies,

                required_capability=item["capability"]

            )

            subtasks.append(subtask)

            previous_id = subtask_id


        return subtasks


    # ------------------------------------------------
    # 生成总任务对象
    # ------------------------------------------------

    def create_task(self, goal):

        """
        DecisionAgent核心入口。

        输入：
            字符串或结构化目标

        输出：
            Task任务对象
        """

        # 第一步：识别目标
        goal_info = self.recognize_goal(goal)


        # 第二步：选择任务模板
        template = self.select_template(
            goal_info["goal_type"]
        )


        # 第三步：分解子任务
        subtasks = self.decompose_task(
            template
        )


        # 第四步：生成总任务对象
        task = Task(

            task_id=f"TASK-{uuid4().hex[:8]}",

            goal_type=goal_info["goal_type"],

            description=goal_info["description"],

            priority=goal_info["priority"],

            constraints=goal_info["constraints"],

            subtasks=subtasks

        )

        return task


    # ------------------------------------------------
    # 转换成字典，方便传给其他Agent
    # ------------------------------------------------

    def task_to_dict(self, task):

        return asdict(task)