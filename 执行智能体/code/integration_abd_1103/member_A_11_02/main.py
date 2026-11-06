"""A-only demonstration: generate and print an HTN task plan.

This script does NOT connect to SMPL and does NOT execute environment actions.
"""
import argparse
import json

from agents.decision_agent import DecisionAgent


def build_demo_goal():
    return {
        "goal_type": "control_cycle",
        "description": "完成一次仿真控制任务",
        "priority": 4,
        "constraints": {
            "environment": "AtropineEnv",
            "require_monitoring": True,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="成员A：DecisionAgent + HTN任务分解演示")
    parser.add_argument("--goal", default=None, help="可选的目标字符串")
    args = parser.parse_args()

    agent = DecisionAgent()
    goal = args.goal if args.goal else build_demo_goal()
    task = agent.create_task(goal)
    result = agent.task_to_dict(task)

    print("说明：以下输出是任务规划结果，不代表已经连接或运行 SMPL。")
    print(json.dumps(result, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    main()
