from agents.decision_agent import DecisionAgent
import json


# 创建决策Agent
agent = DecisionAgent()

# 输入目标goal = "执行一次仿真控制任务"

# 输入目标
goal = {
    "goal_type": "control_cycle",
    "description": "完成一次仿真控制任务",
    "priority": 4,
    "constraints": {
        "environment": "AtropineEnv",
        "require_monitoring": True
    }
}


# 生成任务对象
task = agent.create_task(goal)



# 转换成字典
result = agent.task_to_dict(task)


# 打印结果
#print(
#    json.dumps(
#        result,
#        ensure_ascii=False,
#        indent=4
#    )
#)
print(agent.task_to_dict(task))