import numpy as np

class MonitoringAgent:
    def __init__(self):
        """
        初始化监管智能体
        注意：action_dim 和 min/max 只是初始占位符，
        等负责环境的同学(B)跑通了，一定要把真实数值同步过来！
        """
        self.action_dim = 4      # 假设环境动作是4维
        self.action_min = -1.0   # 假设动作下限
        self.action_max = 1.0    # 假设动作上限
        
        self.logs = []

    def check_action(self, state, candidate_action):
        """
        核心监管函数：在动作真正下发给仿真环境前进行拦截
        返回: 字典格式，包含 status(状态), level(危险等级), reason(原因)
        """
        # ================= R-01: L1 数据校验层 =================
        if candidate_action is None or len(candidate_action) != self.action_dim:
            return self._response("Reject", 3, f"R-01拦截: 动作维度不足或为空，期望{self.action_dim}维")
        
        if np.any(np.isnan(candidate_action)):
            return self._response("Reject", 3, "R-01拦截: 动作包含NaN空值")

        # ================= R-02: L3 操作沙箱层 (边界拦截) =================
        for i, act_val in enumerate(candidate_action):
            if act_val < self.action_min or act_val > self.action_max:
                return self._response("Reject", 3, f"R-02拦截: 动作[{i}]={act_val} 超出安全范围")

        # ================= R-03: L2 状态逻辑层 (预警) =================
        # 预留位置：等B给出真实状态格式后，在这里写 if state['temp'] > 85: return self._response("Warning", 2, "...")

        # 如果都没问题，放行
        return self._response("Pass", 0, "所有安全检查通过")

    def _response(self, status, level, reason):
        """统一封装返回格式"""
        res = {"status": status, "level": level, "reason": reason}
        self.logs.append(res)
        print(f"👉 监管日志: {res}") # 方便在终端打印出来看
        return res

# 测试代码
if __name__ == "__main__":
    agent = MonitoringAgent()
    fake_state = {"temperature": 75.0}
    print("--- 开始测试 ---")
    agent.check_action(fake_state, [0.5, 0.2, 0.1, 0.0]) # 测试正常
    agent.check_action(fake_state, [0.5, 1.5, 0.0, -2.0]) # 测试越界