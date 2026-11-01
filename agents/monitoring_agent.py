import numpy as np

class MonitoringAgent:
    def __init__(self):
        """
        初始化监管智能体
        注意：这里的 action_dim (维度) 和 min/max (上下界) 只是初始占位符，
        等负责环境的同学(B)跑通了 AtropineEnv，一定要把真实数值同步过来修改这里！
        """
        self.action_dim = 4      # 假设环境动作是4维
        self.action_min = -1.0   # 假设动作下限是 -1.0
        self.action_max = 1.0    # 假设动作上限是 1.0
        
        # 用于记录监管日志，后续写入 CSV 或报告里用
        self.logs = []

    def check_action(self, state, candidate_action):
        """
        核心监管函数：在动作真正下发给仿真环境前进行拦截
        输入: state (当前环境状态), candidate_action (执行Agent想要执行的动作)
        输出: (监管结果 "Pass"/"Reject", 具体原因 reason)
        """
        # ================= L1 数据校验层 =================
        # 检查数据格式是否正确，防止引发系统崩溃
        if candidate_action is None or len(candidate_action) != self.action_dim:
            reason = f"L1拦截: 动作维度不足或为空，期望{self.action_dim}维"
            self._log(state, candidate_action, "Reject", reason)
            return "Reject", reason
        
        if np.any(np.isnan(candidate_action)):
            reason = "L1拦截: 动作包含NaN空值"
            self._log(state, candidate_action, "Reject", reason)
            return "Reject", reason

        # ================= L3 操作沙箱层 (边界拦截) =================
        # 检查动作是否超出了仿真环境允许的安全边界
        for i, act_val in enumerate(candidate_action):
            if act_val < self.action_min or act_val > self.action_max:
                reason = f"L3拦截: 动作[{i}]={act_val} 超出安全范围 [{self.action_min}, {self.action_max}]"
                self._log(state, candidate_action, "Reject", reason)
                return "Reject", reason

        # ================= L2 状态逻辑层 (业务规则拦截) =================
        # 这里预留了一个基于当前状态的逻辑检查（等后续有了具体的 state 数据格式再打开注释修改）
        # 示例：如果当前温度过高，就不允许继续加热
        # if state.get("temperature", 0) > 90.0 and candidate_action[0] > 0:
        #     reason = "L2拦截: 温度已超90度，拒绝加热指令"
        #     self._log(state, candidate_action, "Reject", reason)
        #     return "Reject", reason

        # 如果以上所有检查都通过了，说明动作是安全的
        reason = "安全检查通过"
        self._log(state, candidate_action, "Pass", reason)
        return "Pass", reason

    def _log(self, state, action, result, reason):
        """记录监管日志的内部方法"""
        log_entry = {
            "state": state,
            "action": action,
            "result": result,
            "reason": reason
        }
        self.logs.append(log_entry)


# ================= 本地测试代码 =================
# 只有直接运行这个文件时，才会执行下面的测试
if __name__ == "__main__":
    print("--- 开始测试 MonitoringAgent 监管模块 ---\n")
    agent = MonitoringAgent()
    
    # 造假数据：假设环境返回了一个正常的状态（温度75，压力1.2）
    fake_state = {"temperature": 75.0, "pressure": 1.2} 
    
    # 测试1：提交合法动作
    print("[测试1] 提交合法动作 [0.5, -0.2, 0.0, 0.1]")
    safe_action = [0.5, -0.2, 0.0, 0.1]
    result, reason = agent.check_action(fake_state, safe_action)
    print(f"👉 结果: {result}, 原因: {reason}\n")
    
    # 测试2：提交危险动作（动作越界）
    print("[测试2] 提交越界动作 [0.5, 1.5, 0.0, -2.0]")
    danger_action = [0.5, 1.5, 0.0, -2.0] # 1.5和-2.0越界了
    result, reason = agent.check_action(fake_state, danger_action)
    print(f"👉 结果: {result}, 原因: {reason}\n")
    
    # 测试3：提交格式错误的数据（维度不对）
    print("[测试3] 提交维度错误动作 [0.5]")
    bad_action = [0.5] # 只有1维，但环境需要4维
    result, reason = agent.check_action(fake_state, bad_action)
    print(f"👉 结果: {result}, 原因: {reason}\n")
    
    print("--- 测试结束 ---")