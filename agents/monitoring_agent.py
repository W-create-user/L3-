
import numpy as np
from datetime import datetime


class MonitoringAgent:
    """
    L3 多智能体系统安全监管 Agent

    主要功能：
    R-01：输入数据合法性检查
    R-02：候选动作边界检查
    R-03：过程状态异常预警（等待真实状态规则配置）

    风险等级：
    Level 0：正常放行
    Level 1：低风险提示
    Level 2：中风险预警/限制
    Level 3：高风险阻断

    注意：
    不传入 action_space 时使用演示占位参数，
    不能将演示模式用于正式 AtropineEnv 安全验证。
    """

    def __init__(self, action_space=None):
        """
        初始化监管智能体。

        参数：
            action_space:
                正式运行时传入 env.action_space。
                目前预期为 Gymnasium/Gym Box 类型动作空间。
        """

        self.logs = []

        if action_space is not None:
            # 使用仿真环境真实动作空间
            if not all(
                hasattr(action_space, attr)
                for attr in ("shape", "low", "high")
            ):
                raise TypeError(
                    "当前仅支持具有 shape、low、high 属性的 Box 动作空间"
                )

            self.action_shape = tuple(action_space.shape)

            self.action_min = np.asarray(
                action_space.low, dtype=float
            )

            self.action_max = np.asarray(
                action_space.high, dtype=float
            )

            self.demo_mode = False

        else:
            # 演示占位参数：等待成员 B 提供真实环境数据
            self.action_shape = (4,)

            self.action_min = np.full(
                self.action_shape, -1.0
            )

            self.action_max = np.full(
                self.action_shape, 1.0
            )

            self.demo_mode = True

            print(
                "[WARNING] 当前处于演示模式，"
                "动作空间参数尚未使用真实环境数据！"
            )

        # 检查动作空间配置是否有效
        if (
            self.action_min.shape != self.action_shape
            or self.action_max.shape != self.action_shape
        ):
            raise ValueError("动作空间上下界与动作形状不一致")

        if (
            np.any(np.isnan(self.action_min))
            or np.any(np.isnan(self.action_max))
            or np.any(self.action_min > self.action_max)
        ):
            raise ValueError("动作空间上下界配置错误")

        self.action_dim = int(np.prod(self.action_shape))

    # ==================================================
    # 核心监管接口
    # ==================================================

    def check_action(self, state, candidate_action):
        """
        对候选动作执行安全检查。

        输入：
            state：当前仿真状态
            candidate_action：执行 Agent 生成的候选动作

        返回：
            dict:
                approved：是否允许执行
                status：Pass / Warning / Reject
                level：危险等级 0-3
                rule_id：触发规则编号
                intervention：建议干预方式
                reason：监管原因
        """

        # ================= R-01：输入合法性检查 =================

        if candidate_action is None:
            return self._response(
                status="Reject",
                level=3,
                rule_id="R-01",
                intervention="BLOCK",
                reason="候选动作为空"
            )

        try:
            action = np.asarray(
                candidate_action,
                dtype=float
            )

        except (TypeError, ValueError):
            return self._response(
                status="Reject",
                level=3,
                rule_id="R-01",
                intervention="BLOCK",
                reason="动作数据格式错误，无法转换为数值数组"
            )

        # 检查动作形状
        if action.shape != self.action_shape:
            return self._response(
                status="Reject",
                level=3,
                rule_id="R-01",
                intervention="BLOCK",
                reason=(
                    f"动作维度或形状错误："
                    f"期望 {self.action_shape}，"
                    f"实际 {action.shape}"
                )
            )

        # 检查 NaN、Inf
        if not np.all(np.isfinite(action)):
            return self._response(
                status="Reject",
                level=3,
                rule_id="R-01",
                intervention="BLOCK",
                reason="动作包含 NaN 或 Inf 非有限数值"
            )

        # ================= R-02：动作边界检查 =================

        below_min = action < self.action_min
        above_max = action > self.action_max

        if np.any(below_min) or np.any(above_max):

            # 找出所有越界的动作维度
            invalid_indices = np.where(
                below_min | above_max
            )[0]

            details = []

            for i in invalid_indices:
                details.append(
                    f"动作[{i}]={action[i]:.6g}, "
                    f"允许范围=[{self.action_min[i]:.6g}, "
                    f"{self.action_max[i]:.6g}]"
                )

            return self._response(
                status="Reject",
                level=3,
                rule_id="R-02",
                intervention="BLOCK",
                reason="动作超出允许范围：" + "; ".join(details)
            )

        # ================= R-03：状态异常监管 =================

        # 当前尚未配置真实状态规则。
        # 等待成员 B 提供 AtropineEnv 的真实状态字段、
        # 变量含义、单位和有依据的实验阈值。
        #
        # 后续可以在这里加入：
        # 1. 温度异常检查
        # 2. 压力异常检查（以实际字段为准）
        # 3. 过程状态变化率检查
        # 4. 异常风险等级评估
        #
        # 未配置规则时，不得声称已完成状态异常检测。

        # ================= 检查通过 =================

        return self._response(
            status="Pass",
            level=0,
            rule_id=None,
            intervention="ALLOW",
            reason="所有已启用的安全规则检查通过"
        )

    # ==================================================
    # 统一返回接口与日志记录
    # ==================================================

    def _response(
        self,
        status,
        level,
        rule_id,
        intervention,
        reason
    ):
        """
        统一封装监管结果并记录日志。
        """

        result = {
            "timestamp": datetime.now().isoformat(
                timespec="seconds"
            ),
            "approved": status == "Pass",
            "status": status,
            "level": level,
            "rule_id": rule_id,
            "intervention": intervention,
            "reason": reason,
            "demo_mode": self.demo_mode
        }

        self.logs.append(result)

        print(
            f"[MonitoringAgent] "
            f"Level {level} | "
            f"{status} | "
            f"{reason}"
        )

        return result

    # ==================================================
    # 日志管理
    # ==================================================

    def get_logs(self):
        """获取全部监管日志。"""
        return self.logs.copy()

    def clear_logs(self):
        """清空监管日志。"""
        self.logs.clear()

    def get_statistics(self):
        """统计监管运行情况。"""

        total = len(self.logs)

        passed = sum(
            log["status"] == "Pass"
            for log in self.logs
        )

        rejected = sum(
            log["status"] == "Reject"
            for log in self.logs
        )

        warnings = sum(
            log["status"] == "Warning"
            for log in self.logs
        )

        return {
            "total": total,
            "passed": passed,
            "rejected": rejected,
            "warnings": warnings,
            "rejection_rate": (
                rejected / total if total > 0 else 0.0
            )
        }


# ======================================================
# 安全执行接口
# ======================================================

def safe_step(env, monitor, state, candidate_action):
    """
    监管通过后才允许执行 env.step()。

    返回：
        dict:
            executed：是否执行
            monitoring_result：监管结果
            env_result：仿真环境执行结果

    被拒绝的动作不会调用 env.step()。
    """

    monitoring_result = monitor.check_action(
        state,
        candidate_action
    )

    if not monitoring_result["approved"]:

        print("[安全门控] 动作已被阻断，未进入 env.step()")

        return {
            "executed": False,
            "monitoring_result": monitoring_result,
            "env_result": None
        }

    # 只有通过监管检查才进入环境
    env_result = env.step(candidate_action)

    return {
        "executed": True,
        "monitoring_result": monitoring_result,
        "env_result": env_result
    }


# ======================================================
# 独立测试
# ======================================================

if __name__ == "__main__":

    print("=" * 55)
    print("MonitoringAgent v1.1 独立测试")
    print("=" * 55)

    # 使用演示模式
    agent = MonitoringAgent()

    fake_state = {
        "temperature": 75.0
    }

    # 测试1：正常动作
    print("\n[测试1] 正常动作")

    agent.check_action(
        fake_state,
        [0.5, 0.2, 0.1, 0.0]
    )

    # 测试2：动作越界
    print("\n[测试2] 动作越界")

    agent.check_action(
        fake_state,
        [0.5, 1.5, 0.0, -2.0]
    )

    # 测试3：NaN
    print("\n[测试3] NaN")

    agent.check_action(
        fake_state,
        [0.5, np.nan, 0.1, 0.0]
    )

    # 测试4：Inf
    print("\n[测试4] Inf")

    agent.check_action(
        fake_state,
        [0.5, np.inf, 0.1, 0.0]
    )

    # 测试5：维度错误
    print("\n[测试5] 维度错误")

    agent.check_action(
        fake_state,
        [0.5, 0.2]
    )

    # 测试6：None
    print("\n[测试6] 空动作")

    agent.check_action(
        fake_state,
        None
    )

    # 输出统计
    print("\n" + "=" * 55)
    print("测试统计")
    print("=" * 55)

    print(agent.get_statistics())

    # 检查结果
    assert agent.logs[0]["status"] == "Pass"
    assert agent.logs[1]["status"] == "Reject"
    assert agent.logs[2]["status"] == "Reject"
    assert agent.logs[3]["status"] == "Reject"
    assert agent.logs[4]["status"] == "Reject"
    assert agent.logs[5]["status"] == "Reject"

    print("\n[测试完成] 基础安全规则测试全部通过！")


# ======================================================
# 安全拦截机制测试
# ======================================================

class FakeEnv:
    """
    模拟仿真环境，用于验证危险动作是否被阻断。
    """

    def __init__(self):
        self.step_calls = 0

    def step(self, action):
        self.step_calls += 1
        return "仿真执行成功"


def test_safe_step():

    print("\n" + "=" * 55)
    print("安全执行拦截测试")
    print("=" * 55)

    env = FakeEnv()
    monitor = MonitoringAgent()

    state = {"temperature": 75.0}

    # 正常动作
    result1 = safe_step(
        env,
        monitor,
        state,
        [0.5, 0.2, 0.1, 0.0]
    )

    assert result1["executed"] is True
    assert env.step_calls == 1

    print("正常动作：已允许执行")

    # 危险动作
    result2 = safe_step(
        env,
        monitor,
        state,
        [0.5, 2.0, 0.1, 0.0]
    )

    assert result2["executed"] is False
    assert env.step_calls == 1

    print("危险动作：已成功阻断")
    print("env.step() 调用次数：", env.step_calls)

    print("\n[测试完成] 安全执行拦截验证通过！")


if __name__ == "__main__":
    test_safe_step()
