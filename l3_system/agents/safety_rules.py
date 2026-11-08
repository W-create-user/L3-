"""D 模块规则库：R-01 输入检查，R-02 Box 动作边界检查。
R-03 状态预警待 B 提供真实 observation 字段与阈值后启用。
"""
import numpy as np

RULES = {
    'R-01': '动作缺失、类型/形状不符或含 NaN/Inf：拒绝，Level 3',
    'R-02': '动作超出 Box action_space 上下界：拒绝，Level 3',
    'R-03': '状态异常预警：尚未启用，不参与当前判断',
}


def validate_action(action, shape):
    """返回 (转换后的动作数组或 None, 错误原因或 None)。"""
    if action is None:
        return None, 'R-01: 候选动作为空'
    try:
        arr = np.asarray(action, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return None, 'R-01: 动作无法转换为数值数组'
    if arr.shape != tuple(shape):
        return None, f'R-01: 期望形状 {tuple(shape)}，实际 {arr.shape}'
    if not np.all(np.isfinite(arr)):
        return None, 'R-01: 动作包含 NaN 或 Inf'
    return arr, None


def check_bounds(action, low, high):
    """返回越界原因或 None；支持任意 Box shape。"""
    invalid = (action < low) | (action > high)
    if not np.any(invalid):
        return None
    parts = []
    for index in np.argwhere(invalid):
        idx = tuple(int(x) for x in index)
        label = '[' + ','.join(map(str, idx)) + ']'
        parts.append(f'动作{label}={action[idx]:.6g}，允许范围=[{low[idx]:.6g}, {high[idx]:.6g}]')
    return 'R-02: 动作超出允许范围：' + '; '.join(parts)
