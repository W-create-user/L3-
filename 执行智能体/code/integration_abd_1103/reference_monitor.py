"""B's reference checker for D's described API. This is NOT D's source code."""
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np


class ReferenceMonitoringAgent:
    implementation_source = 'B reference checker from D interface description; D source not supplied'

    def __init__(self, action_space):
        self.action_space = action_space

    def check_action(self, state, candidate_action):
        def decision(approved, rule, reason):
            return {
                'timestamp': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                'approved': approved,
                'status': 'Pass' if approved else 'Reject',
                'level': 0 if approved else 3,
                'rule_id': rule,
                'intervention': 'ALLOW' if approved else 'BLOCK',
                'reason': reason,
                # False means real bounds were injected, not that this is D's implementation.
                'demo_mode': False,
                'implementation_source': self.implementation_source,
                'state_rule_R03_enabled': False,
            }
        try:
            action = np.asarray(candidate_action)
            if (action.shape != self.action_space.shape or action.dtype.kind not in 'fiu'
                    or not np.isfinite(action).all()):
                return decision(False, 'R-01', 'Action shape, data type, or finite-value check failed')
        except (ValueError, TypeError):
            return decision(False, 'R-01', 'Action cannot be converted to a numeric vector')
        if np.any(action < self.action_space.low) or np.any(action > self.action_space.high):
            return decision(False, 'R-02', 'Action exceeds the actual environment action bounds')
        return decision(True, None, 'Action passes R-01 and R-02 using actual environment bounds')
