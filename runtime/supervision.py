"""Latched human-over-the-loop control for the simulation prototype."""
from datetime import datetime, timezone


class Supervision:
    def __init__(self):
        self.mode = 'autonomous'
        self.events = []

    def request_takeover(self, reason):
        self.mode = 'takeover_requested'
        self.events.append({'event': 'takeover_requested', 'reason': reason,
                            'time': datetime.now(timezone.utc).isoformat()})

    def confirm_recovery(self, operator, reason):
        if not isinstance(operator, str) or not operator.strip():
            raise ValueError('A named operator is required')
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('A recovery reason is required')
        self.events.append({'event': 'recovery_confirmed', 'operator': operator,
                            'reason': reason, 'time': datetime.now(timezone.utc).isoformat()})
        self.mode = 'autonomous'

    def require_autonomous(self):
        if self.mode != 'autonomous':
            raise RuntimeError('Human recovery confirmation is required before another task')
