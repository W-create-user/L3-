"""Backward-compatible exports for runtime integration adapters."""
from .b_to_d_adapter import make_d_safety_checker
from .feedback_adapter import execution_result_to_feedback, update_scheduler_from_result


def make_c_feedback_callback(scheduler):
    def callback(result):
        return update_scheduler_from_result(scheduler, result)
    return callback
