"""A local interface baseline, not a process-specific feedback controller."""
import numpy as np


def space_midpoint(observation, action_space):
    # Observation is used to require a valid current state, not to optimize a target.
    state = np.asarray(observation)
    if state.ndim != 1 or not state.size or state.dtype.kind not in 'fiu' or not np.isfinite(state).all():
        raise ValueError('A finite current observation is required for local candidate generation')
    if (not np.isfinite(action_space.low).all() or not np.isfinite(action_space.high).all()
            or np.any(action_space.low > action_space.high)):
        raise ValueError('The midpoint baseline requires finite ordered action bounds')
    midpoint = (action_space.low.astype('float64') + action_space.high.astype('float64')) / 2
    return midpoint.astype(action_space.dtype)
