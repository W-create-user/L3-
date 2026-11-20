"""Common simulation adapter contract and explicit action coordinate conversions."""
from __future__ import annotations
from typing import Protocol, runtime_checkable
import numpy as np

@runtime_checkable
class BaseAdapter(Protocol):
    @property
    def metadata(self) -> dict: ...
    @property
    def action_space(self): ...
    @property
    def observation_space(self): ...
    @property
    def state_version(self) -> int: ...
    @property
    def done(self) -> bool: ...
    def reset(self, seed=1101): ...
    def get_observation(self): ...
    def step(self, action): ...
    def close(self): ...

class ActionCoordinates:
    """physical = model * scale + offset; normalization is a separate transform."""
    def __init__(self, low, high, scale, offset):
        self.low=np.asarray(low,dtype=float).copy();self.high=np.asarray(high,dtype=float).copy()
        self.scale=np.asarray(scale,dtype=float).copy();self.offset=np.asarray(offset,dtype=float).copy()
        if (self.low.ndim!=1 or any(a.shape!=self.low.shape for a in [self.high,self.scale,self.offset])
                or not all(np.isfinite(a).all() for a in [self.low,self.high,self.scale,self.offset])
                or np.any(self.low>=self.high) or np.any(self.scale<=0)):
            raise ValueError('Invalid action coordinate parameters')
    def _vector(self,value):
        raw=np.asarray(value)
        if raw.shape!=self.low.shape or raw.dtype.kind not in 'fiu' or not np.isfinite(raw).all():
            raise ValueError('Expected finite action vector with live dimension')
        return raw.astype(float)
    def to_physical(self,model): return self._vector(model)*self.scale+self.offset
    def from_physical(self,physical): return (self._vector(physical)-self.offset)/self.scale
    def to_normalized(self,model):
        # Algebraic [-1,1] transform; independent of backend's epsilon-based normalization.
        return 2*(self._vector(model)-self.low)/(self.high-self.low)-1
    def from_normalized(self,normalized):
        value=self._vector(normalized)
        if np.any(np.abs(value)>1):raise ValueError('Normalized action outside [-1,1]')
        return self.low+(value+1)*(self.high-self.low)/2
