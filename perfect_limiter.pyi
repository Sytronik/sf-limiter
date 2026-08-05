from typing import TypeAlias, final

import numpy as np
from numpy.typing import ArrayLike, NDArray

_ProcessOutput: TypeAlias = tuple[NDArray[np.float32], NDArray[np.float32]]

@final
class PerfectLimiter:
    def __init__(
        self,
        sample_rate: int,
        threshold: float = 1.0,
        attack_ms: float = 5.0,
        hold_ms: float = 15.0,
        release_ms: float = 40.0,
    ) -> None: ...
    def process(self, audio: ArrayLike, channel_axis: int = -1) -> _ProcessOutput: ...
    def reset(self) -> None: ...
    @property
    def sample_rate(self) -> int: ...
    @property
    def threshold(self) -> float: ...
    @property
    def latency_samples(self) -> int: ...
    @property
    def hold_samples(self) -> int: ...
    @property
    def release_samples(self) -> float: ...

def limit(
    audio: ArrayLike,
    sample_rate: int,
    threshold: float = 1.0,
    attack_ms: float = 5.0,
    hold_ms: float = 15.0,
    release_ms: float = 40.0,
    channel_axis: int = -1,
) -> _ProcessOutput: ...
