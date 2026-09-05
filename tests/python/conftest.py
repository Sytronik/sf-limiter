from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import ArrayLike

import sf_limiter


@pytest.fixture(params=["function", "object"])
def entrypoint(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def process_audio() -> Callable[..., tuple[np.ndarray, np.ndarray]]:
    def helper(
        entrypoint: str,
        audio: ArrayLike,
        *,
        axis: int = -1,
        sample_rate: int = 48_000,
        **settings: float | bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        if entrypoint == "function":
            return sf_limiter.limit(
                audio, sample_rate=sample_rate, axis=axis, **settings
            )
        return sf_limiter.SFLimiter(sample_rate, **settings).process(audio, axis=axis)

    return helper


@pytest.fixture
def assert_never_clips() -> Callable[..., None]:
    def helper(audio: np.ndarray, ceiling: float = 1.0) -> None:
        assert np.isfinite(audio).all()
        assert np.max(np.abs(audio), initial=0.0) <= ceiling

    return helper
