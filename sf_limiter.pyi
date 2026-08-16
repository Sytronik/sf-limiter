from typing import TypeAlias, final

import numpy as np
from numpy.typing import ArrayLike, NDArray

_ProcessOutput: TypeAlias = tuple[NDArray[np.float32], NDArray[np.float32]]

@final
class SFLimiter:
    """A configurable look-ahead brick-wall limiter.

    The limiter applies one linked gain value to all channels in each frame,
    preserving their relative balance. Processing is offline and every call
    starts with a neutral gain envelope.

    Args:
        sample_rate: Sample rate in hertz. Must be greater than zero.
        threshold: Maximum absolute output sample, in ``(0, 1]``.
        attack_ms: Look-ahead attack time in milliseconds. It must round to at
            least one sample at ``sample_rate``.
        hold_ms: Hold time in milliseconds. May be zero.
        release_ms: Release time in milliseconds. May be zero.

    Raises:
        ValueError: If any configuration value is invalid.
    """

    def __init__(
        self,
        sample_rate: int,
        threshold: float = 1.0,
        attack_ms: float = 5.0,
        hold_ms: float = 15.0,
        release_ms: float = 40.0,
    ) -> None: ...
    def process(self, audio: ArrayLike, axis: int = -1) -> _ProcessOutput:
        """Process mono or multichannel audio.

        Args:
            audio: A one-dimensional mono array or a two-dimensional
                multichannel array. Values are converted to ``numpy.float32``;
                the input is not modified.
            axis: Frame axis. The default of ``-1`` expects
                ``(channels, frames)``. Use ``0`` for ``(frames, channels)``.

        Returns:
            A tuple ``(audio, frame_gains)`` containing the limited audio and one
            linked gain value per frame as ``numpy.float32`` arrays.

        Raises:
            ValueError: If the input shape, frame axis, or samples are invalid.
        """

    @property
    def sample_rate(self) -> int:
        """Configured sample rate in hertz."""

    @property
    def threshold(self) -> float:
        """Configured maximum absolute output sample."""

    @property
    def lookahead_samples(self) -> int:
        """Look-ahead latency in samples."""

    @property
    def attack_samples(self) -> int:
        """Look-ahead latency in samples. (alias for ``lookahead_samples``)"""

    @property
    def hold_samples(self) -> int:
        """Configured hold duration in samples."""

    @property
    def release_samples(self) -> float:
        """Configured release duration in samples."""

def limit(
    audio: ArrayLike,
    sample_rate: int,
    threshold: float = 1.0,
    attack_ms: float = 5.0,
    hold_ms: float = 15.0,
    release_ms: float = 40.0,
    axis: int = -1,
) -> _ProcessOutput:
    """Limit mono or multichannel audio in one call.

    Args:
        audio: A one-dimensional mono array or a two-dimensional multichannel
            array. Values are converted to ``numpy.float32``; the input is not
            modified.
        sample_rate: Sample rate in hertz. Must be greater than zero.
        threshold: Maximum absolute output sample, in ``(0, 1]``.
        attack_ms: Look-ahead attack time in milliseconds.
        hold_ms: Hold time in milliseconds. May be zero.
        release_ms: Release time in milliseconds. May be zero.
        axis: Frame axis. The default of ``-1`` expects
            ``(channels, frames)``. Use ``0`` for ``(frames, channels)``.

    Returns:
        A tuple ``(audio, frame_gains)`` containing the limited audio and one linked
        gain value per frame as ``numpy.float32`` arrays.

    Raises:
        ValueError: If a configuration value or input is invalid.
    """
