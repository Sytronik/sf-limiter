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
        sample_rate: Sample rate in hertz. Must be a positive 32-bit integer.
        threshold_dBFS: Output ceiling in dBFS. Must be finite and at most ``0.0``;
            ``0.0`` is full scale and approximately ``-6.02`` is half scale.
        attack_ms: Look-ahead attack time in milliseconds. It must round to at
            least one sample at ``sample_rate``.
        hold_ms: Hold time in milliseconds. May be zero.
        release_ms: Release time in milliseconds. May be zero.
        true_peak: If true, use ITU-R BS.1770-5 true-peak detection to calculate
            limiter gain. The processed output true peak is not guaranteed to
            remain below the ceiling. Supported sample rates are 8, 11.025, 12,
            16, 22.05, 24, 32, 44.1, 48, 88.2, and 96 kHz, plus every rate at or
            above 176.4 kHz.

    Raises:
        ValueError: If any configuration value is invalid.
    """

    def __init__(
        self,
        sample_rate: int,
        threshold_dBFS: float = 0.0,
        attack_ms: float = 5.0,
        hold_ms: float = 15.0,
        release_ms: float = 40.0,
        true_peak: bool = False,
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
        """Configured output ceiling as a linear amplitude."""

    @property
    def threshold_dBFS(self) -> float:
        """Configured output ceiling in dBFS."""

    @property
    def true_peak(self) -> bool:
        """Whether ITU-R BS.1770-5 true-peak limiting is enabled."""

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
    threshold_dBFS: float = 0.0,
    attack_ms: float = 5.0,
    hold_ms: float = 15.0,
    release_ms: float = 40.0,
    axis: int = -1,
    true_peak: bool = False,
) -> _ProcessOutput:
    """Limit mono or multichannel audio in one call.

    Args:
        audio: A one-dimensional mono array or a two-dimensional multichannel
            array. Values are converted to ``numpy.float32``; the input is not
            modified.
        sample_rate: Sample rate in hertz. Must be a positive 32-bit integer.
        threshold_dBFS: Output ceiling in dBFS. Must be finite and at most ``0.0``;
            ``0.0`` is full scale and approximately ``-6.02`` is half scale.
        attack_ms: Look-ahead attack time in milliseconds.
        hold_ms: Hold time in milliseconds. May be zero.
        release_ms: Release time in milliseconds. May be zero.
        axis: Frame axis. The default of ``-1`` expects
            ``(channels, frames)``. Use ``0`` for ``(frames, channels)``.
        true_peak: If true, use ITU-R BS.1770-5 true-peak detection to calculate
            limiter gain. The processed output true peak is not guaranteed to
            remain below the ceiling. Supported sample rates are 8, 11.025, 12,
            16, 22.05, 24, 32, 44.1, 48, 88.2, and 96 kHz, plus every rate at or
            above 176.4 kHz.

    Returns:
        A tuple ``(audio, frame_gains)`` containing the limited audio and one linked
        gain value per frame as ``numpy.float32`` arrays.

    Raises:
        ValueError: If a configuration value or input is invalid.
    """
