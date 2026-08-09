import numpy as np
import pytest

import sf_limiter


def assert_never_clips(audio: np.ndarray, ceiling: float = 1.0) -> None:
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio), initial=0.0) <= ceiling


def test_float64_mono_input_never_clips() -> None:
    time = np.arange(48_000, dtype=np.float64)
    audio = np.sin(time * 0.071) * 8.0
    audio[::997] = 32.0
    original = audio.copy()

    output, gains = sf_limiter.limit(audio, sample_rate=48_000)

    assert output.shape == audio.shape
    assert output.dtype == np.float32
    assert gains.shape == audio.shape
    assert np.array_equal(audio, original)
    assert_never_clips(output)
    assert np.all((0.0 <= gains) & (gains <= 1.0))


def test_frame_major_multichannel_input_never_clips() -> None:
    rng = np.random.default_rng(42)
    audio = rng.normal(0.0, 12.0, size=(24_000, 6)).astype(np.float32)
    audio[::4093, :] = 64.0

    limiter = sf_limiter.SFLimiter(
        48_000,
        threshold=0.8,
        attack_ms=5.0,
        hold_ms=15.0,
        release_ms=40.0,
    )
    output, gains = limiter.process(audio)

    assert output.shape == audio.shape
    assert gains.shape == (audio.shape[0],)
    assert limiter.latency_samples == 240
    assert_never_clips(output, 0.8)


def test_channel_major_shape_is_preserved() -> None:
    audio = np.array(
        [[0.0, 4.0, -3.0, 0.2], [0.0, -5.0, 2.0, -0.2]],
        dtype=np.float32,
    )

    output, gains = sf_limiter.limit(
        audio,
        sample_rate=1_000,
        attack_ms=1.0,
        hold_ms=0.0,
        release_ms=0.0,
        channel_axis=0,
    )

    assert output.shape == audio.shape
    assert gains.shape == (audio.shape[1],)
    assert_never_clips(output)


def test_invalid_dimensions_are_rejected() -> None:
    with pytest.raises(ValueError, match="1D or 2D"):
        sf_limiter.limit(np.zeros((2, 3, 4)), sample_rate=48_000)
