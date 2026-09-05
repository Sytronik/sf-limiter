from collections.abc import Callable

import numpy as np
import pytest

import sf_limiter

threshold_dBFS = -2.0
threshold = float(np.float32(10.0 ** (threshold_dBFS / 20.0)))


def test_true_peak_option_limits_inter_sample_peaks() -> None:
    frames = np.arange(256, dtype=np.float64)
    audio = (
        1.1 * np.sin(2.0 * np.pi * 12_000.0 * frames / 48_000.0 + np.pi / 4.0)
    ).astype(np.float32)

    sample_peak_output, _ = sf_limiter.limit(
        audio,
        sample_rate=48_000,
        attack_ms=1.0,
        hold_ms=0.0,
        release_ms=0.0,
    )
    limiter = sf_limiter.SFLimiter(
        48_000,
        attack_ms=1.0,
        hold_ms=0.0,
        release_ms=0.0,
        true_peak=True,
    )
    true_peak_output, frame_gains = limiter.process(audio)

    assert np.array_equal(sample_peak_output, audio)
    assert np.any(np.abs(true_peak_output) < np.abs(audio))
    assert np.all((0.0 <= frame_gains) & (frame_gains <= 1.0))
    assert limiter.true_peak is True
    assert "true_peak=true" in repr(limiter).lower()


@pytest.mark.parametrize("true_peak", [False, True], ids=["sample-peak", "true-peak"])
@pytest.mark.parametrize("layout", ["mono", "frame-major", "channel-major"])
def test_function_and_object_outputs_match(
    process_audio: Callable[..., tuple[np.ndarray, np.ndarray]],
    assert_never_clips: Callable[..., None],
    layout: str,
    true_peak: bool,
) -> None:
    mono = (1.1 * np.sin(np.arange(128) * np.pi / 2 + np.pi / 4)).astype(np.float32)
    audio = mono if layout == "mono" else np.stack([mono, -0.5 * mono])
    axis = -1
    if layout == "frame-major":
        audio = audio.T.copy()
        axis = 0
    settings = {
        "threshold_dBFS": -2.0,
        "attack_ms": 1.0,
        "hold_ms": 2.0,
        "release_ms": 3.5,
        "true_peak": true_peak,
    }
    output, gains = process_audio("function", audio, axis=axis, **settings)
    object_output, object_gains = process_audio("object", audio, axis=axis, **settings)
    assert output.shape == audio.shape
    assert gains.shape == (audio.shape[axis],)
    np.testing.assert_array_equal(output, object_output)
    np.testing.assert_array_equal(gains, object_gains)
    assert_never_clips(output, threshold)


@pytest.mark.parametrize("true_peak", [False, True], ids=["sample-peak", "true-peak"])
@pytest.mark.parametrize(
    "invalid", [None, np.nan, 2**33], ids=["hot-only", "nan", "out-of-range"]
)
def test_reused_object_starts_neutral_after_processing_or_error(
    true_peak: bool, invalid: float | None
) -> None:
    settings = {
        "attack_ms": 1.0,
        "hold_ms": 2.0,
        "release_ms": 1_000.0,
        "true_peak": true_peak,
    }
    reused = sf_limiter.SFLimiter(48_000, **settings)
    reused.process(np.full(128, 8.0, dtype=np.float32))
    if invalid is not None:
        with pytest.raises(ValueError):
            reused.process(np.array([0.0, invalid], dtype=np.float32))
    quiet = np.full(64, 0.25, dtype=np.float32)
    output, gains = reused.process(quiet)
    fresh_output, fresh_gains = sf_limiter.SFLimiter(48_000, **settings).process(quiet)
    np.testing.assert_array_equal(output, fresh_output)
    np.testing.assert_array_equal(gains, fresh_gains)


@pytest.mark.parametrize("axis", [0, 1], ids=["frame-major", "channel-major"])
def test_reported_gain_is_linked_across_channels(
    process_audio: Callable[..., tuple[np.ndarray, np.ndarray]],
    assert_never_clips: Callable[..., None],
    entrypoint: str,
    axis: int,
) -> None:
    audio = np.array(
        [[4.0, -1.0, 0.5], [-0.25, 2.0, -1.0], [0.1, -0.2, 0.3]], dtype=np.float32
    )
    if axis == 1:
        audio = audio.T.copy()
    output, gains = process_audio(
        entrypoint,
        audio,
        axis=axis,
        threshold_dBFS=-2.0,
        attack_ms=1.0,
        hold_ms=0.0,
        release_ms=0.0,
    )
    broadcast_gains = gains[:, None] if axis == 0 else gains[None, :]
    np.testing.assert_allclose(
        output, audio * broadcast_gains, rtol=2 * np.finfo(np.float32).eps, atol=0
    )
    assert np.all(np.isfinite(gains) & (gains >= 0.0) & (gains <= 1.0))
    assert np.any(gains < 1.0)
    assert_never_clips(output, threshold)
