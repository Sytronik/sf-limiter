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
    output, gains = limiter.process(audio, axis=0)

    assert output.shape == audio.shape
    assert gains.shape == (audio.shape[0],)
    assert limiter.latency_samples == 240
    assert_never_clips(output, 0.8)


def test_channel_major_shape_is_preserved_by_default() -> None:
    audio = np.array(
        [[0.0, 4.0, -3.0, 0.2], [0.0, -5.0, 2.0, -0.2]],
        dtype=np.float32,
    )

    limiter = sf_limiter.SFLimiter(
        1_000,
        attack_ms=1.0,
        hold_ms=0.0,
        release_ms=0.0,
    )
    output, gains = limiter.process(audio)

    assert output.shape == audio.shape
    assert gains.shape == (audio.shape[1],)
    assert_never_clips(output)


def test_channel_major_processing_matches_frame_major_processing() -> None:
    frame_major = np.array(
        [[0.0, 0.0], [4.0, -5.0], [-3.0, 2.0], [0.2, -0.2]],
        dtype=np.float32,
    )
    channel_major = frame_major.T

    frame_output, frame_gains = sf_limiter.limit(
        frame_major,
        sample_rate=1_000,
        threshold=0.8,
        attack_ms=1.0,
        hold_ms=0.0,
        release_ms=0.0,
        axis=0,
    )
    channel_output, channel_gains = sf_limiter.limit(
        channel_major,
        sample_rate=1_000,
        threshold=0.8,
        attack_ms=1.0,
        hold_ms=0.0,
        release_ms=0.0,
    )

    assert np.array_equal(channel_gains, frame_gains)
    assert np.array_equal(channel_output.T, frame_output)


def assert_non_contiguous_matches_contiguous(
    audio: np.ndarray, *, axis: int = -1
) -> None:
    assert not audio.flags.c_contiguous
    original = audio.copy()
    contiguous = np.ascontiguousarray(audio)

    output, gains = sf_limiter.limit(
        audio,
        sample_rate=1_000,
        threshold=0.8,
        attack_ms=1.0,
        hold_ms=0.0,
        release_ms=0.0,
        axis=axis,
    )
    expected_output, expected_gains = sf_limiter.limit(
        contiguous,
        sample_rate=1_000,
        threshold=0.8,
        attack_ms=1.0,
        hold_ms=0.0,
        release_ms=0.0,
        axis=axis,
    )

    assert np.array_equal(audio, original)
    assert np.array_equal(output, expected_output)
    assert np.array_equal(gains, expected_gains)


@pytest.mark.parametrize(
    "audio",
    [
        np.linspace(-4.0, 4.0, 48, dtype=np.float32)[1::3],
        np.linspace(-4.0, 4.0, 48, dtype=np.float32)[::-2],
    ],
    ids=["positive-stride", "negative-stride"],
)
def test_non_contiguous_mono_matches_contiguous_input(audio: np.ndarray) -> None:
    assert_non_contiguous_matches_contiguous(audio)


@pytest.mark.parametrize(
    "audio",
    [
        np.asfortranarray(np.linspace(-4.0, 4.0, 72, dtype=np.float32).reshape(24, 3)),
        np.linspace(-4.0, 4.0, 192, dtype=np.float32).reshape(32, 6)[::2, 1::2],
        np.linspace(-4.0, 4.0, 72, dtype=np.float32).reshape(24, 3)[::-1, ::-1],
    ],
    ids=["fortran-order", "strided-axes", "reversed-axes"],
)
def test_non_contiguous_frame_major_matches_contiguous_input(
    audio: np.ndarray,
) -> None:
    assert_non_contiguous_matches_contiguous(audio, axis=0)


@pytest.mark.parametrize(
    "audio",
    [
        np.linspace(-4.0, 4.0, 72, dtype=np.float32).reshape(24, 3).T,
        np.linspace(-4.0, 4.0, 192, dtype=np.float32).reshape(6, 32)[1::2, ::2],
        np.linspace(-4.0, 4.0, 72, dtype=np.float32).reshape(3, 24)[::-1, ::-1],
    ],
    ids=["transposed", "strided-axes", "reversed-axes"],
)
def test_non_contiguous_channel_major_matches_contiguous_input(
    audio: np.ndarray,
) -> None:
    assert_non_contiguous_matches_contiguous(audio)


def test_invalid_dimensions_are_rejected() -> None:
    with pytest.raises(ValueError, match="1D or 2D"):
        sf_limiter.limit(np.zeros((2, 3, 4)), sample_rate=48_000)


@pytest.mark.parametrize(
    ("audio", "axis", "dimensions"),
    [
        (np.zeros(4, dtype=np.float32), 1, 1),
        (np.zeros(4, dtype=np.float32), -2, 1),
        (np.zeros((2, 3), dtype=np.float32), 2, 2),
        (np.zeros((2, 3), dtype=np.float32), -3, 2),
    ],
)
def test_invalid_frame_axes_are_rejected(
    audio: np.ndarray, axis: int, dimensions: int
) -> None:
    with pytest.raises(
        ValueError, match=rf"axis={axis} is invalid for a {dimensions}D array"
    ):
        sf_limiter.limit(audio, sample_rate=48_000, axis=axis)


@pytest.mark.parametrize(
    ("shape", "axis"),
    [
        ((2, 0), -1),
        ((0, 2), 0),
    ],
    ids=["channel-major", "frame-major"],
)
def test_empty_frame_dimension_is_supported(shape: tuple[int, int], axis: int) -> None:
    audio = np.zeros(shape, dtype=np.float32)

    output, gains = sf_limiter.limit(audio, sample_rate=48_000, axis=axis)

    assert output.shape == audio.shape
    assert output.dtype == np.float32
    assert gains.shape == (0,)
    assert gains.dtype == np.float32


@pytest.mark.parametrize(
    ("shape", "axis"),
    [
        ((0, 2), -1),
        ((2, 0), 0),
    ],
    ids=["channel-major", "frame-major"],
)
def test_empty_channel_dimension_is_rejected(shape: tuple[int, int], axis: int) -> None:
    audio = np.zeros(shape, dtype=np.float32)

    with pytest.raises(ValueError, match="channel dimension must not be empty"):
        sf_limiter.limit(audio, sample_rate=48_000, axis=axis)


def test_frame_major_non_finite_index_uses_input_order() -> None:
    audio = np.array([[0.0, np.nan], [1.0, 2.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="flat index 1"):
        sf_limiter.limit(audio, sample_rate=48_000, axis=0)
