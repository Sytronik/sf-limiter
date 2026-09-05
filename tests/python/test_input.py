from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import ArrayLike

import sf_limiter

threshold_dBFS = -2.0
threshold = float(np.float32(10.0 ** (threshold_dBFS / 20.0)))


def test_audio_keyword_argument_is_supported() -> None:
    audio = np.array([0.0, 2.0, -3.0], dtype=np.float32)

    output, frame_gains = sf_limiter.limit(audio=audio, sample_rate=1_000)
    assert output.shape == audio.shape
    assert frame_gains.shape == audio.shape

    limiter = sf_limiter.SFLimiter(1_000)
    output, frame_gains = limiter.process(audio=audio)
    assert output.shape == audio.shape
    assert frame_gains.shape == audio.shape


def test_supported_sample_boundaries_are_accepted(
    process_audio: Callable[..., tuple[np.ndarray, np.ndarray]],
    assert_never_clips: Callable[..., None],
    entrypoint: str,
) -> None:
    maximum = np.float32(2**32)
    audio = np.array([-maximum, maximum], dtype=np.float32)

    output, frame_gains = process_audio(entrypoint, audio)

    assert output.shape == audio.shape
    assert frame_gains.shape == audio.shape
    assert_never_clips(output)


def test_channel_major_shape_is_preserved_by_default(
    assert_never_clips: Callable[..., None],
) -> None:
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
    output, frame_gains = limiter.process(audio)

    assert output.shape == audio.shape
    assert frame_gains.shape == (audio.shape[1],)
    assert_never_clips(output)


def test_channel_major_processing_matches_frame_major_processing() -> None:
    frame_major = np.array(
        [[0.0, 0.0], [4.0, -5.0], [-3.0, 2.0], [0.2, -0.2]],
        dtype=np.float32,
    )
    channel_major = frame_major.T

    frame_major_output, frame_gains_from_frame_major = sf_limiter.limit(
        frame_major,
        sample_rate=1_000,
        threshold_dBFS=threshold_dBFS,
        attack_ms=1.0,
        hold_ms=0.0,
        release_ms=0.0,
        axis=0,
    )
    channel_major_output, frame_gains_from_channel_major = sf_limiter.limit(
        channel_major,
        sample_rate=1_000,
        threshold_dBFS=threshold_dBFS,
        attack_ms=1.0,
        hold_ms=0.0,
        release_ms=0.0,
    )

    assert np.array_equal(frame_gains_from_channel_major, frame_gains_from_frame_major)
    assert np.array_equal(channel_major_output.T, frame_major_output)


def assert_non_contiguous_matches_contiguous(
    process_audio: Callable[..., tuple[np.ndarray, np.ndarray]],
    entrypoint: str,
    audio: np.ndarray,
    *,
    axis: int = -1,
) -> None:
    assert not audio.flags.c_contiguous
    original = audio.copy()
    contiguous = np.ascontiguousarray(audio)

    output, frame_gains = process_audio(
        entrypoint,
        audio,
        sample_rate=1_000,
        threshold_dBFS=threshold_dBFS,
        attack_ms=1.0,
        hold_ms=0.0,
        release_ms=0.0,
        axis=axis,
    )
    expected_output, expected_frame_gains = process_audio(
        entrypoint,
        contiguous,
        sample_rate=1_000,
        threshold_dBFS=threshold_dBFS,
        attack_ms=1.0,
        hold_ms=0.0,
        release_ms=0.0,
        axis=axis,
    )

    assert np.array_equal(audio, original)
    assert np.array_equal(output, expected_output)
    assert np.array_equal(frame_gains, expected_frame_gains)


@pytest.mark.parametrize(
    "audio",
    [
        np.linspace(-4.0, 4.0, 48, dtype=np.float32)[1::3],
        np.linspace(-4.0, 4.0, 48, dtype=np.float32)[::-2],
    ],
    ids=["positive-stride", "negative-stride"],
)
def test_non_contiguous_mono_matches_contiguous_input(
    process_audio: Callable[..., tuple[np.ndarray, np.ndarray]],
    entrypoint: str,
    audio: np.ndarray,
) -> None:
    assert_non_contiguous_matches_contiguous(process_audio, entrypoint, audio)


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
    process_audio: Callable[..., tuple[np.ndarray, np.ndarray]],
    entrypoint: str,
    audio: np.ndarray,
) -> None:
    assert_non_contiguous_matches_contiguous(process_audio, entrypoint, audio, axis=0)


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
    process_audio: Callable[..., tuple[np.ndarray, np.ndarray]],
    entrypoint: str,
    audio: np.ndarray,
) -> None:
    assert_non_contiguous_matches_contiguous(process_audio, entrypoint, audio)


@pytest.mark.parametrize("shape", [(), (2, 3, 4)], ids=["scalar", "3d"])
def test_invalid_dimensions_are_rejected(
    process_audio: Callable[..., tuple[np.ndarray, np.ndarray]],
    entrypoint: str,
    shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="1D or 2D"):
        process_audio(entrypoint, np.zeros(shape))


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
    process_audio: Callable[..., tuple[np.ndarray, np.ndarray]],
    entrypoint: str,
    audio: np.ndarray,
    axis: int,
    dimensions: int,
) -> None:
    with pytest.raises(
        ValueError, match=rf"axis={axis} is invalid for a {dimensions}D array"
    ):
        process_audio(entrypoint, audio, axis=axis)


@pytest.mark.parametrize(
    ("shape", "axis"),
    [
        ((0,), -1),
        ((2, 0), -1),
        ((0, 2), 0),
    ],
    ids=["mono", "channel-major", "frame-major"],
)
def test_empty_frame_dimension_is_supported(
    process_audio: Callable[..., tuple[np.ndarray, np.ndarray]],
    entrypoint: str,
    shape: tuple[int, ...],
    axis: int,
) -> None:
    audio = np.zeros(shape, dtype=np.float32)

    output, frame_gains = process_audio(entrypoint, audio, axis=axis)

    assert output.shape == audio.shape
    assert output.dtype == np.float32
    assert frame_gains.shape == (0,)
    assert frame_gains.dtype == np.float32


@pytest.mark.parametrize(
    ("shape", "axis"),
    [
        ((0, 2), -1),
        ((2, 0), 0),
    ],
    ids=["channel-major", "frame-major"],
)
def test_empty_channel_dimension_is_rejected(
    process_audio: Callable[..., tuple[np.ndarray, np.ndarray]],
    entrypoint: str,
    shape: tuple[int, int],
    axis: int,
) -> None:
    audio = np.zeros(shape, dtype=np.float32)

    with pytest.raises(ValueError, match="channel dimension must not be empty"):
        process_audio(entrypoint, audio, axis=axis)


@pytest.mark.parametrize(
    "audio",
    [
        [0.25, 2.0, -3.0],
        [[0.25, 2.0, -3.0], [-0.5, 1.0, 0.0]],
        np.array([0, 2, -3], dtype=np.int16),
        np.array([[0.1, 2.0, -3.0], [-0.2, 1.0, 0.0]], dtype=np.float64),
    ],
    ids=["mono-list", "nested-list", "int16", "float64"],
)
def test_arraylike_matches_float32_input(
    process_audio: Callable[..., tuple[np.ndarray, np.ndarray]],
    entrypoint: str,
    audio: ArrayLike,
) -> None:
    converted = np.asarray(audio, dtype=np.float32)
    original = np.array(audio, copy=True)
    output, gains = process_audio(entrypoint, audio)
    np.testing.assert_array_equal(audio, original)
    expected_output, expected_gains = process_audio(entrypoint, converted)
    assert output.dtype == gains.dtype == np.float32
    assert output.shape == converted.shape
    assert gains.shape == (converted.shape[-1],)
    np.testing.assert_array_equal(output, expected_output)
    np.testing.assert_array_equal(gains, expected_gains)


@pytest.mark.parametrize(
    ("shape", "axis", "alias"),
    [((6,), 0, -1), ((3, 2), 0, -2), ((2, 3), 1, -1)],
    ids=["mono", "frame-major", "channel-major"],
)
def test_valid_axis_aliases_match(
    process_audio: Callable[..., tuple[np.ndarray, np.ndarray]],
    entrypoint: str,
    shape: tuple[int, ...],
    axis: int,
    alias: int,
) -> None:
    audio = np.array([0.0, 2.0, -3.0, 0.5, 1.0, -0.25], dtype=np.float32).reshape(shape)
    output, gains = process_audio(entrypoint, audio, axis=axis)
    alias_output, alias_gains = process_audio(entrypoint, audio, axis=alias)
    np.testing.assert_array_equal(output, alias_output)
    np.testing.assert_array_equal(gains, alias_gains)


@pytest.mark.parametrize("readonly", [False, True], ids=["writable", "readonly"])
@pytest.mark.parametrize(
    ("shape", "axis"),
    [((6,), -1), ((3, 2), 0), ((2, 3), -1)],
    ids=["mono", "frame-major", "channel-major"],
)
def test_contiguous_input_is_not_modified_or_shared(
    process_audio: Callable[..., tuple[np.ndarray, np.ndarray]],
    assert_never_clips: Callable[..., None],
    entrypoint: str,
    shape: tuple[int, ...],
    axis: int,
    readonly: bool,
) -> None:
    audio = np.linspace(-3.0, 3.0, 6, dtype=np.float32).reshape(shape)
    original = audio.copy()
    audio.flags.writeable = not readonly
    assert audio.flags.c_contiguous
    output, gains = process_audio(entrypoint, audio, axis=axis)
    np.testing.assert_array_equal(audio, original)
    assert not np.shares_memory(output, audio)
    assert not np.shares_memory(gains, audio)
    assert output.dtype == gains.dtype == np.float32
    assert output.shape == shape
    assert gains.shape == (shape[axis],)
    assert_never_clips(output)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (np.nan, "is not finite"),
        (np.inf, "is not finite"),
        (-np.inf, "is not finite"),
        (np.nextafter(np.float32(2**32), np.float32(np.inf)), "must be between"),
        (np.nextafter(np.float32(-(2**32)), np.float32(-np.inf)), "must be between"),
    ],
    ids=["nan", "inf", "neg-inf", "above-max", "below-min"],
)
@pytest.mark.parametrize("reversed_view", [False, True], ids=["contiguous", "reversed"])
@pytest.mark.parametrize(
    ("shape", "axis"),
    [((6,), -1), ((3, 2), 0), ((2, 3), -1)],
    ids=["mono", "frame-major", "channel-major"],
)
def test_invalid_sample_reports_logical_input_index(
    process_audio: Callable[..., tuple[np.ndarray, np.ndarray]],
    entrypoint: str,
    shape: tuple[int, ...],
    axis: int,
    reversed_view: bool,
    value: float,
    message: str,
) -> None:
    audio = np.zeros(shape, dtype=np.float32)
    if reversed_view:
        audio = audio[(slice(None, None, -1),) * audio.ndim]
        assert not audio.flags.c_contiguous
    audio.flat[3] = value
    with pytest.raises(ValueError, match=rf"flat index 3 {message}"):
        process_audio(entrypoint, audio, axis=axis)
