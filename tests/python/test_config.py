from collections.abc import Callable

import numpy as np
import pytest

import sf_limiter


@pytest.mark.parametrize("sample_rate", [-1, 0, 2**32, 2**128])
def test_invalid_sample_rate_raises_value_error(sample_rate: int) -> None:
    message = "sample_rate must be a positive 32-bit integer"

    with pytest.raises(ValueError, match=message):
        sf_limiter.SFLimiter(sample_rate)

    with pytest.raises(ValueError, match=message):
        sf_limiter.limit(np.zeros(1, dtype=np.float32), sample_rate=sample_rate)


@pytest.mark.parametrize("sample_rate", [10_000, 47_999, 176_399])
def test_true_peak_rejects_unsupported_sample_rate(sample_rate: int) -> None:
    message = "true_peak does not support sample_rate"

    with pytest.raises(ValueError, match=message):
        sf_limiter.SFLimiter(sample_rate, true_peak=True)

    with pytest.raises(ValueError, match=message):
        sf_limiter.limit(
            np.zeros(1, dtype=np.float32),
            sample_rate=sample_rate,
            true_peak=True,
        )

    sf_limiter.SFLimiter(sample_rate, true_peak=False)


@pytest.mark.parametrize("threshold_dBFS", [0.1, np.inf, -np.inf, np.nan])
def test_invalid_dBFS_threshold_is_rejected(
    process_audio: Callable[..., tuple[np.ndarray, np.ndarray]],
    entrypoint: str,
    threshold_dBFS: float,
) -> None:
    with pytest.raises(ValueError, match="finite dBFS value"):
        process_audio(entrypoint, [0.0], threshold_dBFS=threshold_dBFS)


@pytest.mark.parametrize("parameter", ["attack_ms", "hold_ms", "release_ms"])
@pytest.mark.parametrize(
    "value", [-1.0, np.nan, np.inf, -np.inf], ids=["negative", "nan", "inf", "neg-inf"]
)
def test_invalid_timing_is_rejected(
    process_audio: Callable[..., tuple[np.ndarray, np.ndarray]],
    entrypoint: str,
    parameter: str,
    value: float,
) -> None:
    with pytest.raises(
        ValueError, match=rf"{parameter} must be finite and non-negative"
    ):
        process_audio(entrypoint, [0.0], **{parameter: value})


@pytest.mark.parametrize("attack_ms", [0.0, 0.49], ids=["zero", "rounds-to-zero"])
def test_sub_sample_attack_is_rejected(
    process_audio: Callable[..., tuple[np.ndarray, np.ndarray]],
    entrypoint: str,
    attack_ms: float,
) -> None:
    with pytest.raises(ValueError, match="attack_ms=.*rounds to zero samples"):
        process_audio(entrypoint, [0.0], sample_rate=1_000, attack_ms=attack_ms)


def test_half_sample_attack_and_zero_hold_release_are_accepted(
    process_audio: Callable[..., tuple[np.ndarray, np.ndarray]],
    assert_never_clips: Callable[..., None],
    entrypoint: str,
) -> None:
    settings = {"attack_ms": 0.5, "hold_ms": 0.0, "release_ms": 0.0}
    limiter = sf_limiter.SFLimiter(1_000, **settings)
    assert limiter.attack_samples == limiter.lookahead_samples == 1
    assert limiter.hold_samples == limiter.release_samples == 0
    output, gains = process_audio(
        entrypoint, [2.0, -2.0], sample_rate=1_000, **settings
    )
    assert_never_clips(output)
    assert np.all((0.0 <= gains) & (gains < 1.0))


def test_default_configuration_properties() -> None:
    limiter = sf_limiter.SFLimiter(48_000)
    assert limiter.sample_rate == 48_000
    assert limiter.threshold == 1.0
    assert limiter.threshold_dBFS == 0.0
    assert limiter.true_peak is False
    assert limiter.attack_samples == limiter.lookahead_samples == 240
    assert limiter.hold_samples == 720
    assert limiter.release_samples == 1920.0


def test_custom_configuration_properties() -> None:
    limiter = sf_limiter.SFLimiter(
        1_000, threshold_dBFS=-6.0, attack_ms=2.0, hold_ms=3.0, release_ms=2.5
    )
    assert limiter.sample_rate == 1_000
    assert limiter.threshold_dBFS == -6.0
    assert limiter.threshold == pytest.approx(10.0 ** (-6.0 / 20.0))
    assert limiter.attack_samples == limiter.lookahead_samples == 2
    assert limiter.hold_samples == 3
    assert limiter.release_samples == 2.5
