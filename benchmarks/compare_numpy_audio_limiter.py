"""Compare planar sf-limiter and numpy-audio-limiter Python API performance."""

import argparse
import gc
import math
import platform
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import numpy_audio_limiter

import sf_limiter

DEFAULT_DURATIONS = (0.1, 1.0, 10.0)
DEFAULT_CHANNELS = (1, 2)


@dataclass(frozen=True)
class Timing:
    median_seconds: float
    minimum_seconds: float
    maximum_seconds: float
    iterations: int


def package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


def time_constant_coefficient(milliseconds: float, sample_rate: int) -> float:
    samples = milliseconds * sample_rate / 1_000.0
    return 0.0 if samples == 0.0 else math.exp(-1.0 / samples)


def make_audio(frames: int, channels: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    planar = rng.normal(0.0, 0.35, size=(channels, frames)).astype(np.float32)
    planar[:, ::997] *= 4.0
    return planar


def measure(
    function: Callable[[], object],
    *,
    warmups: int,
    repeats: int,
    minimum_sample_seconds: float,
) -> Timing:
    for _ in range(warmups):
        function()

    started = time.perf_counter()
    function()
    elapsed = max(time.perf_counter() - started, sys.float_info.epsilon)
    iterations = max(1, math.ceil(minimum_sample_seconds / elapsed))

    timings: list[float] = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            started_ns = time.perf_counter_ns()
            for _ in range(iterations):
                function()
            elapsed_seconds = (time.perf_counter_ns() - started_ns) / 1e9
            timings.append(elapsed_seconds / iterations)
    finally:
        if gc_was_enabled:
            gc.enable()

    return Timing(
        median_seconds=statistics.median(timings),
        minimum_seconds=min(timings),
        maximum_seconds=max(timings),
        iterations=iterations,
    )


def validate_outputs(
    audio: np.ndarray,
    sf_reused: Callable[[], tuple[np.ndarray, np.ndarray]],
    sf_one_shot: Callable[[], tuple[np.ndarray, np.ndarray]],
    numpy_one_shot: Callable[[], np.ndarray],
) -> None:
    for name, function in (
        ("sf_limiter.SFLimiter.process", sf_reused),
        ("sf_limiter.limit", sf_one_shot),
    ):
        result = function()
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError(f"{name} returned an unexpected value")
        output, frame_gains = result
        if output.shape != audio.shape or frame_gains.shape != (audio.shape[1],):
            raise RuntimeError(f"{name} returned an unexpected shape")
        if output.dtype != np.float32 or not np.isfinite(output).all():
            raise RuntimeError(f"{name} returned invalid samples")

    reference_output = numpy_one_shot()
    if reference_output.shape != audio.shape:
        raise RuntimeError("numpy_audio_limiter.limit returned an unexpected shape")
    if reference_output.dtype != np.float32 or not np.isfinite(reference_output).all():
        raise RuntimeError("numpy_audio_limiter.limit returned invalid samples")


def print_header(*, sf_true_peak: bool) -> None:
    print(f"Python {platform.python_version()} ({platform.machine()})")
    print(f"NumPy {np.__version__}")
    print(f"sf-limiter {package_version('sf-limiter')}")
    print(f"numpy-audio-limiter {package_version('numpy-audio-limiter')}")
    print()
    print("Times include Python binding overhead and output allocation.")
    print("Inputs are contiguous channel-planar arrays shaped (channels, frames).")
    print("Deterministic input generation is excluded.")
    print(
        "sf-limiter returns audio and frame_gains; numpy-audio-limiter returns audio only."
    )
    print(
        f"sf-limiter true-peak detection is {'enabled' if sf_true_peak else 'disabled'}; "
        "numpy-audio-limiter always uses sample peaks."
    )
    print(
        "Limiter parameters are analogous, but the algorithms are not numerically equivalent."
    )
    print()


def print_result(
    frames: int, channels: int, sample_rate: int, timings: Sequence[tuple[str, Timing]]
) -> None:
    audio_seconds = frames / sample_rate
    baseline = next(
        timing.median_seconds
        for name, timing in timings
        if name == "numpy_audio_limiter.limit"
    )
    print(f"{audio_seconds:g} s, {channels} channel(s), {frames:,} frames")
    print(
        f"{'implementation':<34} {'median':>10} {'range':>21} "
        f"{'x realtime':>12} {'vs numpy':>10}"
    )
    for name, timing in timings:
        median_ms = timing.median_seconds * 1_000.0
        minimum_ms = timing.minimum_seconds * 1_000.0
        maximum_ms = timing.maximum_seconds * 1_000.0
        realtime = audio_seconds / timing.median_seconds
        relative = baseline / timing.median_seconds
        print(
            f"{name:<34} {median_ms:>8.3f} ms "
            f"{minimum_ms:>8.3f}..{maximum_ms:<8.3f} "
            f"{realtime:>11.1f}x {relative:>9.2f}x"
        )
    print()


def run(args: argparse.Namespace) -> None:
    attack_samples = max(1, round(args.attack_ms * args.sample_rate / 1_000.0))
    attack_coefficient = time_constant_coefficient(args.attack_ms, args.sample_rate)
    release_coefficient = time_constant_coefficient(args.release_ms, args.sample_rate)

    print_header(sf_true_peak=args.sf_true_peak)
    for i_duration, duration in enumerate(args.durations):
        frames = max(1, round(duration * args.sample_rate))
        for channels in args.channels:
            audio = make_audio(
                frames,
                channels,
                seed=args.seed + i_duration * 1_000 + channels,
            )
            limiter = sf_limiter.SFLimiter(
                args.sample_rate,
                threshold_dBFS=args.threshold_dBFS,
                attack_ms=args.attack_ms,
                hold_ms=0.0,
                release_ms=args.release_ms,
                true_peak=args.sf_true_peak,
            )

            def sf_reused(
                planar_audio: np.ndarray = audio,
                configured_limiter: sf_limiter.SFLimiter = limiter,
            ) -> tuple[np.ndarray, np.ndarray]:
                return configured_limiter.process(planar_audio)

            def sf_one_shot(
                planar_audio: np.ndarray = audio,
            ) -> tuple[np.ndarray, np.ndarray]:
                return sf_limiter.limit(
                    planar_audio,
                    sample_rate=args.sample_rate,
                    threshold_dBFS=args.threshold_dBFS,
                    attack_ms=args.attack_ms,
                    hold_ms=0.0,
                    release_ms=args.release_ms,
                    true_peak=args.sf_true_peak,
                )

            def numpy_one_shot(planar_audio: np.ndarray = audio) -> np.ndarray:
                # pyrefly: ignore [missing-attribute]
                return numpy_audio_limiter.limit(
                    signal=planar_audio,
                    attack_coeff=attack_coefficient,
                    release_coeff=release_coefficient,
                    delay=attack_samples,
                    threshold=10.0 ** (args.threshold_dBFS / 20.0),
                )

            validate_outputs(
                audio,
                sf_reused,
                sf_one_shot,
                numpy_one_shot,
            )
            implementations = (
                ("sf_limiter.SFLimiter.process", sf_reused),
                ("sf_limiter.limit", sf_one_shot),
                ("numpy_audio_limiter.limit", numpy_one_shot),
            )
            timings = [
                (
                    name,
                    measure(
                        function,
                        warmups=args.warmups,
                        repeats=args.repeats,
                        minimum_sample_seconds=args.minimum_sample_seconds,
                    ),
                )
                for name, function in implementations
            ]
            print_result(frames, channels, args.sample_rate, timings)


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError(
            "must be a finite number greater than or equal to zero"
        )
    return parsed


def threshold_dBFS_float(value: str) -> float:
    threshold_dBFS = float(value)
    if not math.isfinite(threshold_dBFS) or threshold_dBFS > 0.0:
        raise argparse.ArgumentTypeError(
            "must be a finite dBFS value less than or equal to zero"
        )
    return threshold_dBFS


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-rate", type=positive_int, default=48_000)
    parser.add_argument(
        "--durations", type=positive_float, nargs="+", default=DEFAULT_DURATIONS
    )
    parser.add_argument(
        "--channels", type=positive_int, nargs="+", default=DEFAULT_CHANNELS
    )
    parser.add_argument(
        "--threshold-dBFS",
        type=threshold_dBFS_float,
        default=-1.0,
        help="output ceiling in dBFS (default: %(default)s)",
    )
    parser.add_argument(
        "--sf-true-peak",
        action="store_true",
        help="enable true-peak detection for sf-limiter only",
    )
    parser.add_argument("--attack-ms", type=positive_float, default=5.0)
    parser.add_argument("--release-ms", type=nonnegative_float, default=40.0)
    parser.add_argument("--warmups", type=positive_int, default=3)
    parser.add_argument("--repeats", type=positive_int, default=7)
    parser.add_argument("--minimum-sample-seconds", type=positive_float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
