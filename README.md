# sf-limiter

[![PyPI version](https://img.shields.io/pypi/v/sf-limiter)](https://pypi.org/project/sf-limiter/)
[![PyPI downloads](https://img.shields.io/pypi/dm/sf-limiter)](https://pypi.org/project/sf-limiter/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Sytronik/sf-limiter/blob/main/LICENSE)

`sf-limiter` (short for “straightforward limiter”) is a look-ahead brick-wall
audio limiter for Python and NumPy, implemented in Rust.

For the Rust crate, see the [Rust documentation](https://github.com/Sytronik/sf-limiter/blob/main/README-rust.md).

It applies one linked gain value to every channel in a frame, preserving the
relative balance between channels.

> **Non-streaming:** The current API processes a complete audio buffer offline.
> Each call starts with a fresh gain envelope, so limiter state does not carry
> across chunks or successive calls. The output has the same shape and length
> as the input, while the limiter uses future samples equal to its attack-time
> look-ahead.

> **Note:** The limiter algorithm itself was not AI-generated. Codex was used
> only to help with API design, test codes, documentation, and packaging.

## Python 3.11+

Install the package from PyPI:

```shell
python -m pip install sf-limiter
```

Limit a mono NumPy array:

```python
import numpy as np
import sf_limiter

audio = np.array([0.0, 0.5, 3.0, -4.0, 0.25], dtype=np.float64)
limited, frame_gains = sf_limiter.limit(audio, sample_rate=48_000)

assert limited.dtype == np.float32
assert np.max(np.abs(limited), initial=0.0) <= 1.0
```

The one-shot `limit` function accepts these keyword parameters:

- `threshold_dBFS=0.0` (dBFS)
- `attack_ms=5.0`
- `hold_ms=15.0`
- `release_ms=40.0`
- `axis=-1`
- `true_peak=False`

For repeated use, configure a limiter object once:

```python
limiter = sf_limiter.SFLimiter(
    48_000,
    threshold_dBFS=-1.0,
    attack_ms=5.0,
    hold_ms=15.0,
    release_ms=40.0,
    true_peak=True,
)
limited, frame_gains = limiter.process(audio)
```

`limiter.threshold_dBFS` returns the configured dBFS value, while
`limiter.threshold` returns the corresponding linear amplitude and
`limiter.true_peak` reports whether true-peak limiting is enabled. With
`true_peak=True`, the limiter uses the
[ITU-R BS.1770-5](https://www.itu.int/rec/R-REC-BS.1770-5-202311-I/en)
Annex 2 estimator when calculating gain. It does not remeasure and correct the
processed output, so `threshold_dBFS` is not a guaranteed dBTP ceiling.
With `true_peak=True`, supported sample rates are 8, 11.025, 12, 16, 22.05,
24, 32, 44.1, 48, 88.2, and 96 kHz, as well as 176.4 kHz and higher.

Input may be a one-dimensional mono array or a two-dimensional multichannel
array. The last axis is interpreted as frames by default, so the usual shape is
`(channels, frames)`. Pass `axis=0` for `(frames, channels)`. The
input is converted to `float32` without being mutated; the returned audio is a
new `float32` array, and `frame_gains` contains one value per frame. Input
samples must be finite and within the inclusive range `[-2 ** 32, 2 ** 32]`.

Each call starts from a neutral gain envelope. Non-finite or out-of-range samples
and invalid configuration values raise `ValueError`.

## Ceiling guarantee

For input samples within `[-2 ** 32, 2 ** 32]`, a valid channel count, and a
finite `threshold_dBFS` no greater than `0.0` dBFS, every returned discrete
sample is finite and has an absolute value no greater than the corresponding
linear ceiling (`10 ** (threshold_dBFS / 20)`). The test suite checks this with
large impulses, high-level deterministic noise, mono input, and linked
multichannel input.

This is a brick-wall guarantee for discrete sample peaks in both modes. With
true-peak limiting enabled, the BS.1770-5 estimate is used to calculate the
gain envelope and generally reduces inter-sample peaks. Gain changes can create
new reconstructed peaks, however, and the output is not remeasured or corrected.
The output true peak is therefore not guaranteed to stay below the configured
ceiling.

## Design reference

The limiter design was informed by Geraint Luff's
[“Designing a straightforward limiter”](https://signalsmith-audio.co.uk/writing/2022/limiter/)
(Signalsmith Audio, 2022). In particular, this implementation follows the
article's look-ahead structure: a moving minimum of permissible gain, an
exponential release, and finite-length cascaded box-filter smoothing.

The Rust implementation was extracted from `limiter.rs` in
[thesia](https://github.com/Sytronik/thesia) and adapted into a standalone
crate with a dependency-free core.

## Development

Developing the Python package requires a Rust toolchain because `uv sync`
builds the native extension from source. Install the stable toolchain with
[`rustup`](https://rust-lang.org/tools/install/). On macOS or Linux:

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

On Windows, use the installer on the same page and install the Visual Studio
C++ Build Tools when prompted. Restart your terminal after installation so
that `rustc` and `cargo` are available on `PATH`, then verify:

```shell
rustc --version
cargo --version
```

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
using the instructions for your operating system. Clone the repository and
change to its root directory:

```shell
git clone https://github.com/Sytronik/sf-limiter.git
cd sf-limiter
```

Run the following commands from the repository root to create the Python
environment with the test dependencies, build the extension, and run its tests:

```shell
uv sync
uv run pytest -q
```

CI runs the Python suite on Python 3.11–3.14 and Rust tests in debug and
release profiles. It also builds wheel and source distributions on Linux,
macOS, and Windows, installs each in a separate temporary environment, and
runs the Python suite outside the checkout. These tests check installed
metadata, typing files, public API signatures, and the README Python examples.
To run the distribution checks locally (requires Python with `venv` and Rust):

```shell
uv build
python3 scripts/test_distribution.py "dist/*.whl"
python3 scripts/test_distribution.py "dist/*.tar.gz"
```

Each pattern must match exactly one distribution. Release tags run the same
CI checks, then test every release wheel on its native architecture and
rebuild/install the release sdist before allowing publication to PyPI.

Compare the Python API performance with
[`numpy-audio-limiter`](https://github.com/iver56/numpy-audio-limiter):

```shell
uv sync --group benchmark
uv run --group benchmark python benchmarks/compare_numpy_audio_limiter.py
```

The benchmark uses contiguous channel-planar arrays shaped
`(channels, frames)` for both implementations and measures reusable and
one-shot `sf_limiter` calls separately. Use `--help` to select durations,
channel counts, timing repetitions, and limiter settings.

## TODO

- [ ] Add a streaming API
