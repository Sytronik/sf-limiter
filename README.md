# sf-limiter

[![PyPI version](https://img.shields.io/pypi/v/sf-limiter)](https://pypi.org/project/sf-limiter/)
[![PyPI downloads](https://img.shields.io/pypi/dm/sf-limiter)](https://pypi.org/project/sf-limiter/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Sytronik/sf-limiter/blob/main/LICENSE)
<!-- [![Crates.io version](https://img.shields.io/crates/v/sf-limiter)](https://crates.io/crates/sf-limiter) -->

`sf-limiter` (short for “straightforward limiter”) is a look-ahead brick-wall
audio limiter with a Rust core and Python bindings for NumPy.

It applies one linked gain value to every channel in a frame, preserving the
relative balance between channels.

> **Non-streaming:** The current API processes a complete audio buffer offline.
> Each call starts with a fresh gain envelope, so limiter state does not carry
> across chunks or successive calls. The output has the same shape and length
> as the input, while the limiter uses future samples equal to its attack-time
> look-ahead.

> **Note:** The limiter algorithm itself was not AI-generated. Codex was used
> only to help with API design, documentation, and packaging.

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
new `float32` array, and `frame_gains` contains one value per frame.

Each call starts from a neutral gain envelope. Non-finite samples and invalid
configuration values raise `ValueError`.

## Rust

The core Rust API accepts flat `f32` samples in either of these layouts:

- **Frame-interleaved:** each frame contains one sample per channel. Use
  `process_interleaved` or `process_interleaved_inplace`.
- **Channel-planar:** all frames of the first channel are followed by all
  frames of the next channel. Use `process_planar` or
  `process_planar_inplace`.

For frame-interleaved audio:

```rust
use sf_limiter::SFLimiter;

let input = [0.0, 0.5, 3.0, -4.0, 0.25];
let mut limiter = SFLimiter::with_default(48_000)?;
let output = limiter.process_interleaved(&input, 1)?;

assert!(output.audio.iter().all(|sample| sample.abs() <= 1.0));
# Ok::<(), sf_limiter::LimiterError>(())
```

Use `process_interleaved_inplace` to reuse the input allocation. Both methods
return one linked gain value per frame and reset the envelope on every call.

For channel-planar audio:

```rust
let mut planar = [0.0, 0.5, 3.0, -4.0, 0.25, -0.25];
let mut limiter = SFLimiter::with_default(48_000)?;
let frame_gains = limiter.process_planar_inplace(&mut planar, 2)?;

assert_eq!(frame_gains.len(), 3);
# Ok::<(), sf_limiter::LimiterError>(())
```

The Python binding always uses the planar core path. Default two-dimensional
`axis=-1` input is already planar and is processed directly;
`(frames, channels)` input is transposed to planar layout for processing and
then restored to its original layout for the returned array. The interleaved
core path remains available to Rust callers.

`SFLimiter::new` accepts `true_peak` as its final boolean argument. The
`with_default` constructor keeps true-peak processing disabled. True-peak
detection supports 8, 11.025, 12, 16, 22.05, 24, 32, 44.1, 48, 88.2, and 96 kHz,
as well as 176.4 kHz and higher. Other sample rates are rejected when
`true_peak` is enabled; sample-peak mode accepts any positive `u32` sample
rate.

## Ceiling guarantee

For finite input, a valid channel count, and a finite `threshold_dBFS` no greater
than `0.0` dBFS, every returned discrete sample is finite and has an absolute
value no greater than the corresponding linear ceiling
(`10 ** (threshold_dBFS / 20)`). The test suite checks this with large impulses,
high-level deterministic noise, mono input, and linked multichannel input.

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

Run the Rust tests:

```shell
cargo test
```

Create the Python environment with the test dependencies, build the extension,
and run its tests:

```shell
uv sync
uv run pytest -q
```

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

- [ ] Refine the Rust API
- [ ] Add a streaming API
- [ ] Publish the crate to crates.io
