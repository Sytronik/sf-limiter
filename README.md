# sf-limiter (WIP)

`sf-limiter` (short for “straightforward limiter”) is a look-ahead brick-wall audio limiter with a
dependency-free Rust core and optional Python bindings for NumPy.

> **Note:** The limiter algorithm itself was not AI-generated. Codex assisted only with
> documentation and packaging-related work.

It applies one linked gain value to every channel in a frame, preserving the
relative balance between channels. Processing is offline: the output has the
same shape and length as the input, while the limiter uses future samples equal
to its attack-time look-ahead.

## Python 3.11+

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then
create the Python 3.11 virtual environment and install the package with:

```shell
uv sync --no-dev
```

`uv` installs a compatible Python 3.11 interpreter when needed, creates the
project's `.venv`, builds the Rust extension, and installs the locked
dependencies.

Limit a mono NumPy array:

```python
import numpy as np
import sf_limiter

audio = np.array([0.0, 0.5, 3.0, -4.0, 0.25], dtype=np.float64)
limited, gains = sf_limiter.limit(audio, sample_rate=48_000)

assert limited.dtype == np.float32
assert np.max(np.abs(limited), initial=0.0) <= 1.0
```

The one-shot `limit` function accepts these keyword parameters:

- `threshold=1.0`
- `attack_ms=5.0`
- `hold_ms=15.0`
- `release_ms=40.0`
- `channel_axis=-1`

For repeated use, configure a limiter object once:

```python
limiter = sf_limiter.SFLimiter(
    48_000,
    threshold=0.95,
    attack_ms=5.0,
    hold_ms=15.0,
    release_ms=40.0,
)
limited, gains = limiter.process(audio)
```

Input may be a one-dimensional mono array or a two-dimensional multichannel
array. The last axis is interpreted as channels by default, so the usual shape
is `(frames, channels)`. Pass `channel_axis=0` for `(channels, frames)`. The
input is converted to `float32` without being mutated; the returned audio is a
new `float32` array, and `gains` contains one value per frame.

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

assert!(output.samples.iter().all(|sample| sample.abs() <= 1.0));
# Ok::<(), sf_limiter::LimiterError>(())
```

Use `process_interleaved_inplace` to reuse the input allocation. Both methods
return one linked gain value per frame and reset the envelope on every call.

For channel-planar audio:

```rust
let mut planar = [0.0, 0.5, 3.0, -4.0, 0.25, -0.25];
let mut limiter = SFLimiter::with_default(48_000)?;
let gains = limiter.process_planar_inplace(&mut planar, 2)?;

assert_eq!(gains.len(), 3);
# Ok::<(), sf_limiter::LimiterError>(())
```

The Python binding always uses the planar core path. Two-dimensional
`channel_axis=0` input is already planar and is processed directly;
`(frames, channels)` input is transposed to planar layout for processing and
then restored to its original layout for the returned array. The interleaved
core path remains available to Rust callers.

## Ceiling guarantee

For finite input, a valid channel count, and a threshold in `(0, 1]`, every
returned discrete sample is finite and has an absolute value no greater than
the configured threshold. The test suite checks this with large impulses,
high-level deterministic noise, mono input, and linked multichannel input.

This is a sample-peak limiter. It does not oversample to detect reconstructed
inter-sample (true-peak) excursions.

## Design reference

The limiter design was informed by Geraint Luff's
[“Designing a straightforward limiter”](https://signalsmith-audio.co.uk/writing/2022/limiter/)
(Signalsmith Audio, 2022). In particular, this implementation follows the
article's look-ahead structure: a moving minimum of permissible gain, an
exponential release, and finite-length cascaded box-filter smoothing.

The Rust implementation was extracted from `limiter.rs` in [thesia](https://github.com/Sytronik/thesia) and adapted into a standalone crate with a dependency-free core.

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
