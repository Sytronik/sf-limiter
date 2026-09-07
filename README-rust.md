# sf-limiter

<!-- [![Crates.io version](https://img.shields.io/crates/v/sf-limiter)](https://crates.io/crates/sf-limiter) -->
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Sytronik/sf-limiter/blob/main/LICENSE)

`sf-limiter` (short for “straightforward limiter”) is a look-ahead brick-wall
audio limiter with optional runtime SIMD acceleration.

For Python and NumPy, see the [Python documentation](https://github.com/Sytronik/sf-limiter/blob/main/README.md).

It applies one linked gain value to every channel in a frame, preserving the
relative balance between channels.

> **Non-streaming:** The current API processes a complete audio buffer offline.
> Each call starts with a fresh gain envelope, so limiter state does not carry
> across chunks or successive calls. The output has the same shape and length
> as the input, while the limiter uses future samples equal to its attack-time
> look-ahead.

> **Note:** The limiter algorithm itself was not AI-generated. Codex was used
> only to help with API design, test codes, documentation, and packaging.

## Usage

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
use sf_limiter::SFLimiter;

let mut planar = [0.0, 0.5, 3.0, -4.0, 0.25, -0.25];
let mut limiter = SFLimiter::with_default(48_000)?;
let frame_gains = limiter.process_planar_inplace(&mut planar, 2)?;

assert_eq!(frame_gains.len(), 3);
# Ok::<(), sf_limiter::LimiterError>(())
```

`SFLimiter::new(sample_rate, threshold_dBFS, attack_ms, hold_ms, release_ms, true_peak)`
accepts the sample rate in Hz, the ceiling in dBFS, times in milliseconds, and
a final boolean to enable true-peak limiting using the
[ITU-R BS.1770-5](https://www.itu.int/rec/R-REC-BS.1770-5-202311-I/en)
Annex 2 estimator. The
`with_default` constructor uses a 0 dBFS ceiling with 5/15/40 ms
attack/hold/release timing and keeps true-peak processing disabled. True-peak
detection supports 8, 11.025, 12, 16, 22.05, 24, 32, 44.1, 48, 88.2, and 96 kHz,
as well as 176.4 kHz and higher. Other sample rates are rejected when
`true_peak` is enabled; sample-peak mode accepts any positive `u32` sample
rate.

Input samples must be finite and within the inclusive range `[-2^32, 2^32]`.
Invalid samples, channel counts, or configuration values return `LimiterError`.

## Ceiling guarantee

For input samples within `[-2^32, 2^32]`, a valid channel count, and a
finite `threshold_dBFS` no greater than `0.0` dBFS, every returned discrete
sample is finite and has an absolute value no greater than the corresponding
linear ceiling (`10^(threshold_dBFS / 20)`). The test suite checks this with
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
crate with optional runtime SIMD acceleration.

## Development

Run the Rust tests, formatting check, and linter:

```shell
cargo test
cargo fmt -- --check
cargo clippy --all-targets -- -D warnings
```

The optional `python` feature enables
PyO3/NumPy bindings; see the [Python documentation](https://github.com/Sytronik/sf-limiter/blob/main/README.md)
for their development and distribution workflow.

## TODO

- [ ] Refine the Rust API
- [ ] Add a streaming API
- [ ] Publish the crate to crates.io

## SIMD acceleration

The default `simd` Cargo feature uses `pulp` to select CPU-specific code at
runtime for FIR interpolation, pre-upsampling, and gain application. Python
wheels include this feature; no CPU-specific build flags are required. Unsupported
CPUs use the baseline implementation. SIMD uses the existing auto-vectorizable
loops, so the compiler determines which operations are vectorized.

Rust users can disable runtime SIMD and its dependencies:

```toml
sf-limiter = { version = "0.2", default-features = false }
```

With no features enabled, the Rust core has no external dependencies. Its loops
can still be auto-vectorized for the chosen compilation target. For portable
wheels, do not globally enable AVX2 or use `target-cpu=native`.
