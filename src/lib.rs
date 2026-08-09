//! A dependency-free look-ahead brick-wall limiter core.
//!
//! `SFLimiter` accepts frame-interleaved `f32` audio. It computes one gain
//! value per frame across every channel, so the stereo image is preserved.
//!
//! The design was informed by Geraint Luff's article
//! [“Designing a straightforward limiter”](https://signalsmith-audio.co.uk/writing/2022/limiter/).

mod envelope;

#[cfg(feature = "python")]
mod python;

use std::error::Error;
use std::fmt::{self, Display, Formatter};

use envelope::{BoxStackFilter, ExponentialRelease, MovingMinimum};

/// Validation and input errors returned by [`SFLimiter`].
#[derive(Clone, Debug, PartialEq)]
pub enum LimiterError {
    InvalidSampleRate,
    InvalidThreshold(f64),
    InvalidTime {
        parameter: &'static str,
        value_ms: f64,
    },
    AttackTooShort {
        value_ms: f64,
        sample_rate: u32,
    },
    InvalidChannelCount,
    InputNotFrameAligned {
        sample_count: usize,
        channels: usize,
    },
    NonFiniteSample {
        index: usize,
    },
}

impl Display for LimiterError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSampleRate => formatter.write_str("sample_rate must be greater than zero"),
            Self::InvalidThreshold(value) => {
                write!(
                    formatter,
                    "threshold must be finite and in (0, 1], got {value}"
                )
            }
            Self::InvalidTime {
                parameter,
                value_ms,
            } => write!(
                formatter,
                "{parameter} must be finite and non-negative, got {value_ms}"
            ),
            Self::AttackTooShort {
                value_ms,
                sample_rate,
            } => write!(
                formatter,
                "attack_ms={value_ms} rounds to zero samples at {sample_rate} Hz"
            ),
            Self::InvalidChannelCount => formatter.write_str("channels must be greater than zero"),
            Self::InputNotFrameAligned {
                sample_count,
                channels,
            } => write!(
                formatter,
                "{sample_count} samples cannot be divided into frames with {channels} channels"
            ),
            Self::NonFiniteSample { index } => {
                write!(
                    formatter,
                    "input sample at flat index {index} is not finite"
                )
            }
        }
    }
}

impl Error for LimiterError {}

/// The result of an allocating limiter call.
#[derive(Clone, Debug, PartialEq)]
pub struct LimiterOutput {
    /// Limited frame-interleaved audio.
    pub samples: Vec<f32>,
    /// The gain applied to each frame.
    pub gains: Vec<f32>,
}

/// A look-ahead limiter with a finite-length, smoothly varying gain envelope.
#[derive(Clone, Debug)]
pub struct SFLimiter {
    sample_rate: u32,
    threshold: f64,
    attack_samples: usize,
    hold_samples: usize,
    moving_minimum: MovingMinimum,
    release: ExponentialRelease,
    smoother: BoxStackFilter,
}

impl SFLimiter {
    /// Builds a limiter from times expressed in milliseconds.
    ///
    /// `threshold` must be in `(0, 1]`. `attack_ms` must round to at least one
    /// sample; hold and release may be zero.
    pub fn new(
        sample_rate: u32,
        threshold: f64,
        attack_ms: f64,
        hold_ms: f64,
        release_ms: f64,
    ) -> Result<Self, LimiterError> {
        if sample_rate == 0 {
            return Err(LimiterError::InvalidSampleRate);
        }
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) || threshold == 0.0 {
            return Err(LimiterError::InvalidThreshold(threshold));
        }
        validate_time("attack_ms", attack_ms)?;
        validate_time("hold_ms", hold_ms)?;
        validate_time("release_ms", release_ms)?;

        let attack_samples = milliseconds_to_samples(sample_rate, attack_ms);
        if attack_samples == 0 {
            return Err(LimiterError::AttackTooShort {
                value_ms: attack_ms,
                sample_rate,
            });
        }
        let total_hold_samples =
            milliseconds_to_samples(sample_rate, attack_ms + hold_ms).max(attack_samples);
        let hold_samples = total_hold_samples - attack_samples;
        let release_samples = release_ms * sample_rate as f64 / 1000.0;

        let mut limiter = Self {
            sample_rate,
            threshold,
            attack_samples,
            hold_samples,
            moving_minimum: MovingMinimum::new(total_hold_samples),
            release: ExponentialRelease::new(release_samples),
            smoother: BoxStackFilter::new(attack_samples),
        };
        limiter.reset();
        Ok(limiter)
    }

    /// Builds the limiter with a 1.0 ceiling and 5/15/40 ms
    /// attack/hold/release timing.
    pub fn with_default(sample_rate: u32) -> Result<Self, LimiterError> {
        Self::new(sample_rate, 1.0, 5.0, 15.0, 40.0)
    }

    /// Processes a copy of frame-interleaved audio.
    pub fn process_interleaved(
        &mut self,
        samples: &[f32],
        channels: usize,
    ) -> Result<LimiterOutput, LimiterError> {
        let mut output = samples.to_vec();
        let gains = self.process_interleaved_inplace(&mut output, channels)?;
        Ok(LimiterOutput {
            samples: output,
            gains,
        })
    }

    /// Processes frame-interleaved audio in place and returns one gain per frame.
    ///
    /// The limiter resets at the start of every call. This is an offline API:
    /// it uses future frames within [`Self::latency_samples`] but returns an
    /// array with the same length as the input.
    pub fn process_interleaved_inplace(
        &mut self,
        samples: &mut [f32],
        channels: usize,
    ) -> Result<Vec<f32>, LimiterError> {
        validate_samples(samples, channels)?;
        self.reset();

        if samples.is_empty() {
            return Ok(Vec::new());
        }

        let frame_count = samples.len() / channels;
        let mut envelope = Vec::with_capacity(frame_count + self.attack_samples);
        for frame in samples.chunks_exact(channels) {
            envelope.push(self.calculate_gain(frame));
        }
        for _ in 0..self.attack_samples {
            envelope.push(self.calculate_gain_from_peak(0.0));
        }

        let mut gains: Vec<f32> = envelope
            .into_iter()
            .skip(self.attack_samples)
            .map(|gain| gain as f32)
            .collect();

        for (frame_index, frame) in samples.chunks_exact_mut(channels).enumerate() {
            let peak = frame
                .iter()
                .map(|sample| f64::from(sample.abs()))
                .fold(0.0, f64::max);
            let mut gain = f64::from(gains[frame_index]);

            // The envelope construction should already satisfy this condition.
            // Lowering the whole frame is a numerical safety guard which keeps
            // rounding drift from ever crossing the configured ceiling.
            if peak * gain > self.threshold {
                gain = self.threshold / (peak + f64::EPSILON);
                gains[frame_index] = gain as f32;
            }

            for sample in frame {
                let limited = f64::from(*sample) * gain;
                *sample = limited.clamp(-self.threshold, self.threshold) as f32;
            }
        }

        Ok(gains)
    }

    /// Restores the internal gain envelope to its neutral state.
    pub fn reset(&mut self) {
        self.moving_minimum.reset();
        self.release.reset();
        self.smoother.reset(1.0);
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Look-ahead latency in samples.
    pub fn latency_samples(&self) -> usize {
        self.attack_samples
    }

    pub fn hold_samples(&self) -> usize {
        self.hold_samples
    }

    pub fn release_samples(&self) -> f64 {
        self.release.release_samples()
    }

    fn calculate_gain(&mut self, frame: &[f32]) -> f64 {
        let peak = frame
            .iter()
            .map(|sample| f64::from(sample.abs()))
            .fold(0.0, f64::max);
        self.calculate_gain_from_peak(peak)
    }

    fn calculate_gain_from_peak(&mut self, peak: f64) -> f64 {
        let raw_gain = if peak > self.threshold {
            self.threshold / (peak + f64::EPSILON)
        } else {
            1.0
        };
        let held = self.moving_minimum.step(raw_gain);
        let released = self.release.step(held);
        self.smoother.step(released).min(1.0)
    }
}

fn validate_time(parameter: &'static str, value_ms: f64) -> Result<(), LimiterError> {
    if value_ms.is_finite() && value_ms >= 0.0 {
        Ok(())
    } else {
        Err(LimiterError::InvalidTime {
            parameter,
            value_ms,
        })
    }
}

fn milliseconds_to_samples(sample_rate: u32, value_ms: f64) -> usize {
    (value_ms * sample_rate as f64 / 1000.0).round() as usize
}

fn validate_samples(samples: &[f32], channels: usize) -> Result<(), LimiterError> {
    if channels == 0 {
        return Err(LimiterError::InvalidChannelCount);
    }
    if !samples.len().is_multiple_of(channels) {
        return Err(LimiterError::InputNotFrameAligned {
            sample_count: samples.len(),
            channels,
        });
    }
    if let Some(index) = samples.iter().position(|sample| !sample.is_finite()) {
        return Err(LimiterError::NonFiniteSample { index });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_configuration() {
        assert_eq!(
            SFLimiter::with_default(0).unwrap_err(),
            LimiterError::InvalidSampleRate
        );
        assert!(matches!(
            SFLimiter::new(48_000, 1.1, 5.0, 15.0, 40.0),
            Err(LimiterError::InvalidThreshold(1.1))
        ));
        assert!(matches!(
            SFLimiter::new(48_000, 1.0, 0.0, 15.0, 40.0),
            Err(LimiterError::AttackTooShort { .. })
        ));
    }

    #[test]
    fn rejects_non_finite_audio() {
        let mut limiter = SFLimiter::with_default(48_000).unwrap();
        let error = limiter
            .process_interleaved(&[0.0, f32::NAN], 1)
            .unwrap_err();
        assert_eq!(error, LimiterError::NonFiniteSample { index: 1 });
    }

    #[test]
    fn empty_audio_is_supported() {
        let mut limiter = SFLimiter::with_default(48_000).unwrap();
        let output = limiter.process_interleaved(&[], 2).unwrap();
        assert!(output.samples.is_empty());
        assert!(output.gains.is_empty());
    }
}
