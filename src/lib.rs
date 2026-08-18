//! A dependency-free look-ahead brick-wall limiter core.
//!
//! `SFLimiter` accepts frame-interleaved or channel-planar `f32` audio. It
//! computes one gain value per frame across every channel, so the stereo image
//! is preserved.
//!
//! The design was informed by Geraint Luff's article
//! [“Designing a straightforward limiter”](https://signalsmith-audio.co.uk/writing/2022/limiter/).

mod envelope;
mod peak;

#[cfg(feature = "python")]
mod python;

use std::error::Error;
use std::fmt::{self, Display, Formatter};

use envelope::{BoxStackFilter, ExponentialRelease, MovingMinimum};

/// Validation and input errors returned by [`SFLimiter`].
#[derive(Clone, Debug, PartialEq)]
pub enum LimiterError {
    InvalidSampleRate,
    UnsupportedTruePeakSampleRate(u32),
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
    #[allow(non_snake_case)]
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSampleRate => formatter.write_str("sample_rate must be greater than zero"),
            Self::UnsupportedTruePeakSampleRate(sample_rate) => write!(
                formatter,
                "true_peak does not support sample_rate={sample_rate}; \
                supported rates are 8000, 11025, 12000, 16000, 22050, 24000, 32000, \
                44100, 48000, 88200, 96000, and every rate at or above 176400 Hz"
            ),
            Self::InvalidThreshold(threshold_dBFS) => {
                write!(
                    formatter,
                    "threshold_dBFS must be a finite dBFS value less than or equal to 0, \
                    got {threshold_dBFS}"
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
    /// Limited audio in the same layout as the input.
    pub audio: Vec<f32>,
    /// The gain applied to each frame.
    pub frame_gains: Vec<f32>,
}

/// A look-ahead limiter with a finite-length, smoothly varying gain envelope.
#[derive(Clone, Debug)]
#[allow(non_snake_case)]
pub struct SFLimiter {
    sample_rate: u32,
    threshold_dBFS: f64,
    threshold: f64,
    true_peak: bool,
    attack_samples: usize,
    hold_samples: usize,
    moving_minimum: MovingMinimum,
    release: ExponentialRelease,
    smoother: BoxStackFilter,
}

impl SFLimiter {
    /// Builds a limiter from times expressed in milliseconds.
    ///
    /// `threshold_dBFS` is expressed in dBFS and must be finite and at most
    /// `0.0`. It is converted to a linear amplitude with
    /// `10^(threshold_dBFS / 20)`.
    /// `attack_ms` must round to at least one sample; hold and release may be
    /// zero. When `true_peak` is enabled, an ITU-R BS.1770-5 true-peak estimate
    /// is used to calculate the limiter gain. The processed output is not
    /// remeasured, so its true peak is not guaranteed to remain below the
    /// threshold. True-peak mode supports 8, 11.025, 12, 16, 22.05, 24, 32,
    /// 44.1, 48, 88.2, and 96 kHz, plus every sample rate at or above 176.4 kHz.
    #[allow(non_snake_case)]
    pub fn new(
        sample_rate: u32,
        threshold_dBFS: f64,
        attack_ms: f64,
        hold_ms: f64,
        release_ms: f64,
        true_peak: bool,
    ) -> Result<Self, LimiterError> {
        if sample_rate == 0 {
            return Err(LimiterError::InvalidSampleRate);
        }
        if true_peak && !peak::supports_true_peak_sample_rate(sample_rate) {
            return Err(LimiterError::UnsupportedTruePeakSampleRate(sample_rate));
        }
        if !threshold_dBFS.is_finite() || threshold_dBFS > 0.0 {
            return Err(LimiterError::InvalidThreshold(threshold_dBFS));
        }
        let threshold = dBFS_to_amplitude(threshold_dBFS);
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
            threshold_dBFS,
            threshold,
            true_peak,
            attack_samples,
            hold_samples,
            moving_minimum: MovingMinimum::new(total_hold_samples),
            release: ExponentialRelease::new(release_samples),
            smoother: BoxStackFilter::new(attack_samples),
        };
        limiter.reset();
        Ok(limiter)
    }

    /// Builds the limiter with a 0 dBFS ceiling and 5/15/40 ms
    /// attack/hold/release timing.
    pub fn with_default(sample_rate: u32) -> Result<Self, LimiterError> {
        Self::new(sample_rate, 0.0, 5.0, 15.0, 40.0, false)
    }

    /// Processes a copy of frame-interleaved audio.
    pub fn process_interleaved(
        &mut self,
        audio: &[f32],
        channels: usize,
    ) -> Result<LimiterOutput, LimiterError> {
        let mut output = audio.to_vec();
        let frame_gains = self.process_interleaved_inplace(&mut output, channels)?;
        Ok(LimiterOutput {
            audio: output,
            frame_gains,
        })
    }

    /// Processes frame-interleaved audio in place and returns one gain per frame.
    ///
    /// The limiter resets at the start of every call. This is an offline API:
    /// it uses future frames within [`Self::lookahead_samples`] but returns an
    /// array with the same length as the input.
    pub fn process_interleaved_inplace(
        &mut self,
        audio: &mut [f32],
        channels: usize,
    ) -> Result<Vec<f32>, LimiterError> {
        let frame_peaks = if self.true_peak {
            peak::collect_true_peaks_from_interleaved(audio, channels, self.sample_rate)?
        } else {
            peak::collect_sample_peaks_from_interleaved(audio, channels)?
        };
        let frame_gains = self.calculate_frame_gains(frame_peaks);
        apply_frame_gains_to_interleaved(audio, &frame_gains, channels, self.threshold as f32);
        Ok(frame_gains)
    }

    /// Processes a copy of channel-planar audio.
    ///
    /// The flat input contains every frame of the first channel, followed by
    /// every frame of the second channel, and so on.
    pub fn process_planar(
        &mut self,
        audio: &[f32],
        channels: usize,
    ) -> Result<LimiterOutput, LimiterError> {
        let mut output = audio.to_vec();
        let frame_gains = self.process_planar_inplace(&mut output, channels)?;
        Ok(LimiterOutput {
            audio: output,
            frame_gains,
        })
    }

    /// Processes channel-planar audio in place and returns one gain per frame.
    ///
    /// The flat input contains every frame of the first channel, followed by
    /// every frame of the second channel, and so on. The limiter resets at the
    /// start of every call.
    pub fn process_planar_inplace(
        &mut self,
        audio: &mut [f32],
        channels: usize,
    ) -> Result<Vec<f32>, LimiterError> {
        let frame_peaks = if self.true_peak {
            peak::collect_true_peaks_from_planar(audio, channels, self.sample_rate)?
        } else {
            peak::collect_sample_peaks_from_planar(audio, channels)?
        };
        let frame_gains = self.calculate_frame_gains(frame_peaks);
        apply_frame_gains_to_planar(audio, &frame_gains, self.threshold as f32);
        Ok(frame_gains)
    }

    fn calculate_frame_gains(&mut self, frame_peaks: Vec<f32>) -> Vec<f32> {
        self.reset();

        if frame_peaks.is_empty() {
            return frame_peaks;
        }

        let mut frame_gains = frame_peaks;
        let frame_count = frame_gains.len();
        for i_lookahead in 0..frame_count + self.attack_samples {
            let lookahead_peak = frame_gains
                .get(i_lookahead)
                .map_or(0.0, |peak| *peak as f64);
            let envelope_gain = self.calculate_gain_from_peak(lookahead_peak);

            if i_lookahead < self.attack_samples {
                continue;
            }

            let i_current_frame = i_lookahead - self.attack_samples;
            let current_peak = frame_gains[i_current_frame] as f64;

            // The envelope construction should already satisfy this condition.
            // Lowering the whole frame is a numerical safety guard which keeps
            // rounding drift from ever crossing the configured ceiling.
            if current_peak * envelope_gain > self.threshold {
                frame_gains[i_current_frame] =
                    (self.threshold / (current_peak + f64::EPSILON)) as f32;
            } else {
                frame_gains[i_current_frame] = envelope_gain as f32;
            }
        }

        frame_gains
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Returns the configured output ceiling as a linear amplitude.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Returns the configured output ceiling in dBFS.
    #[allow(non_snake_case)]
    pub fn threshold_dBFS(&self) -> f64 {
        self.threshold_dBFS
    }

    /// Whether ITU-R BS.1770-5 true-peak detection is enabled.
    pub fn true_peak(&self) -> bool {
        self.true_peak
    }

    /// Look-ahead latency in samples.
    pub fn lookahead_samples(&self) -> usize {
        self.attack_samples
    }

    /// Look-ahead latency in samples. (alias for [`Self::lookahead_samples`])
    pub fn attack_samples(&self) -> usize {
        self.attack_samples
    }

    pub fn hold_samples(&self) -> usize {
        self.hold_samples
    }

    pub fn release_samples(&self) -> f64 {
        self.release.release_samples()
    }

    /// Restores the internal gain envelope to its neutral state.
    fn reset(&mut self) {
        self.moving_minimum.reset();
        self.release.reset();
        self.smoother.reset(1.0);
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

#[allow(non_snake_case)]
fn dBFS_to_amplitude(threshold_dBFS: f64) -> f64 {
    10.0_f64.powf(threshold_dBFS / 20.0)
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

fn validate_layout(sample_count: usize, channels: usize) -> Result<usize, LimiterError> {
    if channels == 0 {
        return Err(LimiterError::InvalidChannelCount);
    }
    if !sample_count.is_multiple_of(channels) {
        return Err(LimiterError::InputNotFrameAligned {
            sample_count,
            channels,
        });
    }
    Ok(sample_count / channels)
}

fn apply_frame_gains_to_interleaved(
    audio: &mut [f32],
    frame_gains: &[f32],
    channels: usize,
    threshold: f32,
) {
    for (frame, gain) in audio.chunks_exact_mut(channels).zip(frame_gains) {
        for sample in frame {
            *sample = (*sample * *gain).clamp(-threshold, threshold);
        }
    }
}

fn apply_frame_gains_to_planar(audio: &mut [f32], frame_gains: &[f32], threshold: f32) {
    if frame_gains.is_empty() {
        return;
    }

    for channel in audio.chunks_exact_mut(frame_gains.len()) {
        for (sample, gain) in channel.iter_mut().zip(frame_gains) {
            *sample = (*sample * *gain).clamp(-threshold, threshold);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(non_snake_case)]
    fn rejects_invalid_configuration() {
        assert_eq!(
            SFLimiter::with_default(0).unwrap_err(),
            LimiterError::InvalidSampleRate
        );
        assert!(matches!(
            SFLimiter::new(48_000, 0.1, 5.0, 15.0, 40.0, false),
            Err(LimiterError::InvalidThreshold(0.1))
        ));
        assert!(matches!(
            SFLimiter::new(48_000, f64::NEG_INFINITY, 5.0, 15.0, 40.0, false),
            Err(LimiterError::InvalidThreshold(threshold_dBFS))
                if threshold_dBFS == f64::NEG_INFINITY
        ));
        assert!(matches!(
            SFLimiter::new(48_000, f64::NAN, 5.0, 15.0, 40.0, false),
            Err(LimiterError::InvalidThreshold(threshold_dBFS)) if threshold_dBFS.is_nan()
        ));
        assert!(matches!(
            SFLimiter::new(48_000, 0.0, 0.0, 15.0, 40.0, false),
            Err(LimiterError::AttackTooShort { .. })
        ));
    }

    #[test]
    fn true_peak_mode_accepts_only_supported_sample_rates() {
        for sample_rate in [
            8_000, 11_025, 12_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000, 88_200, 96_000,
            176_400, 192_000,
        ] {
            assert!(SFLimiter::new(sample_rate, 0.0, 5.0, 15.0, 40.0, true).is_ok());
        }

        for sample_rate in [10_000, 47_999, 176_399] {
            assert_eq!(
                SFLimiter::new(sample_rate, 0.0, 5.0, 15.0, 40.0, true).unwrap_err(),
                LimiterError::UnsupportedTruePeakSampleRate(sample_rate)
            );
            assert!(SFLimiter::new(sample_rate, 0.0, 5.0, 15.0, 40.0, false).is_ok());
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn threshold_is_configured_in_dBFS() {
        let threshold_dBFS = -6.0;
        let limiter = SFLimiter::new(48_000, threshold_dBFS, 5.0, 15.0, 40.0, false).unwrap();

        assert_eq!(limiter.threshold_dBFS(), threshold_dBFS);
        assert_eq!(limiter.threshold(), 10.0_f64.powf(threshold_dBFS / 20.0));
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
        assert!(output.audio.is_empty());
        assert!(output.frame_gains.is_empty());
    }

    #[test]
    fn attack_samples_equals_to_lookahead_samples() {
        let limiter = SFLimiter::new(48_000, 0.0, 12.0, 15.0, 40.0, false).unwrap();
        assert_eq!(limiter.attack_samples(), limiter.lookahead_samples());
    }

    #[test]
    fn true_peak_mode_uses_inter_sample_peaks_for_gain_control() {
        let input: Vec<_> = (0..256)
            .map(|index| {
                ((2.0 * std::f64::consts::PI * 12_000.0 * index as f64 / 48_000.0)
                    + std::f64::consts::FRAC_PI_4)
                    .sin() as f32
                    * 1.1
            })
            .collect();
        let mut sample_peak_limiter = SFLimiter::new(48_000, 0.0, 1.0, 0.0, 0.0, false).unwrap();
        let mut true_peak_limiter = SFLimiter::new(48_000, 0.0, 1.0, 0.0, 0.0, true).unwrap();

        let sample_peak_output = sample_peak_limiter.process_interleaved(&input, 1).unwrap();
        let true_peak_output = true_peak_limiter.process_interleaved(&input, 1).unwrap();
        assert_eq!(sample_peak_output.audio, input);
        assert!(
            true_peak_output
                .audio
                .iter()
                .zip(&input)
                .any(|(output, input)| output.abs() < input.abs())
        );
        assert!(true_peak_limiter.true_peak());
        assert!(!sample_peak_limiter.true_peak());
    }

    #[test]
    fn true_peak_mode_reports_the_gain_applied_to_an_over_ceiling_sample() {
        let mut input = vec![0.0; 128];
        input[64] = 1.02;
        let mut limiter = SFLimiter::new(48_000, 0.0, 1.0, 0.0, 0.0, true).unwrap();

        let output = limiter.process_interleaved(&input, 1).unwrap();

        assert!(output.frame_gains[64] < 1.0);
        assert_eq!(output.audio[64], input[64] * output.frame_gains[64]);
    }

    #[test]
    fn true_peak_mode_enforces_a_negative_sample_peak_ceiling() {
        let input: Vec<_> = (0..256)
            .map(|index| {
                ((2.0 * std::f64::consts::PI * 12_000.0 * index as f64 / 48_000.0)
                    + std::f64::consts::FRAC_PI_4)
                    .sin() as f32
                    * 1.1
            })
            .collect();
        let mut limiter = SFLimiter::new(48_000, -2.0, 1.0, 0.0, 0.0, true).unwrap();

        let output = limiter.process_planar(&input, 1).unwrap();
        assert!(
            output
                .audio
                .iter()
                .all(|sample| f64::from(sample.abs()) <= limiter.threshold()),
            "sample peak exceeds threshold {}",
            limiter.threshold()
        );
    }

    #[test]
    fn true_peak_mode_processes_every_supported_sample_rate() {
        for sample_rate in [
            8_000, 11_025, 12_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000, 88_200, 96_000,
            176_400, 192_000,
        ] {
            let input: Vec<_> = (0..256)
                .map(|index| {
                    ((2.0 * std::f64::consts::PI * (sample_rate as f64 / 4.0) * index as f64
                        / sample_rate as f64)
                        + std::f64::consts::FRAC_PI_4)
                        .sin() as f32
                        * 1.1
                })
                .collect();
            let mut limiter = SFLimiter::new(sample_rate, -1.0, 1.0, 0.0, 0.0, true).unwrap();

            let output = limiter.process_planar(&input, 1).unwrap();
            assert!(
                output
                    .audio
                    .iter()
                    .all(|sample| sample.is_finite()
                        && f64::from(sample.abs()) <= limiter.threshold()),
                "sample_rate={sample_rate}, threshold={}",
                limiter.threshold()
            );
            assert_eq!(output.frame_gains.len(), input.len());
            assert!(
                output
                    .frame_gains
                    .iter()
                    .all(|gain| gain.is_finite() && (0.0..=1.0).contains(gain))
            );
        }
    }
}
