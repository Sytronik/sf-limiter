//! Sample-peak collection and ITU-R BS.1770-5 Annex 2 true-peak estimation.

mod interleaved;
mod planar;

use crate::{LimiterError, layout::AudioLayout};

const FIR_PHASE_COUNT: usize = 4;
pub(super) const FIR_TAP_COUNT: usize = 12;
pub(super) const CENTER_TAP: usize = FIR_TAP_COUNT / 2;
pub(super) const TRAILING_BOUNDARY_FRAMES: usize = FIR_TAP_COUNT - CENTER_TAP - 1;
pub(super) const TWO_X_PHASES: [usize; 2] = [0, FIR_PHASE_COUNT / 2];
pub(super) const FOUR_X_ADDITIONAL_PHASES: [usize; 2] = [1, FIR_PHASE_COUNT - 1];

// The order-48, four-phase FIR interpolator published in BS.1770-5. Coefficients
// are phase-major so the planar hot loop can convolve consecutive frames with
// one invariant coefficient set. Every published coefficient is exactly
// representable as f32.
#[allow(clippy::excessive_precision)]
#[rustfmt::skip]
pub(super) const COEFFICIENTS: [[f32; FIR_TAP_COUNT]; FIR_PHASE_COUNT] = [
    [
         0.0017089843750,  0.0109863281250, -0.0196533203125,  0.0332031250000,
        -0.0594482421875,  0.1373291015625,  0.9721679687500, -0.1022949218750,
         0.0476074218750, -0.0266113281250,  0.0148925781250, -0.0083007812500,
    ],
    [
        -0.0291748046875,  0.0292968750000, -0.0517578125000,  0.0891113281250,
        -0.1665039062500,  0.4650878906250,  0.7797851562500, -0.2003173828125,
         0.1015625000000, -0.0582275390625,  0.0330810546875, -0.0189208984375,
    ],
    [
        -0.0189208984375,  0.0330810546875, -0.0582275390625,  0.1015625000000,
        -0.2003173828125,  0.7797851562500,  0.4650878906250, -0.1665039062500,
         0.0891113281250, -0.0517578125000,  0.0292968750000, -0.0291748046875,
    ],
    [
        -0.0083007812500,  0.0148925781250, -0.0266113281250,  0.0476074218750,
        -0.1022949218750,  0.9721679687500,  0.1373291015625, -0.0594482421875,
         0.0332031250000, -0.0196533203125,  0.0109863281250,  0.0017089843750,
    ],
];

pub(super) const PRE_UPSAMPLE_TAPS: usize = 24;
pub(super) const PRE_UPSAMPLE_CENTER: usize = 11;
pub(super) type PreUpsampleCoefficients = Box<[[f32; PRE_UPSAMPLE_TAPS]]>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum InterpolationFactor {
    Two,
    Four,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum PeakConfig {
    /// sample-peak collection without true-peak estimation
    SamplePeak,

    /// true-peak estimation using a polyphase FIR from BS.1770
    Interpolated(InterpolationFactor),

    /// true-peak estimation using a polyphase FIR from BS.1770 with pre-upsampling
    PreUpsampled {
        interpolation: InterpolationFactor,
        coefficients: PreUpsampleCoefficients,
    },
}

impl PeakConfig {
    pub(crate) fn new(sample_rate: u32, true_peak: bool) -> Result<Self, LimiterError> {
        if true_peak {
            Self::try_from_sample_rate_for_true_peak(sample_rate)
        } else {
            Ok(Self::SamplePeak)
        }
    }

    fn try_from_sample_rate_for_true_peak(sample_rate: u32) -> Result<Self, LimiterError> {
        match sample_rate {
            8_000 => Ok(Self::pre_upsampled(6, InterpolationFactor::Four)),
            11_025 | 12_000 => Ok(Self::pre_upsampled(4, InterpolationFactor::Four)),
            16_000 => Ok(Self::pre_upsampled(3, InterpolationFactor::Four)),
            22_050 | 24_000 => Ok(Self::pre_upsampled(2, InterpolationFactor::Four)),
            32_000 => Ok(Self::pre_upsampled(3, InterpolationFactor::Two)),
            44_100 | 48_000 => Ok(Self::Interpolated(InterpolationFactor::Four)),
            88_200 | 96_000 => Ok(Self::Interpolated(InterpolationFactor::Two)),
            176_400.. => Ok(Self::SamplePeak),
            _ => Err(LimiterError::UnsupportedTruePeakSampleRate(sample_rate)),
        }
    }

    fn pre_upsampled(factor: usize, interpolation: InterpolationFactor) -> Self {
        Self::PreUpsampled {
            interpolation,
            coefficients: build_pre_upsample_coefficients(factor),
        }
    }

    pub(crate) fn collect_frame_peaks(
        &self,
        audio: &[f32],
        channels: usize,
        layout: AudioLayout,
    ) -> Result<Vec<f32>, LimiterError> {
        match layout {
            AudioLayout::Interleaved => interleaved::collect(audio, channels, self),
            AudioLayout::Planar => planar::collect(audio, channels, self),
        }
    }
}

pub(super) fn calc_pre_upsample_interior_bounds(frame_count: usize) -> (usize, usize) {
    if frame_count < PRE_UPSAMPLE_TAPS {
        (frame_count, frame_count)
    } else {
        (
            PRE_UPSAMPLE_CENTER,
            frame_count - (PRE_UPSAMPLE_TAPS - PRE_UPSAMPLE_CENTER - 1),
        )
    }
}

pub(super) fn calc_interpolated_peak_at_frame(
    mut sample_at: impl FnMut(usize) -> f32,
    frame_count: usize,
    frame: usize,
    interpolation: InterpolationFactor,
) -> f32 {
    // An even-length FIR window spans six frames before this frame and five after it.
    // Samples beyond either signal boundary are zero-padded.
    let samples: [f32; FIR_TAP_COUNT] = std::array::from_fn(|tap| {
        let source_frame = frame as isize + tap as isize - CENTER_TAP as isize;
        if let Ok(source_frame) = usize::try_from(source_frame)
            && source_frame < frame_count
        {
            sample_at(source_frame)
        } else {
            0.0
        }
    });
    match interpolation {
        InterpolationFactor::Two => interpolate_two_phases(&samples),
        InterpolationFactor::Four => interpolate_four_phases(&samples),
    }
}

pub(super) fn pre_upsample_sample(
    mut sample: impl FnMut(usize) -> f32,
    frame: usize,
    frame_count: usize,
    coefficients: &[f32; PRE_UPSAMPLE_TAPS],
) -> f32 {
    let mut sum = 0.0_f32;
    for (tap, coefficient) in coefficients.iter().enumerate() {
        let source_frame = frame as isize + tap as isize - PRE_UPSAMPLE_CENTER as isize;
        if let Ok(source_frame) = usize::try_from(source_frame)
            && source_frame < frame_count
        {
            sum += sample(source_frame) * coefficient;
        }
    }
    sum
}

/// Builds a 24-tap polyphase FIR bank for band-limited pre-upsampling.
///
/// Each active phase is a Hann-windowed sinc fractional-delay filter. Phase
/// zero is an impulse so that samples already present in the input are copied
/// exactly, while phase `p` reconstructs the sample at the fractional offset
/// `p / factor`. Every interpolating phase is normalized to unity DC gain.
///
/// The returned bank contains exactly `factor` phases.
pub(super) fn build_pre_upsample_coefficients(factor: usize) -> PreUpsampleCoefficients {
    debug_assert!(matches!(factor, 2 | 3 | 4 | 6));
    let mut coefficients = vec![[0.0; PRE_UPSAMPLE_TAPS]; factor].into_boxed_slice();
    coefficients[0][PRE_UPSAMPLE_CENTER] = 1.0;

    for (i_phase, phase_coefficients) in coefficients.iter_mut().enumerate().skip(1) {
        let fraction = i_phase as f32 / factor as f32;
        let mut normalization = 0.0_f32;
        for (tap, coefficient) in phase_coefficients.iter_mut().enumerate() {
            let distance = fraction - (tap as isize - PRE_UPSAMPLE_CENTER as isize) as f32;
            let sinc = (std::f32::consts::PI * distance).sin() / (std::f32::consts::PI * distance);
            let window = 0.5
                * (1.0
                    + (std::f32::consts::PI * distance / (PRE_UPSAMPLE_TAPS as f32 / 2.0)).cos());
            let value = sinc * window;
            *coefficient = value;
            normalization += value;
        }
        for coefficient in phase_coefficients {
            *coefficient /= normalization;
        }
    }
    coefficients
}

pub(super) fn reduce_upsampled_peaks(upsampled_peaks: &[f32], factor: usize) -> Vec<f32> {
    upsampled_peaks
        .chunks_exact(factor)
        .map(|peaks| peaks.iter().copied().fold(0.0, f32::max))
        .collect()
}

fn interpolate_four_phases(samples: &[f32; FIR_TAP_COUNT]) -> f32 {
    (0..FIR_PHASE_COUNT)
        .map(|phase| convolve_scalar(samples, &COEFFICIENTS[phase]).abs())
        .fold(0.0, f32::max)
}

#[inline(always)]
fn interpolate_two_phases(samples: &[f32; FIR_TAP_COUNT]) -> f32 {
    TWO_X_PHASES
        .map(|phase| convolve_scalar(samples, &COEFFICIENTS[phase]).abs())
        .into_iter()
        .fold(0.0, f32::max)
}

#[inline(always)]
fn convolve_scalar(samples: &[f32; FIR_TAP_COUNT], coefficients: &[f32; FIR_TAP_COUNT]) -> f32 {
    samples
        .iter()
        .zip(coefficients.iter().rev())
        .map(|(sample, coefficient)| sample * coefficient)
        .sum()
}

pub(super) fn validate_finite_samples(audio: &[f32]) -> Result<(), LimiterError> {
    if let Some(index) = audio.iter().position(|sample| !sample.is_finite()) {
        Err(LimiterError::NonFiniteSample { index })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect_true_peaks(
        audio: &[f32],
        channels: usize,
        sample_rate: u32,
        layout: AudioLayout,
    ) -> Result<Vec<f32>, LimiterError> {
        PeakConfig::new(sample_rate, true)?.collect_frame_peaks(audio, channels, layout)
    }

    fn scalar_peak(samples: &[f32; FIR_TAP_COUNT], phases: &[usize]) -> f32 {
        phases
            .iter()
            .map(|&phase| {
                samples
                    .iter()
                    .zip(COEFFICIENTS[phase].iter().rev())
                    .map(|(sample, coefficient)| sample * coefficient)
                    .sum::<f32>()
                    .abs()
            })
            .fold(0.0, f32::max)
    }

    #[test]
    fn scalar_convolution_matches_reference() {
        let samples = std::array::from_fn(|index| {
            (((index as f64 * 0.731).sin() * 1.7) + ((index as f64 * 0.193).cos() * 0.4)) as f32
        });

        let four_phase = interpolate_four_phases(&samples);
        let two_phase = interpolate_two_phases(&samples);

        assert_eq!(four_phase, scalar_peak(&samples, &[0, 1, 2, 3]));
        assert_eq!(two_phase, scalar_peak(&samples, &[0, 2]));
    }

    #[test]
    fn scalar_convolution_uses_fir_time_order() {
        let mut samples = [0.0; FIR_TAP_COUNT];
        samples[FIR_TAP_COUNT - 1] = 1.0;
        for coefficients in &COEFFICIENTS {
            assert_eq!(convolve_scalar(&samples, coefficients), coefficients[0]);
        }

        samples = [0.0; FIR_TAP_COUNT];
        samples[0] = 1.0;
        for coefficients in &COEFFICIENTS {
            assert_eq!(
                convolve_scalar(&samples, coefficients),
                coefficients[FIR_TAP_COUNT - 1]
            );
        }
    }

    #[test]
    fn interpolated_peak_zero_pads_both_boundaries() {
        let audio = [0.25, -0.5, 0.75, -1.0];

        let mut accessed_frames = Vec::new();
        let leading_peak = calc_interpolated_peak_at_frame(
            |source_frame| {
                accessed_frames.push(source_frame);
                audio[source_frame]
            },
            audio.len(),
            0,
            InterpolationFactor::Four,
        );
        let mut leading_window = [0.0; FIR_TAP_COUNT];
        leading_window[CENTER_TAP..CENTER_TAP + audio.len()].copy_from_slice(&audio);
        assert_eq!(leading_peak, scalar_peak(&leading_window, &[0, 1, 2, 3]));
        assert_eq!(accessed_frames, [0, 1, 2, 3]);

        accessed_frames.clear();
        let trailing_peak = calc_interpolated_peak_at_frame(
            |source_frame| {
                accessed_frames.push(source_frame);
                audio[source_frame]
            },
            audio.len(),
            audio.len() - 1,
            InterpolationFactor::Four,
        );
        let mut trailing_window = [0.0; FIR_TAP_COUNT];
        let trailing_start = CENTER_TAP + 1 - audio.len();
        trailing_window[trailing_start..trailing_start + audio.len()].copy_from_slice(&audio);
        assert_eq!(trailing_peak, scalar_peak(&trailing_window, &[0, 1, 2, 3]));
        assert_eq!(accessed_frames, [0, 1, 2, 3]);
    }

    #[test]
    fn two_phase_convolution_detects_a_quarter_rate_inter_sample_peak() {
        let audio: Vec<_> = (0..128)
            .map(|index| {
                (2.0 * std::f64::consts::PI * 24_000.0 * index as f64 / 96_000.0
                    + std::f64::consts::FRAC_PI_6)
                    .sin() as f32
            })
            .collect();

        let peaks = collect_true_peaks(&audio, 1, 96_000, AudioLayout::Interleaved).unwrap();
        let interior_peak = peaks[24..audio.len() - 24]
            .iter()
            .copied()
            .fold(0.0, f32::max);

        assert!(audio.iter().all(|sample| sample.abs() < 1.0));
        assert!(interior_peak > 1.0, "true peak={interior_peak}");
    }

    #[test]
    fn planar_peak_collection_matches_interleaved_peak_collection() {
        let frame_count = 257;
        for channels in [1, 2, 3, 4, 5, 8] {
            let planar_audio: Vec<_> = (0..frame_count * channels)
                .map(|i_sample| {
                    (((i_sample as f64 * 0.731).sin() * 1.7)
                        + ((i_sample as f64 * 0.193).cos() * 0.4)) as f32
                })
                .collect();
            let planar_samples = &planar_audio;
            let interleaved_audio: Vec<_> = (0..frame_count)
                .flat_map(|i_frame| {
                    (0..channels)
                        .map(move |i_channel| planar_samples[i_channel * frame_count + i_frame])
                })
                .collect();

            for sample_rate in [
                8_000, 11_025, 12_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000, 88_200,
                96_000, 176_400, 192_000,
            ] {
                let interleaved = collect_true_peaks(
                    &interleaved_audio,
                    channels,
                    sample_rate,
                    AudioLayout::Interleaved,
                )
                .unwrap();
                let planar =
                    collect_true_peaks(&planar_audio, channels, sample_rate, AudioLayout::Planar)
                        .unwrap();
                assert_eq!(interleaved.len(), planar.len());
                for (interleaved_peak, planar_peak) in interleaved.iter().zip(planar) {
                    assert!(
                        (interleaved_peak - planar_peak).abs() <= 2.0 * f32::EPSILON,
                        "channels={channels}, sample_rate={sample_rate}, interleaved={interleaved_peak}, planar={planar_peak}"
                    );
                }
            }
        }
    }

    #[test]
    fn constant_signal_has_unity_gain_away_from_boundaries() {
        let audio = vec![1.0; 128];
        for sample_rate in [
            8_000, 11_025, 12_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000, 88_200, 96_000,
            176_400, 192_000,
        ] {
            let peaks =
                collect_true_peaks(&audio, 1, sample_rate, AudioLayout::Interleaved).unwrap();
            for peak in &peaks[24..audio.len() - 24] {
                assert!(
                    (*peak - 1.0).abs() < 0.002,
                    "sample_rate={sample_rate}, peak={peak}"
                );
            }
        }
    }

    #[test]
    fn true_peak_sample_rate_configuration_is_explicit() {
        for (sample_rate, factor, interpolation) in [
            (8_000, 6, InterpolationFactor::Four),
            (11_025, 4, InterpolationFactor::Four),
            (12_000, 4, InterpolationFactor::Four),
            (16_000, 3, InterpolationFactor::Four),
            (22_050, 2, InterpolationFactor::Four),
            (24_000, 2, InterpolationFactor::Four),
            (32_000, 3, InterpolationFactor::Two),
        ] {
            assert_eq!(
                PeakConfig::try_from_sample_rate_for_true_peak(sample_rate),
                Ok(PeakConfig::pre_upsampled(factor, interpolation))
            );
        }
        for (sample_rate, interpolation) in [
            (44_100, InterpolationFactor::Four),
            (48_000, InterpolationFactor::Four),
            (88_200, InterpolationFactor::Two),
            (96_000, InterpolationFactor::Two),
        ] {
            assert_eq!(
                PeakConfig::try_from_sample_rate_for_true_peak(sample_rate),
                Ok(PeakConfig::Interpolated(interpolation))
            );
        }
        for sample_rate in [176_400, 192_000, u32::MAX] {
            assert_eq!(
                PeakConfig::try_from_sample_rate_for_true_peak(sample_rate),
                Ok(PeakConfig::SamplePeak)
            );
        }
        for sample_rate in [1, 10_000, 47_999, 176_399] {
            assert_eq!(
                PeakConfig::try_from_sample_rate_for_true_peak(sample_rate),
                Err(LimiterError::UnsupportedTruePeakSampleRate(sample_rate))
            );
        }
    }

    #[test]
    fn pre_upsample_coefficients_are_precomputed_for_each_factor() {
        for (sample_rate, factor) in [
            (8_000, 6),
            (11_025, 4),
            (12_000, 4),
            (16_000, 3),
            (22_050, 2),
            (24_000, 2),
            (32_000, 3),
        ] {
            let config = PeakConfig::new(sample_rate, true).unwrap();
            let PeakConfig::PreUpsampled { coefficients, .. } = config else {
                panic!("sample_rate={sample_rate} did not select pre-upsampling");
            };
            assert_eq!(coefficients, build_pre_upsample_coefficients(factor));
        }

        for sample_rate in [44_100, 48_000, 88_200, 96_000, 176_400, 192_000] {
            let config = PeakConfig::new(sample_rate, true).unwrap();
            assert!(!matches!(config, PeakConfig::PreUpsampled { .. }));
        }
    }

    #[test]
    fn unsupported_true_peak_sample_rate_is_rejected() {
        assert_eq!(
            collect_true_peaks(&[0.0], 1, 176_399, AudioLayout::Interleaved).unwrap_err(),
            LimiterError::UnsupportedTruePeakSampleRate(176_399)
        );
        assert_eq!(
            collect_true_peaks(&[0.0], 1, 10_000, AudioLayout::Planar).unwrap_err(),
            LimiterError::UnsupportedTruePeakSampleRate(10_000)
        );
    }

    #[test]
    fn detects_an_inter_sample_peak() {
        let audio: Vec<_> = (0..64)
            .map(|index| {
                ((2.0 * std::f64::consts::PI * 12_000.0 * index as f64 / 48_000.0)
                    + std::f64::consts::FRAC_PI_4)
                    .sin() as f32
                    * 1.1
            })
            .collect();
        let peaks = collect_true_peaks(&audio, 1, 48_000, AudioLayout::Interleaved).unwrap();
        assert!(audio.iter().all(|sample| sample.abs() < 1.0));
        assert!(
            peaks[6..audio.len() - 6]
                .iter()
                .copied()
                .fold(0.0, f32::max)
                > 1.0
        );
    }
}
