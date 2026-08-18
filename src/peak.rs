//! Sample-peak collection and ITU-R BS.1770-5 Annex 2 true-peak estimation.

use crate::{LimiterError, validate_layout};

const FIR_PHASE_COUNT: usize = 4;
const FIR_TAP_COUNT: usize = 12;
const CENTER_TAP: usize = FIR_TAP_COUNT / 2;
const TRAILING_BOUNDARY_FRAMES: usize = FIR_TAP_COUNT - CENTER_TAP - 1;
const TWO_X_PHASES: [usize; 2] = [0, FIR_PHASE_COUNT / 2];
const FOUR_X_ADDITIONAL_PHASES: [usize; 2] = [1, FIR_PHASE_COUNT - 1];

// The order-48, four-phase FIR interpolator published in BS.1770-5. Coefficients
// are phase-major so the planar hot loop can convolve consecutive frames with
// one invariant coefficient set. Every published coefficient is exactly
// representable as f32.
#[allow(clippy::excessive_precision)]
#[rustfmt::skip]
const COEFFICIENTS: [[f32; FIR_TAP_COUNT]; FIR_PHASE_COUNT] = [
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

const PRE_UPSAMPLE_TAPS: usize = 24;
const PRE_UPSAMPLE_CENTER: isize = 11;
const MAX_PRE_UPSAMPLE_FACTOR: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InterpolationFactor {
    Two,
    Four,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PeakStrategy {
    SamplePeak,
    Interpolated(InterpolationFactor),
    PreUpsampled {
        factor: usize,
        interpolation: InterpolationFactor,
    },
}

fn true_peak_config(sample_rate: u32) -> Option<PeakStrategy> {
    match sample_rate {
        8_000 => Some(PeakStrategy::PreUpsampled {
            factor: 6,
            interpolation: InterpolationFactor::Four,
        }),
        11_025 => Some(PeakStrategy::PreUpsampled {
            factor: 4,
            interpolation: InterpolationFactor::Four,
        }),
        12_000 => Some(PeakStrategy::PreUpsampled {
            factor: 4,
            interpolation: InterpolationFactor::Four,
        }),
        16_000 => Some(PeakStrategy::PreUpsampled {
            factor: 3,
            interpolation: InterpolationFactor::Four,
        }),
        22_050 => Some(PeakStrategy::PreUpsampled {
            factor: 2,
            interpolation: InterpolationFactor::Four,
        }),
        24_000 => Some(PeakStrategy::PreUpsampled {
            factor: 2,
            interpolation: InterpolationFactor::Four,
        }),
        32_000 => Some(PeakStrategy::PreUpsampled {
            factor: 3,
            interpolation: InterpolationFactor::Two,
        }),
        44_100 | 48_000 => Some(PeakStrategy::Interpolated(InterpolationFactor::Four)),
        88_200 | 96_000 => Some(PeakStrategy::Interpolated(InterpolationFactor::Two)),
        176_400.. => Some(PeakStrategy::SamplePeak),
        _ => None,
    }
}

pub(crate) fn supports_true_peak_sample_rate(sample_rate: u32) -> bool {
    true_peak_config(sample_rate).is_some()
}

pub(crate) fn collect_true_peaks_from_interleaved(
    audio: &[f32],
    channels: usize,
    sample_rate: u32,
) -> Result<Vec<f32>, LimiterError> {
    match true_peak_config(sample_rate)
        .ok_or(LimiterError::UnsupportedTruePeakSampleRate(sample_rate))?
    {
        PeakStrategy::SamplePeak => collect_sample_peaks_from_interleaved(audio, channels),
        PeakStrategy::Interpolated(interpolation) => {
            collect_interpolated_peaks_from_interleaved(audio, channels, interpolation)
        }
        PeakStrategy::PreUpsampled {
            factor,
            interpolation,
        } => {
            let frame_count = validate_layout(audio.len(), channels)?;
            validate_finite_samples(audio)?;
            let upsampled = pre_upsample_interleaved(audio, channels, frame_count, factor);
            let upsampled_peaks =
                collect_interpolated_peaks_from_interleaved(&upsampled, channels, interpolation)?;
            Ok(reduce_upsampled_peaks(&upsampled_peaks, factor))
        }
    }
}

fn collect_interpolated_peaks_from_interleaved(
    audio: &[f32],
    channels: usize,
    interpolation: InterpolationFactor,
) -> Result<Vec<f32>, LimiterError> {
    let mut frame_peaks = collect_sample_peaks_from_interleaved(audio, channels)?;
    let frame_count = frame_peaks.len();
    for (i_frame, frame_peak) in frame_peaks.iter_mut().enumerate() {
        for channel in 0..channels {
            let peak = interpolated_peak_at_frame(
                |source_frame| audio[source_frame * channels + channel],
                frame_count,
                i_frame,
                interpolation,
            );
            *frame_peak = frame_peak.max(peak);
        }
    }
    Ok(frame_peaks)
}

pub(crate) fn collect_true_peaks_from_planar(
    audio: &[f32],
    channels: usize,
    sample_rate: u32,
) -> Result<Vec<f32>, LimiterError> {
    match true_peak_config(sample_rate)
        .ok_or(LimiterError::UnsupportedTruePeakSampleRate(sample_rate))?
    {
        PeakStrategy::SamplePeak => collect_sample_peaks_from_planar(audio, channels),
        PeakStrategy::Interpolated(interpolation) => {
            collect_interpolated_peaks_from_planar(audio, channels, interpolation)
        }
        PeakStrategy::PreUpsampled {
            factor,
            interpolation,
        } => {
            let frame_count = validate_layout(audio.len(), channels)?;
            validate_finite_samples(audio)?;
            let upsampled = pre_upsample_planar(audio, channels, frame_count, factor);
            let upsampled_peaks =
                collect_interpolated_peaks_from_planar(&upsampled, channels, interpolation)?;
            Ok(reduce_upsampled_peaks(&upsampled_peaks, factor))
        }
    }
}

fn collect_interpolated_peaks_from_planar(
    audio: &[f32],
    channels: usize,
    interpolation: InterpolationFactor,
) -> Result<Vec<f32>, LimiterError> {
    let mut frame_peaks = collect_sample_peaks_from_planar(audio, channels)?;
    let frame_count = frame_peaks.len();
    for i_channel in 0..channels {
        let channel_offset = i_channel * frame_count;
        let channel_audio = &audio[channel_offset..channel_offset + frame_count];

        if frame_count < FIR_TAP_COUNT {
            collect_planar_boundary_peaks(
                channel_audio,
                &mut frame_peaks,
                0..frame_count,
                interpolation,
            );
            continue;
        }

        // The vectorizable path requires a complete FIR window; edge frames use
        // the scalar path so out-of-range taps can be treated as zero.
        collect_planar_boundary_peaks(
            channel_audio,
            &mut frame_peaks,
            0..CENTER_TAP,
            interpolation,
        );
        let trailing_start = frame_count - TRAILING_BOUNDARY_FRAMES;
        collect_planar_boundary_peaks(
            channel_audio,
            &mut frame_peaks,
            trailing_start..frame_count,
            interpolation,
        );

        for phase in TWO_X_PHASES {
            convolve_planar_phase(channel_audio, &mut frame_peaks, &COEFFICIENTS[phase]);
        }
        if interpolation == InterpolationFactor::Four {
            for phase in FOUR_X_ADDITIONAL_PHASES {
                convolve_planar_phase(channel_audio, &mut frame_peaks, &COEFFICIENTS[phase]);
            }
        }
    }
    Ok(frame_peaks)
}

fn collect_planar_boundary_peaks(
    channel_audio: &[f32],
    frame_peaks: &mut [f32],
    frames: std::ops::Range<usize>,
    interpolation: InterpolationFactor,
) {
    for frame in frames {
        let peak = interpolated_peak_at_frame(
            |source_frame| channel_audio[source_frame],
            channel_audio.len(),
            frame,
            interpolation,
        );
        frame_peaks[frame] = frame_peaks[frame].max(peak);
    }
}

// Keeping the tap expression fixed and making consecutive frames independent
// lets LLVM vectorize this loop across frames without relaxed floating-point
// reductions or architecture-specific intrinsics.
#[inline(never)]
fn convolve_planar_phase(
    channel_audio: &[f32],
    frame_peaks: &mut [f32],
    coefficients: &[f32; FIR_TAP_COUNT],
) {
    debug_assert_eq!(channel_audio.len(), frame_peaks.len());
    debug_assert!(channel_audio.len() >= FIR_TAP_COUNT);

    let interior_len = channel_audio.len() - (FIR_TAP_COUNT - 1);
    for index in 0..interior_len {
        let sum = channel_audio[index] * coefficients[11]
            + channel_audio[index + 1] * coefficients[10]
            + channel_audio[index + 2] * coefficients[9]
            + channel_audio[index + 3] * coefficients[8]
            + channel_audio[index + 4] * coefficients[7]
            + channel_audio[index + 5] * coefficients[6]
            + channel_audio[index + 6] * coefficients[5]
            + channel_audio[index + 7] * coefficients[4]
            + channel_audio[index + 8] * coefficients[3]
            + channel_audio[index + 9] * coefficients[2]
            + channel_audio[index + 10] * coefficients[1]
            + channel_audio[index + 11] * coefficients[0];
        let frame_peak = &mut frame_peaks[index + CENTER_TAP];
        *frame_peak = frame_peak.max(sum.abs());
    }
}

pub(crate) fn collect_sample_peaks_from_interleaved(
    audio: &[f32],
    channels: usize,
) -> Result<Vec<f32>, LimiterError> {
    let frame_count = validate_layout(audio.len(), channels)?;
    validate_finite_samples(audio)?;

    let mut frame_peaks = Vec::with_capacity(frame_count);
    for frame in audio.chunks_exact(channels) {
        frame_peaks.push(frame.iter().map(|sample| sample.abs()).fold(0.0, f32::max));
    }
    Ok(frame_peaks)
}

pub(crate) fn collect_sample_peaks_from_planar(
    audio: &[f32],
    channels: usize,
) -> Result<Vec<f32>, LimiterError> {
    let frame_count = validate_layout(audio.len(), channels)?;
    validate_finite_samples(audio)?;

    let mut frame_peaks = vec![0.0_f32; frame_count];
    if frame_count == 0 {
        return Ok(frame_peaks);
    }

    for channel in audio.chunks_exact(frame_count) {
        for (peak, sample) in frame_peaks.iter_mut().zip(channel) {
            *peak = peak.max(sample.abs());
        }
    }
    Ok(frame_peaks)
}

fn interpolated_peak_at_frame(
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

fn pre_upsample_interleaved(
    audio: &[f32],
    channels: usize,
    frame_count: usize,
    factor: usize,
) -> Vec<f32> {
    let coefficients = pre_upsample_coefficients(factor);
    let mut upsampled = vec![0.0; audio.len() * factor];
    for i_channel in 0..channels {
        for i_frame in 0..frame_count {
            for (i_phase, phase_coefficients) in coefficients.iter().enumerate().take(factor) {
                let output_frame = i_frame * factor + i_phase;
                upsampled[output_frame * channels + i_channel] = pre_upsample_sample(
                    |source_frame| audio[source_frame * channels + i_channel],
                    i_frame,
                    frame_count,
                    phase_coefficients,
                );
            }
        }
    }
    upsampled
}

fn pre_upsample_planar(
    audio: &[f32],
    channels: usize,
    frame_count: usize,
    factor: usize,
) -> Vec<f32> {
    let coefficients = pre_upsample_coefficients(factor);
    let upsampled_frame_count = frame_count * factor;
    let mut upsampled = vec![0.0; audio.len() * factor];
    for i_channel in 0..channels {
        let input_offset = i_channel * frame_count;
        let output_offset = i_channel * upsampled_frame_count;
        let channel_audio = &audio[input_offset..input_offset + frame_count];
        let channel_output = &mut upsampled[output_offset..output_offset + upsampled_frame_count];
        for frame in 0..frame_count {
            for (i_phase, phase_coefficients) in coefficients.iter().enumerate().take(factor) {
                channel_output[frame * factor + i_phase] = pre_upsample_sample(
                    |source_frame| channel_audio[source_frame],
                    frame,
                    frame_count,
                    phase_coefficients,
                );
            }
        }
    }
    upsampled
}

fn pre_upsample_sample(
    mut sample: impl FnMut(usize) -> f32,
    frame: usize,
    frame_count: usize,
    coefficients: &[f32; PRE_UPSAMPLE_TAPS],
) -> f32 {
    let mut sum = 0.0_f32;
    for (tap, coefficient) in coefficients.iter().enumerate() {
        let source_frame = frame as isize + tap as isize - PRE_UPSAMPLE_CENTER;
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
/// Only the first `factor` entries in the returned array are populated.
fn pre_upsample_coefficients(factor: usize) -> [[f32; PRE_UPSAMPLE_TAPS]; MAX_PRE_UPSAMPLE_FACTOR] {
    debug_assert!(matches!(factor, 2 | 3 | 4 | 6));
    let mut coefficients = [[0.0; PRE_UPSAMPLE_TAPS]; MAX_PRE_UPSAMPLE_FACTOR];
    coefficients[0][PRE_UPSAMPLE_CENTER as usize] = 1.0;

    for (i_phase, phase_coefficients) in coefficients.iter_mut().enumerate().take(factor).skip(1) {
        let fraction = i_phase as f32 / factor as f32;
        let mut normalization = 0.0_f32;
        for (tap, coefficient) in phase_coefficients.iter_mut().enumerate() {
            let distance = fraction - (tap as isize - PRE_UPSAMPLE_CENTER) as f32;
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

fn reduce_upsampled_peaks(upsampled_peaks: &[f32], factor: usize) -> Vec<f32> {
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

fn validate_finite_samples(audio: &[f32]) -> Result<(), LimiterError> {
    if let Some(index) = audio.iter().position(|sample| !sample.is_finite()) {
        Err(LimiterError::NonFiniteSample { index })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let leading_peak = interpolated_peak_at_frame(
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
        let trailing_peak = interpolated_peak_at_frame(
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
    fn planar_phase_convolution_uses_the_same_fir_order() {
        let audio: Vec<_> = (0..32)
            .map(|index| ((index as f64 * 0.731).sin() * 1.7) as f32)
            .collect();
        let mut peaks = vec![0.0; audio.len()];

        convolve_planar_phase(&audio, &mut peaks, &COEFFICIENTS[1]);

        for index in 0..audio.len() - (FIR_TAP_COUNT - 1) {
            let samples: [f32; FIR_TAP_COUNT] =
                audio[index..index + FIR_TAP_COUNT].try_into().unwrap();
            assert_eq!(
                peaks[index + CENTER_TAP],
                convolve_scalar(&samples, &COEFFICIENTS[1]).abs()
            );
        }
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

        let peaks = collect_true_peaks_from_interleaved(&audio, 1, 96_000).unwrap();
        let interior_peak = peaks[24..audio.len() - 24]
            .iter()
            .copied()
            .fold(0.0, f32::max);

        assert!(audio.iter().all(|sample| sample.abs() < 1.0));
        assert!(interior_peak > 1.0, "true peak={interior_peak}");
    }

    #[test]
    fn planar_convolution_matches_interleaved_convolution() {
        let first_channel: Vec<_> = (0..257)
            .map(|index| {
                (((index as f64 * 0.731).sin() * 1.7) + ((index as f64 * 0.193).cos() * 0.4)) as f32
            })
            .collect();
        let second_channel: Vec<_> = (0..257)
            .map(|index| {
                (((index as f64 * 0.417).cos() * 0.9) - ((index as f64 * 0.137).sin() * 0.6)) as f32
            })
            .collect();
        let interleaved_audio: Vec<_> = first_channel
            .iter()
            .zip(&second_channel)
            .flat_map(|(&first, &second)| [first, second])
            .collect();
        let mut planar_audio = first_channel;
        planar_audio.extend(second_channel);

        for sample_rate in [
            8_000, 11_025, 12_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000, 88_200, 96_000,
            176_400, 192_000,
        ] {
            let interleaved =
                collect_true_peaks_from_interleaved(&interleaved_audio, 2, sample_rate).unwrap();
            let planar = collect_true_peaks_from_planar(&planar_audio, 2, sample_rate).unwrap();
            for (interleaved, planar) in interleaved.iter().zip(planar) {
                assert!((interleaved - planar).abs() <= 2.0 * f32::EPSILON);
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
            let peaks = collect_true_peaks_from_interleaved(&audio, 1, sample_rate).unwrap();
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
                true_peak_config(sample_rate),
                Some(PeakStrategy::PreUpsampled {
                    factor,
                    interpolation,
                })
            );
        }
        for (sample_rate, interpolation) in [
            (44_100, InterpolationFactor::Four),
            (48_000, InterpolationFactor::Four),
            (88_200, InterpolationFactor::Two),
            (96_000, InterpolationFactor::Two),
        ] {
            assert_eq!(
                true_peak_config(sample_rate),
                Some(PeakStrategy::Interpolated(interpolation))
            );
        }
        for sample_rate in [176_400, 192_000, u32::MAX] {
            assert_eq!(
                true_peak_config(sample_rate),
                Some(PeakStrategy::SamplePeak)
            );
        }
        for sample_rate in [1, 10_000, 47_999, 176_399] {
            assert_eq!(true_peak_config(sample_rate), None);
        }
    }

    #[test]
    fn pre_upsampling_preserves_original_samples() {
        let audio = [0.25, -0.5, 0.75, -1.0];
        for factor in [2, 3, 4, 6] {
            let upsampled = pre_upsample_interleaved(&audio, 1, audio.len(), factor);
            let original_phases: Vec<_> = upsampled.iter().step_by(factor).copied().collect();
            assert_eq!(original_phases, audio);
        }
    }

    #[test]
    fn unsupported_true_peak_sample_rate_is_rejected() {
        assert_eq!(
            collect_true_peaks_from_interleaved(&[0.0], 1, 176_399).unwrap_err(),
            LimiterError::UnsupportedTruePeakSampleRate(176_399)
        );
        assert_eq!(
            collect_true_peaks_from_planar(&[0.0], 1, 10_000).unwrap_err(),
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
        let peaks = collect_true_peaks_from_interleaved(&audio, 1, 48_000).unwrap();
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
