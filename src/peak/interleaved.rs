use std::ops::Range;

use crate::{LimiterError, layout::validate_layout};

use super::{
    CENTER_TAP, COEFFICIENTS, FIR_TAP_COUNT, FOUR_X_ADDITIONAL_PHASES, InterpolationFactor,
    PRE_UPSAMPLE_CENTER, PRE_UPSAMPLE_TAPS, PeakConfig, TRAILING_BOUNDARY_FRAMES, TWO_X_PHASES,
    calc_interpolated_peak_at_frame, calc_pre_upsample_interior_bounds, pre_upsample_sample,
    reduce_upsampled_peaks, validate_finite_samples,
};

pub(super) fn collect(
    audio: &[f32],
    channels: usize,
    config: &PeakConfig,
) -> Result<Vec<f32>, LimiterError> {
    match config {
        PeakConfig::SamplePeak => collect_sample_peaks(audio, channels),
        PeakConfig::Interpolated(interpolation) => {
            collect_interpolated_peaks(audio, channels, *interpolation)
        }
        PeakConfig::PreUpsampled {
            interpolation,
            coefficients,
        } => {
            let frame_count = validate_layout(audio.len(), channels)?;
            validate_finite_samples(audio)?;
            let factor = coefficients.len();
            let upsampled = pre_upsample(audio, channels, frame_count, coefficients);
            let upsampled_peaks = collect_interpolated_peaks(&upsampled, channels, *interpolation)?;
            Ok(reduce_upsampled_peaks(&upsampled_peaks, factor))
        }
    }
}

fn collect_interpolated_peaks(
    audio: &[f32],
    channels: usize,
    interpolation: InterpolationFactor,
) -> Result<Vec<f32>, LimiterError> {
    let mut frame_peaks = collect_sample_peaks(audio, channels)?;
    let frame_count = frame_peaks.len();

    if channels < 4 || frame_count < FIR_TAP_COUNT {
        collect_boundary_peaks(
            audio,
            &mut frame_peaks,
            channels,
            0..frame_count,
            interpolation,
        );
        return Ok(frame_peaks);
    }

    // Complete FIR windows use a channel-contiguous loop so LLVM can
    // vectorize across channels. Edge frames retain the scalar zero-padding
    // path because some taps are outside the signal.
    collect_boundary_peaks(
        audio,
        &mut frame_peaks,
        channels,
        0..CENTER_TAP,
        interpolation,
    );
    let trailing_start = frame_count - TRAILING_BOUNDARY_FRAMES;
    collect_boundary_peaks(
        audio,
        &mut frame_peaks,
        channels,
        trailing_start..frame_count,
        interpolation,
    );

    for phase in TWO_X_PHASES {
        convolve_phase(audio, &mut frame_peaks, channels, &COEFFICIENTS[phase]);
    }
    if interpolation == InterpolationFactor::Four {
        for phase in FOUR_X_ADDITIONAL_PHASES {
            convolve_phase(audio, &mut frame_peaks, channels, &COEFFICIENTS[phase]);
        }
    }
    Ok(frame_peaks)
}

fn collect_boundary_peaks(
    audio: &[f32],
    frame_peaks: &mut [f32],
    channels: usize,
    frames: Range<usize>,
    interpolation: InterpolationFactor,
) {
    let frame_count = frame_peaks.len();
    for i_frame in frames {
        for i_channel in 0..channels {
            let peak = calc_interpolated_peak_at_frame(
                |source_frame| audio[source_frame * channels + i_channel],
                frame_count,
                i_frame,
                interpolation,
            );
            frame_peaks[i_frame] = frame_peaks[i_frame].max(peak);
        }
    }
}

// Each FIR tap addresses a contiguous channel span. Keeping the tap expression
// fixed lets LLVM vectorize this loop across channels without changing the
// floating-point accumulation order within a channel.
#[inline(never)]
fn convolve_phase(
    audio: &[f32],
    frame_peaks: &mut [f32],
    channels: usize,
    coefficients: &[f32; FIR_TAP_COUNT],
) {
    debug_assert_eq!(audio.len(), frame_peaks.len() * channels);
    debug_assert!(channels >= 4);
    debug_assert!(frame_peaks.len() >= FIR_TAP_COUNT);

    let interior_len = frame_peaks.len() - (FIR_TAP_COUNT - 1);
    for i_window in 0..interior_len {
        let window_start = i_window * channels;
        let window = &audio[window_start..window_start + FIR_TAP_COUNT * channels];
        let mut peak = frame_peaks[i_window + CENTER_TAP];
        for i_channel in 0..channels {
            let sum = window[i_channel] * coefficients[11]
                + window[channels + i_channel] * coefficients[10]
                + window[2 * channels + i_channel] * coefficients[9]
                + window[3 * channels + i_channel] * coefficients[8]
                + window[4 * channels + i_channel] * coefficients[7]
                + window[5 * channels + i_channel] * coefficients[6]
                + window[6 * channels + i_channel] * coefficients[5]
                + window[7 * channels + i_channel] * coefficients[4]
                + window[8 * channels + i_channel] * coefficients[3]
                + window[9 * channels + i_channel] * coefficients[2]
                + window[10 * channels + i_channel] * coefficients[1]
                + window[11 * channels + i_channel] * coefficients[0];
            peak = peak.max(sum.abs());
        }
        frame_peaks[i_window + CENTER_TAP] = peak;
    }
}

fn collect_sample_peaks(audio: &[f32], channels: usize) -> Result<Vec<f32>, LimiterError> {
    let frame_count = validate_layout(audio.len(), channels)?;
    validate_finite_samples(audio)?;

    let mut frame_peaks = Vec::with_capacity(frame_count);
    for frame in audio.chunks_exact(channels) {
        frame_peaks.push(frame.iter().map(|sample| sample.abs()).fold(0.0, f32::max));
    }
    Ok(frame_peaks)
}

fn pre_upsample(
    audio: &[f32],
    channels: usize,
    frame_count: usize,
    coefficients: &[[f32; PRE_UPSAMPLE_TAPS]],
) -> Vec<f32> {
    let factor = coefficients.len();
    if channels < 4 {
        return pre_upsample_scalar(audio, channels, frame_count, coefficients);
    }

    let mut upsampled = vec![0.0; audio.len() * factor];

    for (i_frame, input_frame) in audio.chunks_exact(channels).enumerate() {
        let output_start = i_frame * factor * channels;
        upsampled[output_start..output_start + channels].copy_from_slice(input_frame);
    }

    let (interior_start, interior_end) = calc_pre_upsample_interior_bounds(frame_count);
    for i_frame in 0..interior_start {
        pre_upsample_boundary_frame(
            audio,
            &mut upsampled,
            channels,
            frame_count,
            i_frame,
            coefficients,
        );
    }

    for i_frame in interior_start..interior_end {
        for (i_phase, phase_coefficients) in coefficients.iter().enumerate().skip(1) {
            let output_frame = i_frame * factor + i_phase;
            let output_start = output_frame * channels;
            let output = &mut upsampled[output_start..output_start + channels];

            for (tap, &coefficient) in phase_coefficients.iter().enumerate() {
                let source_frame = i_frame + tap - PRE_UPSAMPLE_CENTER;
                let input_start = source_frame * channels;
                let input = &audio[input_start..input_start + channels];
                for (output_sample, &input_sample) in output.iter_mut().zip(input) {
                    *output_sample += input_sample * coefficient;
                }
            }
        }
    }

    for i_frame in interior_end..frame_count {
        pre_upsample_boundary_frame(
            audio,
            &mut upsampled,
            channels,
            frame_count,
            i_frame,
            coefficients,
        );
    }

    upsampled
}

fn pre_upsample_scalar(
    audio: &[f32],
    channels: usize,
    frame_count: usize,
    coefficients: &[[f32; PRE_UPSAMPLE_TAPS]],
) -> Vec<f32> {
    let factor = coefficients.len();
    let mut upsampled = vec![0.0; audio.len() * factor];
    for i_channel in 0..channels {
        for i_frame in 0..frame_count {
            upsampled[i_frame * factor * channels + i_channel] =
                audio[i_frame * channels + i_channel];
            for (i_phase, phase_coefficients) in coefficients.iter().enumerate().skip(1) {
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

fn pre_upsample_boundary_frame(
    audio: &[f32],
    upsampled: &mut [f32],
    channels: usize,
    frame_count: usize,
    i_frame: usize,
    coefficients: &[[f32; PRE_UPSAMPLE_TAPS]],
) {
    let factor = coefficients.len();
    for (i_phase, phase_coefficients) in coefficients.iter().enumerate().skip(1) {
        let output_frame = i_frame * factor + i_phase;
        for i_channel in 0..channels {
            upsampled[output_frame * channels + i_channel] = pre_upsample_sample(
                |source_frame| audio[source_frame * channels + i_channel],
                i_frame,
                frame_count,
                phase_coefficients,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::peak::convolve_scalar;

    fn collect_interpolated_peaks_scalar_reference(
        audio: &[f32],
        channels: usize,
        interpolation: InterpolationFactor,
    ) -> Vec<f32> {
        let mut frame_peaks = collect_sample_peaks(audio, channels).unwrap();
        let frame_count = frame_peaks.len();
        for (i_frame, frame_peak) in frame_peaks.iter_mut().enumerate() {
            for i_channel in 0..channels {
                let peak = calc_interpolated_peak_at_frame(
                    |source_frame| audio[source_frame * channels + i_channel],
                    frame_count,
                    i_frame,
                    interpolation,
                );
                *frame_peak = frame_peak.max(peak);
            }
        }
        frame_peaks
    }

    fn pre_upsample_reference(
        audio: &[f32],
        channels: usize,
        frame_count: usize,
        coefficients: &[[f32; PRE_UPSAMPLE_TAPS]],
    ) -> Vec<f32> {
        let factor = coefficients.len();
        let mut upsampled = vec![0.0; audio.len() * factor];
        for i_frame in 0..frame_count {
            for i_channel in 0..channels {
                for (i_phase, phase_coefficients) in coefficients.iter().enumerate() {
                    upsampled[(i_frame * factor + i_phase) * channels + i_channel] =
                        pre_upsample_sample(
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

    #[test]
    fn phase_convolution_uses_the_same_fir_order() {
        let channels = 4;
        let frame_count = 32;
        let audio: Vec<_> = (0..frame_count * channels)
            .map(|i_sample| ((i_sample as f64 * 0.731).sin() * 1.7) as f32)
            .collect();
        let mut peaks = vec![0.0; frame_count];

        convolve_phase(&audio, &mut peaks, channels, &COEFFICIENTS[1]);

        for i_window in 0..frame_count - (FIR_TAP_COUNT - 1) {
            let expected = (0..channels)
                .map(|i_channel| {
                    let samples =
                        std::array::from_fn(|tap| audio[(i_window + tap) * channels + i_channel]);
                    convolve_scalar(&samples, &COEFFICIENTS[1]).abs()
                })
                .fold(0.0, f32::max);
            assert_eq!(peaks[i_window + CENTER_TAP], expected);
        }
    }

    #[test]
    fn pre_upsampling_preserves_original_samples() {
        let audio = [0.25, -0.5, 0.75, -1.0];
        for factor in [2, 3, 4, 6] {
            let coefficients = super::super::build_pre_upsample_coefficients(factor);
            let upsampled = pre_upsample(&audio, 1, audio.len(), &coefficients);
            let original_phases: Vec<_> = upsampled.iter().step_by(factor).copied().collect();
            assert_eq!(original_phases, audio);
        }
    }

    #[test]
    fn vectorizable_interpolation_matches_scalar_reference() {
        for channels in [1, 2, 3, 4, 5, 8] {
            for frame_count in [0, 1, 11, 12, 13, 257] {
                let audio: Vec<_> = (0..frame_count * channels)
                    .map(|i_sample| {
                        ((i_sample as f64 * 0.731).sin() + (i_sample as f64 * 0.193).cos()) as f32
                    })
                    .collect();
                for interpolation in [InterpolationFactor::Two, InterpolationFactor::Four] {
                    assert_eq!(
                        collect_interpolated_peaks(&audio, channels, interpolation).unwrap(),
                        collect_interpolated_peaks_scalar_reference(
                            &audio,
                            channels,
                            interpolation,
                        ),
                        "channels={channels}, frame_count={frame_count}, interpolation={interpolation:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn vectorizable_pre_upsampling_matches_scalar_reference() {
        for channels in [1, 2, 3, 4, 8] {
            for frame_count in [0, 1, 11, 12, 23, 24, 25, 257] {
                let audio: Vec<_> = (0..frame_count * channels)
                    .map(|i_sample| {
                        ((i_sample as f64 * 0.731).sin() + (i_sample as f64 * 0.193).cos()) as f32
                    })
                    .collect();
                for factor in [2, 3, 4, 6] {
                    let coefficients = super::super::build_pre_upsample_coefficients(factor);
                    assert_eq!(
                        pre_upsample(&audio, channels, frame_count, &coefficients),
                        pre_upsample_reference(&audio, channels, frame_count, &coefficients),
                        "channels={channels}, frame_count={frame_count}, factor={factor}"
                    );
                }
            }
        }
    }
}
