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
    for i_channel in 0..channels {
        let channel_offset = i_channel * frame_count;
        let channel_audio = &audio[channel_offset..channel_offset + frame_count];

        if frame_count < FIR_TAP_COUNT {
            collect_boundary_peaks(
                channel_audio,
                &mut frame_peaks,
                0..frame_count,
                interpolation,
            );
            continue;
        }

        // The vectorizable path requires a complete FIR window; edge frames use
        // the scalar path so out-of-range taps can be treated as zero.
        collect_boundary_peaks(
            channel_audio,
            &mut frame_peaks,
            0..CENTER_TAP,
            interpolation,
        );
        let trailing_start = frame_count - TRAILING_BOUNDARY_FRAMES;
        collect_boundary_peaks(
            channel_audio,
            &mut frame_peaks,
            trailing_start..frame_count,
            interpolation,
        );

        for phase in TWO_X_PHASES {
            convolve_phase(channel_audio, &mut frame_peaks, &COEFFICIENTS[phase]);
        }
        if interpolation == InterpolationFactor::Four {
            for phase in FOUR_X_ADDITIONAL_PHASES {
                convolve_phase(channel_audio, &mut frame_peaks, &COEFFICIENTS[phase]);
            }
        }
    }
    Ok(frame_peaks)
}

fn collect_boundary_peaks(
    channel_audio: &[f32],
    frame_peaks: &mut [f32],
    frames: Range<usize>,
    interpolation: InterpolationFactor,
) {
    for i_frame in frames {
        let peak = calc_interpolated_peak_at_frame(
            |source_frame| channel_audio[source_frame],
            channel_audio.len(),
            i_frame,
            interpolation,
        );
        frame_peaks[i_frame] = frame_peaks[i_frame].max(peak);
    }
}

// Keeping the tap expression fixed and making consecutive frames independent
// lets LLVM vectorize this loop across frames without relaxed floating-point
// reductions or architecture-specific intrinsics.
#[inline(never)]
fn convolve_phase(
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

fn collect_sample_peaks(audio: &[f32], channels: usize) -> Result<Vec<f32>, LimiterError> {
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

fn pre_upsample(
    audio: &[f32],
    channels: usize,
    frame_count: usize,
    coefficients: &[[f32; PRE_UPSAMPLE_TAPS]],
) -> Vec<f32> {
    let factor = coefficients.len();
    let upsampled_frame_count = frame_count * factor;
    let mut upsampled = vec![0.0; audio.len() * factor];
    let mut phase_output = vec![0.0; frame_count];
    let (interior_start, interior_end) = calc_pre_upsample_interior_bounds(frame_count);

    for i_channel in 0..channels {
        let input_offset = i_channel * frame_count;
        let output_offset = i_channel * upsampled_frame_count;
        let channel_audio = &audio[input_offset..input_offset + frame_count];
        let channel_output = &mut upsampled[output_offset..output_offset + upsampled_frame_count];

        for (output_frame, &sample) in channel_output.chunks_exact_mut(factor).zip(channel_audio) {
            output_frame[0] = sample;
        }

        for (i_phase, phase_coefficients) in coefficients.iter().enumerate().skip(1) {
            for (i_frame, output_sample) in phase_output[..interior_start].iter_mut().enumerate() {
                *output_sample = pre_upsample_sample(
                    |source_frame| channel_audio[source_frame],
                    i_frame,
                    frame_count,
                    phase_coefficients,
                );
            }

            if interior_start < interior_end {
                phase_output[interior_start..interior_end].fill(0.0);
                let interior_length = interior_end - interior_start;
                for (tap, &coefficient) in phase_coefficients.iter().enumerate() {
                    let source_start = interior_start + tap - PRE_UPSAMPLE_CENTER;
                    let source = &channel_audio[source_start..source_start + interior_length];
                    for (output_sample, &input_sample) in phase_output[interior_start..interior_end]
                        .iter_mut()
                        .zip(source)
                    {
                        *output_sample += input_sample * coefficient;
                    }
                }
            }

            for (i_frame, output_sample) in phase_output[interior_end..].iter_mut().enumerate() {
                *output_sample = pre_upsample_sample(
                    |source_frame| channel_audio[source_frame],
                    interior_end + i_frame,
                    frame_count,
                    phase_coefficients,
                );
            }

            for (output_frame, &sample) in
                channel_output.chunks_exact_mut(factor).zip(&phase_output)
            {
                output_frame[i_phase] = sample;
            }
        }
    }
    upsampled
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
        for i_channel in 0..channels {
            let channel_offset = i_channel * frame_count;
            let channel_audio = &audio[channel_offset..channel_offset + frame_count];
            for (i_frame, frame_peak) in frame_peaks.iter_mut().enumerate() {
                let peak = calc_interpolated_peak_at_frame(
                    |source_frame| channel_audio[source_frame],
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
        let upsampled_frame_count = frame_count * factor;
        let mut upsampled = vec![0.0; audio.len() * factor];
        for i_channel in 0..channels {
            let input_offset = i_channel * frame_count;
            let output_offset = i_channel * upsampled_frame_count;
            let channel_audio = &audio[input_offset..input_offset + frame_count];
            let channel_output =
                &mut upsampled[output_offset..output_offset + upsampled_frame_count];
            for i_frame in 0..frame_count {
                for (i_phase, phase_coefficients) in coefficients.iter().enumerate() {
                    channel_output[i_frame * factor + i_phase] = pre_upsample_sample(
                        |source_frame| channel_audio[source_frame],
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
        let audio: Vec<_> = (0..32)
            .map(|index| ((index as f64 * 0.731).sin() * 1.7) as f32)
            .collect();
        let mut peaks = vec![0.0; audio.len()];

        convolve_phase(&audio, &mut peaks, &COEFFICIENTS[1]);

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
            for frame_count in [0, 1, 31, 32, 63, 64, 65, 257] {
                let channel_samples: Vec<Vec<f32>> = (0..channels)
                    .map(|channel| {
                        (0..frame_count)
                            .map(|frame| {
                                let index = channel * frame_count + frame;
                                ((index as f64 * 0.731).sin() + (index as f64 * 0.193).cos()) as f32
                            })
                            .collect()
                    })
                    .collect();
                let audio: Vec<_> = channel_samples.into_iter().flatten().collect();
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
