use crate::{LimiterError, layout::validate_layout};

use super::{
    InterpolationFactor, PRE_UPSAMPLE_CENTER, PRE_UPSAMPLE_TAPS, PeakConfig, PeakStrategy,
    PreUpsampleCoefficients, interpolated_peak_at_frame, pre_upsample_sample,
    reduce_upsampled_peaks, validate_finite_samples,
};

pub(super) fn collect(
    audio: &[f32],
    channels: usize,
    config: &PeakConfig,
) -> Result<Vec<f32>, LimiterError> {
    match config.strategy() {
        PeakStrategy::SamplePeak => collect_sample_peaks(audio, channels),
        PeakStrategy::Interpolated(interpolation) => {
            collect_interpolated_peaks(audio, channels, interpolation)
        }
        PeakStrategy::PreUpsampled {
            factor,
            interpolation,
        } => {
            let frame_count = validate_layout(audio.len(), channels)?;
            validate_finite_samples(audio)?;
            let coefficients = config
                .pre_upsample_coefficients()
                .expect("pre-upsampling requires precomputed coefficients");
            let upsampled = pre_upsample(audio, channels, frame_count, factor, coefficients);
            let upsampled_peaks = collect_interpolated_peaks(&upsampled, channels, interpolation)?;
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
    factor: usize,
    coefficients: &PreUpsampleCoefficients,
) -> Vec<f32> {
    if channels < 4 {
        return pre_upsample_scalar(audio, channels, frame_count, factor, coefficients);
    }

    let mut upsampled = vec![0.0; audio.len() * factor];

    for (i_frame, input_frame) in audio.chunks_exact(channels).enumerate() {
        let output_start = i_frame * factor * channels;
        upsampled[output_start..output_start + channels].copy_from_slice(input_frame);
    }

    let (interior_start, interior_end) = if frame_count < PRE_UPSAMPLE_TAPS {
        (frame_count, frame_count)
    } else {
        (
            PRE_UPSAMPLE_CENTER,
            frame_count - (PRE_UPSAMPLE_TAPS - PRE_UPSAMPLE_CENTER - 1),
        )
    };
    for i_frame in 0..interior_start {
        pre_upsample_boundary_frame(
            audio,
            &mut upsampled,
            channels,
            frame_count,
            factor,
            i_frame,
            coefficients,
        );
    }

    for i_frame in interior_start..interior_end {
        for (i_phase, phase_coefficients) in coefficients.iter().enumerate().take(factor).skip(1) {
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
            factor,
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
    factor: usize,
    coefficients: &PreUpsampleCoefficients,
) -> Vec<f32> {
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

fn pre_upsample_boundary_frame(
    audio: &[f32],
    upsampled: &mut [f32],
    channels: usize,
    frame_count: usize,
    factor: usize,
    i_frame: usize,
    coefficients: &[[f32; PRE_UPSAMPLE_TAPS]],
) {
    for (i_phase, phase_coefficients) in coefficients.iter().enumerate().take(factor).skip(1) {
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

    fn pre_upsample_reference(
        audio: &[f32],
        channels: usize,
        frame_count: usize,
        factor: usize,
        coefficients: &PreUpsampleCoefficients,
    ) -> Vec<f32> {
        pre_upsample_scalar(audio, channels, frame_count, factor, coefficients)
    }

    #[test]
    fn pre_upsampling_preserves_original_samples() {
        let audio = [0.25, -0.5, 0.75, -1.0];
        for factor in [2, 3, 4, 6] {
            let coefficients = super::super::pre_upsample_coefficients(factor);
            let upsampled = pre_upsample(&audio, 1, audio.len(), factor, &coefficients);
            let original_phases: Vec<_> = upsampled.iter().step_by(factor).copied().collect();
            assert_eq!(original_phases, audio);
        }
    }

    #[test]
    fn vectorizable_pre_upsampling_matches_scalar_reference() {
        for channels in [1, 2, 3, 4, 8] {
            for frame_count in [0, 1, 11, 12, 23, 24, 25, 257] {
                let audio: Vec<_> = (0..frame_count * channels)
                    .map(|index| {
                        ((index as f64 * 0.731).sin() + (index as f64 * 0.193).cos()) as f32
                    })
                    .collect();
                for factor in [2, 3, 4, 6] {
                    let coefficients = super::super::pre_upsample_coefficients(factor);
                    assert_eq!(
                        pre_upsample(&audio, channels, frame_count, factor, &coefficients),
                        pre_upsample_reference(
                            &audio,
                            channels,
                            frame_count,
                            factor,
                            &coefficients,
                        ),
                        "channels={channels}, frame_count={frame_count}, factor={factor}"
                    );
                }
            }
        }
    }
}
