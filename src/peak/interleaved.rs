use crate::{LimiterError, layout::validate_layout};

use super::{
    InterpolationFactor, PeakStrategy, interpolated_peak_at_frame, pre_upsample_coefficients,
    pre_upsample_sample, reduce_upsampled_peaks, validate_finite_samples,
};

pub(super) fn collect(
    audio: &[f32],
    channels: usize,
    strategy: PeakStrategy,
) -> Result<Vec<f32>, LimiterError> {
    match strategy {
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
            let upsampled = pre_upsample(audio, channels, frame_count, factor);
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

fn pre_upsample(audio: &[f32], channels: usize, frame_count: usize, factor: usize) -> Vec<f32> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pre_upsampling_preserves_original_samples() {
        let audio = [0.25, -0.5, 0.75, -1.0];
        for factor in [2, 3, 4, 6] {
            let upsampled = pre_upsample(&audio, 1, audio.len(), factor);
            let original_phases: Vec<_> = upsampled.iter().step_by(factor).copied().collect();
            assert_eq!(original_phases, audio);
        }
    }
}
