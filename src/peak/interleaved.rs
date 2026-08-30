use std::ops::Range;

use crate::{LimiterError, layout::validate_layout};

use super::convolve::{calc_mirrored_terms, convolve_fir_12, convolve_mirrored_fir_12};
use super::{
    BS1770_2X_PHASES, BS1770_CENTER_TAP, BS1770_COEFFICIENTS, BS1770_MIRRORED_COEFFS,
    BS1770_N_TAPS, BS1770_TRAILING_BOUNDARY_FRAMES, InterpolationFactor, PRE_UPSAMPLE_CENTER_TAP,
    PRE_UPSAMPLE_N_TAPS, PeakConfig, PreUpsampleCoefficients, calc_interpolated_peak_at_frame,
    calc_pre_upsample_interior_bounds, pre_upsample_mirrored_samples,
    pre_upsample_symmetric_sample, reduce_upsampled_peaks, validate_input_sample,
};

pub(super) fn collect(
    audio: &[f32],
    channels: usize,
    config: &PeakConfig,
) -> Result<Vec<f32>, LimiterError> {
    match config {
        PeakConfig::SamplePeak => collect_validated_sample_peaks(audio, channels),
        PeakConfig::Interpolated(interpolation) => {
            collect_interpolated_peaks(audio, channels, *interpolation)
        }
        PeakConfig::PreUpsampled {
            interpolation,
            coefficients,
        } => {
            let frame_count = validate_layout(audio.len(), channels)?;
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

    if channels < 4 || frame_count < BS1770_N_TAPS {
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
        0..BS1770_CENTER_TAP,
        interpolation,
    );
    let trailing_start = frame_count - BS1770_TRAILING_BOUNDARY_FRAMES;
    collect_boundary_peaks(
        audio,
        &mut frame_peaks,
        channels,
        trailing_start..frame_count,
        interpolation,
    );

    match interpolation {
        InterpolationFactor::Two => {
            for phase in BS1770_2X_PHASES {
                convolve_phase(
                    audio,
                    &mut frame_peaks,
                    channels,
                    &BS1770_COEFFICIENTS[phase],
                );
            }
        }
        InterpolationFactor::Four => {
            for coeff_pairs in &BS1770_MIRRORED_COEFFS {
                convolve_mirrored_phases(audio, &mut frame_peaks, channels, coeff_pairs);
            }
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
                |i_src_frame| audio[i_src_frame * channels + i_channel],
                i_frame,
                frame_count,
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
    coefficients: &[f32; BS1770_N_TAPS],
) {
    debug_assert_eq!(audio.len(), frame_peaks.len() * channels);
    debug_assert!(channels >= 4);
    debug_assert!(frame_peaks.len() >= BS1770_N_TAPS);

    let interior_len = frame_peaks.len() - (BS1770_N_TAPS - 1);
    for i_window in 0..interior_len {
        let window_start = i_window * channels;
        let window = &audio[window_start..window_start + BS1770_N_TAPS * channels];
        let mut peak = frame_peaks[i_window + BS1770_CENTER_TAP];
        for i_channel in 0..channels {
            let sum = convolve_fir_12!(|tap| window[tap * channels + i_channel], coefficients);
            peak = peak.max(sum.abs());
        }
        frame_peaks[i_window + BS1770_CENTER_TAP] = peak;
    }
}

// The two output phases use reversed coefficient orders. Keeping each tap at a
// fixed channel offset lets LLVM vectorize this loop across channels.
#[inline(never)]
fn convolve_mirrored_phases(
    audio: &[f32],
    frame_peaks: &mut [f32],
    channels: usize,
    coeff_pairs: &[(f32, f32); BS1770_N_TAPS / 2],
) {
    debug_assert_eq!(audio.len(), frame_peaks.len() * channels);
    debug_assert!(channels >= 4);
    debug_assert!(frame_peaks.len() >= BS1770_N_TAPS);

    let interior_len = frame_peaks.len() - (BS1770_N_TAPS - 1);
    for i_window in 0..interior_len {
        let window_start = i_window * channels;
        let window = &audio[window_start..window_start + BS1770_N_TAPS * channels];
        let mut peak = frame_peaks[i_window + BS1770_CENTER_TAP];
        for i_channel in 0..channels {
            let (sample, mirror_sample) =
                convolve_mirrored_fir_12!(|tap| window[tap * channels + i_channel], coeff_pairs);
            peak = peak.max(sample.abs()).max(mirror_sample.abs());
        }
        frame_peaks[i_window + BS1770_CENTER_TAP] = peak;
    }
}

fn collect_sample_peaks(audio: &[f32], channels: usize) -> Result<Vec<f32>, LimiterError> {
    let frame_count = validate_layout(audio.len(), channels)?;

    let mut frame_peaks = Vec::with_capacity(frame_count);
    for frame in audio.chunks_exact(channels) {
        frame_peaks.push(frame.iter().map(|sample| sample.abs()).fold(0.0, f32::max));
    }
    Ok(frame_peaks)
}

fn collect_validated_sample_peaks(
    audio: &[f32],
    channels: usize,
) -> Result<Vec<f32>, LimiterError> {
    let frame_count = validate_layout(audio.len(), channels)?;

    let mut frame_peaks = Vec::with_capacity(frame_count);
    for (i_frame, frame) in audio.chunks_exact(channels).enumerate() {
        let mut peak = 0.0_f32;
        for (i_channel, &sample) in frame.iter().enumerate() {
            validate_input_sample(i_frame * channels + i_channel, sample)?;
            peak = peak.max(sample.abs());
        }
        frame_peaks.push(peak);
    }
    Ok(frame_peaks)
}

fn pre_upsample(
    audio: &[f32],
    channels: usize,
    frame_count: usize,
    coefficients: &PreUpsampleCoefficients,
) -> Vec<f32> {
    let factor = coefficients.len();
    if channels < 4 {
        return pre_upsample_scalar(audio, channels, frame_count, coefficients);
    }

    let mut upsampled = vec![0.0; audio.len() * factor];

    for (i_frame, frame) in audio.chunks_exact(channels).enumerate() {
        let out_start = i_frame * factor * channels;
        upsampled[out_start..out_start + channels].copy_from_slice(frame);
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
        for (i_phase, coeff_pairs) in coefficients.mirrored_phases() {
            convolve_mirrored_phase_frame(
                audio,
                &mut upsampled,
                channels,
                i_frame,
                factor,
                i_phase,
                coeff_pairs,
            );
        }
        if let Some((i_phase, phase_coefficients)) = coefficients.symmetric_phase() {
            convolve_symmetric_phase_frame(
                audio,
                &mut upsampled,
                channels,
                i_frame,
                factor,
                i_phase,
                phase_coefficients,
            );
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

fn convolve_mirrored_phase_frame(
    audio: &[f32],
    upsampled: &mut [f32],
    channels: usize,
    i_frame: usize,
    factor: usize,
    i_phase: usize,
    coeff_pairs: &[(f32, f32); PRE_UPSAMPLE_N_TAPS / 2],
) {
    // Phase p and phase factor-p use reversed coefficients. Pairing mirrored
    // samples lets both outputs share two products instead of using four.
    let mirror_phase = factor - i_phase;
    let out_start = (i_frame * factor + i_phase) * channels;
    let mirror_out_start = (i_frame * factor + mirror_phase) * channels;
    let [primary_out, mirror_out] = upsampled
        .get_disjoint_mut([
            out_start..out_start + channels,
            mirror_out_start..mirror_out_start + channels,
        ])
        .expect(
            "The output slices for the primary and mirror phases must not overlap. \
            Check the calculation of out_start and mirror_out_start.",
        );

    for (tap, &(common_coefficient, differential_coefficient)) in coeff_pairs.iter().enumerate() {
        let mirror_tap = PRE_UPSAMPLE_N_TAPS - 1 - tap;
        let i_src_frame = i_frame + tap - PRE_UPSAMPLE_CENTER_TAP;
        let i_mirror_src_frame = i_frame + mirror_tap - PRE_UPSAMPLE_CENTER_TAP;
        let input = &audio[i_src_frame * channels..(i_src_frame + 1) * channels];
        let mirror_input =
            &audio[i_mirror_src_frame * channels..(i_mirror_src_frame + 1) * channels];
        for i_channel in 0..channels {
            let (common, differential) = calc_mirrored_terms(
                input[i_channel],
                mirror_input[i_channel],
                common_coefficient,
                differential_coefficient,
            );
            primary_out[i_channel] += common + differential;
            mirror_out[i_channel] += common - differential;
        }
    }
}

fn convolve_symmetric_phase_frame(
    audio: &[f32],
    upsampled: &mut [f32],
    channels: usize,
    i_frame: usize,
    factor: usize,
    i_phase: usize,
    coefficients: &[f32; PRE_UPSAMPLE_N_TAPS],
) {
    // The half-sample phase is self-symmetric, so each input pair shares one
    // coefficient and needs one product instead of two.
    let out_start = (i_frame * factor + i_phase) * channels;
    let output = &mut upsampled[out_start..out_start + channels];

    for (tap, &coefficient) in coefficients
        .iter()
        .take(PRE_UPSAMPLE_N_TAPS / 2)
        .enumerate()
    {
        let mirror_tap = PRE_UPSAMPLE_N_TAPS - 1 - tap;
        let i_src_frame = i_frame + tap - PRE_UPSAMPLE_CENTER_TAP;
        let i_mirror_src_frame = i_frame + mirror_tap - PRE_UPSAMPLE_CENTER_TAP;
        let input = &audio[i_src_frame * channels..(i_src_frame + 1) * channels];
        let mirror_input =
            &audio[i_mirror_src_frame * channels..(i_mirror_src_frame + 1) * channels];

        for i_channel in 0..channels {
            output[i_channel] += (input[i_channel] + mirror_input[i_channel]) * coefficient;
        }
    }
}

fn pre_upsample_scalar(
    audio: &[f32],
    channels: usize,
    frame_count: usize,
    coefficients: &PreUpsampleCoefficients,
) -> Vec<f32> {
    let factor = coefficients.len();
    let mut upsampled = vec![0.0; audio.len() * factor];
    for i_channel in 0..channels {
        for i_frame in 0..frame_count {
            upsampled[i_frame * factor * channels + i_channel] =
                audio[i_frame * channels + i_channel];
            for (i_phase, coeff_pairs) in coefficients.mirrored_phases() {
                let mirror_phase = factor - i_phase;
                let (sample, mirror_sample) = pre_upsample_mirrored_samples(
                    |i_src_frame| audio[i_src_frame * channels + i_channel],
                    i_frame,
                    frame_count,
                    coeff_pairs,
                );
                upsampled[(i_frame * factor + i_phase) * channels + i_channel] = sample;
                upsampled[(i_frame * factor + mirror_phase) * channels + i_channel] = mirror_sample;
            }
            if let Some((i_phase, phase_coefficients)) = coefficients.symmetric_phase() {
                upsampled[(i_frame * factor + i_phase) * channels + i_channel] =
                    pre_upsample_symmetric_sample(
                        |i_src_frame| audio[i_src_frame * channels + i_channel],
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
    coefficients: &PreUpsampleCoefficients,
) {
    let factor = coefficients.len();
    for (i_phase, coeff_pairs) in coefficients.mirrored_phases() {
        let mirror_phase = factor - i_phase;
        for i_channel in 0..channels {
            let (sample, mirror_sample) = pre_upsample_mirrored_samples(
                |i_src_frame| audio[i_src_frame * channels + i_channel],
                i_frame,
                frame_count,
                coeff_pairs,
            );
            upsampled[(i_frame * factor + i_phase) * channels + i_channel] = sample;
            upsampled[(i_frame * factor + mirror_phase) * channels + i_channel] = mirror_sample;
        }
    }
    if let Some((i_phase, phase_coefficients)) = coefficients.symmetric_phase() {
        for i_channel in 0..channels {
            upsampled[(i_frame * factor + i_phase) * channels + i_channel] =
                pre_upsample_symmetric_sample(
                    |i_src_frame| audio[i_src_frame * channels + i_channel],
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
    use crate::peak::{
        calc_interpolated_peak_at_frame_reference, convolve_scalar, pre_upsample_sample,
    };

    fn assert_peaks_close(actual: &[f32], expected: &[f32], context: &str) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            let tolerance = 16.0 * f32::EPSILON * expected.abs().max(1.0);
            assert!(
                (*actual - expected).abs() <= tolerance,
                "{context}, actual={actual}, expected={expected}, tolerance={tolerance}"
            );
        }
    }

    fn collect_interpolated_peaks_scalar_reference(
        audio: &[f32],
        channels: usize,
        interpolation: InterpolationFactor,
    ) -> Vec<f32> {
        let mut frame_peaks = collect_sample_peaks(audio, channels).unwrap();
        let frame_count = frame_peaks.len();
        for (i_frame, frame_peak) in frame_peaks.iter_mut().enumerate() {
            for i_channel in 0..channels {
                let peak = calc_interpolated_peak_at_frame_reference(
                    |i_src_frame| audio[i_src_frame * channels + i_channel],
                    i_frame,
                    frame_count,
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
        coefficients: &[[f32; PRE_UPSAMPLE_N_TAPS]],
    ) -> Vec<f32> {
        let factor = coefficients.len();
        let mut upsampled = vec![0.0; audio.len() * factor];
        for i_frame in 0..frame_count {
            for i_channel in 0..channels {
                for (i_phase, phase_coefficients) in coefficients.iter().enumerate() {
                    upsampled[(i_frame * factor + i_phase) * channels + i_channel] =
                        pre_upsample_sample(
                            |i_src_frame| audio[i_src_frame * channels + i_channel],
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

        convolve_phase(&audio, &mut peaks, channels, &BS1770_COEFFICIENTS[1]);

        for i_window in 0..frame_count - (BS1770_N_TAPS - 1) {
            let expected = (0..channels)
                .map(|i_channel| {
                    let samples =
                        std::array::from_fn(|tap| audio[(i_window + tap) * channels + i_channel]);
                    convolve_scalar(&samples, &BS1770_COEFFICIENTS[1]).abs()
                })
                .fold(0.0, f32::max);
            assert_eq!(peaks[i_window + BS1770_CENTER_TAP], expected);
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
                    let actual =
                        collect_interpolated_peaks(&audio, channels, interpolation).unwrap();
                    let expected = collect_interpolated_peaks_scalar_reference(
                        &audio,
                        channels,
                        interpolation,
                    );
                    let context = format!(
                        "channels={channels}, frame_count={frame_count}, interpolation={interpolation:?}"
                    );
                    match interpolation {
                        InterpolationFactor::Two => assert_eq!(actual, expected, "{context}"),
                        InterpolationFactor::Four => {
                            assert_peaks_close(&actual, &expected, &context)
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn vectorizable_pre_upsampling_matches_scalar_reference() {
        for channels in [1, 2, 3, 4, 8] {
            for frame_count in [0, 1, 31, 32, 63, 64, 65, 257] {
                let audio: Vec<_> = (0..frame_count * channels)
                    .map(|i_sample| {
                        ((i_sample as f64 * 0.731).sin() + (i_sample as f64 * 0.193).cos()) as f32
                    })
                    .collect();
                for factor in [2, 3, 4, 6] {
                    let coefficients = super::super::build_pre_upsample_coefficients(factor);
                    let actual = pre_upsample(&audio, channels, frame_count, &coefficients);
                    let expected =
                        pre_upsample_reference(&audio, channels, frame_count, &coefficients);
                    assert_eq!(actual.len(), expected.len());
                    for (actual, expected) in actual.iter().zip(expected) {
                        let tolerance = 16.0 * f32::EPSILON * expected.abs().max(1.0);
                        assert!(
                            (*actual - expected).abs() <= tolerance,
                            "channels={channels}, frame_count={frame_count}, factor={factor}, actual={actual}, expected={expected}"
                        );
                    }
                }
            }
        }
    }
}
