use std::ops::Range;

use crate::{LimiterError, layout::validate_layout};

use super::convolve::{calc_mirrored_terms, convolve_fir_12, convolve_mirrored_fir_12};
use super::{
    BS1770_2X_PHASES, BS1770_CENTER_TAP, BS1770_COEFFICIENTS, BS1770_MIRRORED_COEFFS,
    BS1770_N_TAPS, BS1770_TRAILING_BOUNDARY_FRAMES, InterpolationFactor, PRE_UPSAMPLE_CENTER_TAP,
    PRE_UPSAMPLE_N_TAPS, PeakConfig, PreUpsampleCoefficients, calc_interpolated_peak_at_frame,
    calc_pre_upsample_interior_bounds, pre_upsample_mirrored_samples,
    pre_upsample_symmetric_sample, reduce_upsampled_peaks,
};

// Bounds the mirrored-phase scratch space to 1 KiB while leaving enough
// consecutive frames for LLVM to vectorize the interior convolution loops.
const PRE_UPSAMPLE_BLOCK_FRAMES: usize = 128;

pub(super) fn collect(
    audio: &[f32],
    channels: usize,
    config: &PeakConfig,
) -> Result<Vec<f32>, LimiterError> {
    if audio.is_empty() {
        validate_layout(audio.len(), channels)?;
        return Ok(Vec::new());
    }

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
        let channel = &audio[channel_offset..channel_offset + frame_count];

        if frame_count < BS1770_N_TAPS {
            collect_boundary_peaks(channel, &mut frame_peaks, 0..frame_count, interpolation);
            continue;
        }

        // The vectorizable path requires a complete FIR window; edge frames use
        // the scalar path so out-of-range taps can be treated as zero.
        collect_boundary_peaks(
            channel,
            &mut frame_peaks,
            0..BS1770_CENTER_TAP,
            interpolation,
        );
        let trailing_start = frame_count - BS1770_TRAILING_BOUNDARY_FRAMES;
        collect_boundary_peaks(
            channel,
            &mut frame_peaks,
            trailing_start..frame_count,
            interpolation,
        );

        match interpolation {
            InterpolationFactor::Two => {
                for phase in BS1770_2X_PHASES {
                    convolve_phase(channel, &mut frame_peaks, &BS1770_COEFFICIENTS[phase]);
                }
            }
            InterpolationFactor::Four => {
                for coeff_pairs in &BS1770_MIRRORED_COEFFS {
                    convolve_mirrored_phases(channel, &mut frame_peaks, coeff_pairs);
                }
            }
        }
    }
    Ok(frame_peaks)
}

fn collect_boundary_peaks(
    audio_channel: &[f32],
    frame_peaks: &mut [f32],
    frames: Range<usize>,
    interpolation: InterpolationFactor,
) {
    for i_frame in frames {
        let peak = calc_interpolated_peak_at_frame(
            |i_src_frame| audio_channel[i_src_frame],
            i_frame,
            audio_channel.len(),
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
    audio_channel: &[f32],
    frame_peaks: &mut [f32],
    coefficients: &[f32; BS1770_N_TAPS],
) {
    debug_assert_eq!(audio_channel.len(), frame_peaks.len());
    debug_assert!(audio_channel.len() >= BS1770_N_TAPS);

    let interior_len = audio_channel.len() - (BS1770_N_TAPS - 1);
    for index in 0..interior_len {
        let sum = convolve_fir_12!(|tap| audio_channel[index + tap], coefficients);
        let frame_peak = &mut frame_peaks[index + BS1770_CENTER_TAP];
        *frame_peak = frame_peak.max(sum.abs());
    }
}

// The two output phases use reversed coefficient orders. The tap expression is
// kept explicit so LLVM can continue vectorizing independent output frames.
#[inline(never)]
fn convolve_mirrored_phases(
    channel_audio: &[f32],
    frame_peaks: &mut [f32],
    coeff_pairs: &[(f32, f32); BS1770_N_TAPS / 2],
) {
    debug_assert_eq!(channel_audio.len(), frame_peaks.len());
    debug_assert!(channel_audio.len() >= BS1770_N_TAPS);

    let interior_len = channel_audio.len() - (BS1770_N_TAPS - 1);
    for index in 0..interior_len {
        let (sample, mirror_sample) =
            convolve_mirrored_fir_12!(|tap| channel_audio[index + tap], coeff_pairs);
        let frame_peak = &mut frame_peaks[index + BS1770_CENTER_TAP];
        *frame_peak = frame_peak.max(sample.abs()).max(mirror_sample.abs());
    }
}

fn collect_sample_peaks(audio: &[f32], channels: usize) -> Result<Vec<f32>, LimiterError> {
    let frame_count = validate_layout(audio.len(), channels)?;

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
    coefficients: &PreUpsampleCoefficients,
) -> Vec<f32> {
    let factor = coefficients.len();
    let upsampled_frame_count = frame_count * factor;
    let mut upsampled = vec![0.0; audio.len() * factor];
    let (interior_start, interior_end) = calc_pre_upsample_interior_bounds(frame_count);

    for i_channel in 0..channels {
        let in_offset = i_channel * frame_count;
        let out_offset = i_channel * upsampled_frame_count;
        let channel = &audio[in_offset..in_offset + frame_count];
        let out_channel = &mut upsampled[out_offset..out_offset + upsampled_frame_count];

        for (out_frame, &sample) in out_channel.chunks_exact_mut(factor).zip(channel) {
            out_frame[0] = sample;
        }

        for (i_phase, coeff_pairs) in coefficients.mirrored_phases() {
            calculate_mirrored_phase(
                channel,
                out_channel,
                factor,
                i_phase,
                interior_start,
                interior_end,
                coeff_pairs,
            );
        }

        if let Some((i_phase, phase_coefficients)) = coefficients.symmetric_phase() {
            calculate_symmetric_phase(
                channel,
                out_channel,
                factor,
                i_phase,
                interior_start,
                interior_end,
                phase_coefficients,
            );
        }
    }
    upsampled
}

// Keep the fixed scratch arrays in a reusable phase-call stack frame.
#[inline(never)]
fn calculate_mirrored_phase(
    audio: &[f32],
    output: &mut [f32],
    factor: usize,
    i_phase: usize,
    interior_start: usize,
    interior_end: usize,
    coeff_pairs: &[(f32, f32); PRE_UPSAMPLE_N_TAPS / 2],
) {
    // Phase p and phase factor-p use reversed coefficients. Pairing mirrored
    // samples lets both outputs share two products instead of using four.
    let frame_count = audio.len();
    let i_mirror_phase = factor - i_phase;
    debug_assert_eq!(output.len(), frame_count * factor);

    for i_frame in 0..interior_start {
        let (sample, mirror_sample) = pre_upsample_mirrored_samples(
            |i_src_frame| audio[i_src_frame],
            i_frame,
            frame_count,
            coeff_pairs,
        );
        let out_frame = &mut output[i_frame * factor..(i_frame + 1) * factor];
        out_frame[i_phase] = sample;
        out_frame[i_mirror_phase] = mirror_sample;
    }
    if interior_start == interior_end {
        return;
    }

    let mut phase_out = [0.0; PRE_UPSAMPLE_BLOCK_FRAMES];
    let mut mirror_out = [0.0; PRE_UPSAMPLE_BLOCK_FRAMES];
    for block_start in (interior_start..interior_end).step_by(PRE_UPSAMPLE_BLOCK_FRAMES) {
        let block_end = (block_start + PRE_UPSAMPLE_BLOCK_FRAMES).min(interior_end);
        let block_length = block_end - block_start;
        let phase_block = &mut phase_out[..block_length];
        let mirror_block = &mut mirror_out[..block_length];
        convolve_mirrored_phase_block(audio, phase_block, mirror_block, block_start, coeff_pairs);
        let output_block = &mut output[block_start * factor..block_end * factor];
        for (out_frame, (&sample, &mirror_sample)) in output_block
            .chunks_exact_mut(factor)
            .zip(phase_block.iter().zip(mirror_block.iter()))
        {
            out_frame[i_phase] = sample;
            out_frame[i_mirror_phase] = mirror_sample;
        }
    }

    for i_frame in interior_end..frame_count {
        let (sample, mirror_sample) = pre_upsample_mirrored_samples(
            |i_src_frame| audio[i_src_frame],
            i_frame,
            frame_count,
            coeff_pairs,
        );
        let out_frame = &mut output[i_frame * factor..(i_frame + 1) * factor];
        out_frame[i_phase] = sample;
        out_frame[i_mirror_phase] = mirror_sample;
    }
}

// Avoid a call per block and expose consecutive frames to the loop vectorizer.
#[inline(always)]
fn convolve_mirrored_phase_block(
    audio: &[f32],
    phase_out: &mut [f32],
    mirror_out: &mut [f32],
    block_start: usize,
    coeff_pairs: &[(f32, f32); PRE_UPSAMPLE_N_TAPS / 2],
) {
    debug_assert_eq!(phase_out.len(), mirror_out.len());
    phase_out.fill(0.0);
    mirror_out.fill(0.0);
    let block_length = phase_out.len();
    for (tap, &(common_coefficient, differential_coefficient)) in coeff_pairs.iter().enumerate() {
        let mirror_tap = PRE_UPSAMPLE_N_TAPS - 1 - tap;
        let source_start = block_start + tap - PRE_UPSAMPLE_CENTER_TAP;
        let mirror_source_start = block_start + mirror_tap - PRE_UPSAMPLE_CENTER_TAP;
        let source = &audio[source_start..source_start + block_length];
        let mirror_source = &audio[mirror_source_start..mirror_source_start + block_length];

        for i_frame in 0..block_length {
            let (common, differential) = calc_mirrored_terms(
                source[i_frame],
                mirror_source[i_frame],
                common_coefficient,
                differential_coefficient,
            );
            phase_out[i_frame] += common + differential;
            mirror_out[i_frame] += common - differential;
        }
    }
}

#[inline(never)]
fn calculate_symmetric_phase(
    audio: &[f32],
    output: &mut [f32],
    factor: usize,
    i_phase: usize,
    interior_start: usize,
    interior_end: usize,
    coefficients: &[f32; PRE_UPSAMPLE_N_TAPS],
) {
    // The half-sample phase is self-symmetric, so each input pair shares one
    // coefficient and needs one product instead of two.
    let frame_count = audio.len();
    debug_assert_eq!(output.len(), frame_count * factor);
    for i_frame in 0..interior_start {
        output[i_frame * factor + i_phase] = pre_upsample_symmetric_sample(
            |i_src_frame| audio[i_src_frame],
            i_frame,
            frame_count,
            coefficients,
        );
    }
    if interior_start == interior_end {
        return;
    }

    let mut phase_out = [0.0; PRE_UPSAMPLE_BLOCK_FRAMES];
    for block_start in (interior_start..interior_end).step_by(PRE_UPSAMPLE_BLOCK_FRAMES) {
        let block_end = (block_start + PRE_UPSAMPLE_BLOCK_FRAMES).min(interior_end);
        let block_length = block_end - block_start;
        let phase_block = &mut phase_out[..block_length];
        convolve_symmetric_phase_block(audio, phase_block, block_start, coefficients);
        let output_block = &mut output[block_start * factor..block_end * factor];
        for (out_frame, &sample) in output_block
            .chunks_exact_mut(factor)
            .zip(phase_block.iter())
        {
            out_frame[i_phase] = sample;
        }
    }

    for i_frame in interior_end..frame_count {
        output[i_frame * factor + i_phase] = pre_upsample_symmetric_sample(
            |i_src_frame| audio[i_src_frame],
            i_frame,
            frame_count,
            coefficients,
        );
    }
}

#[inline(always)]
fn convolve_symmetric_phase_block(
    audio: &[f32],
    output: &mut [f32],
    block_start: usize,
    coefficients: &[f32; PRE_UPSAMPLE_N_TAPS],
) {
    output.fill(0.0);
    let block_length = output.len();
    for (tap, &coefficient) in coefficients
        .iter()
        .take(PRE_UPSAMPLE_N_TAPS / 2)
        .enumerate()
    {
        let mirror_tap = PRE_UPSAMPLE_N_TAPS - 1 - tap;
        let source_start = block_start + tap - PRE_UPSAMPLE_CENTER_TAP;
        let mirror_source_start = block_start + mirror_tap - PRE_UPSAMPLE_CENTER_TAP;
        let source = &audio[source_start..source_start + block_length];
        let mirror_source = &audio[mirror_source_start..mirror_source_start + block_length];
        for i_frame in 0..block_length {
            output[i_frame] += (source[i_frame] + mirror_source[i_frame]) * coefficient;
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
        for i_channel in 0..channels {
            let channel_offset = i_channel * frame_count;
            let channel = &audio[channel_offset..channel_offset + frame_count];
            for (i_frame, frame_peak) in frame_peaks.iter_mut().enumerate() {
                let peak = calc_interpolated_peak_at_frame_reference(
                    |i_src_frame| channel[i_src_frame],
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
        let upsampled_frame_count = frame_count * factor;
        let mut upsampled = vec![0.0; audio.len() * factor];
        for i_channel in 0..channels {
            let in_offset = i_channel * frame_count;
            let out_offset = i_channel * upsampled_frame_count;
            let channel = &audio[in_offset..in_offset + frame_count];
            let out_channel = &mut upsampled[out_offset..out_offset + upsampled_frame_count];
            for i_frame in 0..frame_count {
                for (i_phase, phase_coefficients) in coefficients.iter().enumerate() {
                    out_channel[i_frame * factor + i_phase] = pre_upsample_sample(
                        |i_src_frame| channel[i_src_frame],
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

        convolve_phase(&audio, &mut peaks, &BS1770_COEFFICIENTS[1]);

        for index in 0..audio.len() - (BS1770_N_TAPS - 1) {
            let samples: [f32; BS1770_N_TAPS] =
                audio[index..index + BS1770_N_TAPS].try_into().unwrap();
            assert_eq!(
                peaks[index + BS1770_CENTER_TAP],
                convolve_scalar(&samples, &BS1770_COEFFICIENTS[1]).abs()
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
        let first_block_boundary = PRE_UPSAMPLE_BLOCK_FRAMES + PRE_UPSAMPLE_N_TAPS - 1;
        let second_block_boundary = 2 * PRE_UPSAMPLE_BLOCK_FRAMES + PRE_UPSAMPLE_N_TAPS - 1;
        for channels in [1, 2, 3, 4, 8] {
            for frame_count in [
                0,
                1,
                31,
                32,
                63,
                64,
                65,
                257,
                first_block_boundary - 1,
                first_block_boundary,
                first_block_boundary + 1,
                second_block_boundary - 1,
                second_block_boundary,
                second_block_boundary + 1,
            ] {
                let channel_samples: Vec<Vec<f32>> = (0..channels)
                    .map(|i_channel| {
                        (0..frame_count)
                            .map(|i_frame| {
                                let i = i_channel * frame_count + i_frame;
                                ((i as f64 * 0.731).sin() + (i as f64 * 0.193).cos()) as f32
                            })
                            .collect()
                    })
                    .collect();
                let audio: Vec<_> = channel_samples.into_iter().flatten().collect();
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
