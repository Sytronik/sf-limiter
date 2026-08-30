use sf_limiter::{LimiterError, SFLimiter};

#[allow(non_snake_case)]
fn threshold_from_dBFS(threshold_dBFS: f64) -> f32 {
    10.0_f64.powf(threshold_dBFS / 20.0) as f32
}

fn assert_never_clips(audio: &[f32], ceiling: f32) {
    assert!(
        audio
            .iter()
            .all(|sample| sample.is_finite() && sample.abs() <= ceiling),
        "peak={} exceeds ceiling={ceiling}",
        audio.iter().map(|sample| sample.abs()).fold(0.0, f32::max)
    );
}

#[test]
fn adversarial_mono_signal_never_clips() {
    let mut input = vec![0.0; 48_000];
    for (index, sample) in input.iter_mut().enumerate() {
        *sample = match index % 997 {
            0 => 32.0,
            1 => -24.0,
            _ => ((index as f32 * 0.071).sin() * 3.5) + ((index as f32 * 0.013).cos() * 1.5),
        };
    }

    let mut limiter = SFLimiter::with_default(48_000).unwrap();
    let output = limiter.process_interleaved(&input, 1).unwrap();

    assert_eq!(output.audio.len(), input.len());
    assert_eq!(output.frame_gains.len(), input.len());
    assert!(
        output
            .frame_gains
            .iter()
            .all(|gain| (0.0..=1.0).contains(gain))
    );
    assert_never_clips(&output.audio, 1.0);
}

#[test]
fn linked_multichannel_signal_never_clips() {
    let channels = 6;
    let frames = 24_000;
    let mut state = 0x9e37_79b9_u32;
    let mut input = Vec::with_capacity(frames * channels);

    for i_frame in 0..frames {
        for i_channel in 0..channels {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
            let impulse = if (i_frame + i_channel * 131) % 4093 == 0 {
                64.0
            } else {
                0.0
            };
            input.push(noise * 12.0 + impulse);
        }
    }

    let mut limiter = SFLimiter::new(48_000, -2.0, 5.0, 15.0, 40.0, false).unwrap();
    let output = limiter.process_interleaved(&input, channels).unwrap();

    assert_eq!(output.frame_gains.len(), frames);
    assert_never_clips(&output.audio, threshold_from_dBFS(-2.0));
}

#[test]
fn processing_in_place_has_the_same_hard_ceiling() {
    let mut audio = vec![4.0, -3.0, 2.0, -5.0, 0.25, -0.25];
    let mut limiter = SFLimiter::new(1_000, 0.0, 1.0, 0.0, 0.0, false).unwrap();

    let frame_gains = limiter.process_interleaved_inplace(&mut audio, 2).unwrap();

    assert_eq!(frame_gains.len(), 3);
    assert_never_clips(&audio, 1.0);
}

#[test]
fn repeated_calls_start_from_a_neutral_envelope() {
    let hot_input = vec![8.0; 16];
    let quiet_input = vec![0.25; 8];
    let mut reused_limiter = SFLimiter::new(1_000, 0.0, 3.0, 2.0, 1_000.0, false).unwrap();
    let mut fresh_limiter = reused_limiter.clone();

    reused_limiter.process_interleaved(&hot_input, 1).unwrap();
    let reused_output = reused_limiter.process_interleaved(&quiet_input, 1).unwrap();
    let fresh_output = fresh_limiter.process_interleaved(&quiet_input, 1).unwrap();

    assert_eq!(reused_output, fresh_output);
}

#[test]
fn samples_use_the_reported_f32_gain() {
    let input = [1.1_f32, 0.1_f32];
    let mut limiter = SFLimiter::new(1_000, -10.0, 1.0, 0.0, 0.0, false).unwrap();

    let output = limiter.process_interleaved(&input, 2).unwrap();

    assert_eq!(output.audio[1], input[1] * output.frame_gains[0]);
    assert_ne!(output.frame_gains[0], 1.0);
    assert_never_clips(&output.audio, threshold_from_dBFS(-10.0));
}

#[test]
fn planar_processing_matches_interleaved_processing() {
    let interleaved = [0.0, 0.0, 4.0, -5.0, -3.0, 2.0, 0.2, -0.2];
    let mut planar = [0.0, 4.0, -3.0, 0.2, 0.0, -5.0, 2.0, -0.2];
    let mut interleaved_limiter = SFLimiter::new(1_000, -2.0, 1.0, 0.0, 0.0, false).unwrap();
    let mut planar_limiter = interleaved_limiter.clone();

    let interleaved_output = interleaved_limiter
        .process_interleaved(&interleaved, 2)
        .unwrap();
    let planar_frame_gains = planar_limiter
        .process_planar_inplace(&mut planar, 2)
        .unwrap();

    let planar_as_interleaved: Vec<_> = (0..4)
        .flat_map(|i_frame| [planar[i_frame], planar[4 + i_frame]])
        .collect();
    assert_eq!(planar_frame_gains, interleaved_output.frame_gains);
    assert_eq!(planar_as_interleaved, interleaved_output.audio);
    assert_never_clips(&planar, threshold_from_dBFS(-2.0));
}

#[test]
fn true_peak_planar_processing_matches_interleaved_processing() {
    let interleaved: Vec<_> = (0..128)
        .flat_map(|i_frame| {
            let sample = (((2.0 * std::f64::consts::PI * 12_000.0 * i_frame as f64 / 48_000.0)
                + std::f64::consts::FRAC_PI_4)
                .sin() as f32)
                * 1.1;
            [sample, -0.75 * sample]
        })
        .collect();
    let mut planar = Vec::with_capacity(interleaved.len());
    planar.extend(interleaved.iter().step_by(2).copied());
    planar.extend(interleaved.iter().skip(1).step_by(2).copied());
    let mut interleaved_limiter = SFLimiter::new(48_000, -2.0, 1.0, 0.0, 0.0, true).unwrap();
    let mut planar_limiter = interleaved_limiter.clone();

    let interleaved_output = interleaved_limiter
        .process_interleaved(&interleaved, 2)
        .unwrap();
    let planar_frame_gains = planar_limiter
        .process_planar_inplace(&mut planar, 2)
        .unwrap();
    let planar_as_interleaved: Vec<_> = (0..128)
        .flat_map(|i_frame| [planar[i_frame], planar[128 + i_frame]])
        .collect();

    assert_eq!(planar_frame_gains, interleaved_output.frame_gains);
    assert_eq!(planar_as_interleaved, interleaved_output.audio);
}

#[test]
fn true_peak_low_rate_processing_accepts_supported_sample_boundaries() {
    let maximum = 4_294_967_296.0_f32;
    let input: Vec<_> = (0..128)
        .map(|i_frame| if i_frame % 2 == 0 { maximum } else { -maximum })
        .collect();
    let mut interleaved_limiter = SFLimiter::new(8_000, 0.0, 1.0, 0.0, 0.0, true).unwrap();
    let mut planar_limiter = interleaved_limiter.clone();

    let interleaved_output = interleaved_limiter.process_interleaved(&input, 1).unwrap();
    let planar_output = planar_limiter.process_planar(&input, 1).unwrap();

    for output in [interleaved_output, planar_output] {
        assert_eq!(output.frame_gains.len(), input.len());
        assert!(output.frame_gains.iter().all(|gain| gain.is_finite()));
        assert_never_clips(&output.audio, 1.0);
    }
}

#[test]
fn samples_outside_the_supported_range_are_rejected_without_mutation() {
    let mut audio = [0.0, 8_589_934_592.0_f32];
    let original = audio;
    let mut limiter = SFLimiter::with_default(48_000).unwrap();

    let error = limiter
        .process_interleaved_inplace(&mut audio, 1)
        .unwrap_err();

    assert_eq!(
        error,
        LimiterError::InputSampleOutOfRange {
            index: 1,
            value: original[1],
        }
    );
    assert_eq!(audio, original);
}

#[test]
fn planar_non_finite_index_uses_flat_channel_major_order() {
    let mut planar = [0.0, 1.0, 2.0, f32::INFINITY];
    let mut limiter = SFLimiter::with_default(48_000).unwrap();

    let error = limiter.process_planar_inplace(&mut planar, 2).unwrap_err();

    assert_eq!(
        error,
        sf_limiter::LimiterError::NonFiniteSample { index: 3 }
    );
}

#[test]
fn planar_processing_rejects_invalid_layout_without_mutating_input() {
    let mut zero_channel_input = [1.0, -1.0];
    let original_zero_channel_input = zero_channel_input;
    let mut limiter = SFLimiter::with_default(48_000).unwrap();

    let error = limiter
        .process_planar_inplace(&mut zero_channel_input, 0)
        .unwrap_err();

    assert_eq!(error, LimiterError::InvalidChannelCount);
    assert_eq!(zero_channel_input, original_zero_channel_input);

    let mut misaligned_input = [1.0, -1.0, 0.5];
    let original_misaligned_input = misaligned_input;
    let error = limiter
        .process_planar_inplace(&mut misaligned_input, 2)
        .unwrap_err();

    assert_eq!(
        error,
        LimiterError::InputNotFrameAligned {
            sample_count: 3,
            channels: 2,
        }
    );
    assert_eq!(misaligned_input, original_misaligned_input);
}
