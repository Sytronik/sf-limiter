use sf_limiter::SFLimiter;

fn assert_never_clips(samples: &[f32], ceiling: f32) {
    assert!(
        samples
            .iter()
            .all(|sample| sample.is_finite() && sample.abs() <= ceiling),
        "peak={} exceeds ceiling={ceiling}",
        samples
            .iter()
            .map(|sample| sample.abs())
            .fold(0.0, f32::max)
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

    assert_eq!(output.samples.len(), input.len());
    assert_eq!(output.gains.len(), input.len());
    assert!(output.gains.iter().all(|gain| (0.0..=1.0).contains(gain)));
    assert_never_clips(&output.samples, 1.0);
}

#[test]
fn linked_multichannel_signal_never_clips() {
    let channels = 6;
    let frames = 24_000;
    let mut state = 0x9e37_79b9_u32;
    let mut input = Vec::with_capacity(frames * channels);

    for frame in 0..frames {
        for channel in 0..channels {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
            let impulse = if (frame + channel * 131) % 4093 == 0 {
                64.0
            } else {
                0.0
            };
            input.push(noise * 12.0 + impulse);
        }
    }

    let mut limiter = SFLimiter::new(48_000, 0.8, 5.0, 15.0, 40.0).unwrap();
    let output = limiter.process_interleaved(&input, channels).unwrap();

    assert_eq!(output.gains.len(), frames);
    assert_never_clips(&output.samples, 0.8);
}

#[test]
fn processing_in_place_has_the_same_hard_ceiling() {
    let mut audio = vec![4.0, -3.0, 2.0, -5.0, 0.25, -0.25];
    let mut limiter = SFLimiter::new(1_000, 1.0, 1.0, 0.0, 0.0).unwrap();

    let gains = limiter.process_interleaved_inplace(&mut audio, 2).unwrap();

    assert_eq!(gains.len(), 3);
    assert_never_clips(&audio, 1.0);
}

#[test]
fn samples_use_the_reported_f32_gain() {
    let input = [1.1_f32, 0.1_f32];
    let mut limiter = SFLimiter::new(1_000, 0.3, 1.0, 0.0, 0.0).unwrap();

    let output = limiter.process_interleaved(&input, 2).unwrap();

    assert_eq!(output.samples[1], input[1] * output.gains[0]);
    assert_ne!(output.gains[0], 1.0);
    assert_never_clips(&output.samples, 0.3);
}

#[test]
fn planar_processing_matches_interleaved_processing() {
    let interleaved = [0.0, 0.0, 4.0, -5.0, -3.0, 2.0, 0.2, -0.2];
    let mut planar = [0.0, 4.0, -3.0, 0.2, 0.0, -5.0, 2.0, -0.2];
    let mut interleaved_limiter = SFLimiter::new(1_000, 0.8, 1.0, 0.0, 0.0).unwrap();
    let mut planar_limiter = interleaved_limiter.clone();

    let interleaved_output = interleaved_limiter
        .process_interleaved(&interleaved, 2)
        .unwrap();
    let planar_gains = planar_limiter
        .process_planar_inplace(&mut planar, 2)
        .unwrap();

    let planar_as_interleaved: Vec<_> = (0..4)
        .flat_map(|frame| [planar[frame], planar[4 + frame]])
        .collect();
    assert_eq!(planar_gains, interleaved_output.gains);
    assert_eq!(planar_as_interleaved, interleaved_output.samples);
    assert_never_clips(&planar, 0.8);
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
