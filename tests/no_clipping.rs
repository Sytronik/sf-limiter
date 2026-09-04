use sf_limiter::SFLimiter;

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
