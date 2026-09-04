use sf_limiter::{LimiterError, SFLimiter};

use super::common::{
    ProcessingApi, SR_PEAK_MODES_COMBINATIONS, assert_output_contract, assert_same_bits,
    planar_to_interleaved,
};

#[test]
fn processing_apis_preserve_output_contract_across_layouts_and_configurations() {
    let configurations = [
        (1_000, 0.0, 1.0, 0.0, 0.0, false),
        (1_000, -2.0, 1.0, 0.0, 0.0, false),
        (1_000, -60.0, 1.0, 0.0, 0.0, false),
        (1_000, 0.0, 3.0, 2.0, 5.0, false),
        (1_000, -2.0, 3.0, 2.0, 5.0, false),
        (1_000, -60.0, 3.0, 2.0, 5.0, false),
        (8_000, -2.0, 1.0, 0.0, 0.0, true),
        (48_000, -2.0, 1.0, 0.0, 0.0, true),
        (96_000, -2.0, 1.0, 0.0, 0.0, true),
        (192_000, -2.0, 1.0, 0.0, 0.0, true),
    ];
    for (sample_rate, threshold_db, attack_ms, hold_ms, release_ms, true_peak) in configurations {
        let template = SFLimiter::new(
            sample_rate,
            threshold_db,
            attack_ms,
            hold_ms,
            release_ms,
            true_peak,
        )
        .unwrap();
        for channels in [1, 2, 6] {
            for frame_count in [0, 1, 2, 7, 257] {
                let mut interleaved: Vec<_> = (0..frame_count * channels)
                    .map(|i_sample| [0.0, 0.125, -0.25, 2.0, -3.0, 0.5, -0.0625][i_sample % 7])
                    .collect();
                if frame_count > 0 {
                    for (i_frame, i_channel, value) in [
                        (0, 0, 8.0),
                        (frame_count / 2, channels / 2, -16.0),
                        (frame_count - 1, channels - 1, 32.0),
                    ] {
                        interleaved[i_frame * channels + i_channel] = value;
                    }
                }
                let context = format!(
                    "sample_rate={sample_rate}, threshold={threshold_db}, \
                    attack/hold/release={attack_ms}/{hold_ms}/{release_ms}, \
                    true_peak={true_peak}, channels={channels}, frames={frame_count}"
                );
                let run_api = |api: ProcessingApi| {
                    let api_context = format!("{api:?}, {context}");
                    let original = api.arrange_input(&interleaved, channels);
                    let mut audio = original.clone();
                    let output = api
                        .process(&mut template.clone(), &mut audio, channels)
                        .unwrap_or_else(|error| panic!("{api_context}: {error}"));
                    assert_output_contract(
                        api,
                        &original,
                        &output,
                        channels,
                        template.threshold() as f32,
                        &api_context,
                    );
                    if !api.is_inplace() {
                        assert_same_bits(&audio, &original, &api_context);
                    }
                    output
                };
                let interleaved_allocating_output = run_api(ProcessingApi::Interleaved);
                let interleaved_inplace_output = run_api(ProcessingApi::InterleavedInplace);
                let planar_allocating_output = run_api(ProcessingApi::Planar);
                let planar_inplace_output = run_api(ProcessingApi::PlanarInplace);

                assert_eq!(
                    interleaved_allocating_output, interleaved_inplace_output,
                    "interleaved allocating/in-place: {context}"
                );
                assert_eq!(
                    planar_allocating_output, planar_inplace_output,
                    "planar allocating/in-place: {context}"
                );
                let planar_audio = planar_to_interleaved(&planar_allocating_output.audio, channels);
                for (name, actual, expected) in [
                    ("audio", &planar_audio, &interleaved_allocating_output.audio),
                    (
                        "frame_gains",
                        &planar_allocating_output.frame_gains,
                        &interleaved_allocating_output.frame_gains,
                    ),
                ] {
                    if true_peak {
                        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate()
                        {
                            assert!(
                                approx::relative_eq!(
                                    actual,
                                    expected,
                                    epsilon = 16.0 * f32::EPSILON,
                                    max_relative = 16.0 * f32::EPSILON
                                ),
                                "{context}, {name}[{index}]: planar={actual}, interleaved={expected}"
                            );
                        }
                    } else {
                        assert_eq!(actual, expected, "{context}, {name}");
                    }
                }
            }
        }
    }
}

#[test]
fn repeated_calls_start_from_a_neutral_envelope() {
    for (sample_rate, true_peak) in SR_PEAK_MODES_COMBINATIONS {
        let template = SFLimiter::new(sample_rate, 0.0, 3.0, 2.0, 1_000.0, true_peak).unwrap();
        for api in ProcessingApi::ALL {
            let mut reused_limiter = template.clone();
            let quiet_input = api.arrange_input(&[0.25, -0.125, 0.0625].repeat(9), 3);
            let fresh_output = api
                .process(&mut template.clone(), &mut quiet_input.clone(), 3)
                .unwrap();
            for (insert_error, insert_empty) in
                [(false, false), (true, false), (false, true), (true, true)]
            {
                let context = format!(
                    "{api:?}, sample_rate={sample_rate}, error={insert_error}, empty={insert_empty}"
                );
                api.process(&mut reused_limiter, &mut [8.0; 16], 1).unwrap();
                if insert_error {
                    let mut invalid_input = [0.25, f32::NAN, 0.5, -0.5];
                    let original = invalid_input;
                    assert_eq!(
                        api.process(&mut reused_limiter, &mut invalid_input, 2)
                            .unwrap_err(),
                        LimiterError::NonFiniteSample { index: 1 },
                        "{context}"
                    );
                    assert_same_bits(&invalid_input, &original, &context);
                }
                if insert_empty {
                    let output = api.process(&mut reused_limiter, &mut [], 6).unwrap();
                    assert!(
                        output.audio.is_empty() && output.frame_gains.is_empty(),
                        "{context}"
                    );
                }
                let reused_output = api
                    .process(&mut reused_limiter, &mut quiet_input.clone(), 3)
                    .unwrap();
                assert_eq!(reused_output, fresh_output, "{context}");
            }
        }
    }
}

#[test]
fn samples_use_the_reported_f32_gain() {
    // Only the first channel exceeds the ceiling; both quiet channels must
    // follow its gain without relying on the final sample clamp. Separate the
    // hot frames to preserve the original 1.1/0.1 rounding regression case.
    let interleaved = [
        [0.1, 0.05, -0.05],
        [0.1, 0.05, -0.05],
        [1.1, 0.1, -0.2],
        [0.1, 0.05, -0.05],
        [-2.0, 0.1, -0.1],
        [0.1, 0.05, -0.05],
        [0.1, 0.05, -0.05],
    ]
    .concat();
    let channels = 3;
    let frame_count = interleaved.len() / channels;
    for api in ProcessingApi::ALL {
        let mut limiter = SFLimiter::new(1_000, -10.0, 1.0, 0.0, 0.0, false).unwrap();
        let input = api.arrange_input(&interleaved, channels);
        let output = api
            .process(&mut limiter, &mut input.clone(), channels)
            .unwrap();
        let context = format!("{api:?}");
        assert_output_contract(
            api,
            &input,
            &output,
            channels,
            limiter.threshold() as f32,
            &context,
        );
        assert_eq!(output.frame_gains[0], 1.0);
        for i_frame in [2, 4] {
            let gain = output.frame_gains[i_frame];
            assert!(gain > 0.0 && gain < 1.0, "{context}, i_frame={i_frame}");
            for i_channel in [1, 2] {
                let i_sample = if api.is_planar() {
                    i_channel * frame_count + i_frame
                } else {
                    i_frame * channels + i_channel
                };
                assert_eq!(
                    output.audio[i_sample],
                    input[i_sample] * gain,
                    "{context}, i_frame={i_frame}, i_channel={i_channel}"
                );
                assert!(
                    output.audio[i_sample].abs() < input[i_sample].abs(),
                    "{context}"
                );
            }
        }
    }
}

#[test]
fn sample_peak_processing_preserves_silence_and_below_ceiling_audio() {
    for threshold_db in [0.0, -2.0, -60.0] {
        let template = SFLimiter::new(1_000, threshold_db, 3.0, 2.0, 5.0, false).unwrap();
        let ceiling = template.threshold() as f32;
        for channels in [1, 2, 6] {
            for scale in [0.0, 0.5 * ceiling] {
                let interleaved: Vec<_> = (0..9 * channels)
                    .map(|i_sample| [0.0, 1.0, -1.0, 0.5][i_sample % 4] * scale)
                    .collect();
                for api in ProcessingApi::ALL {
                    let context = format!(
                        "{api:?}, threshold={threshold_db}, channels={channels}, scale={scale}"
                    );
                    let original = api.arrange_input(&interleaved, channels);
                    let mut audio = original.clone();
                    let output = api
                        .process(&mut template.clone(), &mut audio, channels)
                        .unwrap();
                    assert_same_bits(&output.audio, &original, &context);
                    assert_same_bits(&audio, &original, &context);
                    assert_eq!(output.frame_gains, vec![1.0; 9], "{context}");
                }
            }
        }
    }
}
