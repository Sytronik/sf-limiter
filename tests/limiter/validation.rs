use sf_limiter::{LimiterError, SFLimiter};

use super::common::{
    MAXIMUM_INPUT_AMPLITUDE, ProcessingApi, SR_PEAK_MODES_COMBINATIONS, assert_output_contract,
    assert_same_bits,
};

#[test]
fn processing_apis_accept_supported_sample_boundaries() {
    for (sample_rate, true_peak) in SR_PEAK_MODES_COMBINATIONS {
        let template = SFLimiter::new(sample_rate, 0.0, 1.0, 0.0, 0.0, true_peak).unwrap();
        for channels in [1, 6] {
            let interleaved: Vec<_> = (0..128)
                .flat_map(|i_frame| {
                    (0..channels).map(move |i_channel| {
                        if (i_frame + i_channel) % 2 == 0 {
                            MAXIMUM_INPUT_AMPLITUDE
                        } else {
                            -MAXIMUM_INPUT_AMPLITUDE
                        }
                    })
                })
                .collect();
            for api in ProcessingApi::ALL {
                let context = format!("{api:?}, sample_rate={sample_rate}, channels={channels}");
                let original = api.arrange_input(&interleaved, channels);
                let mut audio = original.clone();
                let output = api
                    .process(&mut template.clone(), &mut audio, channels)
                    .unwrap();
                assert_output_contract(api, &original, &output, channels, 1.0, &context);
                if !api.is_inplace() {
                    assert_same_bits(&audio, &original, &context);
                }
            }
        }
    }
}

#[test]
fn processing_apis_reject_invalid_samples_at_the_first_flat_index_without_mutation() {
    let outside = f32::from_bits(MAXIMUM_INPUT_AMPLITUDE.to_bits() + 1);
    for value in [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        outside,
        -outside,
        f32::MAX,
        -f32::MAX,
    ] {
        for (sample_rate, true_peak) in SR_PEAK_MODES_COMBINATIONS {
            let template = SFLimiter::new(sample_rate, 0.0, 1.0, 0.0, 0.0, true_peak).unwrap();
            for api in ProcessingApi::ALL {
                for index in [0, 5, 11] {
                    // These are flat buffers in the API's own layout. A later
                    // error of the other kind must not hide the earlier one.
                    let mut audio = [0.25; 12];
                    audio[11] = if value.is_finite() { f32::NAN } else { outside };
                    audio[index] = value;
                    let original = audio;
                    let context =
                        format!("{api:?}, sample_rate={sample_rate}, value={value}, index={index}");
                    let expected = if value.is_finite() {
                        LimiterError::InputSampleOutOfRange { index, value }
                    } else {
                        LimiterError::NonFiniteSample { index }
                    };
                    let error = api
                        .process(&mut template.clone(), &mut audio, 3)
                        .unwrap_err();
                    assert_eq!(error, expected, "{context}");
                    assert_same_bits(&audio, &original, &context);
                }
            }
        }
    }
}

#[test]
fn processing_apis_reject_invalid_layouts_without_mutation() {
    for (sample_rate, true_peak) in SR_PEAK_MODES_COMBINATIONS {
        let template = SFLimiter::new(sample_rate, 0.0, 1.0, 0.0, 0.0, true_peak).unwrap();
        for api in ProcessingApi::ALL {
            for (original, channels, expected) in [
                (vec![], 0, LimiterError::InvalidChannelCount),
                (vec![1.0, -1.0], 0, LimiterError::InvalidChannelCount),
                (
                    vec![1.0, -1.0, 0.5],
                    2,
                    LimiterError::InputNotFrameAligned {
                        sample_count: 3,
                        channels: 2,
                    },
                ),
            ] {
                let context = format!(
                    "{api:?}, sample_rate={sample_rate}, channels={channels}, len={}",
                    original.len()
                );
                let mut audio = original.clone();
                let error = api
                    .process(&mut template.clone(), &mut audio, channels)
                    .unwrap_err();
                assert_eq!(error, expected, "{context}");
                assert_same_bits(&audio, &original, &context);
            }
        }
    }
}
