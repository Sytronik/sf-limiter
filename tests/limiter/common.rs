use sf_limiter::{LimiterError, LimiterOutput, SFLimiter};

pub(super) const MAXIMUM_INPUT_AMPLITUDE: f32 = 4_294_967_296.0; // The inclusive input bound is 2^32.
pub(super) const SR_PEAK_MODES_COMBINATIONS: [(u32, bool); 5] = [
    (1_000, false),
    (8_000, true),
    (48_000, true),
    (96_000, true),
    (192_000, true),
];
pub(super) const TRUE_PEAK_SAMPLE_RATES: [u32; 13] = [
    8_000, 11_025, 12_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000, 88_200, 96_000, 176_400,
    192_000,
];

#[derive(Clone, Copy, Debug)]
pub(super) enum ProcessingApi {
    Interleaved,
    InterleavedInplace,
    Planar,
    PlanarInplace,
}

impl ProcessingApi {
    pub(super) const ALL: [Self; 4] = [
        Self::Interleaved,
        Self::InterleavedInplace,
        Self::Planar,
        Self::PlanarInplace,
    ];

    pub(super) fn is_planar(self) -> bool {
        matches!(self, Self::Planar | Self::PlanarInplace)
    }

    pub(super) fn is_inplace(self) -> bool {
        matches!(self, Self::InterleavedInplace | Self::PlanarInplace)
    }

    pub(super) fn arrange_input(self, interleaved: &[f32], channels: usize) -> Vec<f32> {
        if self.is_planar() {
            (0..channels)
                .flat_map(|i_channel| {
                    interleaved
                        .iter()
                        .skip(i_channel)
                        .step_by(channels)
                        .copied()
                })
                .collect()
        } else {
            interleaved.to_vec()
        }
    }

    pub(super) fn process(
        self,
        limiter: &mut SFLimiter,
        audio: &mut [f32],
        channels: usize,
    ) -> Result<LimiterOutput, LimiterError> {
        let frame_gains = match self {
            Self::Interleaved => return limiter.process_interleaved(audio, channels),
            Self::Planar => return limiter.process_planar(audio, channels),
            Self::InterleavedInplace => limiter.process_interleaved_inplace(audio, channels)?,
            Self::PlanarInplace => limiter.process_planar_inplace(audio, channels)?,
        };
        Ok(LimiterOutput {
            audio: audio.to_vec(),
            frame_gains,
        })
    }
}

pub(super) fn planar_to_interleaved(planar: &[f32], channels: usize) -> Vec<f32> {
    let frame_count = planar.len() / channels;
    (0..frame_count)
        .flat_map(|i_frame| {
            (0..channels).map(move |i_channel| planar[i_channel * frame_count + i_frame])
        })
        .collect()
}

pub(super) fn assert_same_bits(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}");
    for (i_sample, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{context}, i_sample={i_sample}"
        );
    }
}

pub(super) fn assert_output_contract(
    api: ProcessingApi,
    input: &[f32],
    output: &LimiterOutput,
    channels: usize,
    ceiling: f32,
    context: &str,
) {
    let frame_count = input.len() / channels;
    assert_eq!(output.audio.len(), input.len(), "{context}");
    assert_eq!(output.frame_gains.len(), frame_count, "{context}");
    for (i_frame, gain) in output.frame_gains.iter().enumerate() {
        assert!(
            gain.is_finite() && (0.0..=1.0).contains(gain),
            "{context}, i_frame={i_frame}, gain={gain}"
        );
    }
    for (i_sample, (&input_sample, &output_sample)) in input.iter().zip(&output.audio).enumerate() {
        let i_frame = if api.is_planar() {
            i_sample % frame_count
        } else {
            i_sample / channels
        };
        assert!(
            output_sample.is_finite() && output_sample.abs() <= ceiling,
            "{context}, i_sample={i_sample}, output={output_sample}, ceiling={ceiling}"
        );
        assert_eq!(
            output_sample,
            (input_sample * output.frame_gains[i_frame]).clamp(-ceiling, ceiling),
            "{context}, i_sample={i_sample}, i_frame={i_frame}"
        );
    }
}
