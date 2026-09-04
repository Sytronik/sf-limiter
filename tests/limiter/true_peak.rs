use sf_limiter::SFLimiter;

use super::common::{ProcessingApi, TRUE_PEAK_SAMPLE_RATES, planar_to_interleaved};

fn inter_sample_peak_signal(i_frame: usize) -> f32 {
    ((2.0 * std::f64::consts::PI * 12_000.0 * i_frame as f64 / 48_000.0)
        + std::f64::consts::FRAC_PI_4)
        .sin() as f32
        * 1.1
}

#[test]
fn true_peak_mode_attenuates_inter_sample_peaks_missed_by_sample_peak_mode() {
    let input: Vec<_> = (0..256).map(inter_sample_peak_signal).collect();
    let mut sample_peak_limiter = SFLimiter::new(48_000, 0.0, 1.0, 0.0, 0.0, false).unwrap();
    let mut true_peak_limiter = SFLimiter::new(48_000, 0.0, 1.0, 0.0, 0.0, true).unwrap();

    let sample_peak_output = sample_peak_limiter.process_interleaved(&input, 1).unwrap();
    let true_peak_output = true_peak_limiter.process_interleaved(&input, 1).unwrap();

    assert_eq!(sample_peak_output.audio, input);
    assert!(
        true_peak_output
            .audio
            .iter()
            .zip(&input)
            .any(|(output, input)| output.abs() < input.abs())
    );
    assert!(true_peak_limiter.true_peak());
    assert!(!sample_peak_limiter.true_peak());
}

#[test]
fn true_peak_mode_reports_the_gain_applied_to_an_over_ceiling_sample() {
    let mut input = vec![0.0; 128];
    input[64] = 1.02;
    let mut limiter = SFLimiter::new(48_000, 0.0, 1.0, 0.0, 0.0, true).unwrap();

    let output = limiter.process_interleaved(&input, 1).unwrap();

    assert!(output.frame_gains[64] < 1.0);
    assert_eq!(output.audio[64], input[64] * output.frame_gains[64]);
}

#[test]
fn true_peak_mode_enforces_a_negative_sample_peak_ceiling() {
    let input: Vec<_> = (0..256).map(inter_sample_peak_signal).collect();
    let mut limiter = SFLimiter::new(48_000, -2.0, 1.0, 0.0, 0.0, true).unwrap();

    let output = limiter.process_planar(&input, 1).unwrap();

    assert!(
        output
            .audio
            .iter()
            .all(|sample| f64::from(sample.abs()) <= limiter.threshold()),
        "sample peak exceeds threshold {}",
        limiter.threshold()
    );
}

#[test]
fn true_peak_mode_processes_every_supported_sample_rate() {
    for sample_rate in TRUE_PEAK_SAMPLE_RATES {
        let input: Vec<_> = (0..256)
            .map(|index| {
                ((2.0 * std::f64::consts::PI * (sample_rate as f64 / 4.0) * index as f64
                    / sample_rate as f64)
                    + std::f64::consts::FRAC_PI_4)
                    .sin() as f32
                    * 1.1
            })
            .collect();
        let mut limiter = SFLimiter::new(sample_rate, -1.0, 1.0, 0.0, 0.0, true).unwrap();

        let output = limiter.process_planar(&input, 1).unwrap();

        assert!(
            output
                .audio
                .iter()
                .all(|sample| sample.is_finite() && f64::from(sample.abs()) <= limiter.threshold()),
            "sample_rate={sample_rate}, threshold={}",
            limiter.threshold()
        );
        assert_eq!(output.frame_gains.len(), input.len());
        assert!(
            output
                .frame_gains
                .iter()
                .all(|gain| gain.is_finite() && (0.0..=1.0).contains(gain))
        );
    }
}

#[test]
fn true_peak_planar_processing_matches_interleaved_processing() {
    let interleaved: Vec<_> = (0..128)
        .flat_map(|i_frame| {
            let sample = inter_sample_peak_signal(i_frame);
            [sample, -0.75 * sample]
        })
        .collect();
    let mut planar = ProcessingApi::Planar.arrange_input(&interleaved, 2);
    let mut interleaved_limiter = SFLimiter::new(48_000, -2.0, 1.0, 0.0, 0.0, true).unwrap();
    let mut planar_limiter = interleaved_limiter.clone();

    let interleaved_output = interleaved_limiter
        .process_interleaved(&interleaved, 2)
        .unwrap();
    let planar_frame_gains = planar_limiter
        .process_planar_inplace(&mut planar, 2)
        .unwrap();
    let planar_as_interleaved = planar_to_interleaved(&planar, 2);

    assert_eq!(planar_frame_gains, interleaved_output.frame_gains);
    assert_eq!(planar_as_interleaved, interleaved_output.audio);
}
