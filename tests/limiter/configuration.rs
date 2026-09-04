use sf_limiter::{LimiterError, SFLimiter};

use super::common::TRUE_PEAK_SAMPLE_RATES;

#[test]
fn rejects_zero_sample_rate() {
    assert_eq!(
        SFLimiter::with_default(0).unwrap_err(),
        LimiterError::InvalidSampleRate
    );
}

#[test]
fn rejects_invalid_thresholds() {
    for value in [0.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let error = SFLimiter::new(48_000, value, 5.0, 15.0, 40.0, false).unwrap_err();
        let LimiterError::InvalidThreshold(actual) = error else {
            panic!("threshold={value}: unexpected error {error:?}");
        };
        if value.is_nan() {
            assert!(actual.is_nan());
        } else {
            assert_eq!(actual, value);
        }
    }
}

#[test]
fn rejects_invalid_time_parameters() {
    for (i_parameter, parameter) in ["attack_ms", "hold_ms", "release_ms"]
        .into_iter()
        .enumerate()
    {
        for value in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut times = [5.0, 15.0, 40.0];
            times[i_parameter] = value;
            let error =
                SFLimiter::new(48_000, 0.0, times[0], times[1], times[2], false).unwrap_err();
            let LimiterError::InvalidTime {
                parameter: actual_parameter,
                value_ms,
            } = error
            else {
                panic!("{parameter}={value}: unexpected error {error:?}");
            };
            assert_eq!(actual_parameter, parameter);
            if value.is_nan() {
                assert!(value_ms.is_nan(), "{parameter}");
            } else {
                assert_eq!(value_ms, value, "{parameter}");
            }
        }
    }
}

#[test]
fn attack_time_rounds_at_half_sample_boundaries() {
    for attack_ms in [0.0, 0.499] {
        assert_eq!(
            SFLimiter::new(1_000, 0.0, attack_ms, 0.0, 0.0, false).unwrap_err(),
            LimiterError::AttackTooShort {
                value_ms: attack_ms,
                sample_rate: 1_000,
            }
        );
    }
    for (attack_ms, expected_samples) in [(0.5, 1), (1.499, 1), (1.5, 2)] {
        let limiter = SFLimiter::new(1_000, 0.0, attack_ms, 0.0, 0.0, false).unwrap();
        assert_eq!(
            limiter.attack_samples(),
            expected_samples,
            "attack_ms={attack_ms}"
        );
        assert_eq!(
            limiter.lookahead_samples(),
            expected_samples,
            "attack_ms={attack_ms}"
        );
        assert_eq!(limiter.hold_samples(), 0);
        assert_eq!(limiter.release_samples(), 0.0);
    }
}

#[test]
fn default_configuration_has_expected_values() {
    let limiter = SFLimiter::with_default(48_000).unwrap();

    assert_eq!(limiter.sample_rate(), 48_000);
    assert_eq!(limiter.threshold_dBFS(), 0.0);
    assert_eq!(limiter.threshold(), 1.0);
    assert!(!limiter.true_peak());
    assert_eq!(limiter.attack_samples(), 240);
    assert_eq!(limiter.lookahead_samples(), 240);
    assert_eq!(limiter.hold_samples(), 720);
    assert_eq!(limiter.release_samples(), 1_920.0);
}

#[test]
fn configured_times_report_rounded_attack_and_hold_but_fractional_release() {
    let limiter = SFLimiter::new(1_000, -6.0, 1.5, 2.0, 2.5, false).unwrap();

    assert_eq!(limiter.sample_rate(), 1_000);
    assert_eq!(limiter.attack_samples(), 2);
    assert_eq!(limiter.lookahead_samples(), 2);
    assert_eq!(limiter.hold_samples(), 2);
    assert_eq!(limiter.release_samples(), 2.5);
}

#[test]
fn true_peak_mode_accepts_only_supported_sample_rates() {
    for sample_rate in TRUE_PEAK_SAMPLE_RATES {
        assert!(SFLimiter::new(sample_rate, 0.0, 5.0, 15.0, 40.0, true).is_ok());
    }

    for sample_rate in [10_000, 47_999, 176_399] {
        assert_eq!(
            SFLimiter::new(sample_rate, 0.0, 5.0, 15.0, 40.0, true).unwrap_err(),
            LimiterError::UnsupportedTruePeakSampleRate(sample_rate)
        );
        assert!(SFLimiter::new(sample_rate, 0.0, 5.0, 15.0, 40.0, false).is_ok());
    }
}

#[test]
#[allow(non_snake_case)]
fn threshold_is_configured_in_dBFS() {
    let threshold_dBFS = -6.0;
    let limiter = SFLimiter::new(48_000, threshold_dBFS, 5.0, 15.0, 40.0, false).unwrap();

    assert_eq!(limiter.threshold_dBFS(), threshold_dBFS);
    assert_eq!(limiter.threshold(), 10.0_f64.powf(threshold_dBFS / 20.0));
}
