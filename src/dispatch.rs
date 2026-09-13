//! Dispatch whole loops so their bodies are compiled with the selected CPU features.
//! Closures must use `#[inline(always)] move ||` so large loop bodies do not
//! escape into baseline functions and their captured slice metadata stays invariant.
//! The scalar fallback can still be auto-vectorized for the build's baseline target.

#[cfg(feature = "simd")]
fn arch() -> pulp::Arch {
    static ARCH: std::sync::OnceLock<pulp::Arch> = std::sync::OnceLock::new();
    *ARCH.get_or_init(pulp::Arch::new)
}

#[inline(always)]
pub(crate) fn dispatch<R>(operation: impl FnOnce() -> R) -> R {
    #[cfg(all(test, feature = "simd"))]
    if let Some(arch) = tests::OVERRIDE.get() {
        return arch.dispatch(operation);
    }
    #[cfg(feature = "simd")]
    {
        arch().dispatch(operation)
    }
    #[cfg(not(feature = "simd"))]
    {
        operation()
    }
}

#[cfg(all(test, feature = "simd"))]
mod tests {
    use super::*;
    use crate::{SFLimiter, TRUE_PEAK_SAMPLE_RATE_CASES};

    thread_local! {
        pub(super) static OVERRIDE: std::cell::Cell<Option<pulp::Arch>> = const { std::cell::Cell::new(None) };
    }

    fn with_arch<R>(arch: pulp::Arch, operation: impl FnOnce() -> R) -> R {
        struct Restore(Option<pulp::Arch>);
        impl Drop for Restore {
            fn drop(&mut self) {
                OVERRIDE.set(self.0);
            }
        }
        let _restore = Restore(OVERRIDE.replace(Some(arch)));
        operation()
    }

    #[test]
    fn dispatched_processing_matches_baseline() {
        for sample_rate in TRUE_PEAK_SAMPLE_RATE_CASES {
            for true_peak in [false, true] {
                for channels in [1, 2, 3, 4, 5, 8] {
                    for frames in [0, 1, 11, 12, 13, 63, 64, 65, 127, 128, 129, 257] {
                        let audio: Vec<f32> = (0..frames * channels)
                            .map(|i| ((i * 127 % 1021) as f32 - 510.0) / 256.0)
                            .collect();
                        for planar in [false, true] {
                            let run = || {
                                let mut limiter =
                                    SFLimiter::new(sample_rate, -3.0, 1.0, 1.0, 10.0, true_peak)
                                        .unwrap();
                                if planar {
                                    limiter.process_planar(&audio, channels)
                                } else {
                                    limiter.process_interleaved(&audio, channels)
                                }
                            };
                            let baseline = with_arch(pulp::Arch::Scalar, run).unwrap();
                            let selected = with_arch(arch(), run).unwrap();
                            assert_eq!(
                                baseline, selected,
                                "rate={sample_rate}, true_peak={true_peak}, channels={channels}, frames={frames}, planar={planar}"
                            );
                        }
                    }
                }
            }
        }
    }
}
