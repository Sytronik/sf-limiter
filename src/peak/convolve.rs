// Keep the fixed tap expressions explicit so LLVM can vectorize the independent
// outer channel or frame loops without relying on nested-loop transformations.
macro_rules! convolve_fir_12 {
    (|$tap:ident| $sample:expr, $coefficients:expr $(,)?) => {{
        let coefficients = $coefficients;
        let mut sum = {
            let $tap = 0_usize;
            $sample * coefficients[11]
        };
        {
            let $tap = 1_usize;
            sum += $sample * coefficients[10];
        }
        {
            let $tap = 2_usize;
            sum += $sample * coefficients[9];
        }
        {
            let $tap = 3_usize;
            sum += $sample * coefficients[8];
        }
        {
            let $tap = 4_usize;
            sum += $sample * coefficients[7];
        }
        {
            let $tap = 5_usize;
            sum += $sample * coefficients[6];
        }
        {
            let $tap = 6_usize;
            sum += $sample * coefficients[5];
        }
        {
            let $tap = 7_usize;
            sum += $sample * coefficients[4];
        }
        {
            let $tap = 8_usize;
            sum += $sample * coefficients[3];
        }
        {
            let $tap = 9_usize;
            sum += $sample * coefficients[2];
        }
        {
            let $tap = 10_usize;
            sum += $sample * coefficients[1];
        }
        {
            let $tap = 11_usize;
            sum += $sample * coefficients[0];
        }
        sum
    }};
}

macro_rules! convolve_mirrored_fir_12 {
    (|$tap:ident| $sample:expr, $coeff_pairs:expr $(,)?) => {{
        let coeff_pairs = $coeff_pairs;
        let (common_0, differential_0) = calc_mirrored_terms(
            {
                let $tap = 0_usize;
                $sample
            },
            {
                let $tap = 11_usize;
                $sample
            },
            coeff_pairs[0].0,
            coeff_pairs[0].1,
        );
        let (common_1, differential_1) = calc_mirrored_terms(
            {
                let $tap = 1_usize;
                $sample
            },
            {
                let $tap = 10_usize;
                $sample
            },
            coeff_pairs[1].0,
            coeff_pairs[1].1,
        );
        let (common_2, differential_2) = calc_mirrored_terms(
            {
                let $tap = 2_usize;
                $sample
            },
            {
                let $tap = 9_usize;
                $sample
            },
            coeff_pairs[2].0,
            coeff_pairs[2].1,
        );
        let (common_3, differential_3) = calc_mirrored_terms(
            {
                let $tap = 3_usize;
                $sample
            },
            {
                let $tap = 8_usize;
                $sample
            },
            coeff_pairs[3].0,
            coeff_pairs[3].1,
        );
        let (common_4, differential_4) = calc_mirrored_terms(
            {
                let $tap = 4_usize;
                $sample
            },
            {
                let $tap = 7_usize;
                $sample
            },
            coeff_pairs[4].0,
            coeff_pairs[4].1,
        );
        let (common_5, differential_5) = calc_mirrored_terms(
            {
                let $tap = 5_usize;
                $sample
            },
            {
                let $tap = 6_usize;
                $sample
            },
            coeff_pairs[5].0,
            coeff_pairs[5].1,
        );
        let common_sum = common_0 + common_1 + common_2 + common_3 + common_4 + common_5;
        let differential_sum = differential_0
            + differential_1
            + differential_2
            + differential_3
            + differential_4
            + differential_5;
        let sample = common_sum + differential_sum;
        let mirror_sample = common_sum - differential_sum;
        (sample, mirror_sample)
    }};
}

pub(super) use {convolve_fir_12, convolve_mirrored_fir_12};

#[inline(always)]
pub(super) fn convolve_scalar<const N: usize>(samples: &[f32; N], coefficients: &[f32; N]) -> f32 {
    samples
        .iter()
        .zip(coefficients.iter().rev())
        .map(|(sample, coefficient)| sample * coefficient)
        .sum()
}

#[inline(always)]
pub(super) const fn decompose_mirrored_coefficients(
    coefficient: f32,
    mirror_coefficient: f32,
) -> (f32, f32) {
    (
        (coefficient + mirror_coefficient) * 0.5,
        (coefficient - mirror_coefficient) * 0.5,
    )
}

#[inline(always)]
pub(super) fn calc_mirrored_terms(
    input: f32,
    mirror_input: f32,
    common_coefficient: f32,
    differential_coefficient: f32,
) -> (f32, f32) {
    (
        (input + mirror_input) * common_coefficient,
        (input - mirror_input) * differential_coefficient,
    )
}

pub(super) fn convolve_mirrored_samples_with_pairs<const PAIR_COUNT: usize>(
    mut sample_at: impl FnMut(usize) -> f32,
    coeff_pairs: &[(f32, f32); PAIR_COUNT],
) -> (f32, f32) {
    let mut primary_sum = 0.0_f32;
    let mut mirror_sum = 0.0_f32;
    for (tap, &(common_coefficient, differential_coefficient)) in coeff_pairs.iter().enumerate() {
        let mirror_tap = PAIR_COUNT * 2 - 1 - tap;
        let (common, differential) = calc_mirrored_terms(
            sample_at(tap),
            sample_at(mirror_tap),
            common_coefficient,
            differential_coefficient,
        );
        primary_sum += common + differential;
        mirror_sum += common - differential;
    }
    (primary_sum, mirror_sum)
}
