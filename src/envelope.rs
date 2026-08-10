use std::collections::VecDeque;

#[derive(Clone, Debug)]
pub(crate) struct MovingMinimum {
    window: usize,
    index: usize,
    values: VecDeque<(usize, f64)>,
}

impl MovingMinimum {
    pub(crate) fn new(window: usize) -> Self {
        debug_assert!(window > 0);
        Self {
            window,
            index: 0,
            values: VecDeque::with_capacity(window),
        }
    }

    pub(crate) fn reset(&mut self) {
        self.index = 0;
        self.values.clear();
    }

    pub(crate) fn step(&mut self, value: f64) -> f64 {
        while self
            .values
            .back()
            .is_some_and(|&(_, previous)| previous >= value)
        {
            self.values.pop_back();
        }
        self.values.push_back((self.index, value));

        let first_valid = (self.index + 1).saturating_sub(self.window);
        while self
            .values
            .front()
            .is_some_and(|&(index, _)| index < first_valid)
        {
            self.values.pop_front();
        }
        self.index += 1;

        self.values
            .front()
            .map(|&(_, minimum)| minimum)
            .expect("the current value is always in the moving window")
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ExponentialRelease {
    release_samples: f64,
    release_slew: f64,
    output: f64,
}

impl ExponentialRelease {
    pub(crate) fn new(release_samples: f64) -> Self {
        Self {
            release_samples,
            release_slew: (release_samples + 1.0).recip(),
            output: 1.0,
        }
    }

    pub(crate) fn reset(&mut self) {
        self.output = 1.0;
    }

    pub(crate) fn step(&mut self, input: f64) -> f64 {
        let output = input.min((input - self.output).mul_add(self.release_slew, self.output));
        self.output = output;
        output
    }

    pub(crate) fn release_samples(&self) -> f64 {
        self.release_samples
    }
}

#[derive(Clone, Debug)]
struct BoxFilter {
    buffer: Vec<f64>,
    index: usize,
    sum: f64,
    inverse_length: f64,
}

impl BoxFilter {
    fn new(length: usize) -> Self {
        debug_assert!(length > 0);
        Self {
            buffer: vec![0.0; length],
            index: 0,
            sum: 0.0,
            inverse_length: 1.0 / length as f64,
        }
    }

    fn reset(&mut self, fill: f64) {
        self.buffer.fill(fill);
        self.index = 0;
        self.sum = fill * self.buffer.len() as f64;
    }

    fn step(&mut self, value: f64) -> f64 {
        self.sum += value - self.buffer[self.index];
        self.buffer[self.index] = value;
        self.index += 1;
        if self.index == self.buffer.len() {
            self.index = 0;
        }
        self.sum * self.inverse_length
    }
}

#[derive(Clone, Debug)]
pub(crate) struct BoxStackFilter {
    layers: [BoxFilter; 3],
}

impl BoxStackFilter {
    const RATIOS: [f64; 3] = [0.404_078_562_416, 0.334_851_475_794, 0.261_069_961_789];

    pub(crate) fn new(size: usize) -> Self {
        debug_assert!(size > 0);
        let order = size - 1;
        let mut lengths = [1; 3];
        let mut errors = [0.0; 3];
        let mut allocated_order = 0;

        for (index, ratio) in Self::RATIOS.into_iter().enumerate() {
            let fractional_order = ratio * order as f64;
            let layer_order = fractional_order.floor() as usize;
            lengths[index] = layer_order + 1;
            errors[index] = layer_order as f64 - fractional_order;
            allocated_order += layer_order;
        }

        for _ in allocated_order..order {
            let index = errors
                .iter()
                .enumerate()
                .min_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(index, _)| index)
                .expect("the fixed-size layer list is not empty");
            lengths[index] += 1;
            errors[index] += 1.0;
        }

        Self {
            layers: lengths.map(BoxFilter::new),
        }
    }

    pub(crate) fn reset(&mut self, fill: f64) {
        for layer in &mut self.layers {
            layer.reset(fill);
        }
    }

    pub(crate) fn step(&mut self, value: f64) -> f64 {
        self.layers
            .iter_mut()
            .fold(value, |current, layer| layer.step(current))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moving_minimum_tracks_a_fixed_window() {
        let mut minimum = MovingMinimum::new(3);
        let output: Vec<_> = [3.0, 2.0, 4.0, 5.0, 1.0]
            .into_iter()
            .map(|value| minimum.step(value))
            .collect();

        assert_eq!(output, [3.0, 2.0, 2.0, 2.0, 1.0]);
    }

    #[test]
    fn three_layer_stack_has_requested_impulse_length() {
        for size in 1..100 {
            let mut filter = BoxStackFilter::new(size);
            filter.reset(0.0);
            let mut output = Vec::new();
            output.push(filter.step(1.0));
            output.extend((1..(size + 3)).map(|_| filter.step(0.0)));

            assert!(output[..size].iter().all(|value| *value > 0.0));
            assert!(output[size..].iter().all(|value| value.abs() < 1e-15));
        }
    }
}
