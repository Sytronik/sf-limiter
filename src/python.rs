use numpy::ndarray::{ArrayD, ArrayViewD, IxDyn};
use numpy::{AllowTypeChange, IntoPyArray, PyArray1, PyArrayDyn, PyArrayLikeDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{LimiterError, LimiterOutput, PerfectLimiter};

type PyProcessOutput<'py> = PyResult<(Bound<'py, PyArrayDyn<f32>>, Bound<'py, PyArray1<f32>>)>;

#[pyclass(name = "PerfectLimiter", module = "perfect_limiter")]
struct PyPerfectLimiter {
    inner: PerfectLimiter,
}

#[pymethods]
impl PyPerfectLimiter {
    #[new]
    #[pyo3(signature = (
        sample_rate,
        threshold = 1.0,
        attack_ms = 5.0,
        hold_ms = 15.0,
        release_ms = 40.0
    ))]
    fn new(
        sample_rate: u32,
        threshold: f64,
        attack_ms: f64,
        hold_ms: f64,
        release_ms: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: PerfectLimiter::new(sample_rate, threshold, attack_ms, hold_ms, release_ms)
                .map_err(value_error)?,
        })
    }

    /// Process mono or multichannel NumPy audio and return `(audio, gains)`.
    #[pyo3(signature = (audio, channel_axis = -1))]
    fn process<'py>(
        &mut self,
        py: Python<'py>,
        audio: PyArrayLikeDyn<'py, f32, AllowTypeChange>,
        channel_axis: isize,
    ) -> PyProcessOutput<'py> {
        process_numpy(py, &mut self.inner, audio.as_array(), channel_axis)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    #[getter]
    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate()
    }

    #[getter]
    fn threshold(&self) -> f64 {
        self.inner.threshold()
    }

    #[getter]
    fn latency_samples(&self) -> usize {
        self.inner.latency_samples()
    }

    #[getter]
    fn hold_samples(&self) -> usize {
        self.inner.hold_samples()
    }

    #[getter]
    fn release_samples(&self) -> f64 {
        self.inner.release_samples()
    }

    fn __repr__(&self) -> String {
        format!(
            "PerfectLimiter(sample_rate={}, threshold={}, latency_samples={})",
            self.inner.sample_rate(),
            self.inner.threshold(),
            self.inner.latency_samples()
        )
    }
}

/// Limit a NumPy array in one call and return `(audio, gains)`.
#[pyfunction]
#[pyo3(signature = (
    audio,
    sample_rate,
    threshold = 1.0,
    attack_ms = 5.0,
    hold_ms = 15.0,
    release_ms = 40.0,
    channel_axis = -1
))]
#[allow(clippy::too_many_arguments)]
fn limit<'py>(
    py: Python<'py>,
    audio: PyArrayLikeDyn<'py, f32, AllowTypeChange>,
    sample_rate: u32,
    threshold: f64,
    attack_ms: f64,
    hold_ms: f64,
    release_ms: f64,
    channel_axis: isize,
) -> PyProcessOutput<'py> {
    let mut limiter = PerfectLimiter::new(sample_rate, threshold, attack_ms, hold_ms, release_ms)
        .map_err(value_error)?;
    process_numpy(py, &mut limiter, audio.as_array(), channel_axis)
}

fn process_numpy<'py>(
    py: Python<'py>,
    limiter: &mut PerfectLimiter,
    audio: ArrayViewD<'_, f32>,
    channel_axis: isize,
) -> PyProcessOutput<'py> {
    let shape = audio.shape().to_vec();
    let (interleaved, channels, normalized_axis) = match audio.ndim() {
        1 => {
            normalize_channel_axis(channel_axis, 1)?;
            (audio.iter().copied().collect(), 1, 0)
        }
        2 => {
            let axis = normalize_channel_axis(channel_axis, 2)?;
            let frames = shape[1 - axis];
            let channels = shape[axis];
            if channels == 0 {
                return Err(PyValueError::new_err(
                    "the channel dimension must not be empty",
                ));
            }

            let mut interleaved = Vec::with_capacity(audio.len());
            for frame in 0..frames {
                for channel in 0..channels {
                    let index = if axis == 0 {
                        [channel, frame]
                    } else {
                        [frame, channel]
                    };
                    interleaved.push(audio[index]);
                }
            }
            (interleaved, channels, axis)
        }
        dimensions => {
            return Err(PyValueError::new_err(format!(
                "audio must be a 1D or 2D array, got {dimensions} dimensions"
            )));
        }
    };

    let LimiterOutput { samples, gains } = limiter
        .process_interleaved(&interleaved, channels)
        .map_err(value_error)?;

    let shaped_samples = if shape.len() == 2 && normalized_axis == 0 {
        let frames = shape[1];
        let mut channel_major = Vec::with_capacity(samples.len());
        for channel in 0..channels {
            for frame in 0..frames {
                channel_major.push(samples[frame * channels + channel]);
            }
        }
        channel_major
    } else {
        samples
    };

    let output = ArrayD::from_shape_vec(IxDyn(&shape), shaped_samples)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((output.into_pyarray(py), gains.into_pyarray(py)))
}

fn normalize_channel_axis(channel_axis: isize, dimensions: usize) -> PyResult<usize> {
    let normalized = if channel_axis < 0 {
        channel_axis + dimensions as isize
    } else {
        channel_axis
    };
    if normalized < 0 || normalized >= dimensions as isize {
        return Err(PyValueError::new_err(format!(
            "channel_axis={channel_axis} is invalid for a {dimensions}D array"
        )));
    }
    Ok(normalized as usize)
}

fn value_error(error: LimiterError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pymodule(name = "perfect_limiter")]
fn python_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyPerfectLimiter>()?;
    module.add_function(wrap_pyfunction!(limit, module)?)?;
    Ok(())
}
