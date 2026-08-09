use numpy::ndarray::{ArrayD, ArrayViewD, IxDyn};
use numpy::{AllowTypeChange, IntoPyArray, PyArray1, PyArrayDyn, PyArrayLikeDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{LimiterError, SFLimiter};

type PyProcessOutput<'py> = PyResult<(Bound<'py, PyArrayDyn<f32>>, Bound<'py, PyArray1<f32>>)>;

/// A configurable look-ahead brick-wall limiter.
///
/// The limiter applies one linked gain value to all channels in each frame,
/// preserving their relative balance. Processing is offline and every call
/// starts with a neutral gain envelope.
///
/// Args:
///     sample_rate: Sample rate in hertz. Must be greater than zero.
///     threshold: Maximum absolute output sample, in ``(0, 1]``.
///     attack_ms: Look-ahead attack time in milliseconds. It must round to at
///         least one sample at ``sample_rate``.
///     hold_ms: Hold time in milliseconds. May be zero.
///     release_ms: Release time in milliseconds. May be zero.
///
/// Raises:
///     ValueError: If any configuration value is invalid.
#[pyclass(name = "SFLimiter", module = "sf_limiter")]
struct PySFLimiter {
    inner: SFLimiter,
}

#[pymethods]
impl PySFLimiter {
    #[new]
    /// Create a limiter with the requested ceiling and envelope timing.
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
            inner: SFLimiter::new(sample_rate, threshold, attack_ms, hold_ms, release_ms)
                .map_err(value_error)?,
        })
    }

    /// Process mono or multichannel audio.
    ///
    /// Args:
    ///     audio: A one-dimensional mono array or a two-dimensional
    ///         multichannel array. Values are converted to ``numpy.float32``;
    ///         the input is not modified.
    ///     channel_axis: Channel axis for two-dimensional input. The default
    ///         of ``-1`` expects ``(frames, channels)``. Use ``0`` for
    ///         ``(channels, frames)``. For mono input, ``0`` and ``-1`` are
    ///         equivalent.
    ///
    /// Returns:
    ///     A tuple ``(audio, gains)``. ``audio`` is a new ``numpy.float32``
    ///     array with the same shape as the input. ``gains`` is a
    ///     one-dimensional ``numpy.float32`` array containing one linked gain
    ///     value per frame.
    ///
    /// Raises:
    ///     ValueError: If the input is not one- or two-dimensional, the
    ///         channel axis is invalid, the channel dimension is empty, or an
    ///         input sample is not finite.
    #[pyo3(signature = (audio, channel_axis = -1))]
    fn process<'py>(
        &mut self,
        py: Python<'py>,
        audio: PyArrayLikeDyn<'py, f32, AllowTypeChange>,
        channel_axis: isize,
    ) -> PyProcessOutput<'py> {
        process_numpy(py, &mut self.inner, audio.as_array(), channel_axis)
    }

    /// Restore the internal gain envelope to its neutral state.
    fn reset(&mut self) {
        self.inner.reset();
    }

    #[getter]
    /// int: Configured sample rate in hertz.
    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate()
    }

    #[getter]
    /// float: Configured maximum absolute output sample.
    fn threshold(&self) -> f64 {
        self.inner.threshold()
    }

    #[getter]
    /// int: Look-ahead latency in samples.
    fn latency_samples(&self) -> usize {
        self.inner.latency_samples()
    }

    #[getter]
    /// int: Configured hold duration in samples.
    fn hold_samples(&self) -> usize {
        self.inner.hold_samples()
    }

    #[getter]
    /// float: Configured release duration in samples.
    fn release_samples(&self) -> f64 {
        self.inner.release_samples()
    }

    fn __repr__(&self) -> String {
        format!(
            "SFLimiter(sample_rate={}, threshold={}, latency_samples={})",
            self.inner.sample_rate(),
            self.inner.threshold(),
            self.inner.latency_samples()
        )
    }
}

/// Limit mono or multichannel audio in one call.
///
/// Args:
///     audio: A one-dimensional mono array or a two-dimensional multichannel
///         array. Values are converted to ``numpy.float32``; the input is not
///         modified.
///     sample_rate: Sample rate in hertz. Must be greater than zero.
///     threshold: Maximum absolute output sample, in ``(0, 1]``.
///     attack_ms: Look-ahead attack time in milliseconds. It must round to at
///         least one sample at ``sample_rate``.
///     hold_ms: Hold time in milliseconds. May be zero.
///     release_ms: Release time in milliseconds. May be zero.
///     channel_axis: Channel axis for two-dimensional input. The default of
///         ``-1`` expects ``(frames, channels)``. Use ``0`` for
///         ``(channels, frames)``. For mono input, ``0`` and ``-1`` are
///         equivalent.
///
/// Returns:
///     A tuple ``(audio, gains)``. ``audio`` is a new ``numpy.float32`` array
///     with the same shape as the input. ``gains`` is a one-dimensional
///     ``numpy.float32`` array containing one linked gain value per frame.
///
/// Raises:
///     ValueError: If a configuration value or input is invalid, or an input
///         sample is not finite.
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
    let mut limiter = SFLimiter::new(sample_rate, threshold, attack_ms, hold_ms, release_ms)
        .map_err(value_error)?;
    process_numpy(py, &mut limiter, audio.as_array(), channel_axis)
}

fn process_numpy<'py>(
    py: Python<'py>,
    limiter: &mut SFLimiter,
    audio: ArrayViewD<'_, f32>,
    channel_axis: isize,
) -> PyProcessOutput<'py> {
    let shape = audio.shape().to_vec();
    let (mut interleaved, channels, normalized_axis) = match audio.ndim() {
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

            let interleaved = if axis == 1 {
                audio.iter().copied().collect()
            } else {
                let mut interleaved = Vec::with_capacity(audio.len());
                for frame in 0..frames {
                    for channel in 0..channels {
                        interleaved.push(audio[[channel, frame]]);
                    }
                }
                interleaved
            };
            (interleaved, channels, axis)
        }
        dimensions => {
            return Err(PyValueError::new_err(format!(
                "audio must be a 1D or 2D array, got {dimensions} dimensions"
            )));
        }
    };

    let gains = limiter
        .process_interleaved_inplace(&mut interleaved, channels)
        .map_err(value_error)?;

    let shaped_samples = if shape.len() == 2 && normalized_axis == 0 {
        let frames = shape[1];
        let mut channel_major = Vec::with_capacity(interleaved.len());
        for channel in 0..channels {
            for frame in 0..frames {
                channel_major.push(interleaved[frame * channels + channel]);
            }
        }
        channel_major
    } else {
        interleaved
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

/// Python bindings for the sf-limiter look-ahead brick-wall audio limiter.
#[pymodule(name = "sf_limiter")]
fn python_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySFLimiter>()?;
    module.add_function(wrap_pyfunction!(limit, module)?)?;
    Ok(())
}
