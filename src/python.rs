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
///     sample_rate: Sample rate in hertz. Must be a positive 32-bit integer.
///     threshold_dBFS: Output ceiling in dBFS. Must be finite and at most ``0.0``;
///         ``0.0`` is full scale and approximately ``-6.02`` is half scale.
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
        threshold_dBFS = 0.0,
        attack_ms = 5.0,
        hold_ms = 15.0,
        release_ms = 40.0
    ))]
    #[allow(non_snake_case)]
    fn new(
        sample_rate: &Bound<'_, PyAny>,
        threshold_dBFS: f64,
        attack_ms: f64,
        hold_ms: f64,
        release_ms: f64,
    ) -> PyResult<Self> {
        let sample_rate = parse_sample_rate(sample_rate)?;
        Ok(Self {
            inner: SFLimiter::new(sample_rate, threshold_dBFS, attack_ms, hold_ms, release_ms)
                .map_err(value_error)?,
        })
    }

    /// Process mono or multichannel audio.
    ///
    /// Args:
    ///     audio: A one-dimensional mono array or a two-dimensional
    ///         multichannel array. Values are converted to ``numpy.float32``;
    ///         the input is not modified.
    ///     axis: Frame axis. The default of ``-1`` expects
    ///         ``(channels, frames)``. Use ``0`` for ``(frames, channels)``.
    ///         For mono input, ``0`` and ``-1`` are
    ///         equivalent.
    ///
    /// Returns:
    ///     A tuple ``(audio, frame_gains)``. ``audio`` is a new ``numpy.float32``
    ///     array with the same shape as the input. ``frame_gains`` is a
    ///     one-dimensional ``numpy.float32`` array containing one linked gain
    ///     value per frame.
    ///
    /// Raises:
    ///     ValueError: If the input is not one- or two-dimensional, the
    ///         frame axis is invalid, the channel dimension is empty, or an
    ///         input sample is not finite.
    #[pyo3(signature = (audio, axis = -1))]
    fn process<'py>(
        &mut self,
        py: Python<'py>,
        audio: PyArrayLikeDyn<'py, f32, AllowTypeChange>,
        axis: isize,
    ) -> PyProcessOutput<'py> {
        process_numpy(py, &mut self.inner, audio.as_array(), axis)
    }

    #[getter]
    /// int: Configured sample rate in hertz.
    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate()
    }

    #[getter]
    /// float: Configured output ceiling as a linear amplitude.
    fn threshold(&self) -> f64 {
        self.inner.threshold()
    }

    #[getter]
    /// float: Configured output ceiling in dBFS.
    #[allow(non_snake_case)]
    fn threshold_dBFS(&self) -> f64 {
        self.inner.threshold_dBFS()
    }

    #[getter]
    /// int: Look-ahead latency in samples.
    fn lookahead_samples(&self) -> usize {
        self.inner.lookahead_samples()
    }

    #[getter]
    /// int: Look-ahead latency in samples. (alias for ``lookahead_samples``)
    fn attack_samples(&self) -> usize {
        self.inner.attack_samples()
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
            "SFLimiter(\
            \n    sample_rate={}, threshold_dBFS={}, threshold={},\
            \n    lookahead_samples={}, hold_samples={}, release_samples={}\
            \n)",
            self.inner.sample_rate(),
            self.inner.threshold_dBFS(),
            self.inner.threshold(),
            self.inner.lookahead_samples(),
            self.inner.hold_samples(),
            self.inner.release_samples()
        )
    }
}

/// Limit mono or multichannel audio in one call.
///
/// Args:
///     audio: A one-dimensional mono array or a two-dimensional multichannel
///         array. Values are converted to ``numpy.float32``; the input is not
///         modified.
///     sample_rate: Sample rate in hertz. Must be a positive 32-bit integer.
///     threshold_dBFS: Output ceiling in dBFS. Must be finite and at most ``0.0``;
///         ``0.0`` is full scale and approximately ``-6.02`` is half scale.
///     attack_ms: Look-ahead attack time in milliseconds. It must round to at
///         least one sample at ``sample_rate``.
///     hold_ms: Hold time in milliseconds. May be zero.
///     release_ms: Release time in milliseconds. May be zero.
///     axis: Frame axis. The default of ``-1`` expects
///         ``(channels, frames)``. Use ``0`` for ``(frames, channels)``. For
///         mono input, ``0`` and ``-1`` are
///         equivalent.
///
/// Returns:
///     A tuple ``(audio, frame_gains)``. ``audio`` is a new ``numpy.float32`` array
///     with the same shape as the input. ``frame_gains`` is a one-dimensional
///     ``numpy.float32`` array containing one linked gain value per frame.
///
/// Raises:
///     ValueError: If a configuration value or input is invalid, or an input
///         sample is not finite.
#[pyfunction]
#[pyo3(signature = (
    audio,
    sample_rate,
    threshold_dBFS = 0.0,
    attack_ms = 5.0,
    hold_ms = 15.0,
    release_ms = 40.0,
    axis = -1
))]
#[allow(clippy::too_many_arguments)]
#[allow(non_snake_case)]
fn limit<'py>(
    py: Python<'py>,
    audio: PyArrayLikeDyn<'py, f32, AllowTypeChange>,
    sample_rate: &Bound<'_, PyAny>,
    threshold_dBFS: f64,
    attack_ms: f64,
    hold_ms: f64,
    release_ms: f64,
    axis: isize,
) -> PyProcessOutput<'py> {
    let sample_rate = parse_sample_rate(sample_rate)?;
    let mut limiter = SFLimiter::new(sample_rate, threshold_dBFS, attack_ms, hold_ms, release_ms)
        .map_err(value_error)?;
    process_numpy(py, &mut limiter, audio.as_array(), axis)
}

fn parse_sample_rate(sample_rate: &Bound<'_, PyAny>) -> PyResult<u32> {
    let sample_rate = sample_rate
        .extract::<u32>()
        .map_err(|_| PyValueError::new_err("sample_rate must be a positive 32-bit integer"))?;
    if sample_rate == 0 {
        return Err(PyValueError::new_err(
            "sample_rate must be a positive 32-bit integer",
        ));
    }
    Ok(sample_rate)
}

fn process_numpy<'py>(
    py: Python<'py>,
    limiter: &mut SFLimiter,
    audio: ArrayViewD<'_, f32>,
    axis: isize,
) -> PyProcessOutput<'py> {
    let shape = audio.shape().to_vec();
    let (mut planar_audio, channels, interleaved_shape) = match audio.ndim() {
        1 => {
            normalize_axis(axis, 1)?;
            let planar = audio.as_slice().map_or_else(
                || (0..shape[0]).map(|frame| audio[[frame]]).collect(),
                <[f32]>::to_vec,
            );
            (planar, 1, None)
        }
        2 => {
            let frame_axis = normalize_axis(axis, 2)?;
            let channel_axis = 1 - frame_axis;
            let channels = shape[channel_axis];
            if channels == 0 {
                return Err(PyValueError::new_err(
                    "the channel dimension must not be empty",
                ));
            }

            let frames = shape[frame_axis];
            let planar = if frame_axis == 1 {
                audio.as_slice().map_or_else(
                    || {
                        let mut planar = Vec::with_capacity(audio.len());
                        for channel in 0..channels {
                            for frame in 0..frames {
                                planar.push(audio[[channel, frame]]);
                            }
                        }
                        planar
                    },
                    <[f32]>::to_vec,
                )
            } else {
                let mut planar = Vec::with_capacity(audio.len());
                for channel in 0..channels {
                    for frame in 0..frames {
                        planar.push(audio[[frame, channel]]);
                    }
                }
                planar
            };
            let interleaved_shape = (frame_axis == 0).then_some((frames, channels));
            (planar, channels, interleaved_shape)
        }
        dimensions => {
            return Err(PyValueError::new_err(format!(
                "audio must be a 1D or 2D array, got {dimensions} dimensions"
            )));
        }
    };

    // The NumPy view and output allocation need the GIL, but the limiter core
    // only works with Rust-owned data. Detach while doing the CPU-intensive
    // part so other Python threads can run during processing.
    let frame_gains = py
        .detach(|| limiter.process_planar_inplace(&mut planar_audio, channels))
        .map_err(|error| match (error, interleaved_shape) {
            (LimiterError::NonFiniteSample { index }, Some((frames, channels))) if frames > 0 => {
                let channel = index / frames;
                let frame = index % frames;
                value_error(LimiterError::NonFiniteSample {
                    index: frame * channels + channel,
                })
            }
            (error, _) => value_error(error),
        })?;

    let output_audio = if let Some((frames, channels)) = interleaved_shape {
        planar_to_interleaved(&planar_audio, frames, channels)
    } else {
        planar_audio
    };

    let output_audio = ArrayD::from_shape_vec(IxDyn(&shape), output_audio)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok((output_audio.into_pyarray(py), frame_gains.into_pyarray(py)))
}

fn planar_to_interleaved(samples: &[f32], frames: usize, channels: usize) -> Vec<f32> {
    let mut interleaved = Vec::with_capacity(samples.len());
    for frame in 0..frames {
        for channel in 0..channels {
            interleaved.push(samples[channel * frames + frame]);
        }
    }
    interleaved
}

fn normalize_axis(axis: isize, dimensions: usize) -> PyResult<usize> {
    let normalized = if axis < 0 {
        axis + dimensions as isize
    } else {
        axis
    };
    if normalized < 0 || normalized >= dimensions as isize {
        return Err(PyValueError::new_err(format!(
            "axis={axis} is invalid for a {dimensions}D array"
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
