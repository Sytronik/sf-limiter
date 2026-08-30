use crate::LimiterError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AudioLayout {
    Interleaved,
    Planar,
}

impl AudioLayout {
    pub(crate) fn apply_frame_gains(
        self,
        audio: &mut [f32],
        frame_gains: &[f32],
        channels: usize,
        threshold: f32,
    ) {
        match self {
            Self::Interleaved => {
                apply_frame_gains_to_interleaved(audio, frame_gains, channels, threshold)
            }
            Self::Planar => apply_frame_gains_to_planar(audio, frame_gains, threshold),
        }
    }
}

pub(crate) fn validate_layout(sample_count: usize, channels: usize) -> Result<usize, LimiterError> {
    if channels == 0 {
        return Err(LimiterError::InvalidChannelCount);
    }
    if !sample_count.is_multiple_of(channels) {
        return Err(LimiterError::InputNotFrameAligned {
            sample_count,
            channels,
        });
    }
    Ok(sample_count / channels)
}

fn apply_frame_gains_to_interleaved(
    audio: &mut [f32],
    frame_gains: &[f32],
    channels: usize,
    threshold: f32,
) {
    for (frame, gain) in audio.chunks_exact_mut(channels).zip(frame_gains) {
        for sample in frame {
            *sample = (*sample * *gain).clamp(-threshold, threshold);
        }
    }
}

fn apply_frame_gains_to_planar(audio: &mut [f32], frame_gains: &[f32], threshold: f32) {
    if frame_gains.is_empty() {
        return;
    }

    for channel in audio.chunks_exact_mut(frame_gains.len()) {
        for (sample, gain) in channel.iter_mut().zip(frame_gains) {
            *sample = (*sample * *gain).clamp(-threshold, threshold);
        }
    }
}
