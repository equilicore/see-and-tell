import os

import numpy as np
from cntell.common.component import Component, Input, Output
from ffmpegio import ffmpeg as run_ffmpeg

from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    ImageClip,
    concatenate_audioclips,
    concatenate_videoclips,
    CompositeAudioClip,
)


class Mixer(Component):
    class MixerInput(Input):
        path_to_orig: str
        voiced_captions: list[np.ndarray]
        timings: list[int]
        save_to: str
        audio_caps_srate: int = 16000

        class Config:
            arbitrary_types_allowed = True

    class MixerOutput(Output):
        saved_to: str

    name = "Mixer"
    input_model = MixerInput
    output_model = MixerOutput

    def __mix_video_and_audio_ifempty(self, video_file, output_file):
        run_ffmpeg(f"-i {video_file} -c:v copy -c:a copy -y {output_file}")

    def _mix_video_and_audio(
        self,
        video_file: str,
        audio_captions: list[np.array],
        pause_at: list[int],
        output_file: str,
        audio_captions_sample_rate: int = 16000,
    ):
        if len(audio_captions) == 0 or len(pause_at) == 0:
            return self.__mix_video_and_audio_ifempty(video_file, output_file)

        # Load the input video
        input_video_clip = VideoFileClip(video_file)
        input_audio_clip = AudioFileClip(video_file)
        audio_captions_clips = [
            AudioArrayClip(np.expand_dims(audio, axis=1), audio_captions_sample_rate)
            for audio in audio_captions
        ]

        output_video_clips = []
        output_audio_clips = []
        output_audio_caption_clips = []
        start = 0
        current_segment = 0

        while current_segment < len(pause_at):
            segment_duration = audio_captions_clips[current_segment].duration
            before_duration = pause_at[current_segment] - start
            video_subclip = input_video_clip.subclip(start, pause_at[current_segment])
            audio_subclip = input_audio_clip.subclip(start, pause_at[current_segment])

            before_silence = AudioArrayClip(
                np.zeros((int(before_duration * audio_captions_sample_rate), 1)),
                audio_captions_sample_rate,
            )
            segment_silence = AudioArrayClip(
                np.zeros((int(segment_duration * input_audio_clip.fps), 2)),
                input_audio_clip.fps,
            )
            segment_repeated_frame = input_video_clip.get_frame(
                pause_at[current_segment] + 1 / input_video_clip.fps
            )
            segment_repeated_clip = ImageClip(
                segment_repeated_frame, duration=segment_duration
            )

            output_video_clips.append(video_subclip)
            output_audio_clips.append(audio_subclip)
            output_audio_caption_clips.append(before_silence)

            output_video_clips.append(segment_repeated_clip)
            output_audio_clips.append(segment_silence)
            output_audio_caption_clips.append(audio_captions_clips[current_segment])

            start = pause_at[current_segment]
            current_segment += 1

        # Add the last segment and silence
        video_subclip = input_video_clip.subclip(start)
        audio_subclip = input_audio_clip.subclip(start)
        silence_til_end = AudioArrayClip(
            np.zeros((int(video_subclip.duration * audio_captions_sample_rate), 1)),
            audio_captions_sample_rate,
        )
        output_video_clips.append(video_subclip)
        output_audio_clips.append(audio_subclip)
        output_audio_caption_clips.append(silence_til_end)

        # Concatenate the clips
        output_video_clip = concatenate_videoclips(output_video_clips)
        output_audio_clip = concatenate_audioclips(output_audio_clips)
        output_audio_caption_clip = concatenate_audioclips(
            output_audio_caption_clips
        )

        # Composite audio clip
        output_audio_clip = CompositeAudioClip(
            [output_audio_clip, output_audio_caption_clip]
        )
        output_audio_clip = output_audio_clip.set_duration(
            output_video_clip.duration
        )

        output_video_clip = output_video_clip.set_audio(output_audio_clip)
        # Write the output file
        output_video_clip.write_videofile(output_file, fps=input_video_clip.fps)

    def prepare(
        self, use_dir: os.PathLike = None, use_gpu: bool = False, *args, **kwargs
    ) -> None:
        return super().prepare(use_dir, use_gpu, *args, **kwargs)

    def _run(self, inputs: MixerInput) -> MixerOutput:
        """Describe a video.

        Args:
            video (str): The path to the video to describe.

        Returns:
            dict: The description of the video.
        """
        self._mix_video_and_audio(
            inputs.path_to_orig,
            inputs.voiced_captions,
            inputs.timings,
            inputs.save_to,
            inputs.audio_caps_srate,
        )

        return self.MixerOutput(saved_to=inputs.save_to)
