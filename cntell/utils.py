"""DEPRECATED"""

import os
import ffmpeg
import numpy as np
from typing import Iterable
from pyannote.core import Segment
from ffmpegio import ffmpeg as run_ffmpeg
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.editor import VideoFileClip, AudioFileClip, ImageClip, concatenate_audioclips, concatenate_videoclips, CompositeAudioClip
from deprecation import deprecated


@deprecated(deprecated_in='1.0.0', current_version='1.0.0', details='Use ~cntell.slicer.slicer.Slicer instead.')
def ffmpeg_path(path: str):
    return path.replace("\\", "/")


@deprecated(deprecated_in='0.1.0a', current_version='0.1.0a', details='Use ~cntell.slicer.slicer.Slicer instead.')
def split_on_frames(video: str, save_dir: str, fps=1) -> None:
    """Splits a video into frames and saves them to a directory.

    Args:
        video (str): The path to the video to split.
        save_dir (str): The path to save the frames to.
    """
    run_ffmpeg(
        f'-i {video} -vf fps={fps} -v quiet {ffmpeg_path(os.path.join(save_dir, "frame_%10d.png"))}'
    )
    return os.listdir(save_dir)


def split_on_audio(video: str, save_dir: str) -> None:
    """Splits a video into audio clips and saves them to a directory.

    Args:
        video (str): The path to the video to split.
        save_dir (str): The path to save the audio clips to.
    """
    run_ffmpeg(f"-i {video} -v quiet -y {ffmpeg_path(save_dir)}")


@deprecated(deprecated_in='1.0.0', current_version='1.0.0', details='Now used internally in pipeline components')
def prune_empty_segments(
    segments: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Removes empty segments from a list of segments.

    Args:
        segments (list[tuple[float, float]]): A list of segments.

    Returns:
        list[tuple[float, float]]: A list of segments with empty segments removed.
    """
    return [segment for segment in segments if segment[0] <= segment[1]]


@deprecated(deprecated_in='1.0.0', current_version='1.0.0', details='Now used internally in pipeline components')
def split_segments(
    segments: list[tuple[int, int]], max_length: int, min_length: int
) -> list[tuple[int, int]]:
    """Splits segments into segments of a maximum length.

    Args:
        segments (list[tuple[int, int]]): A list of segments.
        max_length (int): The maximum length of each segment.
        min_length (int): The minimum length of each segment.

    Returns:
        list[tuple[int, int]]: A list of segments of a maximum length.
    """
    result = []
    for segment in segments:
        start = segment[0]
        end = segment[1]
        for i in range(start, end, max_length):
            if end - i < min_length:
                break
            result.append((i, min(i + max_length, end)))
        else:
            result.append(segment)
    return list(set(prune_empty_segments(result)))


@deprecated(deprecated_in='1.0.0', current_version='1.0.0', details='Now used internally in pipeline components')
def get_frames_with_no_speech(speech_segments: Iterable[Segment], seconds: int):
    """Gets the frames with no speech in them.

    Args:
        speech_segments (Iterable[Segment]): An iterable of speech segments.
        seconds (int): The length of the video in seconds.t] + 1 

    Returns:
        list[tuple[int, int]]: A list of frames with no speech in them.
    """
    second_segments = []
    result = []
    for segment in speech_segments:
        second_segments.append((round(segment.start), round(segment.end)))

    if len(second_segments) == 0:
        return [(0, seconds)]
    
    second_segments.sort(key=lambda x: x[0])

    start = 0
    for i in range(len(second_segments)):
        end = second_segments[i][0]
        if start != end:
            result.append((start, end))
        start = second_segments[i][1]
    return prune_empty_segments(result)


@deprecated(deprecated_in='1.0.0', current_version='1.0.0', details='Now used internally in pipeline components')
def get_length_of_video(video_file: str):
    """Gets the length of a video in seconds.

    Args:
        video_file (str): The path to the video file.

    Returns:
        int: The length of the video in seconds.
    """
    return int(float(ffmpeg.probe(video_file)["format"]["duration"]))


def __mix_video_and_audio_ifempty(video_file, output_file):
    run_ffmpeg(f"-i {video_file} -c:v copy -c:a copy -y {output_file}")


def mix_video_and_audio(
    video_file: str,
    audio_captions: list[np.array],
    pause_at: list[int],
    output_file: str,
    audio_captions_sample_rate: int = 16000,
):
    if len(audio_captions) == 0:
        return __mix_video_and_audio_ifempty(video_file, output_file)
    
     # Load the input video
    input_video_clip = VideoFileClip(video_file)
    input_audio_clip = AudioFileClip(video_file)
    audio_captions_clips = [AudioArrayClip(np.expand_dims(audio, axis=1),  audio_captions_sample_rate) for audio in audio_captions]

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

        before_silence = AudioArrayClip(np.zeros((int(before_duration * audio_captions_sample_rate), 1)), audio_captions_sample_rate)        
        segment_silence = AudioArrayClip(np.zeros((int(segment_duration * input_audio_clip.fps), 2)), input_audio_clip.fps)
        segment_repeated_frame = input_video_clip.get_frame(pause_at[current_segment] + 1 / input_video_clip.fps)
        segment_repeated_clip = ImageClip(segment_repeated_frame, duration=segment_duration)


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
    silence_til_end = AudioArrayClip(np.zeros((int(video_subclip.duration * audio_captions_sample_rate), 1)), audio_captions_sample_rate)
    output_video_clips.append(video_subclip)
    output_audio_clips.append(audio_subclip)
    output_audio_caption_clips.append(silence_til_end)

    # Concatenate the clips
    output_video_clip = concatenate_videoclips(output_video_clips)
    output_audio_clip = concatenate_audioclips(output_audio_clips)
    output_audio_caption_clip = concatenate_audioclips(output_audio_caption_clips)

    # Composite audio clip 
    output_audio_clip = CompositeAudioClip([output_audio_clip, output_audio_caption_clip])
    output_audio_clip = output_audio_clip.set_duration(output_video_clip.duration)

    output_video_clip = output_video_clip.set_audio(output_audio_clip)
    # Write the output file
    output_video_clip.write_videofile(output_file, fps=input_video_clip.fps)
