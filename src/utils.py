import math
import os
from typing import Iterable
from ffmpegio import ffmpeg as run_ffmpeg
import ffmpeg
import numpy as np
import soundfile
from pyannote.core import Segment
from scipy.interpolate import interp1d

def ffmpeg_path(path: str):
    return path.replace("\\", "/")


def split_on_frames(video: str, save_dir: str, fps=1) -> None:
    """Splits a video into frames and saves them to a directory.

    Args:
        video (str): The path to the video to split.
        save_dir (str): The path to save the frames to.
    """
    run_ffmpeg(
        f'-i {video} -vf fps={fps} -v quiet {ffmpeg_path(os.path.join(save_dir, "frame_%d.png"))}'
    )
    return os.listdir(save_dir)


def split_on_audio(video: str, save_dir: str) -> None:
    """Splits a video into audio clips and saves them to a directory.

    Args:
        video (str): The path to the video to split.
        save_dir (str): The path to save the audio clips to.
    """
    run_ffmpeg(f"-i {video} -v quiet -y {ffmpeg_path(save_dir)}")


def prune_empty_segments(
    segments: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Removes empty segments from a list of segments.

    Args:
        segments (list[tuple[float, float]]): A list of segments.

    Returns:
        list[tuple[float, float]]: A list of segments with empty segments removed.
    """
    return [segment for segment in segments if segment[0] < segment[1]]

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


def get_frames_with_no_speech(speech_segments: Iterable[Segment], seconds: int):
    """Gets the frames with no speech in them.

    Args:
        speech_segments (Iterable[Segment]): An iterable of speech segments.
        seconds (int): The length of the video in seconds.

    Returns:
        list[tuple[int, int]]: A list of frames with no speech in them.
    """
    second_segments = []
    result = []
    for segment in speech_segments:
        second_segments.append((math.floor(segment.start), math.ceil(segment.end)))

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


def get_length_of_video(video_file: str):
    """Gets the length of a video in seconds.

    Args:
        video_file (str): The path to the video file.

    Returns:
        int: The length of the video in seconds.
    """
    return int(float(ffmpeg.probe(video_file)["format"]["duration"]))


def mix_video_and_audio(
    video_file: str,
    audio_array: list[np.array],
    segments: list[tuple[int, int]],
    output_file: str,
):
    """Mixes a video and audio files together.

    Args:
        video_file (str): The path to the video file.
        audio_files (str): The path to the audio files.
        audio_starts (list[int]): A list of the start times of the audio files.
        output_file (str): The path to save the output file to.
    """
    audio_starts = [segment[0] for segment in segments]
    audio_lengths = [segment[1] - segment[0] for segment in segments]
    length = get_length_of_video(video_file)
    audio = np.zeros(((length + 1) * 16000))
    for i in range(len(audio_array)):
    #     current_length = audio_array[i].shape[0]
    #     if audio_array[i].shape[0] > audio_lengths[i] * 16000:
    #         new_length = int(audio_lengths[i] * 16000)
    #         time_old = np.arange(current_length) / 16000
    #         t_min, t_max = time_old.min(), time_old.max()
    #         time_new = np.linspace(t_min, t_max, new_length)
    #         compressed = interp1d(time_old, audio_array[i])(time_new)
    #         slice = (audio_starts[i] * 16000, (audio_starts[i] + audio_lengths[i]) * 16000)
    #     else: 
        compressed = audio_array[i]
        if compressed.shape[0] > audio_lengths[i] * 16000:
            compressed = compressed[:audio_lengths[i] * 16000]
        slice = (audio_starts[i] * 16000, audio_starts[i] * 16000 + compressed.shape[0])
        audio[slice[0]:slice[1]] += compressed

    soundfile.write(output_file + " [Audio].mp3", audio, 16000)
    run_ffmpeg(
        f'-i "{ffmpeg_path(video_file)}" -i "{ffmpeg_path(output_file + " [Audio].mp3")}" '
        f'-filter_complex "[1:a]volume=6dB[audio];[audio]aresample=44100, '
        f'aformat=sample_fmts=s16:channel_layouts=mono[audio];[0:a][audio]amerge=inputs=2[a]" '
        f' -map 0:v -map "[a]" -c:v copy -c:a libmp3lame -ac 2 -y -v quiet "{ffmpeg_path(output_file)}"'
    )
