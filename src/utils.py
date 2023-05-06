import math
import os
from typing import Iterable
from ffmpegio import ffmpeg as run_ffmpeg
import ffmpeg
import numpy as np
import soundfile
from pyannote.core import Segment


def ffmpeg_path(path: str):
    return path.replace("\\", "/")

def split_on_frames(video: str, save_dir: str, fps=1) -> None:
    """Splits a video into frames and saves them to a directory.

    Args:
        video (str): The path to the video to split.
        save_dir (str): The path to save the frames to.
    """
    run_ffmpeg(f'-i {video} -vf fps={fps} -v quiet {ffmpeg_path(os.path.join(save_dir, "frame_%d.png"))}')
    return os.listdir(save_dir)


def split_on_audio(video: str, save_dir: str) -> None:
    """Splits a video into audio clips and saves them to a directory.

    Args:
        video (str): The path to the video to split.
        save_dir (str): The path to save the audio clips to.
    """
    print(ffmpeg_path(save_dir))
    run_ffmpeg(f'-i {video} -v quiet -y {ffmpeg_path(save_dir)}')

def prune_empty_segments(
    segments: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Removes empty segments from a list of segments.

    Args:
        segments (list[tuple[float, float]]): A list of segments.

    Returns:
        list[tuple[float, float]]: A list of segments with empty segments removed.
    """
    return [segment for segment in segments if segment[0] != segment[1]]


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
    return list(set(result))


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
        second_segments.append((math.ceil(segment.start), math.floor(segment.end)))

    if len(second_segments) == 0:
        return [(0, seconds)]

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
    audio_starts: list[int],
    output_file: str,
):
    """Mixes a video and audio files together.

    Args:
        video_file (str): The path to the video file.
        audio_files (str): The path to the audio files.
        audio_starts (list[int]): A list of the start times of the audio files.
        output_file (str): The path to save the output file to.
    """
    length = get_length_of_video(video_file)
    audio = np.zeros((length * 16000))
    for i in range(len(audio_array)):
        audio[audio_starts[i] : audio_starts[i] + len(audio_array[i])] += audio_array[i]

    soundfile.write(output_file + " [Audio].mp3", audio, 16000)
    run_ffmpeg(
        f'-i "{video_file}" -i "{output_file + " [Audio].mp3"}" '
        f' -c:v copy -c:a aac -b:a 256k -ac 2 -map 0:v:0 -map 1:a:0 '
        f'"{ffmpeg_path(output_file)}"'
    )
