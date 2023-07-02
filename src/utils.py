import math
import os
from typing import Iterable
from ffmpegio import ffmpeg as run_ffmpeg
import ffmpeg
import numpy as np
import soundfile
import cv2
from pyannote.core import Segment
from scipy.interpolate import interp1d
from moviepy.editor import VideoFileClip, AudioFileClip, ImageClip, concatenate_audioclips, concatenate_videoclips, CompositeAudioClip
from moviepy.audio.AudioClip import AudioArrayClip

def ffmpeg_path(path: str):
    return path.replace("\\", "/")


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


def resize_array(arr, size):
    curr_size = arr.shape[0]
    if curr_size > size:
        # Cut the end of the array
        arr = arr[:size]
    elif curr_size < size:
        # Pad the array with zeros
        pad_width = [(0, size - curr_size)] + [(0, 0)] * (arr.ndim - 1)
        arr = np.pad(arr, pad_width, mode='constant', constant_values=0)
    return arr


def mix_video_and_audio(
    video_file: str,
    audio_captions: list[np.array],
    # segments: list[tuple[int, int]],
    pause_at: list[int],
    output_file: str,
    audio_captions_sample_rate: int = 16000,
):
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
        segment_repeated_frame = input_video_clip.get_frame(pause_at[current_segment])
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




    # video_fps = int(input_video_clip.fps)
    # # Get the duration of the input video
    # segment_lengths = [audio_captions[i].shape[0] / (audio_captions_sample_rate) for i in range(len(audio_captions))]
    # segment_lengths = [math.ceil(x / video_fps) * video_fps for x in segment_lengths]
    # # Set the fps of the output video


    # input_audio_clip.get_frame(0)
    # # Initialize an empty list to store the frames
    # output_video_frames = []
    # output_audio_frames = []
    # output_audio_captions_frames = []

    # current_segment = 0
    # inserting_caption = False
    # # while True:
    # #     video_time = len(output_video_frames) / video_fps
    # #     if video_time >= segments[current_segment][0]:
    # #         current_segment += 1
    # #         inserting_caption = True

    # # Extract frames from the input video
    # segment_times = []
    # for time, frame in input_video_clip.iter_frames(with_times=True):
    #     # Calculate the time of the current frame
    #     if current_segment < len(segments) and time >= segments[current_segment][0]:
    #         output_video_frames.extend([frame] * segment_lengths[current_segment] * video_fps)
    #         current_segment += 1
    #         segment_times.append(time)
    #     else:
    #         output_video_frames.append(frame)

    # # input_video_clip.close()
    # current_segment = 0
    # for time, frame in input_audio_clip.iter_frames(with_times=True):
    #     # Calculate the time of the current frame
    #     if current_segment < len(segments) and time >= segment_times[current_segment]:
    #         output_audio_frames.append(np.zeros((
    #             math.ceil(segment_lengths[current_segment] * input_audio_clip.fps), 2)))
    #         current_segment += 1
    #     else:
    #         output_audio_frames.append(
    #             np.expand_dims(frame, axis=0)
    #         )
    
    # current_segment = 0
    # for time, frame in input_audio_clip.set_fps(audio_captions_sample_rate).iter_frames(with_times=True):
    # # Calculate the time of the current frame
    #     if current_segment < len(segments) and time >= segment_times[current_segment]:
    #         to_insert = resize_array(
    #             audio_captions[current_segment],
    #             math.ceil(segment_lengths[current_segment] * audio_captions_sample_rate)
    #         )
    #         output_audio_captions_frames.append(to_insert)
    #         current_segment += 1
    #     else:
    #         output_audio_captions_frames.append(np.array([0]))

    # # input_audio_clip.close()
    # output_audio_frames = np.concatenate(output_audio_frames, axis=0)
    # output_audio_captions_frames = np.concatenate(output_audio_captions_frames, axis=0)
    # output_audio_captions_frames = np.expand_dims(output_audio_captions_frames, axis=1)
    # 
    # 
    # 
    # output_audio_clip = AudioArrayClip(
    #     output_audio_frames, 
    #     fps=input_audio_clip.fps
    # )
    
    # output_audio_captions_clip = AudioArrayClip(
    #     output_audio_captions_frames, 
    #     fps=audio_captions_sample_rate
    # )

    # output_clip = ImageSequenceClip(output_video_frames, fps=video_fps)
    # composite_audio_clip = \
    #     CompositeAudioClip([output_audio_clip, output_audio_captions_clip])\
    #     .set_duration(output_clip.duration)


    # # Create a new video file from the ImageClips
    # output_clip = output_clip.set_audio(composite_audio_clip)
    # # Write the output video file
    # output_clip.write_videofile(output_file)

    # # cap = cv2.VideoCapture(video_file)
    # # # Load audio from video file
    # # # with soundfile

    # # audio, sample_rate = soundfile.read(video_file)

    # # # Get video properties
    # # fps = cap.get(cv2.CAP_PROP_FPS)
    # # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # # audio = audio.view(-1, int(sample_rate / fps))
    
    

    # # # Define length of pauses between segments
    # # pause_length = math.ceil(fps / 2)

    # # # Define audio and video output settings
    # # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # # out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
    # # out_audio = []
    # # speech_audio = []
    

    # # # Define variable to keep track of the current segment
    # # current_segment = 0
    # # current_segment_frame = 0
    # # current_loop_frame = None
    # # # Loop through each frame of the video
    # # for frame_idx in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
    # #     # Read a frame from the video
    # #     ret, frame = cap.read()
    # #     if not ret:
    # #         break
        
    # #     # Check if we need to insert silence or audio
    # #     if current_segment >= len(segments) or frame_idx < segments[current_segment][0] * fps:
    # #         out.write(frame)
    # #         out_audio.append(audio[frame_idx])
    # #         speech_audio.append(np.zeros(int(16000 / fps)))
    # #         current_loop_frame = None
    # #         current_segment_frame = 0
    # #     elif frame_idx >= segments[current_segment][1] * fps:
    # #         current_segment += 1
    # #         out.write(frame)
    # #         out_audio.append(audio[frame_idx])
    # #         speech_audio.append(np.zeros(int(16000 / fps)))
    # #         current_loop_frame = None
    # #         current_segment_frame = 0
    # #     else:
    # #         if not current_loop_frame:
    # #             current_loop_frame = np.copy(frame)
    # #         out.audio.append(audio[frame_idx])
    # #         out.write(current_loop_frame)
    # #         speech_audio.append(
    # #             audio_array[current_segment][(current_segment_frame * 16000):((current_segment_frame + 1) * 16000)][frame_idx]
    # #         )
    # #         current_segment_frame += 1


    # # # Release the video capture and writer objects
    # # cap.release()
    # # out.release()

    # # # Convert the audio to a numpy array
    # # out_audio = np.array(out_audio)
    # # speech_audio = np.array(speech_audio)

    # # # Save the audio to a file
    # # soundfile.write(output_file + '_audio', out_audio, sample_rate)
    # # soundfile.write(output_file + "_speech", speech_audio, 16000)

    # # # Blend with ffmpeg

    # # run_ffmpeg(
    # #     f'-i "{ffmpeg_path(video_file)}" -i "{ffmpeg_path(output_file + "_speech")}"'
    # #     f'-filter_complex "[1:a]volume=6dB[audio];[audio]aresample=44100, '
    # #     f'aformat=sample_fmts=s16:channel_layouts=mono[audio];[0:a][audio]amerge=inputs=2[a]" '
    # #     f' -map 0:v -map "[a]" -c:v copy -c:a libmp3lame -ac 2 -y -v quiet "{ffmpeg_path(output_file)}"'
    # # )

    # # # audio_starts = [segment[0] for segment in segments]
    # # # audio_lengths = [segment[1] - segment[0] for segment in segments]
    # # # length = get_length_of_video(video_file)
    # # # audio = np.zeros(((length + 1) * sample_rate))
    # # # for i in range(len(audio_array)):
    # # # #     current_length = audio_array[i].shape[0]
    # # # #     if audio_array[i].shape[0] > audio_lengths[i] * 16000:
    # # # #         new_length = int(audio_lengths[i] * 16000)
    # # # #         time_old = np.arange(current_length) / 16000
    # # # #         t_min, t_max = time_old.min(), time_old.max()
    # # # #         time_new = np.linspace(t_min, t_max, new_length)
    # # # #         compressed = interp1d(time_old, audio_array[i])(time_new)
    # # # #         slice = (audio_starts[i] * 16000, (audio_starts[i] + audio_lengths[i]) * 16000)
    # # # #     else: 
    # # #     compressed = audio_array[i]
    # # #     if compressed.shape[0] > audio_lengths[i] * sample_rate:
    # # #         compressed = compressed[:audio_lengths[i] * sample_rate]
    # # #     slice = (audio_starts[i] * sample_rate, audio_starts[i] * sample_rate + compressed.shape[0])
    # # #     audio[slice[0]:slice[1]] += compressed

    # # # soundfile.write(output_file + " [Audio].mp3", audio, 16000)
    # # # run_ffmpeg(
    # # #     f'-i "{ffmpeg_path(video_file)}" -i "{ffmpeg_path(output_file + " [Audio].mp3")}" '
    # # #     f'-filter_complex "[1:a]volume=6dB[audio];[audio]aresample=44100, '
    # # #     f'aformat=sample_fmts=s16:channel_layouts=mono[audio];[0:a][audio]amerge=inputs=2[a]" '
    # # #     f' -map 0:v -map "[a]" -c:v copy -c:a libmp3lame -ac 2 -y -v quiet "{ffmpeg_path(output_file)}"'
    # # # )
