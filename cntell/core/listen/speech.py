"""Pipeline component that detects speech in an audio file.""" ""

from typing import Iterable
import ffmpeg
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.core import Segment
from cntell.common.component import Component, Input, Output


class Listen(Component):
    """Pipeline component that detects speech in an audio file."""

    class PathToAudio(Input):
        path: str

    class SpeechSegments(Output):
        segments: list[tuple[float, float]]

    name = "Listen (Speech Detector)"

    input_model = PathToAudio
    output_model = SpeechSegments

    def __init__(self, max_segment_len=10, min_segment_len=1) -> None:
        self.pipeline = None
        self._maxlen = max_segment_len
        self._minlen = min_segment_len

    def _prune_empty_segments(
        self, segments: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """Removes empty segments from a list of segments.

        Args:
            segments (list[tuple[float, float]]): A list of segments.

        Returns:
            list[tuple[float, float]]: A list of segments with empty segments removed.
        """
        return [segment for segment in segments if segment[0] <= segment[1]]

    def _split_segments(
        self, segments: list[tuple[int, int]], max_length: int, min_length: int
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
        return list(set(self._prune_empty_segments(result)))

    def _get_frames_with_no_speech(
        self, speech_segments: Iterable[Segment], seconds: int
    ):
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
        return self._prune_empty_segments(result)

    def _get_length_of_video(self, video_file: str):
        """Gets the length of a video in seconds.

        Args:
            video_file (str): The path to the video file.

        Returns:
            int: The length of the video in seconds.
        """
        return int(float(ffmpeg.probe(video_file)["format"]["duration"]))

    def prepare(self, *args, **kwargs) -> None:
        """Prepare the component for use.

        Args:
            **kwargs: Keyword arguments to pass to the pipeline.
            See ~pyannote.audio.pipelines.VoiceActivityDetection for more information.
        """
        self.pipeline = VoiceActivityDetection(segmentation="anilbs/segmentation")
        self.pipeline.instantiate(
            {
                "onset": 0.5,
                "offset": 0.5,
                "min_duration_off": 0,
                "min_duration_on": 0,
            }
        )

    def _run(self, audio: PathToAudio) -> SpeechSegments:
        """Detect speech in an audio file.

        Args:
            audio (str): The path to the audio file.

        Returns:
            list[tuple[float, float]]: A list of tuples containing the start and end times of each speech segment.
        """
        speech_segments = list(self.pipeline(audio.path).get_timeline())
        silence_segments = self._get_frames_with_no_speech(
            speech_segments, self._get_length_of_video(audio.path)
        )

        splitted_segments = self._split_segments(
            silence_segments, self._maxlen, self._minlen
        )

        return self.SpeechSegments(
            segments=list(sorted(splitted_segments, key=lambda x: x[0]))
        )

    def run(self, __input: input_model = None, **kwargs) -> output_model:
        return super().run(__input, **kwargs)
