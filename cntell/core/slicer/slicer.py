import os
import shutil
import hashlib
from cntell.common.component import Component, Input, Output
from cntell.common.cms import CacheHitCallback
from ffmpegio import ffmpeg as run_ffmpeg


class Slicer(Component, CacheHitCallback):
    class VideoPath(Input):
        video_path: str

        def checksum(self) -> bytes:
            with open(self.video_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()

    class FramesAndAudio(Output):
        frames_path: list[str]
        audio_path: str

    name = "Slicer"
    input_model = VideoPath
    output_model = FramesAndAudio

    def __init__(
        self, save_dir: str = None, fps: int = 1, naming: str = "frame_%10d.png"
    ) -> None:
        self._save_dir = save_dir
        self._fps = fps
        self._naming = naming

    def on_hit(self, input: VideoPath, output: FramesAndAudio) -> FramesAndAudio:
        frame_paths = []
        for frame_path in output.frames_path:
            frame_filename = os.path.basename(frame_path)
            new_frame_path = os.path.join(self._save_dir, frame_filename)
            shutil.copyfile(frame_path, new_frame_path)
            frame_paths.append(new_frame_path)
        shutil.copyfile(output.audio_path, os.path.join(self._save_dir, "audio.wav"))

        return self.FramesAndAudio(
            run_id=self._bounded_run_id,
            frames_path=frame_paths,
            audio_path=os.path.join(self._save_dir, "audio.wav"),
        )

    def set_save_dir(self, save_dir: str) -> None:
        self._save_dir = save_dir

    def set_fps(self, fps: int) -> None:
        self._fps = fps

    def set_naming(self, naming: str) -> None:
        self._naming = naming

    def _ffmpeg_path(self, path: str):
        return path.replace("\\", "/")

    def _split_on_frames(
        self, video: str, save_dir: str, fps=1, naming="frame_%10d"
    ) -> None:
        """Splits a video into frames and saves them to a directory.

        Args:
            video (str): The path to the video to split.
            save_dir (str): The path to save the frames to.
        """
        run_ffmpeg(
            f"-i {video} -vf fps={fps} -v quiet {self._ffmpeg_path(os.path.join(save_dir, naming))}.png"
        )
        return [os.path.join(save_dir, frame) for frame in os.listdir(save_dir)]

    def _split_on_audio(self, video: str, save_dir: str) -> None:
        """Splits a video into audio clips and saves them to a directory.

        Args:
            video (str): The path to the video to split.
            save_dir (str): The path to save the audio clips to.
        """
        run_ffmpeg(
            f"-i {video} -v quiet -y {self._ffmpeg_path(os.path.join(save_dir, 'audio.wav'))}"
        )
        return os.path.join(save_dir, "audio.wav")

    def _run(
        self,
        video_path: VideoPath,
    ) -> FramesAndAudio:
        return self.FramesAndAudio(
            frames_path=self._split_on_frames(
                video_path.video_path, self._save_dir, self._fps, self._naming
            ),
            audio_path=self._split_on_audio(video_path.video_path, self._save_dir),
        )

    def run(self, __input: input_model = None, **kwargs) -> output_model:
        return super().run(__input, **kwargs)
