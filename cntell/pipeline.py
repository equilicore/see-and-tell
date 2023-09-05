import os
import uuid

from cntell.common.cms import CacheManagementSystem
from cntell.core.slicer.slicer import Slicer
from cntell.core.listen.speech import Listen
from cntell.core.describe.frame import Describe
from cntell.core.face.who import Who
from cntell.core.captions.context import Context
from cntell.core.say.caption import Say
from cntell.core.mixer.mix import Mixer
from cntell.common.component import Component, Input, Output
from cntell.cache.no_cache import NoCMS
from cntell.cache.local_fs import LocalFilesystemCMS


class SeeAndTell(Component):
    class SeeAndTellInput(Input):
        video_path: str
        serie: str = None
        save_to: str = None

    class SeeAndTellOutput(Output):
        video_path: str

    name = "See and Tell"
    input_model = SeeAndTellInput
    output_model = SeeAndTellOutput

    def __init__(
        self, embeddings_folder: str = None, cms: CacheManagementSystem = NoCMS()
    ) -> None:
        self.workdir = None
        self.embeddings_folder = embeddings_folder

        self.slicer = Slicer()
        self.listen = Listen(min_segment_len=3)

        self.describe = Describe(model_name="microsoft/git-large-r-textcaps")

        self.who = Who(embeddings_folder=embeddings_folder)

        self.context = Context()
        self.mixer = Mixer()
        self.say = Say()
        self._CMS = cms

    def prepare(
        self, use_dir: os.PathLike, use_gpu: bool = False, *args, **kwargs
    ) -> None:
        super().prepare(use_dir, use_gpu, *args, **kwargs)

        self.workdir = use_dir

        self.slicer.prepare(use_dir, use_gpu, *args, **kwargs)
        self.listen.prepare(use_dir, use_gpu, *args, **kwargs)
        self.describe.prepare(use_dir, use_gpu, *args, **kwargs)
        self.who.prepare(use_dir, use_gpu, *args, **kwargs)
        self.say.prepare(use_dir, use_gpu, *args, **kwargs)
        self.mixer.prepare(use_dir, use_gpu, *args, **kwargs)

        self._CMS.prepare()

    def _run(self, inputs: SeeAndTellInput) -> SeeAndTellOutput:
        """Describe a video.

        Args:
            video (str): The path to the video to describe.
            output (str): The path to save the output to.
            serie (str, optional): The name of the series to use for face recognition. Defaults to None.
        """
        run_id = str(uuid.uuid4())
        run_path = os.path.join(self.workdir, run_id)
        os.makedirs(run_path, exist_ok=True)

        self.slicer.bound_run(run_id)
        self.listen.bound_run(run_id)
        self.describe.bound_run(run_id)
        self.who.bound_run(run_id)
        self.context.bound_run(run_id)

        self.slicer.set_save_dir(run_path)

        slicer_output: Slicer.FramesAndAudio = self._CMS.run(
            component=self.slicer, inputs=Slicer.VideoPath(video_path=inputs.video_path)
        )

        frame_paths, audio_path = slicer_output.frames_path, slicer_output.audio_path

        listen_output: Listen.SpeechSegments = self._CMS.run(
            component=self.listen, inputs=Listen.PathToAudio(path=audio_path)
        )

        segments = listen_output.segments

        describe_output: Describe.Captions = self._CMS.run(
            self.describe, Describe.Images(images=frame_paths)
        )
    
        if self.who.has_embeddings_for(inputs.serie):
            faces_output: Who.Faces = self._CMS.run(
                component=self.who,
                inputs=Who.WhoInput(images=frame_paths, serie=inputs.serie),
            )

            if len(faces_output.faces) > 0:
                print('faces found')
                captions_with_context: Context.ContextOutput = self._CMS.run(
                    component=self.context,
                    inputs=Context.ContextInput(
                        captions=describe_output.captions, context=faces_output.faces
                    ),
                )
            else:
                captions_with_context = Context.ContextOutput(
                    captions=describe_output.captions,
                    indices=list(range(len(describe_output.captions))),
                    run_id=run_id,
                )

            idxs = captions_with_context.indices
            detections = faces_output.faces
        
            frames_to_proceed = []
            for start, end in segments:
                most_described_frame = max(
                    [
                        (ind, detections[i])
                        for i, ind in enumerate(idxs)
                        if start <= ind < end
                    ],
                    key=lambda x: len(x[1]),
                )
                frames_to_proceed.append(most_described_frame[0])
                
            captions_to_voice = [
                captions_with_context.captions[
                    captions_with_context.indices.index(i)
                ] for i in frames_to_proceed
            ] 
              
        else:
            frames_to_proceed = [int(s[0]) for s in segments]
            captions_to_voice = [describe_output[i] for i in frames_to_proceed]
            
        voiced_captions: Say.AudioArrays = self._CMS.run(
            component=self.say,
            inputs=Say.Captions(texts=captions_to_voice)
        )
        
        assert len(voiced_captions.audio) == len(frames_to_proceed), f"{len(voiced_captions.audio)} != {len(frames_to_proceed)}"
        
        if len(frames_to_proceed) and frames_to_proceed[0] == 0:
            frames_to_proceed = frames_to_proceed[1:]
            voiced_captions.audio = voiced_captions.audio[1:]

        mixed_video: Mixer.MixerOutput = self.mixer.run(
            Mixer.MixerInput(
                path_to_orig=inputs.video_path,
                voiced_captions=voiced_captions.audio,
                timings=frames_to_proceed,
                save_to=inputs.save_to,
            )
        )

        self._CMS.suspend()

        return self.SeeAndTellOutput(video_path=mixed_video.saved_to)


def run_pipeline(
    video: str,
    output: str,
    temporary_folder: str,
    embeddings_folder: str = "./embeddings",
    serie: str = None,
):
    """Run the pipeline on a video."""
    cms = LocalFilesystemCMS.load(temporary_folder)
    # except: cms = LocalFilesystemCMS(temporary_folder)
    see_and_tell = SeeAndTell(embeddings_folder, cms=cms)
    see_and_tell.prepare(use_dir=temporary_folder, use_gpu=True, batch_size=4)
    see_and_tell.run(video_path=video, save_to=output, serie=serie)
