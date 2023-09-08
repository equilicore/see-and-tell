import os

import pandas as pd
from cntell.core.slicer.slicer import Slicer
from jls import jls, Context, DF, APP_DIR
from typing import Any


@jls.processor()
def slice_video(files: DF[Any], context: Context):
    save_dir = os.path.join(APP_DIR, context.app_cfg.get('save_dir', 'frames'))
    slicer = Slicer(
        save_dir=save_dir
    )
    
    out: list[Slicer.FramesAndAudio] = []
    for path in files.get('filename'):
        real_path = context.get_share_path(path)
        out.append(
            slicer.run(Slicer.VideoPath(video_path=real_path))  
        )
        
    return pd.DataFrame({
        'filenames': sum(map(lambda x: x.frames_path, out), []),
        'audio': list(map(lambda x: x.audio_path, out))
    })
        
    
        