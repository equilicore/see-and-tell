import os

import pandas as pd
from cntell.core.slicer.slicer import Slicer
from jls import jls, Context, DF, APP_DIR
from typing import Any


@jls.processor()
def slice_video(files: DF[Any], context: Context):
    save_dir = context.app_cfg.get('save_dir', 'sliced')
    # Real saving dir is APP_DIR/save_dir
    # there files will exist, but the share path will be
    # save_dir/run_id
    real_save_dir = os.path.join(APP_DIR, save_dir)
    os.makedirs(real_save_dir, exist_ok=True)
    
    # Will save
    # APP_DIR/save_dir/*.png
    # APP_DIR/save_dir/audio.wav
    slicer = Slicer(
        save_dir=real_save_dir,
        naming='f_%10d'
    )    
    slicer.bound_run(context.run_id)

    
    frame_paths = []
    
    for path in files.get('filename'):
        real_path = context.get_share_path(path)
        _ = slicer.run(Slicer.VideoPath(video_path=real_path))  
        
    frame_paths = [os.path.join(save_dir, x) for x in os.listdir(real_save_dir) if x.endswith('.png')]
    audio_path = os.path.join(save_dir, 'audio.wav')
    
    
    return_frame_paths = []
    
    for pth in frame_paths:
        context.share(pth)
        return_frame_paths.append(pth)
        
    context.share(audio_path)       
    return_audio_path = [audio_path]
     
        
        
    return [ 
        pd.DataFrame({
            'filename': return_frame_paths
        }),
        pd.DataFrame({
            'audio': return_audio_path
        })
    ]
        
    
        