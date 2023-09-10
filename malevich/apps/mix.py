import json
import os
from jls import jls, Context, DF, APP_DIR
import pandas as pd
from cntell.core.mixer.mix import Mixer


@jls.processor()
def mix_all(
    voiced_captions: DF['audio_arrays'],
    video: DF['filename'],
    timings: DF['timings'],
    context: Context
):
    mixer = Mixer()
    mixer.bound_run(context.run_id)
    
    file = os.path.join(APP_DIR, context.run_id, "output.mp4")
    
    mixer.run(
        Mixer.MixerInput(
            path_to_orig=context.get_share_path(str(video.filename.tolist()[0])),
            voiced_captions=[json.loads(x) for x in voiced_captions.audio_arrays.tolist()],
            timings=timings.timings.tolist(),
            save_to=file,
        )
    )
    
    context.share(
        os.path.join(APP_DIR, context.run_id, "output.mp4"),
    )
    
    return pd.DataFrame({
        'filename': [os.path.join(context.run_id, "output.mp4")],
    })
    
            
            