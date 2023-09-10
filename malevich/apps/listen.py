from jls import jls, Context, DF
import pandas as pd 
from cntell.core.listen.speech import Listen


@jls.init(prepare=True)
def initialize_listen(context: Context):
    if not context.app_cfg.get('initialize_listen', False):
        return
    
    context.common = Listen(
        max_segment_len=context.app_cfg.get('max_segment_length', 10),
        min_segment_len=context.app_cfg.get('min_segment_length', 2),
    )
    context.common.prepare()
    
     
    
@jls.processor()
def listen_for_silence(audio: DF['audio'], context: Context):
    listen: Listen = context.common 
    assert isinstance(listen, Listen), "Module has not initialized properly"
    
    listen.bound_run(context.run_id)
    segments = listen.run(
        Listen.PathToAudio(
            path=context.get_share_path(str(audio.audio[0]))
        )
    ).segments
        
    return pd.DataFrame({
        'start': [x[0] for x in segments],
        'end': [x[1] for x in segments],
    })
       
       
       
    