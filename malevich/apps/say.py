import json
from jls import jls, Context, DF
import pandas as pd 
from cntell.core.say.caption import Say


@jls.init(prepare=True)
def initiliaze_say(context: Context):
    if not context.app_cfg.get('initialize_say', False):
        return
    
    context.common = Say()
    context.common.prepare()
    
    
    
@jls.processor()
def voice_captions(captions: DF['captions'], context: Context):
    say: Say = context.common 
    assert isinstance(say, Say), "Module has not initialized properly"
    
    say.bound_run(context.run_id)
    audio_arrays = say.run(
        Say.Captions(texts=captions.captions.tolist())
    ).audio
    
        
    return pd.DataFrame({
        'audio_arrays': [json.dumps(x.tolist()) for x in audio_arrays]
    })
       
       
       
    