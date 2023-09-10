import json
from jls import jls, Context, DF
import pandas as pd 
from cntell.core.captions.context import Context as CaptionsContext


@jls.init(prepare=True)
def initiliaze_context(context: Context):
    if not context.app_cfg.get('initiliaze_context', False):
        return
    
    context.common = CaptionsContext()
    context.common.prepare()

       
    
@jls.processor()
def enhance_with_context(faces: DF['faces'], captions: DF['captions'], context: Context):
    ctx: CaptionsContext = context.common
    assert isinstance(ctx, CaptionsContext), "Module has not initialized properly"
    
    print(faces)
    print(captions)
    
    ctx.bound_run(context.run_id)
    output: CaptionsContext.ContextOutput = ctx.run(
        CaptionsContext.ContextInput(
            captions=captions['captions'].tolist(),
            context=[json.loads(x) for x in faces['faces'].tolist()]
        )
    )
  
    return pd.DataFrame({
        'captions': output.captions,
        'indices': output.indices,
    })
    
