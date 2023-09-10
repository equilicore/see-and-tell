from jls import jls, Context, DF
import numpy as np
import pandas as pd
from cntell.core.describe.frame import Describe


@jls.init(prepare=True)
def initialize_describe(context: Context):
    if not context.app_cfg.get('initialize_describe', False):
        return
    
    model_name = context.app_cfg.get('model_name', 'microsoft/git-base-textcaps')
    context.common = Describe(
        model_name=model_name
    )
    
    context.common.prepare()
    
    
    
@jls.processor()
def describe_images(frame_paths: DF['filename'], context: Context):
    describe: Describe = context.common
    assert isinstance(describe, Describe), "Module has not initialized properly" 
    
    describe.bound_run(context.run_id)
    describe_output: Describe.Captions = describe.run(
        Describe.Images(
            images=[context.get_share_path(x) for x in frame_paths.filename.tolist()]
        )
    )
    
    return pd.DataFrame({
        'captions': describe_output.captions
    })
    
    
@jls.processor()
def mock_describe_images(frame_paths: DF['filename'], context: Context):
    n = len(frame_paths['filename'])
    
    captions = [
        "a man is standing in front of a building",
        "a woman is standing in front of a mirror",
        "a man and a woman talking to each other",
        "two men are standing in front of a building",
        "a group of people standing in front of a building",
    ]
    
    return pd.DataFrame({
        'captions': np.random.choice(captions, n).tolist()
    })
    