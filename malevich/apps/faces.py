import json
import os
from jls import jls, Context, DF, APP_DIR
import boto3
import pandas as pd 
from cntell.core.face.who import Who


@jls.init(prepare=True)
def initialize_who(context: Context):
    if not context.app_cfg.get('initialize_who', False):
        return
    
    client = boto3.client(
        's3',
        aws_access_key_id=context.app_cfg.get('aws_access_key_id'),
        aws_secret_access_key=context.app_cfg.get('aws_secret_access_key'),
        endpoint_url=context.app_cfg.get('endpoint_url')
    )
    
    print(context.app_cfg)    
    objects = client.list_objects_v2(
        Bucket=context.app_cfg.get('bucket_name'),
        Prefix='embeddings'
    )['Contents']
    
    os.makedirs(
        os.path.join(APP_DIR, 'embeddings'),
        exist_ok=True
    )
    
    for object in objects:
        if object['Size'] == 0:
            continue
        
        embeds_name = object['Key'].split('/')[-1]
        
        with open(os.path.join(APP_DIR, 'embeddings', embeds_name), 'wb') as file:
            client.download_fileobj(
                context.app_cfg.get('bucket_name'), 
                object['Key'], 
                file
            )

    context.common = Who(
        embeddings_folder=os.path.join(APP_DIR, 'embeddings')
    )
    
    context.common.prepare(
        use_dir=APP_DIR,
    )
    
    
    
@jls.processor()
def detect_faces(images: DF['filename'], context: Context):
    who: Who = context.common
    assert isinstance(who, Who), "Module has not initialized properly"
    
    who.bound_run(context.run_id)
    who_output: Who.Faces = who.run(
        Who.WhoInput(
            images=[context.get_share_path(x) for x in images.filename.tolist()],
            serie=context.app_cfg.get('serie')
        )
    )
    
    str_faces = [json.dumps(x) for x in who_output.faces]
    
    return pd.DataFrame({
        'faces': str_faces
    })
    