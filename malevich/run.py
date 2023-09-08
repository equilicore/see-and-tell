import os
import pandas as pd
import boto3
import uuid
import argparse
from jls_utils import *


parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, help='Path to video file')
parser.add_argument(
    '-iU', 
    type=str, 
    help='`utility` image reference', 
    default="registry.gitlab.com/h1054/julius-internal-app-directory/malevich-app-library/utility:latest"
)
parser.add_argument(
    '-iC', 
    type=str, 
    help='`cntell` image reference', 
    default="cntell_malevich"
)
args = parser.parse_args()

def run(path: str):
    malevich_user = os.getenv('MALEVICH_USER')
    malevich_password = os.getenv('MALEVICH_PASSWORD')
    aws_access_key = os.getenv('AWS_ACCESS_KEY')
    aws_secret_key = os.getenv('AWS_SECRET_KEY')
    s3_endpoint_url = os.getenv('S3_ENDPOINT_URL')
    s3_bucket = os.getenv('AWS_BUCKET')

  
    assert malevich_user is not None and malevich_password is not None, \
        'MALEVICH_USER and MALEVICH_PASSWORD must be set as environment variables'
        
    update_core_credentials(malevich_user, malevich_password)
    set_host_port('http://localhost:8080')
    
    try: create_user()
    except: pass
    
    client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        endpoint_url=s3_endpoint_url
    )
    
    cntell_run_id = str(uuid.uuid4())
    s3_video_path = f'{cntell_run_id}/source.mp4'
    
    with open(path, 'rb') as f:
        client.upload_fileobj(f, s3_bucket, s3_video_path)
    
    try: delete_app('slice_video')
    except: pass
    
    
    try: delete_app('download_video')
    except: pass
    
    source_collection = create_collection_from_df(
        pd.DataFrame(
            {
                'filename': ['source.mp4'],
                's3key': [s3_video_path]
            }
        ), 'source_collection'
    )
    
    
    s3_config = {
        'aws_access_key_id': aws_access_key,
        'aws_secret_access_key': aws_secret_key,
        'endpoint_url': s3_endpoint_url,
        'bucket_name': s3_bucket
    }
    
    try:
        create_app(
            app_id='download_video',
            input_id='download_from_collection',
            processor_id='download_files',
            app_cfg={
                **s3_config,
            },
            image_ref=args.iU
        )
    except:
        pass
    
    try:       
        create_app(
            app_id='slice_video',
            processor_id='slice_video',
            image_ref=args.iC
        )
    except:
        pass
    
    try:    
        create_task(
            task_id='slice',
            app_id='slice_video',
            apps_depends=['download_video']
        )
    except:
        pass
    
    cfg = Cfg(
        collections={
            'download_filename_s3': source_collection,
        },
        init_apps_update={
            'connect_to_s3': True
        }
    ) 
    try:
        create_cfg('see-and-tell-config', cfg)
    except:
        update_cfg('see-and-tell-config', cfg)
        pass
        
   
    for file in os.listdir('schema'):
        with open(os.path.join('schema', file), 'r') as f:
            try:
                create_scheme(
                    json.load(f),
                    os.path.splitext(file)[0]
                )
            except:
                pass
       
   
    task_full(
        task_id='slice',
        cfg_id='see-and-tell-config',
        debug_mode=True,
        profile_mode='df_show'
    )
    

    
if __name__ == '__main__':
    run(args.path)