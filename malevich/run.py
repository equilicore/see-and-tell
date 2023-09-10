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


def prune():
    delete_tasks()
    delete_collections()
    delete_apps()
    delete_cfgs()
    delete_schemes()
    
    
def declare_apps(s3_config: dict):
    
    # Download video from S3
    create_app(
        app_id='download_video',
        input_id='download_from_collection',
        processor_id='download_files',
        app_cfg={
            **s3_config,
        },
        image_ref=args.iU
    )
    
    # Slice video into frames and audio
    create_app(
        app_id='slice_video',
        processor_id='slice_video',
        image_ref=args.iC
    )
    
    # [caption, audio] -> [caption]
    create_app(
        app_id='select_filename',
        processor_id='subset',
        app_cfg={
            'expr': '0',
        },
        image_ref=args.iU,
    )
    
    # [caption, audio] -> [audio]
    create_app(
        app_id='select_audio',
        processor_id='subset',
        app_cfg={
            'expr': '1',
        },
        image_ref=args.iU,
    )
    
    # Create captions for each frame
    create_app(
        app_id='describe',
        processor_id='mock_describe_images',
        app_cfg={
            'initialize_describe': True,
        },
        image_ref=args.iC
    )

    # Detect faces in each frame
    create_app(
        app_id='faces',
        processor_id='detect_faces',
        app_cfg={
            **s3_config,
            'serie': 'tbbt',
            'initialize_who': True,
        },
        image_ref=args.iC
    )
    
    # Enhance captions with detected faces
    create_app(
        app_id='enhance_with_context',
        processor_id='enhance_with_context',
        app_cfg={
            'initiliaze_context': True,
        },
        image_ref=args.iC
    )
    
    # Listen for silence
    create_app(
        app_id='listen',
        processor_id='listen_for_silence',
        app_cfg={
            'initialize_listen': True,
        },
        image_ref=args.iC
    )
    
    # Choose captions to voice based on detected faces
    create_app(
        app_id='choose_captions',
        processor_id='choose_captions',
        image_ref=args.iC
    )
    
    # Voice captions
    create_app(
        app_id='say',
        processor_id='voice_captions',
        app_cfg={
            'initialize_say': True,
        },
        image_ref=args.iC
    )
    
    # Produce a final video
    create_app(
        app_id='mix',
        processor_id='mix_all',
        image_ref=args.iC
    )


def declare_tasks():
    # Stage 1: Download and slice
    
    create_task(
        task_id='download_video',
        app_id='download_video',
    )
      
    create_task(
        task_id='slice_video',
        app_id='slice_video',
        tasks_depends=['download_video']
    )
    
    
    # Stage 2.1.1: Describe Frames
    
    create_task(
        task_id='select_filename',
        app_id='select_filename',
        tasks_depends=['slice_video'],
    )
    
    create_task(
        task_id='describe',
        app_id='describe',
        tasks_depends=['select_filename']
    )
    
    # Stage 2.1.2: Detect Faces
    
    create_task(
        task_id='faces',
        app_id='faces',
        tasks_depends=['select_filename']
    )
    
    
    # Stage 2.1.3: Enhance Captions with Faces
    create_task(
        task_id='enhance_with_context',
        app_id='enhance_with_context',
        tasks_depends=['faces', 'describe']
    )
    

    # Stage 2.2: Listen for Silence
    
    create_task(
        task_id='select_audio',
        app_id='select_audio',
        tasks_depends=['slice_video'],
    )
    
    create_task(
        task_id='listen',
        app_id='listen',
        tasks_depends=['select_audio']
    )
  
  
    # Stage 3: Choose Captions to Voice
    create_task(
        task_id='choose_captions',
        app_id='choose_captions',
        tasks_depends=['enhance_with_context', 'listen', 'faces']
    )
  
  
    # Stage 4: Voice Captions
    create_task(
        task_id='say',
        app_id='say',
        tasks_depends=['choose_captions']
    )
    
    
    # Stage 5: Mix all together
    create_task(
        task_id='mix',
        app_id='mix',
        tasks_depends=['say', 'download_video', 'choose_captions']
    )
        
      


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
    
    prune()
    
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
    
    declare_apps(s3_config)
    declare_tasks()

    schema_ids = {}
    for schema in os.listdir('schema'):
        with open(os.path.join('schema', schema), 'r') as f:
            name = schema.split('.')[0]
            schema_ids[name] = create_scheme(json.load(f), name)
    
                   
    create_schemes_mapping(
        scheme_from_id=schema_ids['captions_timings'],
        scheme_to_id=schema_ids['captions'],
        mapping={
            'captions': 'captions',
        }
    )
    
    create_schemes_mapping(
        scheme_from_id=schema_ids['captions_timings'],
        scheme_to_id=schema_ids['timings'],
        mapping={
            'timings': 'timings',
        }
    )
    
    
    cfg = Cfg(
        collections={
            'download_filename_s3': source_collection,
        },
        init_apps_update={
            'connect_to_s3': True
        }
    )
     
    create_cfg('see-and-tell-config', cfg)        
   
    task_full(
        task_id='choose_captions',
        cfg_id='see-and-tell-config',
        debug_mode=True,
        profile_mode='df_show'
    )

    
if __name__ == '__main__':
    run(args.path)