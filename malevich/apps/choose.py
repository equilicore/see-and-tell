import json
from jls import jls, DF, Context
import pandas as pd


@jls.processor()
def choose_captions(
    caption_indices: DF['captions_indices'], 
    segments_: DF['segments'], 
    faces: DF['faces'], 
):
    segments = [
        (row.start, row.end) for _, row in segments_.iterrows()
    ]
    
    detections = [json.loads(x) for x in faces['faces'].tolist()]
    captions = caption_indices['captions'].tolist()
    idxs = caption_indices['indices'].tolist()
    
    frames_to_proceed = []
    for start, end in segments:
        most_described_frame = max(
            [
                (ind, detections[i])
                for i, ind in enumerate(idxs)
                if start <= ind < end
            ],
            key=lambda x: len(x[1]),
        )
        frames_to_proceed.append(most_described_frame[0])
        
    return pd.DataFrame({
        'captions': [captions[idxs.index(i)] for i in frames_to_proceed],
        'timings': frames_to_proceed,
    })