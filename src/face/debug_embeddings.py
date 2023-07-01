import os
import torch.cuda
from embeddings import build_classes_embeddings
from pathlib import Path

HOME = Path(os.getcwd()).parent.parent

path_frames = os.path.join(HOME, 'src', 'frames')
frames = [os.path.join(path_frames, f) for f in os.listdir(path_frames)]

dir_embs = os.path.join(HOME, 'src', 'emd')

os.makedirs(dir_embs, exist_ok=True)

save_emb_file = os.path.join(dir_embs, f'{len(os.listdir(dir_embs))}.json')

BBT_DIR = os.path.join(HOME, 'src', 'dataset', 'BigBangTheory')

print(torch.cuda.is_available())

bbt_embeddings = build_classes_embeddings(directory=BBT_DIR,
                                          save_embedding=save_emb_file,
                                          save_faces=os.path.join(HOME, 'src', 'face_2', 'debug_cropped_faces'),
                                          batch=False)
