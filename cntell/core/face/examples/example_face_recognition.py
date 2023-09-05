import os
from pathlib import Path
from cntell.face.face_recognition import recognize_faces

HOME = Path(os.getcwd()).parent.parent.parent

path_frames = os.path.join(HOME, 'src', 'frames')
frames = [os.path.join(path_frames, f) for f in os.listdir(path_frames)]
dir_embs = os.path.join(HOME, 'embeddings')
os.makedirs(dir_embs, exist_ok=True)
save_emb_file = os.path.join(dir_embs, 'tbbt.json')

BBT_DIR = os.path.join(HOME, 'src', 'dataset', 'BigBangTheory')

# bbt = build_classes_embeddings(directory=BBT_DIR, batch=False)


predictions = recognize_faces(frames,
                              embeddings=save_emb_file,
                              save_faces=os.path.join(HOME, 'src', 'face', 'debug_recognized_faces_test'),
                              debug=False,
                              return_bbox=False)

assert len(predictions) == len(frames)

print("BOUYAH!!!")
