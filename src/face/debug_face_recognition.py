import os
from pathlib import Path
from face_recognition import recognize_one_image, recognize_faces
from embeddings import build_classes_embeddings

HOME = Path(os.getcwd()).parent.parent

path_frames = os.path.join(HOME, 'src', 'frames')
frames = [os.path.join(path_frames, f) for f in os.listdir(path_frames)]
dir_embs = os.path.join(HOME, 'src', 'emd')
os.makedirs(dir_embs, exist_ok=True)
save_emb_file = os.path.join(dir_embs, f'1.json')
BBT_DIR = os.path.join(HOME, 'src', 'dataset', 'BigBangTheory')

# bbt = build_classes_embeddings(directory=BBT_DIR, batch=False)


predictions = recognize_faces(frames,
                              embeddings=save_emb_file,
                              save_faces=os.path.join(HOME, 'src', 'face_2', 'debug_recognized_faces'),
                              debug=False,
                              return_bbox=False)

# for i, f in enumerate(frames):
#     cls = recognize_one_image(f, embeddings=save_emb_file,
#                               save_faces=os.path.join(HOME, 'src', 'face_2', 'debug_recognized_faces', f'image_{i}'),
#                               debug=True,
#                               return_bbox=False)


