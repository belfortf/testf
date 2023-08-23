import threading
from typing import Any
import insightface

import roop.globals
from roop.typing import Frame

FACE_ANALYSER = None
THREAD_LOCK = threading.Lock()


def get_face_analyser() -> Any:
    print('face_analyser.py - get_face_analyser()')
    global FACE_ANALYSER

    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=roop.globals.execution_providers)
            FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
    return FACE_ANALYSER


def get_one_face(frame: Frame) -> Any:
    print('face_analyser.py - get_one_face()')
    face = get_face_analyser().get(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None

#def get_many_faces(frame: Frame) -> Any:
#    try:
#        return get_face_analyser().get(frame)
#    except IndexError:
#        return None

def get_many_faces(frame: Frame) -> Any:
    print('face_analyser.py - get_many_faces()')
    faces = get_face_analyser().get(frame)
    # Print the number of detected faces
    if faces:
        print(f"Detected {len(faces)} faces.")
    if not faces:
        return None
    # Retrieve the face with the largest bounding box
    largest_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    return largest_face
    
    
    
    # try:
    #     return get_face_analyser().get(frame)
    # except IndexError:
    #     return None
