from typing import Any, List, Callable
import cv2
import insightface
import threading

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_one_face, get_many_faces
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'


def get_face_swapper() -> Any:
    print('face_swapper.py - get_face_swapper()')
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=roop.globals.execution_providers)
    return FACE_SWAPPER


def pre_check() -> bool:
    print('face_swapper.py - pre_check()')
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://testinghuggingface-ai.s3.amazonaws.com/inswapper_128.onnx'])
    return True


def pre_start() -> bool:
    print('face_swapper.py - pre_start()')
    if not is_image(roop.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False
    elif not get_one_face(cv2.imread(roop.globals.source_path)):
        update_status('No face in source path detected.', NAME)
        return False
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def post_process() -> None:
    print('face_swapper.py - post_process()')
    global FACE_SWAPPER

    FACE_SWAPPER = None


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    print('face_swapper.py - swap_face()')
    return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)


def process_frame(source_face: Face, temp_frame: Frame) -> Frame:
    print('face_swapper.py - process_frame()')
    if roop.globals.many_faces:
        print("Processing many faces")
        try:
            largest_face = get_many_faces(temp_frame)
            # print('Here is many_faces',largest_face)
            # print('Here is type of many_faces',type(largest_face))
            print("Largest face gender:",largest_face.gender)
            print("Largest face age:",largest_face.age)
            if largest_face:
                temp_frame = swap_face(source_face, largest_face, temp_frame)
        except Exception as e:
            print('Exception was:', e)
    else:
        print("Processing one face")
        target_face = get_one_face(temp_frame)
        if target_face:
            temp_frame = swap_face(source_face, target_face, temp_frame)
    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    print('face_swapper.py - process_frames()')
    source_face = get_one_face(cv2.imread(source_path))
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(source_face, temp_frame)
        cv2.imwrite(temp_frame_path, result)
        if update:
            update()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    print('face_swapper.py - process_image()')
    source_face = get_one_face(cv2.imread(source_path))
    target_frame = cv2.imread(target_path)
    result = process_frame(source_face, target_frame)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    print('face_swapper.py - process_video()')
    roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
