import os
import sys
import cv2
import numpy as np
from params import PARAMS

def get_tensor_size(tensor):
    if tensor is None:
        return 0
    elif 'QuantizedTensor' in str(type(tensor)):
        return tensor.tensor.storage().__sizeof__()

    return tensor.storage().__sizeof__()

def extract_frames(cap, frame_limit, vid_shape = PARAMS['VIDEO_SHAPE'], transpose_frame = False) -> (bool, np.ndarray):
    '''From a cv2 VideoCapture, return a random frame_limit subset of the video'''
    # get 15 frames from random starting point
    video_length = cap.get(7)
    if video_length < frame_limit:
        return False, None

    random_start = int(np.random.random() * (video_length - frame_limit))
    frames = []
    for i in range(random_start, random_start + frame_limit):
        cap.set(1, i)
        ret, frame = cap.read()
        if ret:
            if transpose_frame:
                frame = frame.transpose(1,0,2)

            if not (frame.shape[0] >= vid_shape[1] and frame.shape[1] >= vid_shape[0]):
                return False, None

            frames.append(cv2.resize(frame, dsize=vid_shape))
        else:
            return False, None

    return True, np.array(frames)

def return_frames_as_bytes(frames : np.ndarray, temp_dir = PARAMS['DEV_DIR'], codec='avc1', fps = PARAMS['FPS'],
                           shape = PARAMS['VIDEO_SHAPE'], frame_limit = PARAMS['FRAME_LIMIT']) -> bytes:
    '''From a numpy array of frames (nframes x h x w x 3), return the compressed video (with a specific codec) as bytes'''
    temp_fname = f'{temp_dir}/temp_{codec}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*codec)

    out = cv2.VideoWriter(temp_fname, fourcc, fps, shape)
    for i in range(frame_limit):
        out.write(frames[i, ...])
    out.release()

    with open(temp_fname, 'rb') as f:
        byte_image = f.read()

    os.remove(temp_fname)

    return byte_image

def decode_bytes(byte_video, temp_dir = PARAMS['DEV_DIR']) -> cv2.VideoCapture:
    '''From bytes, return a cv2.VideoCapture'''
    temp_fname = f'{temp_dir}/temp.mp4'
    with open(temp_fname, 'wb') as f:
        f.write(byte_video)

    cap = cv2.VideoCapture(temp_fname)
    os.remove(temp_fname)

    return cap
