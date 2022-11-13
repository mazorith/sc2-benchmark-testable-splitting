import os
import sys
import cv2
import numpy as np
from params import PARAMS, DESIRED_CLASSES
import torch
from copy import deepcopy

def get_tensor_size(tensor):
    if type(tensor) is bytes:
        return len(tensor)
    if type(tensor) is str:
        return sys.getsizeof(tensor)

    if tensor is None:
        return 0
    elif 'QuantizedTensor' in str(type(tensor)):
        return tensor.tensor.storage().__sizeof__()

    return tensor.storage().__sizeof__()

def encode_frame(frame : np.ndarray):
    '''uses cv2.imencode to encode a frame'''
    if frame.shape[2] != 3:
        frame = frame.transpose((1,2,0))

    if frame.dtype != 'uint8':
        frame *= 256
        # frame.astype('uint8')

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    success, frame_bytes = cv2.imencode('.jpg', frame, encode_param)
    if not success:
        raise ValueError('Encoding Failed')

    return frame_bytes

def decode_frame(encoded_frame):
    '''decodes the encode_frame and returns it as a float array (between 0 and 1) and 3xHxW'''
    return cv2.imdecode(encoded_frame, cv2.IMREAD_COLOR).transpose(2,0,1) / 256

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

def calculate_bb_iou(boxA, boxB):
    '''calculates the iou for x0y0x1y1 format, singular box, numpy'''
    # determine the (x, y)-coordinates of the intersection rectangle

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def map_xyxy_to_xyhw(xyxy_box):
    return np.array((xyxy_box[0], xyxy_box[1], xyxy_box[2] - xyxy_box[0], xyxy_box[3] - xyxy_box[1]))

def map_xyhw_to_xyxy(xyhw_box):
    return np.array((xyhw_box[0], xyhw_box[1], xyhw_box[2] + xyhw_box[0], xyhw_box[3] + xyhw_box[1]))

def map_coco_outputs(outputs : {str : torch.Tensor}) -> ({int : [int]}, float):
    '''Maps the model output (dict with keys boxes, labels, scores) to {label : boxes}'''
    boxes = outputs['boxes'].detach().numpy()
    labels = outputs['labels'].detach().numpy()
    scores = outputs['scores'].detach().numpy()
    # ignore scores for now

    d = {}
    for i in range(labels.shape[0]):
        if labels[i] not in DESIRED_CLASSES:
            continue
        if labels[i] in d:
            d[labels[i]].append(boxes[i])
        else:
            d[labels[i]] = [boxes[i]]

    return d, scores

def calc_square_evals(boxes1 : [], boxes2 : []):
    '''Returns a greedy max score for boxes when the correspondence between boxes is unknown'''

    if len(boxes1) > len(boxes2):
        boxes1, boxes2 = boxes2, boxes1

    scores = [] # ordered scores
    for box1 in boxes1:
        highscore, highind = 0, 0
        for j, box2 in enumerate(boxes2):
            curr_score = calculate_bb_iou(box1, box2)

            if curr_score > highscore:
                highscore, highind = curr_score, j

        if highind < len(boxes2)-1:
            boxes2 = boxes2[:highind] + boxes2[highind+1:]
        else:
            boxes2 = boxes2[:highind]

        scores.append(round(highscore, 5))

    return scores


def eval_detections(gt_detections : {int : [[int]]}, generated_detections : {int : [[int]]}) -> {}:
    '''Detections are in the format of {class : [boxes]}
    Returns in the format of {class : (one of IoU, 2 for missed, 3 for extra)}'''
    # assert len(gt_detections) == len(generated_detections)

    unioned_keys = set(gt_detections.keys()).union(set(generated_detections.keys()))
    scores = {}
    gt, pred = deepcopy(gt_detections), deepcopy(generated_detections)

    for key in unioned_keys:
        key_format = f'class_{key}'
        if key not in gt:
            scores[key_format] = [3 for _ in range(len(pred[key]))]
        elif key not in pred:
            scores[key_format] = [2 for _ in range(len(gt[key]))]
        else:
            gt_boxes = gt[key]
            pred_boxes = pred[key]

            scores[key_format] = calc_square_evals(gt_boxes, pred_boxes)

            if len(gt_boxes) > len(pred_boxes): # more gt boxes means scores[key] has missing entries
                scores[key_format] += [2] * (len(gt_boxes) - len(pred_boxes))
            elif len(gt_boxes) < len(pred_boxes): # more gt boxes means scores[key] has missing entries
                scores[key_format] += [3] * (len(pred_boxes) - len(gt_boxes))

    return scores