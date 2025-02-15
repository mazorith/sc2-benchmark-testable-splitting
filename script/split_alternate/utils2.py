import os
import sys
import cv2
import numpy as np
from params import PARAMS, DESIRED_CLASSES
import torch
from copy import deepcopy

def default_detection_postprocessor(d):
    return

def load_model(model_config, device):
    if 'detection_model' not in model_config:
        from sc2bench.models.detection.registry import load_detection_model
        return load_detection_model(model_config, device)
    from sc2bench.models.detection.wrapper import get_wrapped_detection_model
    return get_wrapped_detection_model(model_config, device)

def get_tensor_size(tensor) -> int:
    # "primitive" types
    if tensor is None:
        return 0
    if type(tensor) is bytes:
        return len(tensor)
    if 'QuantizedTensor' in str(type(tensor)):
        return tensor.tensor.storage().__sizeof__()
    if 'Tensor' in str(type(tensor)):
        return tensor.storage().__sizeof__()

    if type(tensor) is dict:
        return sum(get_tensor_size(x) for x in tensor.values())

    if type(tensor) in (list, tuple):
        return sum(get_tensor_size(x) for x in tensor)

    return sys.getsizeof(tensor)

def move_data_list_to_device(data : (), device):
    new_data = []
    for d in data:
        if ('Tensor' in str(type(d))) or ('tensor' in str(type(d))):
            new_data.append(d.to(device))
        else:
            new_data.append(d)

    return new_data

def move_data_dict_to_device(data : {}, device):
    new_data = {}
    for k, v in data.items():
        if ('Tensor' in str(type(v))) or ('tensor' in str(type(v))):
            new_data[k] = v.to(device)
        else:
            new_data[k] = v

    return new_data

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

def extract_frames(cap, frame_limit, vid_shape, transpose_frame = False) -> (bool, np.ndarray):
    '''From a cv2 VideoCapture, return a random frame_limit subset of the video'''
    # get 15 frames from random starting point
    video_length = cap.get(7)
    if video_length < frame_limit:
        return False, None

    if vid_shape is None:
        vid_shape = (720, 1280)

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

def map_coco_outputs(outputs : {str : torch.Tensor}) -> ({int : {int : (int,)}}, [float]):
    '''Maps the model output (dict with keys boxes, labels, scores) to {class_label : {id : boxes}}'''
    boxes = outputs['boxes'].detach().numpy()
    labels = outputs['labels'].detach().numpy()
    scores = outputs['scores'].detach().numpy()
    # ignore scores for now

    d = {}
    curr_id = 0
    for i in range(labels.shape[0]):
        class_id = int(labels[i])
        if class_id not in DESIRED_CLASSES:
            continue

        if class_id in d:
            d[class_id][curr_id] = tuple(boxes[i])
        else:
            d[class_id] = {curr_id : tuple(boxes[i])}
            
        curr_id+=1

    return d, scores

def map_bbox_ids(pred_boxes_allclass : {int : {int : (int,)}}, gt_boxes_allclass : {int : (int,)}) -> ({int: int}, {int: int}):
    '''maps indices of the predicted boxes to the ids of the gt boxes and returns the extraneous items
    input is in the format {class_id : {object_id : bbox_xyxy}}
    return dict in the form of {pred_index : gt_id} and {class_id : len(gt) - len(pred)}'''
    index_mapping = {}

    missing_objects = {} # for every class id, >0 means in gt but not in pred, <0 means in pred but not in gt
    unioned_keys = set(gt_boxes_allclass.keys()).union(set(pred_boxes_allclass.keys()))

    for class_id in unioned_keys:
        if class_id not in pred_boxes_allclass:
            missing_objects[class_id] = len(gt_boxes_allclass[class_id])
            continue

        if class_id not in gt_boxes_allclass:
            missing_objects[class_id] = -len(pred_boxes_allclass[class_id])
            continue

        gt_boxes, pred_boxes = gt_boxes_allclass[class_id], pred_boxes_allclass[class_id]
        missing_objects[class_id] = len(gt_boxes) - len(pred_boxes)

        if len(gt_boxes) < len(pred_boxes):
            for id_gt, gt_box in gt_boxes.items(): # gt boxes on the outside
                highscore, highj = -1, -1
                for id_pred, pred_box in pred_boxes.items():
                    if id_pred in index_mapping:
                        continue

                    curr_score = calculate_bb_iou(gt_box, pred_box)

                    if curr_score >= highscore:
                        highscore, highj, = curr_score, id_pred

                index_mapping[highj] = id_gt
        else:
            used_gt = set()
            for id_pred, pred_box in pred_boxes.items(): # gt boxes on the outside
                highscore, highj = -1, -1
                for id_gt, gt_box in gt_boxes.items():
                    if id_gt in used_gt:
                        continue

                    curr_score = calculate_bb_iou(gt_box, pred_box)

                    if curr_score >= highscore:
                        highscore, highj, = curr_score, id_gt

                index_mapping[id_pred] = highj
                used_gt.add(highj)

    return index_mapping, missing_objects

def remove_classes_from_detections(detections_with_classes : {int : {}}, add_clause = None, return_clause = None):
    '''returns the {object_id : bbox} detections from a detections_with_classes and class map detection'''
    if not add_clause:
        add_clause = lambda x : True
    if not return_clause:
        return_clause = lambda x : False

    detections = {}
    class_map_detection = {}
    for class_id, class_detection in detections_with_classes.items():
        for object_id, bbox in class_detection.items():
            if add_clause(object_id):
                class_map_detection[object_id] : class_id
                detections[object_id] = bbox

            if return_clause(detections):
                return detections


    return detections, class_map_detection

def eval_detections(gt_detections : {int : (int,)}, pred_detections : {int : (int,)},
                    object_id_mapping : {int : int}) -> ({int : float}, {int}):
    '''Detections are in the format of {object_id : [box]} and {pred_oid : gt_oid}
    Returns in the format of {object_id (from either) : score}'''
    # assert len(gt_detections) == len(generated_detections)
    scores = {}

    # see if any pred_detections are outdated
    for object_id in pred_detections.keys():
        assert object_id in object_id_mapping

        gt_object_id = object_id_mapping[object_id]
        key_format = f'pred_{gt_object_id}'

        if gt_object_id not in gt_detections: #in tracker but not in annotations
            scores[key_format] = -1
            continue

        scores[key_format] = calculate_bb_iou(gt_detections[gt_object_id], pred_detections[object_id])

    # check if there are any missing detections
    pred_object_ids = set(object_id_mapping.values())

    missing_detections = set()

    for gt_object_id in gt_detections.keys():
        if gt_object_id not in pred_object_ids:
            missing_detections.add(gt_object_id)

    return scores, missing_detections