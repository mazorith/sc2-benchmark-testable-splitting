import cv2.legacy
import sys
import time
from params import PARAMS
import numpy as np
from utils2 import *
from Logger import ConsoleLogger

def init_tracker(tracker_type="MEDIANFLOW"):
    if tracker_type == 'BOOSTING':
        tracker = cv2.legacy.TrackerBoosting_create()
    elif tracker_type == 'MIL':
        tracker = cv2()
    elif tracker_type == 'KCF':
        tracker = cv2.legacy.TrackerKCF_create()
    elif tracker_type == 'TLD':
        tracker = cv2.legacy.TrackerTLD_create()
    elif tracker_type == 'MEDIANFLOW':
        tracker = cv2.legacy.TrackerMedianFlow_create()
    elif tracker_type == 'GOTURN':
        tracker = cv2.legacy.TrackerGOTURN_create()
    elif tracker_type == 'MOSSE':
        tracker = cv2.legacy.TrackerMOSSE_create()
    elif tracker_type == "CSRT":
        tracker = cv2.legacy.TrackerCSRT_create()
    else:
        raise NotImplementedError('No Tracker Found.')

    return tracker


class Tracker():
    '''Tracker uses format xyhw – change bbox inputs and outputs'''
    def __init__(self, tracker_type = PARAMS['TRACKER']):
        self.tracker_type = tracker_type
        self.trackers = {} # id : tracker
        self.class_map = {} # id : class

    def handle_new_detection(self, frame : np.ndarray, detections : {int : [int]}):
        '''Creates trackers for a new set of Detections
        Detections in the format {class, [bbox_xyxy]}
        '''
        self.trackers = {}
        obj_id = 0
        for key, dets in detections.items():
            for det in dets:
                self.class_map[obj_id] = key
                self.trackers[obj_id] = init_tracker(self.tracker_type)
                self.trackers[obj_id].init(frame, map_xyxy_to_xyhw(det))

                obj_id += 1

    def add_bounding_box(self, frame, bbox_xyxy, object_id):
        raise NotImplementedError
        # self.trackers[object_id] = init_tracker(self.tracker_type)
        # self.trackers[object_id].init(frame, map_xyxy_to_xyhw(bbox_xyxy))

    def update(self, frame) -> {int : [int]}:
        '''returns {object_id, new_bounding_box}'''

        updated_boxes = {}
        for target in self.trackers.keys():
            success, bbox_hxhy = self.trackers[target].update(frame)

            if success:
                target_class = self.class_map[target]
                if target_class in updated_boxes:
                    updated_boxes[target_class].append(map_xyhw_to_xyxy(bbox_hxhy))
                else:
                    updated_boxes[target_class] = [map_xyhw_to_xyxy(bbox_hxhy)]

        return updated_boxes

