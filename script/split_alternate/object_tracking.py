import cv2
import sys
import time


#calculates error b/w actual and predicted via tracking bounding boxes
def bb_intersection_over_union(boxA, boxB):
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


# file to run object tracking return list of all bounding boxes
def parse_visdrone(file_txt, frame):
    lines = None
    with open(file_txt) as f:
        lines = f.readlines()

    parsed_lines = {}
    for l in lines:
        if int(l.split(',')[0]) == frame:
            split = map(int,l.split(',')[2:6])
            parsed_lines[l.split(',')[1]] = tuple(split)
    return parsed_lines

def parse_visdrone_object(file_txt, frame, obj):
    lines = None
    with open(file_txt) as f:
        lines = f.readlines()

    for l in lines:
        if int(l.split(',')[0]) == frame and int(l.split(',')[1]) == obj:
            split = map(int,l.split(',')[2:6])
            return tuple(split)

    return (-1,-1,-1,-1)

# converts (xmin, ymin, w, h) to (xmin, ymin, xmax, ymax)
def convert_visdrone(bb):
    return [bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]]

def init_tracker(tracker_type="MEDIANFLOW"):
    tracker = None
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2() 
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create() 
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create() 
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create() 
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()
    return tracker

class Tracker():

    def __init__(self, frame, bboxes, type="MEDIANFLOW"):
        self.multiTracker = cv2.MultiTracker_create()
        for bbox in bboxes:
            self.multiTracker.add(init_tracker(type), frame, bbox)

    def update(self, frame):
        start_time = time.time()
        success, bboxes = self.multiTracker.update(frame)
        if not success:
            print ('Tracker Update Failed')
            sys.exit()
        print("Total Execution Time = %s seconds" % (time.time() - start_time))
        return bboxes


def run_multitracker():
    video = "/home/ian/dataset/visdrone2019/videos/uav0000013_00000_v.avi"
    annotations = "/home/ian/dataset/visdrone2019/annotations/uav0000013_00000_v.txt"
    
    bboxes = parse_visdrone(annotations,1)
    print("Object Count = " + str(len(bboxes)))

    video = cv2.VideoCapture(video)

    success, frame = video.read()
    if not success:
        print("Video not read")
        return


    multitracker = Tracker(frame, bboxes.values())

    for frame_num in range(2,11):
        print("Frame = " + str(frame_num))
        success, frame = video.read()
        if not success:
            print("Video not read")
            return

        start_time = time.time()
        # print(frame.shape)
        bbox = multitracker.update(frame)


        print("Total Execution Time = %s seconds" % (time.time() - start_time))
        # actual_bbox = parse_visdrone_object(annotations, frame_num, )    

        # accuracy = []
        # for i in range(10):
        #     acc = intersection_over_union(bboxes[i], tracking_results[i])
        #     accuracy.append(acc)
        # print(accuracy)

if __name__ == '__main__':
    print(cv2.__version__)
    video = "/home/ian/dataset/visdrone2019/videos/uav0000013_00000_v.avi"
    annotations = "/home/ian/dataset/visdrone2019/annotations/uav0000013_00000_v.txt"
    run_multitracker()
    # print(bboxes)
