from socket import *
from struct import pack, unpack
import torch
from torch import nn

import pickle
import time
import sys
import argparse
from params import PARAMS, CURR_DATE, DESIRED_CLASSES
from Logger import ConsoleLogger, DictionaryStatsLogger
from utils2 import *
from data import Dataset
from split_models import model_2
from tracker import Tracker
import traceback

from torchdistill.common import yaml_util
from sc2bench.models.detection.registry import load_detection_model
from sc2bench.models.detection.wrapper import get_wrapped_detection_model

def load_model(model_config, device):
    if 'detection_model' not in model_config:
        return load_detection_model(model_config, device)
    return get_wrapped_detection_model(model_config, device)

def create_input(data):
    return data

def get_gt_detections(class_info : ((int,), (int,)), boxes : [[int,],]) -> {int : {int : (int,)}}:
    '''reformats the annotations into eval-friendly output
    class_info is (class_ids, object_ids)
    returns as {class_ids : {object_ids : box}}'''
    assert len(class_info[0]) == len(boxes), f'class_info: {class_info} boxes: {boxes}'
    gt_boxes = {}
    for i in range(len(class_info[0])):
        if class_info[0][i] not in DESIRED_CLASSES:
            continue

        if class_info[0][i] in gt_boxes:
            gt_boxes[class_info[0][i]][class_info[1][i]] = boxes[i]
        else:
            gt_boxes[class_info[0][i]] = {class_info[1][i]: boxes[i]}

    return gt_boxes

def get_gt_detections_as_pred(class_info : ((int,), (int,)), boxes : [[int,],]) -> {int : (int,)}:
    '''reformats the annotations into tracker-prediction format
    class_info is (class_ids, object_ids)
    returns as {object_id : box}'''
    assert len(class_info[0]) == len(boxes), f'class_info: {class_info} boxes: {boxes}'
    gt_boxes = {}
    for i in range(len(class_info[0])):
        if class_info[0][i] not in DESIRED_CLASSES:
            continue

        gt_boxes[class_info[1][i]] = boxes[i]

    return gt_boxes

class Client:

    def _init_tracker(self):
        if self.tracking:
            self._currently_tracking = set()
            self.tracker = Tracker()

    def _init_detector(self):
        self._refresh_iters = 1
        if not self.detection:
            return

        if self.detection_compression == 'model':
            # use student model
            self.logger.log_debug(f'Setting up compression model {self.detector_model}.')

            if self.detector_model == 'faster_rcnn':
                student_model = get_student_model(PARAMS['FASTER_RCNN_YAML'])
                self.client_model = model_2.ClientModel(student_model)
                if self.server_connect:
                    self.server_model = nn.Identity()
                else:
                    self.server_model = model_2.ServerModel(student_model)

                self.client_model.to(self.compressor_device)
                self.server_model.to(self.detector_device)
            else:
                raise NotImplementedError

        else: # classical compression
            if self.server_connect:
                self.logger.log_debug('Classical; connecting to server for detection.')
                pass
            else:
                self.logger.log_debug(f'Classical compression; setting up model {self.detector_model} for offline detection.')
                # offline; get models
                if self.detector_model == 'faster_rcnn':
                    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
                    self.server_model = fasterrcnn_resnet50_fpn_v2(pretrained=True).eval()
                elif self.detector_model == 'mask_rcnn':
                    from torchvision.models.detection import maskrcnn_resnet50_fpn
                    self.server_model = maskrcnn_resnet50_fpn(pretrained=True).eval()
                elif self.detector_model == 'retinanet':
                    from torchvision.models.detection import retinanet_resnet50_fpn
                    self.server_model = retinanet_resnet50_fpn(pretrained=True).eval()
                else:
                    raise NotImplementedError

                self.server_model.to(self.detector_device)

    def __init__(self, server_connect = PARAMS['USE_NETWORK'], run_type = PARAMS['RUN_TYPE'],
                 stats_log_dir = PARAMS['STATS_LOG_DIR'], dataset = PARAMS['DATASET'], tracking = PARAMS['TRACKING'],
                 detection = PARAMS['DETECTION'], detection_compression = PARAMS['DET_COMPRESSOR'],
                 refresh_type = PARAMS['BOX_REFRESH'], run_eval = PARAMS['EVAL'],
                 detector_model = PARAMS['DETECTOR_MODEL'], detector_device = PARAMS['DETECTION_DEVICE'],
                 compressor_device = PARAMS['COMPRESSOR_DEVICE']):

        self.socket, self.message = None, None
        self.logger, self.dataset, self.stats_logger = ConsoleLogger(), Dataset(dataset=dataset), \
                                                       DictionaryStatsLogger(logfile=f"{stats_log_dir}/client-{dataset}-{CURR_DATE}.log")
        self.server_connect, self.tracking, self.detection, self.run_type = server_connect, tracking, detection, run_type
        self.detection_compression, self.refresh_type = detection_compression, refresh_type
        self.run_eval, self.detector_model = run_eval, detector_model

        self.detector_device, self.compressor_device = detector_device, compressor_device

        self.class_map_detection = {}

        self._init_tracker()
        self._init_detector()

    def _connect(self, server_ip, server_port):
        assert self.server_connect
        self.logger.log_debug('Connecting from client')
        self.socket = socket(AF_INET, SOCK_STREAM)
        self.socket.connect((server_ip, server_port))
        self.logger.log_info('Successfully connected to socket')

    def _client_handshake(self):
        assert self.server_connect
        self.logger.log_debug('Sending handshake from client')
        self.socket.sendall(b'\00')
        ack = self.socket.recv(1)
        if(ack == b'\00'):
            self.logger.log_info('Successfully received server Handshake')
        else:
            self.logger.log_info('Message received not server ack')

    def start(self, server_ip, server_port):
        if self.server_connect:
            self._connect(server_ip, server_port)
            self._client_handshake()
            self.logger.log_info('Successfully started client')

        else:
            self.logger.log_info('Starting in offline mode.')

    def _send_encoder_data(self, data):
        '''if connected to the server, formats and sends the data
        else, simply store the data in self.message'''
        if self.server_connect:
            self.logger.log_debug('Sent encoded data to server.')
            data = pickle.dumps(data)
            length = pack('>Q', len(data))

            self.socket.sendall(length)
            self.socket.sendall(data)

            ack = self.socket.recv(1)
            self.logger.log_debug(f'Received server ack: {ack}.')

        else:
            self.message = data

    def _get_server_data(self):
        '''returns the server data in any format'''
        assert self.server_connect

        collected_message = False
        while not collected_message:
            bs = self.socket.recv(8)
            (length,) = unpack('>Q', bs)
            data = b''
            while len(data) < length:
                to_read = length - len(data)
                data += self.socket.recv(
                    4096 if to_read > 4096 else to_read)

            # send our 0 ack
            self.socket.sendall(b'\00')

            self.logger.log_debug('Received message.')

            return pickle.loads(data)

    def get_model_bb(self):
        '''uses the model to get the bounding box
        will never be called if DETECTOR is False'''

        assert self.detection

        if self.server_connect: # connects to server and gets the data from there
            return self._get_server_data()

        else: # this part should be on the server.py for the online case
            self.logger.log_debug('Using offline model for detection.')

            client_data = self.message['data']
            if self.detection_compression == 'model':
                client_data_device = move_data_to_device(client_data, self.detector_device)
                if self.detector_model == 'faster_rcnn':
                    model_outputs = self.server_model(*client_data_device)[0]
                else:
                    raise NotImplementedError('No specified detector model exists.')
            elif self.detection_compression == 'classical':
                decoded_frame = decode_frame(client_data)
                model_outputs = self.server_model([torch.from_numpy(decoded_frame).float().to(self.detector_device)])[0] # first frame
            else:
                raise NotImplementedError('No other compression method exists.')

            self.logger.log_debug("Generated new BB with model (offline).")
            return model_outputs

    def handle_server_data(self, server_data, server_model = PARAMS['DETECTOR_MODEL']):
        '''Handles the server data, ie. formatting the model outputs into eval-friendly
        and tracker-friendly outputs'''
        if self.detection:
            # server_data is in the format of {'boxes' : [], 'labels' : [], 'scores' : []}
            detections, scores = map_coco_outputs(server_data)
            return detections
        else:
            raise NotImplementedError('No other server task other than detection.')

    def get_new_bounding_box(self, d : (), before = None) -> {int : {int : (int,)}}:
        '''gets a new "accurate" bounding box (from a separate detection pipeline)
        returns in the format of {object_id : (xyxy)}'''

        data, size_orig, class_info, gt, fname, _ = d
        gt_detections = get_gt_detections(class_info, gt)

        # if gt is used, get bounding box will simply return the ground truth
        if not self.detection:
            self.logger.log_debug('Using ground truth boxes.')
            # return ground truth boxes
            _, size_orig, class_info, gt, _, _ = d

            self.stats_logger.push_log({'gt' : True, 'original_size' : size_orig})
            return gt_detections

        # otherwise, use an actual detector
        # offline case in model detection is handled in the individual helper functions
        self.logger.log_debug('Creating message for boxes.')
        # uses model for detection; requires some compression (make metrics)
        if self.detection_compression == 'model':  # use a model (bottleneck) to compress it
            self.logger.log_debug('Performing compression using mdoel.')
            data, size_orig, class_info, gt, fname, _ = d
            data_reshape = (data/256).transpose((2,0,1))[np.newaxis, ...]

            # collect model runtime
            now = time.time()
            tensors_to_measure, other_info = self.client_model(torch.from_numpy(data_reshape).float().to(self.compressor_device))
            compression_time = time.time() - now

            size_compressed = get_tensor_size(tensors_to_measure)

            message = {'timestamp': time.time(), 'data': (*tensors_to_measure, *other_info)}

            self.stats_logger.push_log({'compressor' : 'model'})

        else:  # classical compression – compress data (frame) into jpg
            # TODO: make options – option to compress into jpg or compress video inside dataloader
            self.logger.log_debug('Performing classical compression on image.')
            time_from_before = time.time() - before

            # collect compression runtime
            now = time.time()
            encoded_frame = encode_frame(data)
            compression_time = time.time() - now
            size_compressed = sys.getsizeof(encoded_frame)

            message = {'timestamp': time.time(), 'data': encoded_frame}

            self.stats_logger.push_log({'compressor': 'classical'})

        self.stats_logger.push_log({'encode_time': compression_time, 'message_size': size_compressed,
                                    'original_size': size_orig}, append=False)
        self.logger.log_info(f'Generated message with bytesize {size_compressed} and original {size_orig}')

        self._send_encoder_data(message)

        # response_time in the offline case will be the time it takes for the server model to run
        # in the online case it will be 2x latency + response_time
        now = time.time()
        server_data = self.get_model_bb() # batch size 1
        if server_data is None:
            server_data = gt_detections

        self.stats_logger.push_log({'response_time': time.time() - now}, append=False)
        self.logger.log_info(f'Received detection with response time {time.time() - now}')

        return self.handle_server_data(server_data)

    def check_detection_refresh(self, refresh_limit : int = PARAMS['REFRESH_ITERS']) -> bool:
        '''returns true if a refresh is needed, otherwise do nothing'''
        if self.refresh_type == 'fixed':
            if self._refresh_iters >= refresh_limit:
                self._refresh_iters = 1
                return True

            self._refresh_iters += 1
            return False

        else:
            raise NotImplementedError('Invalid Refresh Type')

    def reinit_tracker(self, frame : np.ndarray, detections : {int : (int,)}):
        '''creates a new tracker with the detections and frames
        detections is in the format of {object_id : [box]}'''
        if self.tracking:
            self.tracker.handle_new_detection(frame, detections)

    def get_tracker_bounding_box(self, frame) -> {int : (int,)}:
        if self.tracking:
            now = time.time()
            detections = self.tracker.update(frame)
            self.stats_logger.push_log({'tracker': True, 'tracker_time': time.time() - now})
            return detections

        return None

    def eval_detections(self, gt_detections, pred_detections):
        '''evaluates detections and pushes the logs'''
        self.logger.log_info('Evaluating detections.')
        pred_scores, missing_preds = eval_detections(gt_detections, pred_detections, self.class_map_detection)
        self.stats_logger.push_log({'missing_preds' : missing_preds, **pred_scores}, append=False)
        return

    def start_loop(self):
        try:
            now = time.time()
            for i, d in enumerate(self.dataset.get_dataset()):
                self.logger.log_info(f'Starting iteration {i}')

                if self.run_type == 'BB':

                    data, size_orig, class_info, gt, fname, start = d
                    self.stats_logger.push_log({'iteration' : i, 'fname' : fname})

                    # get the gt detections in {object : bbox} for eval
                    gt_detections = get_gt_detections_as_pred(class_info, gt)
                    self.logger.log_debug(f'gt_detections : {gt_detections}')

                    if start or self.check_detection_refresh(): # get new bounding box
                        # new detections should be in the format of {class : {object : bbox}} for the class matching

                        gt_detections_with_classes = get_gt_detections(class_info, gt)
                        self.logger.log_debug('Re-generating bounding boxes.')

                        # newthread()
                        detections_with_classes = self.get_new_bounding_box(d, now)

                        if self.run_eval:
                            # maps the ground truth labels with the predicted output (for evaluation purposes)
                            self.class_map_detection, extra_detections = map_bbox_ids(detections_with_classes,
                                                                                      gt_detections_with_classes)
                            self.stats_logger.push_log({f'extra_class_{class_id}': extra_detection
                                                        for class_id, extra_detection in extra_detections.items()},
                                                       append=False)

                            # remove the unnecessary boxes (don't bother tracking them)
                            detections = {}
                            for class_id, class_detection in detections_with_classes.items():
                                for object_id, bbox in class_detection.items():
                                    if object_id in self.class_map_detection:
                                        detections[object_id] = bbox

                            self.logger.log_debug(f'Detections made: {len(detections)}, '
                                                  f'with mapping {self.class_map_detection}.')

                        else:
                            detections = {}
                            for class_id, class_detection in detections_with_classes.items():
                                for object_id, bbox in class_detection.items():
                                    detections[object_id] = bbox

                            self.logger.log_debug(f'Detections made: {len(detections)}.')

                        # ith new bboxes, add to tracker
                        # data should be a np array
                        self.reinit_tracker(data, detections)

                        self.stats_logger.push_log({'tracker' : False})

                    else: # use tracker to get new bb
                        # detections here should be in the format of {object : bbox} as there is no
                        # class matching (inside class_map_detection)
                        self.logger.log_debug('Using tracker for bounding boxes.')
                        detections = self.get_tracker_bounding_box(data)
                        if detections is None:
                            detections = gt_detections

                    if self.run_eval:
                        self.eval_detections(gt_detections, detections)

                else:
                    raise NotImplementedError

                # push log
                self.stats_logger.push_log({}, append=True)
                now = time.time()

        except Exception as ex:
            traceback.print_exc()
            self.logger.log_error(str(ex))

        finally:
            self.logger.log_info("Client loop ended.")
            self.close()

    def close(self):
        self.stats_logger.flush()

        if self.server_connect:
            self.socket.shutdown(SHUT_WR)
            self.socket.close()
            self.socket = None
        else:
            pass

def get_student_model(yaml_file = PARAMS['FASTER_RCNN_YAML']):
    if yaml_file is None:
        return None

    config = yaml_util.load_yaml_file(os.path.expanduser(yaml_file))
    models_config = config['models']
    student_model_config = models_config['student_model'] if 'student_model' in models_config else models_config[
        'model']
    student_model = load_model(student_model_config, PARAMS['DETECTION_DEVICE']).eval()

    return student_model

#main functionality for testing/debugging
if __name__ == '__main__':
    cp = Client()

    cp.start(PARAMS['HOST'], PARAMS['PORT'])
    cp.start_loop()