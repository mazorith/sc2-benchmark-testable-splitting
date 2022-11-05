from socket import *
from struct import pack, unpack
import torch
from torch import nn

import pickle
import time
import sys
import argparse
from params import PARAMS, CURR_DATE
from Logger import ConsoleLogger, DictionaryStatsLogger
from utils import *
from data import Dataset
from split_models import model_1
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

def get_gt_detections(class_info : ((int,), (int,)), boxes : [[int,],]) -> {int : [int]}:
    assert len(class_info[0]) == len(boxes), f'class_info: {class_info} boxes: {boxes}'
    gt_boxes = {}
    for i in range(len(class_info[0])):
        gt_boxes[class_info[1][i]] = boxes[i]

    return gt_boxes

def eval_detections(gt_detections : {int : [int]}, generated_detections : {int : [int]}) -> {}:
    # assert len(gt_detections) == len(generated_detections)
    scores = {}
    for object_id in gt_detections.keys():
        if object_id not in generated_detections:
            scores[object_id] = -0.01 # the tracker doesn't know about the object
        elif generated_detections[object_id] is None:
            scores[object_id] = -0.02 # the tracker already let go of the object
        else:
            scores[object_id] = round(calculate_bb_iou(gt_detections[object_id], generated_detections[object_id]), 4)

    return scores

class Client:

    def _init_tracker(self):
        if self.tracking:
            self._currently_tracking = set()
            self.tracker = Tracker()

    def _init_detector(self, student_model):
        self._refresh_iters = 1
        if not self.detection:
            return

        if self.detection_compression == 'model':
            assert student_model
            self.client_model = model_1.ClientModel(student_model)
            if self.server_connect:
                self.server_model = nn.Identity()
            else:
                self.server_model = model_1.ServerModel(student_model)

        else: # classical compression
            self.client_model = nn.Identity()
            if self.detector == 'model':
                self.server_model = student_model
            else:
                self.server_model = nn.Identity()

    def __init__(self, student_model = None, server_connect = PARAMS['USE_NETWORK'], run_type = PARAMS['RUN_TYPE'],
                 stats_log_dir = PARAMS['STATS_LOG_DIR'], dataset = PARAMS['DATASET'], tracking = PARAMS['TRACKING'],
                 detection = PARAMS['DETECTION'], detection_compression = PARAMS['DET_COMPRESSOR'],
                 refresh_type = PARAMS['BOX_REFRESH'], detector = PARAMS['DETECTOR'], run_eval = PARAMS['EVAL']):
        self.socket, self.message = None, None
        self.logger, self.dataset, self.stats_logger = ConsoleLogger(), Dataset(dataset=dataset), \
                                                       DictionaryStatsLogger(logfile=f"{stats_log_dir}/client-{dataset}-{CURR_DATE}.log")
        self.server_connect, self.tracking, self.detection, self.run_type = server_connect, tracking, detection, run_type
        self.detection_compression, self.refresh_type, self.detector = detection_compression, refresh_type, detector
        self.run_eval = run_eval

        self._init_tracker()
        self._init_detector(student_model)

    def _connect(self, server_ip, server_port):
        self.logger.log_debug('Connecting from client')
        self.socket = socket(AF_INET, SOCK_STREAM)
        self.socket.connect((server_ip, server_port))
        self.logger.log_info('Successfully connected to socket')

    def _client_handshake(self):
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
        if self.server_connect:
            data = pickle.dumps(data)
            length = pack('>Q', len(data))

            self.socket.sendall(length)
            self.socket.sendall(data)

            ack = self.socket.recv(1)

        else:
            self.message = data

    def _get_server_data(self):
        '''returns the server data in any format'''
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
            assert len(b'\00') == 1
            collected_message = True
            self.socket.sendall(b'\00')
            return pickle.loads(data)

    def get_model_bb(self):
        '''uses the model to get the bounding box
        will never be called if DETECTOR is False'''

        if self.server_connect:
            return self._get_server_data()

        else:
            client_data = self.message['data']
            model_outputs = self.server_model(*client_data)

            return model_outputs

    def handle_server_data(self, server_data, server_model = PARAMS['DETECTOR_MODEL']):
        '''Handles the server data, ie. formatting the model outputs into eval-friendly outputs'''
        if server_model == 'faster_rcnn':
            return map_faster_rcnn_outputs(server_data)
        else:
            return server_data

    def get_new_bounding_box(self, d : (), before, i : int) -> {}:
        '''gets a new "accurate" bounding box (from a separate detection pipeline)
        returns in the format of {object_id : (xyxy)}'''

        # if gt is used, get bounding box will simply return the ground truth
        if not self.detection:
            self.logger.log_debug('Using ground truth boxes.')
            # return ground truth boxes
            _, size_orig, class_info, gt, _, _ = d

            self.stats_logger.push_log({'gt' : True, 'original_size' : size_orig})
            return get_gt_detections(class_info, gt)

        # otherwise, use an actual detector
        # offline case in model detection is handled in the individual helper functions
        self.logger.log_debug('Creating message for boxes.')
        # uses model for detection; requires some compression (make metrics)
        if self.detection_compression == 'model':  # use a model (bottleneck) to compress it
            data, size_orig, class_info, gt, fname, _ = d
            data_reshape = (data/256).transpose((2,0,1))[np.newaxis, ...]

            # collect model runtime
            now = time.time()
            tensors_to_measure, other_info = self.client_model(torch.from_numpy(data_reshape).float())
            compression_time = time.time() - now

            size_compressed = sum(get_tensor_size(x) for x in (tensors_to_measure))

            message = {'timestamp': time.time(), 'data': (*tensors_to_measure, *other_info)}

            self.stats_logger.push_log({'compressor' : 'model'})

        else:  # classical compression – dataloader should have compressed info
            data, (size_orig, size_compressed), fname = d
            compression_time = time.time() - before
            message = {'timestamp': time.time(), 'data': data}

            self.stats_logger.push_log({'compressor': 'classical'})

        self.stats_logger.push_log({'encode_time': compression_time, 'message_size': size_compressed,
                                    'original_size': size_orig}, append=False)
        self.logger.log_info(f'Generated message with bytesize {size_compressed} and original {size_orig}')

        self._send_encoder_data(message)

        # response_time in the offline case will be the time it takes for the server model to run
        # in the online case it will be 2x latency + response_time
        now = time.time()
        server_data = self.get_model_bb()[0] # batch size 1
        self.stats_logger.push_log({'response_time': time.time() - now}, append=False)

        return self.handle_server_data(server_data)

    def check_detection_refresh(self, refresh_limit : int = PARAMS['REFRESH_ITERS']) -> bool:
        '''returns true if a refresh is needed, otherwise do nothing'''
        if self.refresh_type == 'fixed':
            if self._refresh_iters >= refresh_limit:
                self._refresh_iters = 1
                return True

            self._refresh_iters += 1

        else:
            raise NotImplementedError('Invalid Refresh Type')

    def refresh_tracker(self, frame : np.ndarray, detections : {int : [int]}):
        if self.tracking:
            self.tracker.handle_new_detection(frame, detections)

    def get_tracker_bounding_box(self, frame) -> {int : [int]}:
        if self.tracking:
            now = time.time()
            detections = self.tracker.update(frame)
            self.stats_logger.push_log({'tracker': True, 'tracker_time': time.time() - now})
            return detections

        return None

    def start_loop(self):
        try:
            now = time.time()
            for i, d in enumerate(self.dataset.get_dataset()):
                self.logger.log_info(f'Starting iteration {i}')

                if self.run_type == 'BB':
                    self.logger.log_debug('Starting Bounding Box on frame.')

                    data, size_orig, class_info, gt, fname, start = d
                    self.stats_logger.push_log({'iteration' : i, 'fname' : fname})

                    gt_detections = get_gt_detections(class_info, gt)

                    if start or self.check_detection_refresh():
                        self.logger.log_debug('Re-generating bounding boxes.')
                        detections = self.get_new_bounding_box(d, now, i)

                        # ith new bboxes, add to tracker
                        # data should be a np array
                        self.refresh_tracker(data, detections)

                        self.stats_logger.push_log({'tracker' : False})

                    else: # use tracker to get new bb
                        self.logger.log_debug('Using tracker for bounding boxes.')
                        detections = self.get_tracker_bounding_box(data)
                        if detections is None:
                            detections = gt_detections

                    if self.run_eval:
                        self.stats_logger.push_log(eval_detections(gt_detections, detections), append=False)

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

def get_student_model(yaml_file = PARAMS['DETECTOR_YAML']):
    if yaml_file is None:
        return None

    config = yaml_util.load_yaml_file(os.path.expanduser(yaml_file))
    models_config = config['models']
    student_model_config = models_config['student_model'] if 'student_model' in models_config else models_config[
        'model']
    student_model = load_model(student_model_config, PARAMS['DETECTION_DEVICE']).eval()
    student_model.roi_heads.score_thresh = 0.0005

    return student_model

#main functionality for testing/debugging
if __name__ == '__main__':

    student_model = get_student_model(PARAMS['DETECTOR_YAML'])

    cp = Client(student_model = student_model)
    cp.start(PARAMS['HOST'], PARAMS['PORT'])
    cp.start_loop()