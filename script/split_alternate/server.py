from multiprocessing import connection
import pickle
import time
import sys
from socket import *
from struct import pack, unpack
from params import PARAMS, CURR_DATE
from Logger import ConsoleLogger, DictionaryStatsLogger
from utils2 import *
import traceback

from torchdistill.common import yaml_util
from sc2bench.models.detection.registry import load_detection_model
from sc2bench.models.detection.wrapper import get_wrapped_detection_model

def load_model(model_config, device):
    if 'detection_model' not in model_config:
        return load_detection_model(model_config, device)
    return get_wrapped_detection_model(model_config, device)

def get_student_model(yaml_file = PARAMS['FASTER_RCNN_YAML']):
    if yaml_file is None:
        return None

    config = yaml_util.load_yaml_file(os.path.expanduser(yaml_file))
    models_config = config['models']
    student_model_config = models_config['student_model'] if 'student_model' in models_config else models_config[
        'model']
    student_model = load_model(student_model_config, PARAMS['DETECTION_DEVICE']).eval()

    return student_model

class Server:
    '''Class for server operations. No functionality for offline evaluation (server does not do any eval).'''

    def _init_model(self):
        if self.detection_compression == 'model':
            # all of these should be using a yaml file (student model)
            if self.detector_model == 'faster_rcnn':
                student_model = get_student_model(PARAMS['FASTER_RCNN_YAML'])
                from split_models.model_2 import ServerModel
                self.server_model = ServerModel(student_model)
            else:
                raise NotImplementedError('No other models have been implemented yet.')
        elif self.detection_compression == 'classical':
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

        pass

    def __init__(self, server_connect = PARAMS['USE_NETWORK'], detection_compression = PARAMS['DET_COMPRESSOR'],
                 detector_model = PARAMS['DETECTOR_MODEL']):
        self.socket, self.connection, self.server_connect = None, None, server_connect
        self.data, self.logger, self.stats_logger = None, ConsoleLogger(), DictionaryStatsLogger(f"{PARAMS['STATS_LOG_DIR']}/server-{PARAMS['DATASET']}-{CURR_DATE}.log")
        self.detection_compression, self.detector_model = detection_compression, detector_model
        self._init_model()

    def listen(self, server_ip, server_port):
        self.socket = socket(AF_INET, SOCK_STREAM)
        self.logger.log_info(f"Binding to {server_ip}:{server_port}")
        self.socket.bind((server_ip, server_port))
        self.socket.listen(1)

    def server_handshake(self):
        (connection, addr) = self.socket.accept()
        self.connection = connection
        handshake = self.connection.recv(1)

        if handshake == b'\00':
            self.logger.log_info('Successfully received client Handshake; sending handshake back')
            connection.sendall(b'\00')
        else:
            self.logger.log_error('Message received not client handshake')

    def start(self, server_ip, server_port):
        self.logger.log_info('Starting server')
        self.listen(server_ip, server_port)
        self.server_handshake()
        self.logger.log_info('Successfully started server and handshake')

    def get_client_data(self):
        '''parses message and returns the client data (effectively message['data'])'''

        # collected_message = False
        self.logger.log_info('Waiting for client...')
        while True:
            bs = self.connection.recv(8)
            (length,) = unpack('>Q', bs)
            data = b''
            while len(data) < length:
                to_read = length - len(data)
                data += self.connection.recv(
                    4096 if to_read > 4096 else to_read)

            # send our 0 ack
            self.connection.sendall(b'\00')

            return self.parse_message(pickle.loads(data))

    def parse_message(self, message):
        '''logs the latency (message['latency']) and returns the data (message['data')'''
        timestamp = message['timestamp']
        data = message['data']

        latency = time.time() - timestamp

        if latency <= 1e-2:
            self.logger.log_debug(f'Message sent at timestamp {timestamp} and received with latency {latency}')

        self.logger.log_info(f'Received data with latency {round(latency, 4)}')
        self.stats_logger.push_log({'latency' : round(latency, 4)}, append=False)

        return data

    def process_data(self, client_data):
        '''processes the message using one of the detection models'''
        if self.detection_compression == 'model':
            if self.detector_model == 'faster_rcnn':
                model_outputs = self.server_model(*client_data)[0]
            else:
                raise NotImplementedError('No specified detector model exists.')
        elif self.detection_compression == 'classical':
            decoded_frame = decode_frame(client_data)
            model_outputs = self.server_model([torch.from_numpy(decoded_frame).float()])[0]  # first frame
        else:
            raise NotImplementedError('No other compression method exists.')

        self.logger.log_debug("Generated new BB with model (online).")
        return model_outputs

    def start_server_loop(self):
        '''main loop'''
        try:
            iteration_num = 0
            # effectively time waiting for client
            time_since_processed_lass_message = time.time()
            while True:
                data = self.get_client_data() # data  .shape
                self.logger.log_debug(f'Finished getting client data.')

                curr_time = time.time()
                self.stats_logger.push_log({'client_time' : curr_time - time_since_processed_lass_message})
                response = self.process_data(data)
                process_time = round(time.time() - curr_time, 4)

                self.logger.log_info('Sending response.')

                self.send_response(response)

                self.stats_logger.push_log({'processing_time' : process_time, 'iteration' : iteration_num}, append=True)
                iteration_num += 1

                time_since_processed_lass_message = time.time()

        except Exception as ex:
            traceback.print_exc()
            self.logger.log_error(ex)

        finally:
            self.logger.log_info('Server Loop Ended')
            self.close()

    def send_response(self, data):
        data = pickle.dumps(data)
        length = pack('>Q', len(data))

        self.connection.sendall(length)
        self.connection.sendall(data)

        ack = self.connection.recv(1)
        self.logger.log_debug('Received the ack from the response.')

    def close(self):
        self.stats_logger.flush()
        if not self.server_connect:
            return
        self.connection.shutdown(SHUT_WR)
        self.connection.close()
        self.socket.close()
        self.socket = None

#main functionality for testing/debugging
if __name__ == '__main__':
    server = Server()

    server.start(PARAMS['HOST'], PARAMS['PORT'])
    server.start_server_loop()
