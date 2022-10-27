from socket import *
from struct import pack, unpack
import torch
import pickle
import time
import sys
import argparse
from params import PARAMS, CURR_DATE
from Logger import ConsoleLogger, DictionaryStatsLogger
from utils import *
from data import Dataset
from split_models import model_1

from torchdistill.common import yaml_util
from sc2bench.models.detection.registry import load_detection_model
from sc2bench.models.detection.wrapper import get_wrapped_detection_model

def load_model(model_config, device):
    if 'detection_model' not in model_config:
        return load_detection_model(model_config, device)
    return get_wrapped_detection_model(model_config, device)

def create_input(data):
    return data

class ClientProtocol:

    def __init__(self, model = None, server_connect = PARAMS['USE_NETWORK']):
        self.socket, self.message = None, None
        self.logger = ConsoleLogger()
        self.stats_logger = DictionaryStatsLogger(logfile=f"{PARAMS['STATS_LOG_DIR']}/client-{PARAMS['DATASET']}-{CURR_DATE}.log")
        self.dataset = Dataset()
        self.server_connect = server_connect
        self.model = model

    def connect(self, server_ip, server_port):
        self.logger.log_debug('Connecting from client')
        self.socket = socket(AF_INET, SOCK_STREAM)
        self.socket.connect((server_ip, server_port))
        self.logger.log_info('Successfully connected to socket')

    def client_handshake(self):
        self.logger.log_debug('Sending handshake from client')
        self.socket.sendall(b'\00')
        ack = self.socket.recv(1)
        if(ack == b'\00'):
            self.logger.log_info('Successfully received server Handshake')
        else:
            self.logger.log_info('Message received not server ack')

    def start(self, server_ip, server_port):
        if self.server_connect:
            self.connect(server_ip, server_port)
            self.client_handshake()
            self.logger.log_info('Successfully started client')

        else:
            self.logger.log_info('Starting in offline mode.')

    def send_encoder_data(self, data):
        if self.server_connect:
            data = pickle.dumps(data)
            length = pack('>Q', len(data))

            self.socket.sendall(length)
            self.socket.sendall(data)

            ack = self.socket.recv(1)

        else:
            self.message = data

    def get_server_data(self):
        '''get server data will get the data returned from the server in self.data'''
        if self.server_connect:
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

        else:
            return self.message['data']

    def send_loop(self):
        try:
            now = time.time()
            for i, d in enumerate(self.dataset.get_dataset()):
                self.logger.log_info(f'Starting iteration {i}')

                if PARAMS['RUN_TEST'] == 'model':
                    data, size_orig, fname = d

                    # collect model runtime
                    now = time.time()
                    tensors_to_measure, other_info = self.model(data)
                    compression_time = time.time() - now

                    size_compressed = sum(get_tensor_size(x) for x in (tensors_to_measure))

                    message = {'timestamp' : time.time(), 'data' : (*tensors_to_measure, *other_info)}

                else:
                    data, (size_orig, size_compressed), fname = d
                    compression_time = time.time() - now
                    message = {'timestamp': time.time(), 'data': data}

                self.stats_logger.push_log({'encode_time' : compression_time, 'message_size' : size_compressed,
                                            'original_size' : size_orig, 'iteration' : i, 'fname' : fname}, append=False)
                self.logger.log_info(f'Generated message with bytesize {size_compressed} and original {size_orig}')

                self.send_encoder_data(message)
                # get response and measure time
                now = time.time()
                server_data = self.get_server_data()
                self.stats_logger.push_log({'response_time' : time.time() - now}, append=True)

                now = time.time()

        except Exception as ex:
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

def get_student_model(yaml_file = PARAMS['MODEL_YAML']):
    config = yaml_util.load_yaml_file(os.path.expanduser(yaml_file))
    models_config = config['models']
    student_model_config = models_config['student_model'] if 'student_model' in models_config else models_config[
        'model']
    student_model = load_model(student_model_config, PARAMS['DEVICE']).eval()
    student_model.roi_heads.score_thresh = 0.0005

    return student_model

#main functionality for testing/debugging
if __name__ == '__main__':

    student_model = get_student_model(PARAMS['MODEL_YAML'])
    client_model = model_1.ClientModel(student_model)

    cp = ClientProtocol(model = client_model)
    cp.start(PARAMS['HOST'], PARAMS['PORT'])
    cp.send_loop()