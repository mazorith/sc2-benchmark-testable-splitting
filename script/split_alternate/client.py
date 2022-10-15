from socket import *
from struct import pack, unpack
import torch
import pickle
import time
import sys
from params import PARAMS, CURR_DATE
from Logger import ConsoleLogger, DictionaryStatsLogger
from utils import *
from data import Dataset

def create_input(data):
    return data

class ClientProtocol:

    def __init__(self):
        self.socket = None
        self.data = None
        self.logger = ConsoleLogger()
        self.stats_logger = DictionaryStatsLogger(logfile=f"{PARAMS['STATS_LOG_DIR']}/client-{CURR_DATE}.log")
        self.dataset = Dataset()

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
        self.connect(server_ip, server_port)
        self.client_handshake()
        self.logger.log_info('Successfully started client')

    def send_encoder_data(self, data):
        data = pickle.dumps(data)
        length = pack('>Q', len(data))

        self.socket.sendall(length)
        self.socket.sendall(data)

        ack = self.socket.recv(1)
        #print(ack)

    def handle_input_data(self):
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
            self.data = pickle.loads(data)

    def send_data(self):
        # TODO: replace with actual dataloader
        try:
            for i, d in enumerate(self.dataset.get_dataset()):
                data, (size_orig, size_compressed) = d
                self.logger.log_info(f'Starting iteration {i}')

                now = time.time()
                input_time = time.time() - now
                message = {'timestamp': time.time(), 'data': data}

                self.stats_logger.push_log({'input_time' : input_time, 'message_size' : size_compressed,
                                            'original_size' : size_orig}, append=True)
                self.logger.log_info(f'Generated message with bytesize {size_compressed} and original {size_orig}')

                self.send_encoder_data(message)

            self.close()

        except Exception as ex:
            self.logger.log_error(ex)

        finally:
            self.logger.log_info("Client loop ended.")
            self.close()


    def close(self):
        self.socket.shutdown(SHUT_WR)
        self.socket.close()
        self.stats_logger.flush()
        self.socket = None

#main functionality for testing/debugging
if __name__ == '__main__':

    cp = ClientProtocol()
    cp.start(PARAMS['HOST'], PARAMS['PORT'])
    cp.send_data()