from multiprocessing import connection
import torch
import pickle
import time
import sys
from socket import *
from struct import pack, unpack
from params import PARAMS, CURR_DATE
from Logger import ConsoleLogger, DictionaryStatsLogger
from utils import *

def process_data(data):
    # print(data)
    return 0

class ServerProtocol:

    def __init__(self):
        self.socket = None
        self.connection = None
        self.data = None
        self.logger = ConsoleLogger()
        self.stats_logger = DictionaryStatsLogger(f"{PARAMS['STATS_LOG_DIR']}/server-{CURR_DATE}.log")

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

    def handle_encoder_data(self):
        # collected_message = False
        self.logger.log_info('Waiting for client...')
        while True:
            time.sleep(1)
            bs = self.connection.recv(8)
            (length,) = unpack('>Q', bs)
            data = b''
            while len(data) < length:
                to_read = length - len(data)
                data += self.connection.recv(
                    4096 if to_read > 4096 else to_read)

            # send our 0 ack
            assert len(b'\00') == 1

            return self.parse_message(pickle.loads(data))

    def parse_message(self, message):
        timestamp = message['timestamp']
        data = message['data']

        latency = time.time() - timestamp

        self.logger.log_info(f'Received data with latency {round(latency, 4)}; sending message back')
        self.stats_logger.push_log({'latency' : round(latency, 4)}, append=False)
        self.connection.sendall(b'\00')

        return data

    def start_server_loop(self):
        try:
            while True:
                time.sleep(1)
                data = self.handle_encoder_data() # data  .shape
                self.logger.log_debug(f'Processing data')

                curr_time = time.time()
                process_data(data)
                process_time = round(time.time() - curr_time, 4)

                self.stats_logger.push_log({'processing_time' : process_time}, append=True)

        except Exception as ex:
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
        #print(ack)

    def close(self):
        self.connection.shutdown(SHUT_WR)
        self.connection.close()
        self.socket.close()
        self.stats_logger.flush()
        self.socket = None

#main functionality for testing/debugging
if __name__ == '__main__':
    sp = ServerProtocol()
    sp.start(PARAMS['HOST'], PARAMS['PORT'])
    sp.start_server_loop()
