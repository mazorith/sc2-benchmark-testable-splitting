from multiprocessing import connection
from sqlite3 import connect
import torch
import pickle
import time
import sys
from socket import *
from struct import pack, unpack

class ServerProtocol:

    def __init__(self):
        self.socket = None
        self.connection = None
        self.data = None

    def listen(self, server_ip, server_port):
        self.socket = socket(AF_INET, SOCK_STREAM)
        self.socket.bind((server_ip, server_port))
        self.socket.listen(1)

    def handle_encoder_data(self):
        collected_message = False
        while not collected_message:
            bs = self.connection.recv(8)
            (length,) = unpack('>Q', bs)
            data = b''
            while len(data) < length:
                to_read = length - len(data)
                data += self.connection.recv(
                    4096 if to_read > 4096 else to_read)

            # send our 0 ack
            assert len(b'\00') == 1
            collected_message = True
            self.connection.sendall(b'\00')
            self.data = pickle.loads(data)
        
    def server_handshake(self):
        (connection, addr) = self.socket.accept()
        self.connection = connection
        handshake = self.connection.recv(1)
        if(handshake == b'\00'):
            print('Successfully recived client Handshake')
            print('Sending handshake ack')
            connection.sendall(b'\00')
        else:
            print('Message recived not client handshake')

    def send_input_data(self, data):
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
        self.socket = None

#main functionality for testing/debugging
if __name__ == '__main__':
    sp = ServerProtocol()
    sp.listen('128.195.54.126', 55555)
    sp.server_handshake()
    #time.sleep(30)
    #sp.listen('127.0.0.1', 55556)
    sp.handle_encoder_data()
    print(sp.data.shape)

    print(sys.getsizeof(sp.data))
    print(pack('>Q', len(sp.data)))
    tensor = torch.rand([100,100,100])
    #tensor = pickle.dumps(tensor)

    sp.send_input_data(tensor)
    print(sys.getsizeof(torch.rand([10,10,10])))
    sp.close()
    

# HOST = 'localhost'
# PORT = 50007
# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.bind((HOST, PORT))
# s.listen(1)
# conn, addr = s.accept()
# print ('Connected by', addr)

# bs = conn.recv(8)
# (length,) = unpack('>Q', bs)

# data = b''
# while len(data) < length:
#     to_read = length - len(data)
#     data += conn.recv(4096 if to_read > 4096 else to_read)

# #data = conn.recv(4096)
# #data_variable = pickle.loads(data)
# data_tensor = pickle.loads(data)
# conn.close()
# print (data_tensor.shape)
# # Access the information by doing data_variable.process_id or data_variable.task_id etc..,
# print ('Data received from client')