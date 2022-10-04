import torch
import pickle
from socket import *
from struct import unpack

class ServerProtocol:

    def __init__(self):
        self.socket = None

    def listen(self, server_ip, server_port):
        self.socket = socket(AF_INET, SOCK_STREAM)
        self.socket.bind((server_ip, server_port))
        self.socket.listen(1)

    def handle_tensor(self):

        try:
            while True:
                (connection, addr) = self.socket.accept()
                try:
                    bs = connection.recv(8)
                    (length,) = unpack('>Q', bs)
                    data = b''
                    while len(data) < length:
                        to_read = length - len(data)
                        data += connection.recv(
                            4096 if to_read > 4096 else to_read)

                    # send our 0 ack
                    assert len(b'\00') == 1
                    connection.sendall(b'\00')
                finally:
                    connection.shutdown(SHUT_WR)
                    connection.close()
        finally:
            self.close()

    def close(self):
        self.socket.close()
        self.socket = None

# if __name__ == '__main__':
#     sp = ServerProtocol()
#     sp.listen('127.0.0.1', 55555)
#     sp.handle_tensor()


# class ProcessData:
#     process_id = 0
#     project_id = 0
#     task_id = 0
#     start_time = 0
#     end_time = 0
#     user_id = 0
#     weekend_id = 0


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