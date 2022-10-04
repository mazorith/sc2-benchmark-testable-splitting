from socket import *
from struct import pack
import torch
import pickle

class ClientProtocol:

    def __init__(self):
        self.socket = None

    def connect(self, server_ip, server_port):
        self.socket = socket(AF_INET, SOCK_STREAM)
        self.socket.connect((server_ip, server_port))

    def close(self):
        self.socket.shutdown(SHUT_WR)
        self.socket.close()
        self.socket = None

    def send_tensor(self, tensor_data):
        length = pack('>Q', len(tensor_data))

        self.socket.sendall(length)
        self.socket.sendall(tensor_data)

        ack = self.socket.recv(1)

# if __name__ == '__main__':
#     cp = ClientProtocol()

#     image_data = None
#     with open('IMG_0077.jpg', 'r') as fp:
#         image_data = fp.read()

#     assert(len(image_data))
#     cp.connect('127.0.0.1', 55555)
#     cp.send_image(image_data)
#     cp.close()

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
# # Create a socket connection.
# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.connect((HOST, PORT))

# # Create an instance of ProcessData() to send to server.
# variable = ProcessData()
# variable.process_id = 5

# tensor = torch.rand(100,100,100)
# data_string = pickle.dumps(tensor)

# length = pack('>Q', len(data_string))
# # Pickle the object and send it to the server

# #s.send(data_string)
# s.sendall(length)
# s.sendall(data_string)

# s.close()
# print ('Data Sent to Server')