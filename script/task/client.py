from socket import *
from struct import pack, unpack
import torch
import pickle
import time
import sys

class ClientProtocol:

    def __init__(self):
        self.socket = None
        self.data = None

    def connect(self, server_ip, server_port):
        self.socket = socket(AF_INET, SOCK_STREAM)
        self.socket.connect((server_ip, server_port))

    def close(self):
        self.socket.shutdown(SHUT_WR)
        self.socket.close()
        self.socket = None

    def send_encoder_data(self, data):
        data = pickle.dumps(data)
        length = pack('>Q', len(data))

        self.socket.sendall(length)
        self.socket.sendall(data)

        ack = self.socket.recv(1)
        #print(ack)

    def client_handshake(self):
        print('Sending handshake to server')
        self.socket.sendall(b'\00')
        ack = self.socket.recv(1)
        if(ack == b'\00'):
            print('Successfully recived server Handshake')
        else:
            print('Message recived not server ack')

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

#main functionality for testing/debugging
if __name__ == '__main__':

    cp = ClientProtocol()

    tensor = torch.rand([200,200,200])
    print(tensor.dtype)
    print(tensor.element_size(), tensor.nelement(), tensor.element_size() * tensor.nelement())
    #tensor = pickle.dumps(tensor)
    cp.connect('128.195.54.126', 55555) #128.195.54.126, 127.0.0.1
    cp.client_handshake()
    #time.sleep(30)
    cp.send_encoder_data(tensor)
    cp.handle_input_data()
    print(cp.data.shape)
    print(cp.data.element_size(), cp.data.nelement(), cp.data.element_size() * cp.data.nelement())
    print(sys.getsizeof(cp.data))
    print(pack('>Q', len(cp.data)))
    cp.close()

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