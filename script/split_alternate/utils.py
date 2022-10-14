import sys

def get_message_size(message):
    '''returns memory allocated for a dictionary with a singular tensor stored in message['data']'''
    x = {k : v for k, v in message.items() if k != 'data'}
    data_size = message['data'].nelement() * message['data'].element_size()

    return sys.getsizeof(x) + data_size