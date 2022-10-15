from utils import *
import torch

class Dataset:
    def __init__(self, data_dir = PARAMS['DATA_DIR'], dataset = PARAMS['DATASET']):
        self.data_dir = data_dir
        if dataset == 'toy':
            self.dataset = self.get_toy_dataloader()
        elif dataset == 'bdd':
            self.dataset = self.get_bdd_dataset()
        else:
            raise ValueError('Dataset not found.')

    def get_dataset(self):
        '''Returns byte_data, (x, y) where x is uncompressed data and y is compressed data
        If there is no compression, x=y'''
        return self.dataset

    def get_toy_dataloader(self, shape=(100,100,100)):
        for i in range(100):
            data = torch.rand(shape)
            yield data, (data.nelement() * data.element_size(), data.nelement() * data.element_size())

    def get_bdd_dataset(self, codec = 'avc1', frame_limit = PARAMS['FRAME_LIMIT']):
        fnames = [fname for fname in os.listdir(f'{self.data_dir}/bdd100k/videos/test') if '.mov' in fname] # remove .DS_Store and other misc files

        for fname in fnames[:100]:
            cap = cv2.VideoCapture(f'{self.data_dir}/bdd100k/videos/test/{fname}')
            frames = extract_frames(cap, frame_limit=frame_limit)
            byte_frames = return_frames_as_bytes( frames, codec=codec )
            yield byte_frames, (sys.getsizeof(frames), sys.getsizeof(byte_frames))