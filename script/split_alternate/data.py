from utils import *
import torch
from Logger import ConsoleLogger

class Dataset:
    def __init__(self, data_dir = PARAMS['DATA_DIR'], dataset = PARAMS['DATASET']):
        self.data_dir = data_dir
        self.logger = ConsoleLogger()
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

        num_success = 0
        for fname in fnames:
            if num_success >= 100:  # 100 trials
                break

            full_fname = f'{self.data_dir}/bdd100k/videos/test/{fname}'
            self.logger.log_debug(full_fname)
            cap = cv2.VideoCapture(full_fname)
            success, frames = extract_frames(cap, frame_limit=frame_limit)
            if not success:
                self.logger.log_debug(f'OpenCV Failed on file {fname}')
                continue

            num_success += 1
            byte_frames = return_frames_as_bytes( frames, codec=codec )
            yield byte_frames, (sys.getsizeof(frames), sys.getsizeof(byte_frames))