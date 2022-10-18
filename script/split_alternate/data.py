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
        elif dataset == 'virat':
            self.dataset = self.get_virat_dataset()
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

    def get_bdd_dataset(self, codec = 'avc1', frame_limit = PARAMS['FRAME_LIMIT'], shape = PARAMS['VIDEO_SHAPE']):
        fnames = [fname for fname in os.listdir(f'{self.data_dir}/bdd100k/videos/test') if '.mov' in fname] # remove .DS_Store and other misc files

        num_success = 0
        for fname in fnames:
            if num_success >= 300:  # 100 trials
                break

            full_fname = f'{self.data_dir}/bdd100k/videos/test/{fname}'
            self.logger.log_debug(full_fname)
            cap = cv2.VideoCapture(full_fname)
            success, frames = extract_frames(cap, frame_limit=frame_limit)

            if not success:
                self.logger.log_debug(f'OpenCV Failed on file {fname}')
                continue

            frames_byte_size = sys.getsizeof(frames) * shape[1] * shape[0] / frames.shape[2] / frames.shape[1]

            num_success += 1
            byte_frames = return_frames_as_bytes( frames, codec=codec )
            yield byte_frames, (int(frames_byte_size), sys.getsizeof(byte_frames))

    def get_virat_dataset(self, codec='avc1', frame_limit = PARAMS['FRAME_LIMIT'], shape = PARAMS['VIDEO_SHAPE']):
        virat_dir = f'{self.data_dir}/VIRAT'
        fnames = [fname for fname in os.listdir(virat_dir) if
                  '.mp4' in fname]  # remove .DS_Store and other misc files
        for fname in fnames:
            for i in range(10): # load 10 random frames / video
                full_fname = f'{virat_dir}/{fname}'
                self.logger.log_debug(full_fname)
                cap = cv2.VideoCapture(full_fname)
                success, frames = extract_frames(cap, frame_limit=frame_limit)

                if not success:
                    self.logger.log_debug(f'OpenCV Failed on file {fname}')
                    continue

                frames_byte_size = sys.getsizeof(frames) * shape[1] * shape[0] / frames.shape[2] / frames.shape[1]

                byte_frames = return_frames_as_bytes(frames, codec=codec)
                yield byte_frames, (frames_byte_size, len(byte_frames))

    def get_davis_dataset(self, codec='avc1', frame_limit = PARAMS['FRAME_LIMIT']):
        pass
        # for dtype in
        # frame_i = (9, 9+frame_limit)
        # for i in range(frame_i[0], frame_i[1]):
