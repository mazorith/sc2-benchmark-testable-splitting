from utils import *
import torch
from Logger import ConsoleLogger

class Dataset:
    def __init__(self, data_dir = PARAMS['DATA_DIR'], dataset = PARAMS['DATASET']):
        self.data_dir = data_dir
        self.logger = ConsoleLogger()
        if dataset == 'latency':
            self.dataset = self.get_toy_dataloader()
        elif dataset == 'framelen':
            self.dataset = self.get_framelen_dataset()
        elif dataset == 'bdd':
            self.dataset = self.get_bdd_dataset()
        elif dataset == 'virat':
            self.dataset = self.get_virat_dataset()
        elif dataset == 'yc2':
            self.dataset = self.get_youcook2_dataset()
        elif dataset == 'phone':
            self.dataset = self.get_phone_dataset()
        else:
            raise ValueError('Dataset not found.')

    def get_dataset(self):
        '''Returns byte_data, (x, y) where x is uncompressed data and y is compressed data
        If there is no compression, x=y'''
        return self.dataset

    def open_fname(self, fname, cap = None, codec = 'avc1', frame_limit = PARAMS['FRAME_LIMIT'], shape = PARAMS['VIDEO_SHAPE'],
                   transpose = False):
        '''returns Success, (encoding as bytes, (original_size, size of encoding))'''
        self.logger.log_debug(f'Handling {fname}')
        if not cap:
            cap = cv2.VideoCapture(fname)

        success, frames = extract_frames(cap, frame_limit=frame_limit, transpose_frame=transpose)

        if not success:
            self.logger.log_debug(f'OpenCV Failed on file {fname}')
            return False, None

        frames_byte_size = sys.getsizeof(frames) # frames should be in shape
        byte_frames = return_frames_as_bytes(frames, codec=codec)

        return True, (byte_frames, (frames_byte_size, len(byte_frames)))

    def get_toy_dataloader(self):
        # testing latency
        # 100 arrays split by intervals of 10k: [10k, ..., 1mil]
        # 30 trials each
        array_lens = np.arange(int(1e4), int(1e6 + 1e4), int(1e4), dtype=int)
        for array_len in array_lens:
            arr = np.zeros((array_len,), dtype=np.int64)
            arr_byte_len = sys.getsizeof(arr)
            for trial in range(30):
                yield arr, (arr_byte_len, arr_byte_len)

    def get_framelen_dataset(self):
        '''test framelen vs compression ratio
        using virat dataset because it is the most stable'''
        # 2 ->
        virat_dir = f'{self.data_dir}/VIRAT'
        fnames = [fname for fname in os.listdir(virat_dir) if
                  '.mp4' in fname]  # remove .DS_Store and other misc files

        frame_lens = range(2, 62, 2)
        for frame_len in frame_lens:
            for fname in fnames: # 10 videos inside virat
                full_fname = f'{virat_dir}/{fname}'
                cap = cv2.VideoCapture(full_fname)

                for i in range(10):  # load 10 random frames / video ==> 100 samples / frame length
                    ret, byte_info = self.open_fname(full_fname, cap=cap, frame_limit=frame_len)
                    if not ret:
                        continue

                    byte_encoding, (frames_size, encoding_size) = byte_info

                    yield byte_encoding, (frames_size, encoding_size), full_fname

    def get_bdd_dataset(self):
        fnames = [fname for fname in os.listdir(f'{self.data_dir}/bdd100k/videos/test') if '.mov' in fname] # remove .DS_Store and other misc files

        num_success = 0
        for fname in fnames:
            full_fname = f'{self.data_dir}/bdd100k/videos/test/{fname}'
            cap = cv2.VideoCapture(full_fname)

            for i in range(5):
                if num_success >= 700:  # 700 files x 5 iters/file = 7000 trials
                    break

                ret, byte_info = self.open_fname(full_fname, cap=cap)
                if not ret:
                    continue

                byte_encoding, (frames_size, encoding_size) = byte_info
                num_success+=1

                yield byte_encoding, (frames_size, encoding_size), full_fname

    def get_virat_dataset(self):
        virat_dir = f'{self.data_dir}/VIRAT'
        fnames = [fname for fname in os.listdir(virat_dir) if
                  '.mp4' in fname]  # remove .DS_Store and other misc files
        for fname in fnames:
            full_fname = f'{virat_dir}/{fname}'
            cap = cv2.VideoCapture(full_fname)

            for i in range(20): # load 20 random frames / video
                ret, byte_info = self.open_fname(full_fname, cap=cap)
                if not ret:
                    continue

                byte_encoding, (frames_size, encoding_size) = byte_info

                yield byte_encoding, (frames_size, encoding_size), full_fname

    def get_youcook2_dataset(self):
        yc2_dir = f'{self.data_dir}/YouCook2/raw_videos/validation'
        dirs = [f'{yc2_dir}/{x}' for x in sorted(os.listdir(yc2_dir)) if len(x)==3] # 3 digit codes usually, this is hardcoded but gets past .DS_Store files
        fnames = []
        for dir in dirs:
            fnames += [f'{dir}/{x}' for x in sorted(os.listdir(dir)) if '.webm' in x]

        for fname in fnames:
            cap = cv2.VideoCapture(fname)

            for i in range(70):
                ret, byte_info = self.open_fname(fname, cap=cap)
                if not ret:
                    continue

                byte_encoding, (frames_size, encoding_size) = byte_info

                yield byte_encoding, (frames_size, encoding_size), fname

    def get_phone_dataset(self):
        p_dir = f'{self.data_dir}/Phone'
        fnames = [x for x in os.listdir(p_dir) if '.MOV' in x]

        for fname in fnames:
            full_fname = f'{p_dir}/{fname}'
            cap = cv2.VideoCapture(full_fname)

            for i in range(60):
                ret, byte_info = self.open_fname(f'{self.data_dir}/Phone/{fname}', cap=cap, transpose=True)
                if not ret:
                    continue

                byte_encoding, (frames_size, encoding_size) = byte_info

                yield byte_encoding, (frames_size, encoding_size), full_fname

    def get_davis_dataset(self, codec='avc1', frame_limit = PARAMS['FRAME_LIMIT']):
        pass
        # for dtype in
        # frame_i = (9, 9+frame_limit)
        # for i in range(frame_i[0], frame_i[1]):
