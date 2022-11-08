from datetime import datetime

# constants
LEVELS = {'DEBUG' : 0, 'INFO' : 1, 'ERROR' : 2, 'NOTHING' : 3}
KITTI_CLASSES = {'DontCare' : -1, 'Van': 0, 'Cyclist' : 1, 'Pedestrian' : 2, 'Car' : 3}
CURR_DATE = datetime.now().strftime('%m-%d_%H:%M:%S')

PARAMS = {}
# in the offline case, client.py will have the server model as well
# otherwise
PARAMS['USE_NETWORK'] = False
PARAMS['HOST'] = '127.0.0.1'
PARAMS['PORT'] = 55557

# params for directories + misc things
PARAMS['STATS_LOG_DIR'] = 'Logs'
PARAMS['DATE_FORMAT'] = '%m/%d %H:%M:%S'
PARAMS['LOGGING_LEVEL'] = LEVELS['DEBUG']
PARAMS['FLUSH_LIMIT'] = 100
PARAMS['DEV_DIR'] = 'dev'
PARAMS['DATA_DIR'] = 'Data'

# params for the dataset – used for classical compression methods
PARAMS['DATASET'] = 'kitti'
PARAMS['FRAME_LIMIT'] = 15
PARAMS['FPS'] = 30.
PARAMS['VIDEO_SHAPE'] = (1280,720)
# params for individual datasets
PARAMS['DAVIS_SCENES'] = [1,2,3,4,5,6,7,8,9,10]
PARAMS['KITTI_NAMES'] = ['timestep', 'object_i', 'class_name', '_1', '_2', '_3', 'x0', 'y0', 'x1', 'y1', '_4', '_5', '_6', '_7', '_8', '_9', '_10']

# params for run type
PARAMS['RUN_TYPE'] = 'BB'
PARAMS['EVAL'] = False

# params for the tracking
PARAMS['TRACKING'] = False # execute + evaluate a tracking algorithm; if false, evaluates the detector
PARAMS['TRACKER'] = 'MEDIANFLOW' # tracker algorithm

# params for the object detection
PARAMS['DETECTION'] = True # if false, uses ground truth labels for the detection (to eval tracking) – false makes it offline
PARAMS['DET_COMPRESSOR'] = 'model' # 'model' vs. 'classical': model is bottleneck, classical is classical compression
PARAMS['DETECTION_DEVICE'] = 'cpu'
PARAMS['DETECTOR'] = 'model' # if compressor is model this value is ignored (set to model)
PARAMS['DETECTOR_MODEL'] = 'faster_rcnn'
PARAMS['DETECTOR_YAML'] = '../../configs/coco2017/supervised_compression/ghnd-bq/faster_rcnn_resnet50-bq1ch_fpn_from_faster_rcnn_resnet50_fpn.yaml'

# params for detection 'refreshes'
PARAMS['BOX_REFRESH'] = 'fixed' # method to refresh bbox
PARAMS['REFRESH_ITERS'] = 10 # for fixed method, how many fixed iterations to refresh bb; setting iters to 1 makes detection run 100%