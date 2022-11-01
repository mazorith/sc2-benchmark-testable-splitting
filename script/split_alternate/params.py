from datetime import datetime

LEVELS = {'DEBUG' : 0, 'INFO' : 1, 'ERROR' : 2, 'NOTHING' : 3}

PARAMS = {}
# in the offline case, client.py will have the server model as well
# otherwise
PARAMS['USE_NETWORK'] = False
PARAMS['HOST'] = '127.0.0.1'
PARAMS['PORT'] = 55557

PARAMS['DATE_FORMAT'] = '%m/%d %H:%M:%S'
PARAMS['LOGGING_LEVEL'] = LEVELS['INFO']

PARAMS['STATS_LOG_DIR'] = 'Logs'
PARAMS['FLUSH_LIMIT'] = 100

PARAMS['DEV_DIR'] = 'dev'
PARAMS['DATA_DIR'] = 'Data'

PARAMS['DATASET'] = 'model_toy'
PARAMS['FRAME_LIMIT'] = 15
PARAMS['FPS'] = 30.
PARAMS['VIDEO_SHAPE'] = (1280,720)
PARAMS['DAVIS_SCENES'] = [1,2,3,4,5,6,7,8,9,10]

PARAMS['RUN_TYPE'] = 'model'
PARAMS['DEVICE'] = 'cpu'
PARAMS['MODEL_YAML'] = '../../configs/coco2017/supervised_compression/ghnd-bq/faster_rcnn_resnet50-bq1ch_fpn_from_faster_rcnn_resnet50_fpn.yaml'


CURR_DATE = datetime.now().strftime('%m-%d_%H:%M:%S')