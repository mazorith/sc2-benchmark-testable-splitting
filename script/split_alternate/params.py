from datetime import datetime

LEVELS = {'DEBUG' : 0, 'INFO' : 1, 'ERROR' : 2, 'NOTHING' : 3}

PARAMS = {}
PARAMS['HOST'] = '127.0.0.1'
PARAMS['PORT'] = 55557

PARAMS['DATE_FORMAT'] = '%m/%d %H:%M:%S'
PARAMS['LOGGING_LEVEL'] = LEVELS['DEBUG']

PARAMS['STATS_LOG_DIR'] = 'Logs'
PARAMS['FLUSH_LIMIT'] = 100

PARAMS['DEV_DIR'] = 'dev'
PARAMS['DATA_DIR'] = 'Data'

PARAMS['DATASET'] = 'bdd'
PARAMS['FRAME_LIMIT'] = 15
PARAMS['FPS'] = 30.
PARAMS['VIDEO_SHAPE'] = (1280,720)

CURR_DATE = datetime.now().strftime('%m/%d %H:%M:%S')