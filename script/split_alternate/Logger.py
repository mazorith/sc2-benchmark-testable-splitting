from params import LEVELS, PARAMS
from datetime import datetime

class ConsoleLogger:
    def __init__(self, level = PARAMS['LOGGING_LEVEL'], dateformat = PARAMS['DATE_FORMAT']):
        self.level = level
        self.dateformat = dateformat

    def log_debug(self, message):
        if self.level <= LEVELS['DEBUG']:
            print(f'[DEBUG] {datetime.now().strftime(self.dateformat)}: {message}')

    def log_info(self, message):
        if self.level <= LEVELS['INFO']:
            print(f'[INFO] {datetime.now().strftime(self.dateformat)}: {message}')

    def log_error(self, message):
        if self.level <= LEVELS['ERROR']:
            print(f'[ERROR] {datetime.now().strftime(self.dateformat)}: {message}')

class DictionaryStatsLogger:
    def __init__(self, logfile, flush_limit = PARAMS['FLUSH_LIMIT']):
        self.logfile = logfile
        self.stats = []
        self.flush_counter = 0
        self.flush_limit = flush_limit
        self.curr_dict = {}

    def push_log(self, log : {}, append = False):
        '''adds the log to the list, will automatically periodically flush.
        append = True if you want to append the list to the log'''
        self.curr_dict.update(log)
        if not append:
            return

        if self.flush_counter >= self.flush_limit:
            self.flush() # flush the file
            self.flush_counter = 0
            self.stats = []

        self.stats.append(self.curr_dict)
        self.flush_counter += 1
        self.curr_dict = {}

    def flush(self): # appends to specific file
        if len(self.stats) == 0:
            return

        with open(self.logfile, 'a') as f:
            f.write('\n'.join(str(stat) for stat in self.stats) + '\n')