import numpy as np
from torchdistill.common.constant import def_logger
from torchdistill.common.file_util import get_binary_object_size

logger = def_logger.getChild(__name__)
ANALYZER_CLASS_DICT = dict()


def register_analysis_class(cls):
    ANALYZER_CLASS_DICT[cls.__name__] = cls
    return cls


class BaseAnalyzer(object):
    """
    Base analyzer to analyze and summarize the wrapped modules and intermediate representations.
    """
    def analyze(self, *args, **kwargs):
        raise NotImplementedError()

    def summarize(self):
        raise NotImplementedError()

    def clear(self):
        raise NotImplementedError()


@register_analysis_class
class FileSizeAnalyzer(BaseAnalyzer):
    """
    Analyzer to
    Args:
        unit (str): unit of data size in bytes (`B`, `KB`, `MB`)
        kwargs (dict): keyword arguments
    """
    UNIT_DICT = {'B': 1, 'KB': 1024, 'MB': 1024 * 1024}

    def __init__(self, unit='KB', **kwargs):
        self.unit = unit
        self.unit_size = self.UNIT_DICT[unit]
        self.kwargs = kwargs
        self.file_size_list = list()

    def analyze(self, compressed_obj):
        file_size = get_binary_object_size(compressed_obj, unit_size=self.unit_size)
        self.file_size_list.append(file_size)

    def summarize(self):
        file_sizes = np.array(self.file_size_list)
        logger.info('Bottleneck size [{}]: mean {} std {} for {} samples'.format(self.unit, file_sizes.mean(),
                                                                                 file_sizes.std(), len(file_sizes)))

    def clear(self):
        self.file_size_list.clear()


def get_analyzer(cls_name, **kwargs):
    if cls_name not in ANALYZER_CLASS_DICT:
        return None
    return ANALYZER_CLASS_DICT[cls_name](**kwargs)
