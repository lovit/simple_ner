__title__ = 'simple_ner'
__version__ = '0.0.1'
__author__ = 'Lovit'
__license__ = 'GPL v3'

from .utils import load_sentences
from .utils import get_available_memory
from .utils import get_process_memory
from .utils import remain_time
from .utils import Sentences
from .features import FeatureManager
from .features import ZCorpus
from .ner import FeatureCountingNER