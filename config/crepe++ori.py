import os
from pathlib import Path
import penn
MODULE = 'penn'

# Configuration name
CONFIG = 'crepe++'

# The decoder to use for postprocessing
DECODER = 'argmax'

# Whether to perform local expected value decoding of pitch
LOCAL_EXPECTED_VALUE = False

# The name of the model to use for training
MODEL = 'crepe'

ASSETS_DIR = Path(__file__).parent.parent / 'assets_ori'

# Location of preprocessed features
CACHE_DIR = Path(__file__).parent.parent.parent / 'data' / 'cache'

# Location of datasets on disk
DATA_DIR = Path(__file__).parent.parent.parent / 'data' / 'datasets'