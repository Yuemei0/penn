# Evaluation
# - interpolate unvoiced


###############################################################################
# Configuration
###############################################################################


# # Default configuration parameters to be modified
# from .config import defaults
# # from .config import crepe

# # Modify configuration
# import yapecs
# yapecs.configure('penn', defaults)

# # Import configuration parameters
# from .config.defaults import *
# from .config.static import *

import yapecs
import sys
from pathlib import Path

# -------------------------------------------------------------------
# 默认配置
# -------------------------------------------------------------------
from .config import defaults as default_config
from .config.defaults import *
from .config.static import *

# -------------------------------------------------------------------
# 配置切换函数
# -------------------------------------------------------------------
def configure(config_module=None):
    """
    动态配置 penn 模块。
    config_module: 模块对象，如 defaults.py 或 crepe.py
    """
    import sys
    penn_module = sys.modules[__name__]

    if config_module is None:
        from .config import defaults as config_module

    # 配置 yapecs
    yapecs.configure('penn', config_module)

    # 更新 penn 顶层变量
    for k, v in vars(config_module).items():
        if not k.startswith('__'):
            setattr(penn_module, k, v)

# -------------------------------------------------------------------
# 立即加载默认配置
# -------------------------------------------------------------------
configure()  # 默认加载 fcnf0 配置
###############################################################################
# Module imports
###############################################################################


from .core import *
from .model import Model
from .train import loss, train
from . import convert
from . import data
from . import decode
from . import dsp
from . import evaluate
from . import load
from . import partition
from . import periodicity
from . import plot
from . import train
from . import voicing
