from .courtesy import Courtesy
from .mem_reporter import MemReporter
from .line_profiler import LineProfiler, profile, profile_every, set_target_gpu, reset_global_line_profiler
try:
    from .extension import load_ipython_extension
except ImportError:
    pass
