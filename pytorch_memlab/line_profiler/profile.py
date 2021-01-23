from functools import wraps
from typing import Callable, Tuple
from .line_profiler import LineProfiler, DEFAULT_COLUMNS


global_line_profiler = LineProfiler()
global_line_profiler.enable()


def clear_global_line_profiler():
    """Clears the state of the global line profiler"""
    global_line_profiler.clear()


def set_target_gpu(gpu_id: int):
    """Set the target GPU id to profile memory

    Because of the lack of output space, only one GPU's memory usage is shown
    in line profiler. However you can use this function to switch target GPU
    to profile on. The GPU switch can be performed before profiling and even
    in the profiled functions.

    Args:
        - gpu_id: cuda index to profile the memory on,
                  also accepts `torch.device` object.
    """
    global_line_profiler.target_gpu = gpu_id


def profile(func, columns: Tuple[str, ...] = DEFAULT_COLUMNS):
    """Profile the CUDA memory usage of target function line by line

    The profiling results will be printed at exiting, KeyboardInterrupt raised.
    The CUDA memory is collected only on the **current** cuda device.

    The columns are explained in the PyTorch documentation:
    https://pytorch.org/docs/stable/cuda.html#torch.cuda.memory_stats

    Args:
        func: the function or method to profile on
        columns (list of str): the column names of interest, See PyTorch's doc for available names.

    Usage:
        ```python
        @profile
        def foo():
            linear = torch.nn.Linear(100, 100).cuda()

        foo()

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(100, 100).cuda()

            @profile
            def forward(self, inp):
                return self.linear(inp)

        inp = torch.Tensor(50, 100).cuda()
        foo = Foo()
        foo(inp)
        ```
    """
    import atexit
    global_line_profiler.add_function(func)

    def print_stats_atexit():
        global_line_profiler.print_stats(func, columns)
    atexit.register(print_stats_atexit)

    return func


def profile_every(output_interval: int = 1, enable: bool = True, columns: Tuple[str, ...] = DEFAULT_COLUMNS):
    """Profile the CUDA memory usage of target function line by line

    Prints the profiling output every `output_interval` execution of the target
    function
    The CUDA memory is collected only on the **current** cuda device.

    The columns are explained in the PyTorch documentation:
    https://pytorch.org/docs/stable/cuda.html#torch.cuda.memory_stats

    Args:
        enable (bool): whether to enable the profiling mode, so users don't have to
                       modify any source code for enabling and disabling profiling.
        output_interval (int): frequency of output the profiling results
        columns (list of str): the column names of interest, See PyTorch's doc for available names.
    """

    def inner_decorator(func: Callable):
        func.cur_idx = 1

        if enable:
            global_line_profiler.add_function(func)

        @wraps(func)
        def run_func(*args, **kwargs):
            res = func(*args, **kwargs)
            if enable:
                if func.cur_idx % output_interval == 0:
                    global_line_profiler.print_stats(func, columns)

                func.cur_idx += 1
            return res

        return run_func
    return inner_decorator
