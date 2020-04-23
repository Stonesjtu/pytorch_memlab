import inspect
import sys
from functools import wraps

import torch

from .line_records import LineRecords
from ..utils import readable_size

# Seaborn's `muted` color cycle
DEFAULT_COLUMNS = ['active_bytes.all.peak', 'reserved_bytes.all.peak']


class LineProfiler:
    """Profile the CUDA memory usage info for each line in pytorch

    This class registers callbacks for added functions to profiling them line
    by line, and collects all the statistics in CUDA memory. Usually you may
    want to use simpler wrapper below `profile` or `profile_every`.

    The CUDA memory is collected only on the **current** cuda device.

    Usage:
        ```python
        with LineProfiler(func) as lp:
            func
        lp.display()

        ```python
        lp = LineProfiler(func)
        lp.enable()
        func()
        lp.disable()
        lp.display()
        ```
    """

    def __init__(self, *functions, target_gpu=0, **kwargs):
        self.target_gpu = target_gpu
        self._code_infos = {}
        self._raw_line_records = []
        self.enabled = False
        for func in functions:
            self.add_function(func)

    def add_function(self, func):
        """ Record line profiling information for the given Python function.
        """
        try:
            # We need to use the hash here because pandas will later expect something
            # orderable for its index
            code_hash = hash(func.__code__)
        except AttributeError:
            import warnings
            warnings.warn(
                "Could not extract a code object for the object %r" % (func,))
            return
        if code_hash not in self._code_infos:
            first_line = inspect.getsourcelines(func)[1]
            self._code_infos[code_hash] = {
                'func': func,
                'first_line': first_line,
                'prev_line': first_line,
                'prev_record': -1,
            }

        # re-register the newer trace_callback
        if self.enabled:
            self.register_callback()

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disable()

    def register_callback(self):
        """Register the trace_callback only on demand"""
        if self._code_infos:
            sys.settrace(self._trace_callback)

    def _reset_cuda_stats(self):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

    def enable(self):
        self.enabled = True

        try:
            torch.cuda.empty_cache()
            self._reset_cuda_stats()
        except AssertionError as e:
            print('Could not reset CUDA stats and cache: ' + str(e))

        self.register_callback()

    def disable(self):
        self.enabled = False
        sys.settrace(None)

    def clear(self):
        """Clear the state of the line profiler"""
        self._code_infos = {}
        self._raw_line_records = []

    def _trace_callback(self, frame, event, arg):
        """Trace the execution of python line-by-line"""

        if event == 'call':
            return self._trace_callback

        code_hash = hash(frame.f_code)
        if event in ['line', 'return'] and code_hash in self._code_infos:
            code_info = self._code_infos[code_hash]
            with torch.cuda.device(self.target_gpu):
                self._raw_line_records.append({
                    'code_hash': code_hash,
                    'line': code_info['prev_line'],
                    'prev_record_idx': code_info['prev_record'],
                    **torch.cuda.memory_stats()})
                self._reset_cuda_stats()

            if event == 'line':
                code_info['prev_line'] = frame.f_lineno
                code_info['prev_record'] = len(self._raw_line_records)-1
            elif event == 'return':
                code_info['prev_line'] = code_info['first_line']
                code_info['prev_record'] = -1
            else:
                assert False

    def line_records(self, func=None, columns=DEFAULT_COLUMNS):
        """Get the line records

        The columns are explained in the PyTorch documentation:
        https://pytorch.org/docs/stable/cuda.html#torch.cuda.memory_stats

        Args:
            func (str): the function name of interest, None for all registered function
            columns (list of str): the column names of interest, See PyTorch's doc for available names.

        Returns:
            pd.DataFrame: a (line, statistic)-indexed dataframe of memory stats.
        """
        if len(self._raw_line_records) == 0:
            return pd.DataFrame(index=pd.MultiIndex.from_product([[], []]), columns=columns)

        line_records = _line_records(self._raw_line_records, self._code_infos)
        return _extract_line_records(line_records, func, columns)

    def display(self, func=None, columns=DEFAULT_COLUMNS):
        """Display the profiling results on either IPython or CLI

        The columns are explained in the PyTorch documentation:
        https://pytorch.org/docs/stable/cuda.html#torch.cuda.memory_stats

        .. note:: To work, this needs to be the last thing returned in the IPython statement or cell.

        Args:
            func (str): the function name of interest, None for all registered function
            columns (list of str): the column names of interest, See PyTorch's doc for available names.

        Returns:
            RecordsDisplay: Returns an object that'll display the recorded stats in the IPython console
        """
        return LineRecords(self._raw_line_records, self._code_infos).display(func, columns)

    def print_stats(self, func=None, columns=DEFAULT_COLUMNS, stream=sys.stdout):
        """Print the text profiling results to stream

        The columns are explained in the PyTorch documentation:
        https://pytorch.org/docs/stable/cuda.html#torch.cuda.memory_stats

        Args:
            func (str): the function name of interest, None for all registered function
            columns (list of str): the column names of interest, See PyTorch's doc for available names
            stream (IO-like object): the stream to write to
        """
        stream.write(repr(self.display(func, columns)))


global_line_profiler = LineProfiler()
global_line_profiler.enable()


def clear_global_line_profiler():
    """Clears the state of the global line profiler"""
    global_line_profiler.clear()


def set_target_gpu(gpu_id):
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


def profile(func, columns=DEFAULT_COLUMNS):
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


def profile_every(output_interval=1, enable=True, columns=DEFAULT_COLUMNS):
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

    def inner_decorator(func):
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
