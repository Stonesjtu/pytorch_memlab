import sys
import inspect

import pandas as pd
import torch
from .utils import readable_size

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
        lp.print_stats()

        lp = LineProfiler(func)
        lp.enable()
        func()
        lp.disable()
        lp.print_stats()
        ```
    """

    def __init__(self, *functions, target_gpu=0, **kwargs):
        self.target_gpu = target_gpu
        self.codes = {}
        self._records = []
        self.enabled = False
        for func in functions:
            self.add_function(func)

    def add_function(self, func):
        """ Record line profiling information for the given Python function.
        """
        try:
            codehash = hash(func.__code__)
        except AttributeError:
            import warnings
            warnings.warn("Could not extract a code object for the object %r" % (func,))
            return
        if codehash not in self.codes:
            self.codes[codehash] = {'line': -1, 'func': func}

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
        if self.codes:
            sys.settrace(self.trace_callback)

    def enable(self):
        self.enabled = True
        self.register_callback()

    def disable(self):
        self.enabled = False
        sys.settrace(None)

    def trace_callback(self, frame, event, arg):
        """Trace the execution of python line-by-line"""

        if event == 'call':
            return self.trace_callback

        codehash = hash(frame.f_code)
        if event in ['line', 'return'] and codehash in self.codes:
            last_line = self.codes[codehash]['line']
            
            with torch.cuda.device(self.target_gpu):
                self._records.append({
                    'codehash': codehash, 
                    'line': last_line,
                    **torch.cuda.memory_stats(self.target_gpu)})
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()

            lineno = last_line + 1 if event == 'return' else frame.f_lineno
            self.codes[codehash]['line'] = lineno

        return

    def records(self):
        # Column spec: https://pytorch.org/docs/stable/cuda.html#torch.cuda.memory_stats
        qualnames = {codehash: info['func'].__qualname__ for codehash, info in self.codes.items()}
        records = (pd.DataFrame(self._records)
                        .assign(qualname=lambda df: df.codehash.map(qualnames))
                        .set_index(['qualname', 'line'])
                        .drop(['codehash', 'num_alloc_retries', 'num_ooms'], 1))
        records.columns = pd.MultiIndex.from_tuples([c.split('.') for c in records.columns])
        return records

    def print(self, columns=[('active_bytes', 'all', 'peak'), ('reserved_bytes', 'all', 'peak')]):
        if len(self._records) == 0:
            print('No data collected.')
            return

        formatted = self.records().loc[:, columns].applymap(readable_size)

        for codehash, info in self.codes.items():
            qualname = info['func'].__qualname__
            
            lines, startline = inspect.getsourcelines(info['func'])
            maxlen = max(map(len, lines))
            lines = pd.DataFrame.from_dict({
                        'line': range(startline, startline+len(lines)),
                        '': [f'{{:{maxlen}s}}'.format(l.rstrip('\n\r')) for l in lines]})
            lines.columns = pd.MultiIndex.from_product([lines.columns, [''], ['']])
            
            content = pd.merge(formatted.loc[qualname], lines, right_on='line', left_index=True, how='right').fillna('')
            
            print(content.to_string(index=False, col_space=8))
            print('\n')


global_line_profiler = LineProfiler()
global_line_profiler.enable()

def profile(func, **kwargs):
    """Profile the CUDA memory usage of target function line by line

    The profiling results will be printed at exiting, KeyboardInterrupt raised.
    The CUDA memory is collected only on the **current** cuda device.

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
        global_line_profiler.print(func, **kwargs)
    atexit.register(print_stats_atexit)

    return func


