import os
import sys
import inspect
from collections import defaultdict
from functools import wraps

import torch
from .utils import readable_size

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

    def __init__(self, *functions):
        self.functions = []
        self.code_map = {}
        self.enabled = False
        for func in functions:
            self.add_function(func)

    def add_function(self, func):
        """ Record line profiling information for the given Python function.
        """
        try:
            code = func.__code__
        except AttributeError:
            import warnings
            warnings.warn("Could not extract a code object for the object %r" % (func,))
            return
        if code not in self.code_map:
            self.code_map[code] = {}
            self.code_map[code]['line_stat'] = defaultdict(list)
            self.code_map[code]['func'] = func  # probable memory leak if holding this ref
            self.code_map[code]['func_name'] = func.__name__
            self.functions.append(func)
            self.code_map[code]['source_code'] = inspect.getsourcelines(func)
            self.code_map[code]['last_lineno'] = -1

        # re-register the newer trace_callback
        if self.enabled:
            self.register_callback()

    def __enter__(self):
        self.enable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disable()

    def register_callback(self):
        """Register the trace_callback only on demand"""
        if self.functions:
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

        if event in ['line', 'return'] and frame.f_code in self.code_map:
            line_stat = self.code_map[frame.f_code]['line_stat']
            allocated_memory = torch.cuda.memory_allocated()
            cached_memory = torch.cuda.memory_cached()
            torch.cuda.empty_cache()
            if event == 'return':
                lineno = max(line_stat.keys()) + 1
            else:
                lineno = frame.f_lineno
            last_lineno = self.code_map[frame.f_code]['last_lineno']
            line_stat[last_lineno].append((allocated_memory, cached_memory))
            self.code_map[frame.f_code]['last_lineno'] = lineno
        return

    def print_stats(self):
        """Print the stat of each functions
        """
        for code, stat in self.code_map.items():
            show_func(
                filename=code.co_filename,
                trace_stat=stat,
            )

    def print_func_stats(self, func):
        """Print the stat of a registered function"""
        code = func.__code__
        if code in self.code_map:
            show_func(
                filename=code.co_filename,
                trace_stat=self.code_map[code],
            )


global_line_profiler = LineProfiler()
global_line_profiler.enable()

def profile_every(output_interval=1, enable=True):
    """Profile the CUDA memory usage of target function line by line

    Prints the profiling output every `output_interval` execution of the target
    function
    The CUDA memory is collected only on the **current** cuda device.

    Args:
        - func: the function or method to profile on
        - enable: whether to enable the profiling mode, so users don't have to
        modify any source code for enabling and disabling profiling.
        - output interval: frequency of output the profiling results
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
                    global_line_profiler.print_func_stats(func)
                func.cur_idx += 1
            return res

        return run_func
    return inner_decorator

def profile(func):
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
        global_line_profiler.print_func_stats(func)
    atexit.register(print_stats_atexit)

    return func

def show_func(filename, trace_stat, stream=None):
    """ Show results for a single function.
    """
    if stream is None:
        stream = sys.stdout

    template = '%6s %9s %12s %8s %8s  %-s'
    func_name = trace_stat['func_name']

    linenos = list(trace_stat['line_stat'].keys())
    start_lineno = trace_stat['source_code'][1]

    if os.path.exists(filename):
        stream.write("File: %s\n" % filename)
        stream.write("Function: %s at line %s\n" % (func_name, start_lineno))
        sublines = trace_stat['source_code'][0]
    else:
        stream.write("\n")
        stream.write("Could not find file %s\n" % filename)
        stream.write("Are you sure you are running this program from the same directory\n")
        stream.write("that you ran the profiler from?\n")
        stream.write("Continuing without the function's contents.\n")
        # Fake empty lines so we can see the timings, if not the code.
        nlines = max(linenos) - min(min(linenos), start_lineno) + 1
        sublines = [''] * nlines

    prev_max_allocated = 0
    prev_max_cached = 0
    lineno_mem = {}
    # .items ensure the returned tuple is sorted by key (lineno)
    for lineno, line_stat in trace_stat['line_stat'].items():
        all_allocated_memory = [ls[0] for ls in line_stat]
        all_cached_memory = [ls[1] for ls in line_stat]
        max_allocated = max(all_allocated_memory)
        max_cached = max(all_cached_memory)

        # the first line_stat is for the very beginning of function
        if lineno != -1:
            lineno_mem[lineno] = (
                readable_size(max_allocated),
                readable_size(max_cached),
                readable_size(max_allocated - prev_max_allocated),
                readable_size(max_cached - prev_max_cached),
            )

        prev_max_allocated = max_allocated
        prev_max_cached = max_cached

    linenos = range(start_lineno, start_lineno + len(sublines))
    empty = ('', '', '', '')
    header = template % ('Line #', 'Max usage', 'Peak usage', 'diff max', 'diff peak',
                         'Line Contents')
    stream.write("\n")
    stream.write(header)
    stream.write("\n")
    stream.write('=' * len(header))
    stream.write("\n")
    for lineno, line in zip(linenos, sublines):
        show_line_stat = lineno_mem.get(lineno, empty)
        max_usage, peak_usage, diff_max, diff_peak = show_line_stat
        txt = template % (lineno, max_usage, peak_usage, diff_max, diff_peak,
                          line.rstrip('\n').rstrip('\r'))
        stream.write(txt)
        stream.write("\n")
    stream.write("\n")
    stream.flush()
