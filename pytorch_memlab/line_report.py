import os
import sys
import inspect
import gc
from collections import defaultdict

import torch
from .utils import readable_size

class LineProfiler:
    """ Time the execution of lines of Python code.
    """

    def __init__(self, *functions):
        self.functions = []
        self.code_map = {}
        self.last_time = {}
        self.enable_count = 0
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

    def __enter__(self):
        self.enable_by_count()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disable_by_count()

    def enable(self):
        sys.settrace(self.trace_callback)

    def disable(self):
        self.last_time = {}
        sys.settrace(None)

    def trace_callback(self, frame, event, arg):

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
            line_stat[lineno].append((allocated_memory, cached_memory))
        return

    def print_stat(self):
        """Print the stat of each functions
        """
        for code, stat in self.code_map.items():
            show_func(
                filename=code.co_filename,
                trace_stat=stat,
            )


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
    is_first = True
    d = {}
    # .items ensure the returned tuple is sorted by key (lineno)
    for lineno, line_stat in trace_stat['line_stat'].items():
        all_allocated_memory = [ls[0] for ls in line_stat]
        all_cached_memory = [ls[1] for ls in line_stat]
        max_allocated = max(all_allocated_memory)
        max_cached = max(all_cached_memory)
        if is_first:
            is_first = False
        else:
            d[real_lineno] = (
                readable_size(max_allocated),
                readable_size(max_cached),
                readable_size(max_allocated - prev_max_allocated),
                readable_size(max_cached - prev_max_cached),
            )
        real_lineno = lineno
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
        show_line_stat = d.get(lineno, empty)
        txt = template % (lineno, *show_line_stat,
                          line.rstrip('\n').rstrip('\r'))
        stream.write(txt)
        stream.write("\n")
    stream.write("\n")
