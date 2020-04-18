import sys
import inspect
import numpy as np
import pandas as pd
import torch
from .utils import readable_size
from IPython.display import HTML, display

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
        self._codes = {}
        self._raw = []
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
        if codehash not in self._codes:
            first_line = inspect.getsourcelines(func)[1]
            self._codes[codehash] = {
                'func': func, 
                'first_line': first_line,
                'prev_line': first_line, 
                'prev_record': -1}

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
        if self._codes:
            sys.settrace(self._trace_callback)

    def _reset_cuda_stats(self):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

    def enable(self):
        self.enabled = True

        torch.cuda.empty_cache()
        self._reset_cuda_stats()

        self.register_callback()

    def disable(self):
        self.enabled = False
        sys.settrace(None)

    def _trace_callback(self, frame, event, arg):
        """Trace the execution of python line-by-line"""

        if event == 'call':
            return self._trace_callback

        codehash = hash(frame.f_code)
        if event in ['line', 'return'] and codehash in self._codes:
            codeinfo = self._codes[codehash]
            with torch.cuda.device(self.target_gpu):
                self._raw.append({
                    'codehash': codehash, 
                    'line': codeinfo['prev_line'],
                    'prev': codeinfo['prev_record'],
                    **torch.cuda.memory_stats(self.target_gpu)})
                self._reset_cuda_stats()

            if event == 'line':
                codeinfo['prev_line'] = frame.f_lineno
                codeinfo['prev_record'] = len(self._raw)-1
            elif event == 'return':
                codeinfo['prev_line'] = codeinfo['first_line']
                codeinfo['prev_record'] = -1
            else:
                assert False

    def _refined(self):
        # The records are line-by-line, but the stats we want to report are over periods.
        # So we need to accumulate some stuff.
        # Peak stats are the maximum since `prev`
        # Allocated/freed stats are the sum since `prev` 

        # We'll do this in numpy because indexing lots of rows and columns in pandas is dog-slow. 
        raw = pd.DataFrame(self._raw)
        acc_mask = raw.columns.str.match(r'.*(allocated|freed)$')
        peak_mask = raw.columns.str.match(r'.*(peak)$')
        acc_raw, peak_raw = raw.loc[:, acc_mask].values, raw.loc[:, peak_mask].values
        acc_refined, peak_refined = acc_raw.copy(), peak_raw.copy()

        for i, r in enumerate(self._raw):
            if r['prev'] == -1:
                continue
            if r['prev'] == i-1:
                continue
            acc_refined[i] = acc_raw[r['prev']+1:i+1].sum(0)
            peak_refined[i] = peak_raw[r['prev']+1:i+1].max(0)

        refined = raw.copy()
        refined.loc[:, acc_mask] = acc_refined
        refined.loc[:, peak_mask] = peak_refined    
        return refined

    def records(self):
        # Column spec: https://pytorch.org/docs/stable/cuda.html#torch.cuda.memory_stats
        qualnames = {codehash: info['func'].__qualname__ for codehash, info in self._codes.items()}
        records = (self._refined()
                        .assign(qualname=lambda df: df.codehash.map(qualnames))
                        .set_index(['qualname', 'line'])
                        .drop(['codehash', 'num_alloc_retries', 'num_ooms'], 1))
        records.columns = pd.MultiIndex.from_tuples([c.split('.') for c in records.columns])
        return records

    def print_stats(self, columns=['active_bytes.all.peak', 'reserved_bytes.all.peak'], func=None, stream=None):
        if len(self._raw) == 0:
            print('No data collected.')
            return

        records = self.records().groupby(axis=0, level=[0, 1]).max()
        columns = [tuple(c.split('.')) for c in columns]
        assert all(len(c) == 3 for c in columns), 'Each column name should have three dot-separated parts'
        assert all(c in records.columns for c in columns), 'The column names should come from torch.cuda.memory_stat()\'s output'
        records = records.loc[:, columns]
        if func:
            records = records.loc[[func]]

        bytecols = records.columns[records.columns.get_level_values(0).str.contains('byte')]
        maxes = records.max()

        chunks = []
        for codehash, info in self._codes.items():
            qualname = info['func'].__qualname__
            
            lines, startline = inspect.getsourcelines(info['func'])
            lines = pd.DataFrame.from_dict({'line': range(startline, startline+len(lines)), 'code': lines})
            lines.columns = pd.MultiIndex.from_product([lines.columns, [''], ['']])
            
            content = pd.merge(records.loc[qualname], lines, right_on='line', left_index=True, how='right')
            
            style = content.style
            for c in records.columns:
                style = style.bar([c], color='#5fba7d', width=99, vmin=0, vmax=maxes[c])
            chunk = (style
                        .format({c: readable_size for c in bytecols})
                        .set_properties(subset=['code'], **{'text-align': 'left', 'white-space': 'pre', 'font-family': 'monospace'})
                        .set_table_styles([dict(selector='th', props=[('text-align', 'left')])]) 
                        .render())
            chunks.append((qualname, chunk))

        template = '<h3><span style="font-family: monospace">{q}</span></h3><div>{c}</div>'
        html = '\n'.join(template.format(q=q, c=c) for q, c in chunks)
        if stream is None:
            display(HTML())
        else:
            stream.write(html)

    def print_func_stats(self, stats, func, **kwargs):
        return self.print_stats(func=func, **kwargs)



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


