import sys
import inspect
from functools import wraps
import torch
import pandas as pd
from .utils import readable_size

# Seaborn's `muted` color cycle
COLORS = ['#4878d0', '#ee854a', '#6acc64', '#d65f5f', '#956cb4', '#8c613c', '#dc7ec0', '#797979', '#d5bb67', '#82c6e2']

DEFAULT_COLUMNS = ['active_bytes.all.peak', 'reserved_bytes.all.peak']

def _accumulate_line_records(raw_line_records):
    """The raw records give the memory stats between successive lines executed by the profiler.
    But we want the memory stats between successive lines in our functions! The two diverge when 
    a function we're profiling calls another function we're profiling, since then Torch will have
    its peak/allocated/freed memory stats reset on each line of the called function. 

    To fix that, here we look at each line record in turn, and for peak stats we take the 
    maximum since the last record _in the same function_. For allocated/freed stats, we take the 
    sum since the last record in the same function.
    """

    # We'll do this in numpy because indexing lots of rows and columns in pandas is dog-slow. 
    raw = pd.DataFrame(raw_line_records)
    acc_mask = raw.columns.str.match(r'.*(allocated|freed)$')
    peak_mask = raw.columns.str.match(r'.*(peak)$')
    acc_raw, peak_raw = raw.loc[:, acc_mask].values, raw.loc[:, peak_mask].values
    acc_refined, peak_refined = acc_raw.copy(), peak_raw.copy()

    for row, record in enumerate(raw_line_records):
        if record['prev_record_idx'] == -1:
            # No previous data to accumulate from
            continue
        if record['prev_record_idx'] == row-1:
            # Previous record was the previous line, so no need to accumulate anything
            continue

        # Another profiled function has been called since the last record, so we need to
        # accumulate the allocated/freed/peaks of the intervening records into this one. 
        acc_refined[row] = acc_raw[record['prev_record_idx']+1:row+1].sum(0)
        peak_refined[row] = peak_raw[record['prev_record_idx']+1:row+1].max(0)

    refined = raw.copy()
    refined.loc[:, acc_mask] = acc_refined
    refined.loc[:, peak_mask] = peak_refined    
    return refined

def _line_records(raw_line_records, code_infos):
    """Converts the raw line records to a nicely-shaped dataframe whose values reflect the memory
    usage of lines of _functions_ rather than lines of _execution_. See the `_accumualte_line_records`
    docstring for more detail."""
    # Column spec: https://pytorch.org/docs/stable/cuda.html#torch.cuda.memory_stats
    qual_names = {code_hash: info['func'].__qualname__ for code_hash, info in code_infos.items()}
    records = (_accumulate_line_records(raw_line_records)
                    .assign(qual_name=lambda df: df.code_hash.map(qual_names))
                    .set_index(['qual_name', 'line'])
                    .drop(['code_hash', 'num_alloc_retries', 'num_ooms', 'prev_record_idx'], 1))
    records.columns = pd.MultiIndex.from_tuples([c.split('.') for c in records.columns])

    return records

def _subset_line_records(line_records, func=None, columns=None):
    """Extracts the subset of a line_records dataframe pertinent to a given set of functions and
    columns"""
    if func is not None:
        line_records = line_records.loc[func]

    if columns is not None: 
        columns = [tuple(c.split('.')) for c in columns]
        if not all(len(c) == 3 for c in columns):
            raise ValueError('Each column name should have three dot-separated parts')
        if not all(c in line_records.columns for c in columns):
            options = ", ".join(".".join(c) for c in line_records.columns.tolist())
            raise ValueError('The column names should be fields of torch.cuda.memory_stat(). Options are: ' + options)
        line_records = line_records.loc[:, columns]
    
    return line_records


class RecordsDisplay:
    """IPython's rich display functionality [requires we return](https://ipython.readthedocs.io/en/stable/config/integrating.html)
    an object that has a `_repr_html_` method for when HTML rendering is supported, and 
    a `__repr__` method for when only text is available"""

    def __init__(self, line_records, code_infos):
        self._line_records = line_records
        self._code_infos = code_infos

    def _line_records_merged_with_code(self):
        merged = {}
        for _, info in self._code_infos.items():
            qual_name = info['func'].__qualname__
            
            lines, start_line = inspect.getsourcelines(info['func'])
            lines = pd.DataFrame.from_dict({
                'line': range(start_line, start_line + len(lines)), 
                'code': lines})
            lines.columns = pd.MultiIndex.from_product([lines.columns, [''], ['']])
            
            merged[qual_name] = pd.merge(
                self._line_records.loc[qual_name], lines, 
                right_on='line', left_index=True, how='right')
        return merged

    def __repr__(self):
        """Renders the stats as text"""
        if len(self._line_records) == 0:
            return 'No data collected'

        is_byte_col = self._line_records.columns.get_level_values(0).str.contains('byte')
        byte_cols = self._line_records.columns[is_byte_col]

        string = {}
        for qual_name, merged in self._line_records_merged_with_code().items():
            left_align = '{{:{maxlen}s}}'.format(maxlen=max(map(len, merged.code)))
            merged[byte_cols] = merged[byte_cols].applymap(readable_size)
            merged['code'] = merged['code'].apply(lambda l: left_align.format(l.rstrip('\n\r')))
            string[qual_name] = merged.to_string(index=False)

        return '\n\n\n'.join(['## {q}\n\n{c}'.format(q=q, c=c) for q, c in string.items()])

    def _repr_html_(self):
        """Renders the stats as HTML"""
        if len(self._line_records) == 0:
            return '<p>No data collected</p>'

        is_byte_col = self._line_records.columns.get_level_values(0).str.contains('byte')
        byte_cols = self._line_records.columns[is_byte_col]
        maxes = self._line_records.max()

        html = {}
        for qual_name, merged in self._line_records_merged_with_code().items():

            style = merged.style

            # Style the bar charts
            for i, c in enumerate(self._line_records.columns):
                style = style.bar([c], color=COLORS[i % len(COLORS)], width=99, vmin=0, vmax=maxes[c])

            # Style the text
            html[qual_name] = (style
                                .format({c: readable_size for c in byte_cols})
                                .set_properties(
                                    subset=['code'], 
                                    **{'text-align': 'left', 'white-space': 'pre', 'font-family': 'monospace'})
                                .set_table_styles([{
                                    'selector': 'th', 
                                    'props': [('text-align', 'left')]}]) 
                                .hide_index()
                                .render())

        template = '<h3><span style="font-family: monospace">{q}</span></h3><div>{c}</div>'
        return '\n'.join(template.format(q=q, c=c) for q, c in html.items())


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
            warnings.warn("Could not extract a code object for the object %r" % (func,))
            return
        if code_hash not in self._code_infos:
            first_line = inspect.getsourcelines(func)[1]
            self._code_infos[code_hash] = {
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
        """Returns a (line, statistic)-indexed dataframe of memory stats.
        
        The columns are explained in the PyTorch documentation:
        
        https://pytorch.org/docs/stable/cuda.html#torch.cuda.memory_stats
        """
        if len(self._raw_line_records) == 0:
            return pd.DataFrame(index=pd.MultiIndex.from_product([[], []]), columns=columns)

        line_records = _line_records(self._raw_line_records, self._code_infos)
        return _subset_line_records(line_records, func, columns)

    def display(self, func=None, columns=DEFAULT_COLUMNS):
        """Returns an object that'll display the recorded stats in the IPython console.

        The columns are explained in the PyTorch documentation:
        
        https://pytorch.org/docs/stable/cuda.html#torch.cuda.memory_stats
        
        To work, this needs to be the last thing returned in the IPython statement or cell.
        """ 
        return RecordsDisplay(self.line_records(func, columns), self._code_infos)

    def print_stats(self, func=None, columns=DEFAULT_COLUMNS, stream=sys.stdout):
        stream.write(repr(self.display(func, columns)))

    def print_func_stats(self, func=None, columns=DEFAULT_COLUMNS, stream=sys.stdout):
        return self.print_stats(func, columns, stream)

global_line_profiler = LineProfiler()
global_line_profiler.enable()

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
        global_line_profiler.print_func_stats(func, columns)
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
                    global_line_profiler.print_func_stats(func, columns)
                func.cur_idx += 1
            return res

        return run_func
    return inner_decorator

