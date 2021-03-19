import inspect
import sys
from types import FrameType
import warnings
from typing import Any, Callable, Optional, Tuple

import torch

from .line_records import LineRecords

# Seaborn's `muted` color cycle
DEFAULT_COLUMNS = ('active_bytes.all.peak', 'reserved_bytes.all.peak')


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

    def __init__(self, *functions: Callable, target_gpu: int = 0):
        self.target_gpu = target_gpu
        self._code_infos = {}
        self._raw_line_records = []
        self.enabled = False
        for func in functions:
            self.add_function(func)

    def add_function(self, func: Callable) -> None:
        """ Record line profiling information for the given Python function.
        """
        try:
            # We need to use the hash here because pandas will later expect something
            # orderable for its index
            code_hash = hash(func.__code__)
        except AttributeError:
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
        """Enable the profiler and register trace callback"""
        if not torch.cuda.is_available():
            print('Could not find CUDA deivces and reset CUDA stats and cache')
            return
        torch.cuda.empty_cache()
        self._reset_cuda_stats()
        self.enabled = True
        self.register_callback()

    def disable(self):
        """Disable the profiler and clear trace callback"""
        self.enabled = False
        sys.settrace(None)

    def clear(self):
        """Clear the state of the line profiler"""
        self._code_infos = {}
        self._raw_line_records = []

    def _trace_callback(self, frame: FrameType, event: str, _unused_arg: Tuple[Any, ...]):
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

    def display(self, func: Optional[Callable] = None, columns: Tuple[str, ...] = DEFAULT_COLUMNS) -> LineRecords:
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

    def print_stats(self, func: Optional[Callable] = None, columns: Tuple[str, ...] = DEFAULT_COLUMNS, stream=sys.stdout):
        """Print the text profiling results to stream

        The columns are explained in the PyTorch documentation:
        https://pytorch.org/docs/stable/cuda.html#torch.cuda.memory_stats

        Args:
            func (str): the function name of interest, None for all registered function
            columns (list of str): the column names of interest, See PyTorch's doc for available names
            stream (IO-like object): the stream to write to
        """
        stream.write(str(self.display(func, columns)))
