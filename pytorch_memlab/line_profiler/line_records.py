"""Class and helper functions for processing and displaying line records"""
import inspect
from typing import Callable, Optional, Tuple, List, Dict, Any
import pandas as pd

from ..utils import readable_size


COLORS = [
    '#4878d0', '#ee854a', '#6acc64', '#d65f5f', '#956cb4',
    '#8c613c', '#dc7ec0', '#797979', '#d5bb67', '#82c6e2',
]



def _accumulate_line_records(raw_line_records: List[Dict[str, Any]]) -> pd.DataFrame:
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


def _line_records(raw_line_records: List[Dict[str, Any]], code_infos: List[Dict[str, Any]]) -> pd.DataFrame:
    """Converts the raw line records to a nicely-shaped dataframe whose values reflect
    the memory usage of lines of _functions_ rather than lines of _execution_. See the
    `_accumulate_line_records` docstring for more detail."""
    # Column spec: https://pytorch.org/docs/stable/cuda.html#torch.cuda.memory_stats
    qual_names = {
        code_hash: info['func'].__qualname__ for code_hash, info in code_infos.items()}
    records = (_accumulate_line_records(raw_line_records)
               .assign(qual_name=lambda df: df.code_hash.map(qual_names))
               .set_index(['qual_name', 'line'])
               .drop(['code_hash', 'num_alloc_retries', 'num_ooms', 'prev_record_idx'], 1))
    records.columns = pd.MultiIndex.from_tuples(
        [c.split('.') for c in records.columns])

    return records


class LineRecords:
    """Class for processing raw line records and display on IPython & CLI
    """

    def __init__(self, raw_line_records: List[Dict[str, Any]], code_infos: List[Dict[str, Any]]):
        super().__init__()
        self._raw_line_records = raw_line_records
        self._code_infos = code_infos

    def display(self, func: Callable[..., Any], columns: Tuple[str, ...]):
        """Display the records to either notebook or CLI

        The columns are explained in the PyTorch documentation:
        https://pytorch.org/docs/stable/cuda.html#torch.cuda.memory_stats

        .. note:: Make this call the last one in a notebook cell

        Args:
            func (str): the function name of interest, None for all registered function
            columns (list of str): the column names of interest, See PyTorch's doc for available names.

        Returns:
            RecordsDisplay: a IPython friendly object which converts records to HTML or plain text
        """
        line_records = self._filter_raw_line_records(func, columns)
        return RecordsDisplay(line_records, self._code_infos)

    def _filter_raw_line_records(self, func: Callable[..., Any], columns: Tuple[str, ...]) -> pd.DataFrame:
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
        line_records = _extract_line_records(line_records, func, columns)

        if len(line_records) > 0:
            line_records = line_records.groupby(level=[0, 1]).max()

        return line_records


def _extract_line_records(line_records: LineRecords, func: Optional[Callable] = None, columns: Tuple[str, ...] = None):
    """Extracts the subset of a line_records dataframe pertinent to a given set of functions and
    columns"""
    if func is not None:
        # Support both passing the function directly and passing a qual name/list of qual names
        line_records = line_records.loc[[func.__qualname__] if callable(func) else func]

    if columns is not None:
        columns = [tuple(c.split('.')) for c in columns]
        if not all(len(c) == 3 for c in columns):
            raise ValueError('Each column name should have three dot-separated parts')
        if not all(c in line_records.columns for c in columns):
            options = ", ".join(".".join(c)
                                for c in line_records.columns.tolist())
            raise ValueError(
                'The column names should be fields of torch.cuda.memory_stat(). Options are: ' + options)
        line_records = line_records.loc[:, columns]

    return line_records


class RecordsDisplay:
    """Class for processing raw line records and display on IPython & CLI

    IPython's rich display functionality [requires we return](https://ipython.readthedocs.io/en/stable/config/integrating.html)
    an object that has a `_repr_html_` method for when HTML rendering is supported, and
    a `__repr__` method for when only text is available
    """
    def __init__(self, line_records: LineRecords, code_infos: List[Dict[str, Any]]):
        super().__init__()
        self._line_records = line_records
        self._code_infos = code_infos
        self._merged_line_records = self._merge_line_records_with_code()

    def _merge_line_records_with_code(self) -> Dict[str, Any]:
        merged_records = {}
        for _, info in self._code_infos.items():
            qual_name = info['func'].__qualname__
            if qual_name in self._line_records.index.get_level_values(0):
                lines, start_line = inspect.getsourcelines(info['func'])
                lines = pd.DataFrame.from_dict({
                    'line': range(start_line, start_line + len(lines)),
                    'code': lines})
                lines.columns = pd.MultiIndex.from_product([lines.columns, [''], ['']])

                merged_records[qual_name] = pd.merge(
                    self._line_records.loc[qual_name], lines,
                    right_on='line', left_index=True, how='right')
        return merged_records

    def __repr__(self):
        """Renders the stats as text"""
        if len(self._line_records) == 0:
            return 'No data collected\n'

        is_byte_col = self._line_records.columns.get_level_values(0).str.contains('byte')
        byte_cols = self._line_records.columns[is_byte_col]

        string = {}
        for qual_name, merged in self._merge_line_records_with_code().items():
            maxlen = max(len(c) for c in merged.code)
            left_align = '{{:{maxlen}s}}'.format(maxlen=maxlen)
            merged[byte_cols] = merged[byte_cols].applymap(readable_size)

            # This is a mess, but I can't find any other way to left-align text strings.
            code_header = (left_align.format('code'), '', '')
            merged[code_header] = merged['code'].apply(lambda l: left_align.format(l.rstrip('\n\r')))
            merged = merged.drop('code', 1, level=0)

            string[qual_name] = merged.to_string(index=False)

        return '\n\n'.join(['## {q}\n\n{c}\n'.format(q=q, c=c) for q, c in string.items()])

    def _repr_html_(self):
        """Renders the stats as HTML"""
        if len(self._line_records) == 0:
            return '<p>No data collected</p>'

        is_byte_col = self._line_records.columns.get_level_values(0).str.contains('byte')
        byte_cols = self._line_records.columns[is_byte_col]
        maxes = self._line_records.max()

        html = {}
        for qual_name, merged in self._merge_line_records_with_code().items():

            style = merged.style

            # Style the bar charts
            for i, c in enumerate(self._line_records.columns):
                style = style.bar([c], color=COLORS[i % len(COLORS)],
                                  width=99, vmin=0, vmax=maxes[c])

            # Style the text
            html[qual_name] = (style
                                .format({c: readable_size for c in byte_cols})
                                .set_properties(
                                    subset=['code'], **{
                                        'text-align': 'left',
                                        'white-space': 'pre',
                                        'font-family': 'monospace'})
                                .set_table_styles([{
                                    'selector': 'th',
                                    'props': [('text-align', 'left')]}])
                                .hide_index()
                                .render())

        template = '<h3><span style="font-family: monospace">{q}</span></h3><div>{c}</div>'
        return '\n'.join(template.format(q=q, c=c) for q, c in html.items())
