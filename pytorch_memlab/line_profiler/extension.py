"""IPython & notebook extension interface"""
from IPython.core.magic import (
    Magics,
    magics_class,
    line_cell_magic,
    needs_local_scope,
)
from IPython.core.magic_arguments import magic_arguments, argument, parse_argstring

from .line_profiler import LineProfiler, DEFAULT_COLUMNS


class UsageError(Exception):
    pass


@magics_class
class MemlabMagics(Magics):
    @magic_arguments()
    @argument('--function',
              '-f',
              metavar='FUNC',
              action='append',
              default=[],
              help="""Function to profile. Can be specified multiple times to profile multiple
                   functions""")
    @argument('--column',
              '-c',
              metavar='COLS',
              action='append',
              default=[],
              help="""Columns to display. Can be specified multiple times to profile multiple
                   functions. See the Torch CUDA spec at
                   https://pytorch.org/docs/stable/cuda.html#torch.cuda.memory_stats for details.""")
    @argument('-D',
              '--no_default_columns',
              action='store_true',
              help='Hide the default columns of ' + ", ".join(DEFAULT_COLUMNS))
    @argument('-r',
              '--return-profiler',
              action='store_true',
              help='Return LineProfiler object for introspection')
    @argument('-g',
              '--gpu',
              metavar='GPU_ID',
              default=0,
              type=int,
              help='Profile memory usage of this GPU')
    @argument('-q',
              '--quiet',
              action='store_true',
              help='Don\'t print out profile results')
    @argument('statement',
              nargs='*',
              default=None,
              help='Code to run under profiler. You can omit this in cell magic mode.')
    @argument('-T',
              '--dump-profile',
              metavar='OUTPUT',
              help='Dump text profile output to file')
    @line_cell_magic
    @needs_local_scope
    def mlrun(self, line=None, cell=None, local_ns=None):
        """Execute a statement/cell under the PyTorch Memlab profiler to collect CUDA memory
        allocation information on a per-line basis.
        """
        args = parse_argstring(self.mlrun, line)
        global_ns = self.shell.user_global_ns

        funcs = []
        for name in args.function:
            try:
                fn = eval(name, global_ns, local_ns)
                funcs.append(fn)
            except NameError as e:
                raise UsageError('Could not find function {!r}.\n{}: {}'.format(
                    name, e.__class__.__name__, e)
                )
        profiler = LineProfiler(*funcs, target_gpu=args.gpu)
        if cell is not None:
            code = cell
        else:
            assert args.statement is not None
            code = '\n'.join(args.statement)
        with profiler:
            exec(compile(code, filename='<ipython>', mode='exec'), local_ns)

        if args.dump_profile is not None:
            with open(args.dump_profile, 'w') as f:
                profiler.print_stats(stream=f)

        if args.return_profiler:
            return profiler
        else:
            defaults = [] if args.no_default_columns else list(DEFAULT_COLUMNS)
            return profiler.display(columns=defaults + args.column)


def load_ipython_extension(ipython):
    ipython.register_magics(MemlabMagics)
