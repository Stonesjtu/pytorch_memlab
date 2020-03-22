from IPython.core.magic import (
    Magics,
    magics_class,
    line_cell_magic,
    needs_local_scope,
)
from IPython.core.magic_arguments import magic_arguments, argument, parse_argstring
from .line_profiler import LineProfiler
from tempfile import mkstemp


class UsageError(Exception):
    pass


@magics_class
class MemlabMagics(Magics):
    @magic_arguments()
    @argument('--function', '-f', metavar='FUNC',
              action='append',
              default=[],
              help="""
              Function to profile. Can be specified multiple times
              """)
    @argument('statement', nargs='*', default=None, help="""
              Code to run under profiler.
              You can omit this in cell magic mode.
              """)
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
            except Exception as e:
                raise UsageError('Could not find function {!r}.\n{}: {}'.format(
                    name, e.__class__.__name__, e)
                )
        profiler = LineProfiler(*funcs)
        if cell is not None:
            code = cell
        else:
            assert args.statement is not None
            code = "\n".join(args.statement)
        with profiler:
            exec(compile(code, filename='<ipython>', mode='exec'), local_ns)

        profiler.print_stats()


def load_ipython_extension(ipython):
    ipython.register_magics(MemlabMagics)
