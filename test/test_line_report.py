import torch

from pytorch_memlab.line_report import LineProfiler, profile


def test_line_report():

    def work():
        # comment
        linear = torch.nn.Linear(100, 100).cuda()
        linear_2 = torch.nn.Linear(100, 100).cuda()
        linear_3 = torch.nn.Linear(100, 100).cuda()

    def work_3():
        lstm = torch.nn.LSTM(1000, 1000).cuda()

    def work_2():
        # comment
        linear = torch.nn.Linear(100, 100).cuda()
        linear_2 = torch.nn.Linear(100, 100).cuda()
        linear_3 = torch.nn.Linear(100, 100).cuda()
        work_3()

    line_profiler = LineProfiler(work, work_2)
    line_profiler.enable()

    work()
    work_2()

    line_profiler.disable()
    # line_profiler.print_stat()

def test_line_report_decorator():

    @profile(output_interval=3)
    def work():
        # comment
        linear = torch.nn.Linear(100, 100).cuda()
        linear_2 = torch.nn.Linear(100, 100).cuda()
        linear_3 = torch.nn.Linear(100, 100).cuda()

    work()
    work()
    work()
