import torch

from pytorch_memlab.line_report import LineProfiler


def test_line_report():

    def work():
        # comment
        linear = torch.nn.Linear(100, 100).cuda()
        linear_2 = torch.nn.Linear(100, 100).cuda()
        linear_3 = torch.nn.Linear(100, 100).cuda()
        return None

    def work_2():
        # comment
        def work_3():
            lstm = torch.nn.LSTM(1000, 1000).cuda()
        linear = torch.nn.Linear(100, 100).cuda()
        linear_2 = torch.nn.Linear(100, 100).cuda()
        linear_3 = torch.nn.Linear(100, 100).cuda()
        work_3()
        return None

    line_profiler = LineProfiler(work, work_2)
    line_profiler.enable()

    work()
    work_2()

    line_profiler.disable()
    print(line_profiler.code_map)
    line_profiler.print_stat()

