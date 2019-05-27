import pytest
import torch

from pytorch_memlab import LineProfiler, profile, profile_every


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
    line_profiler.print_stats()

def test_line_report_decorator():

    @profile_every(output_interval=3)
    def work():
        # comment
        linear = torch.nn.Linear(100, 100).cuda()
        linear_2 = torch.nn.Linear(100, 100).cuda()
        linear_3 = torch.nn.Linear(100, 100).cuda()

    @profile_every(output_interval=1)
    def work2():
        # comment
        linear = torch.nn.Linear(100, 100).cuda()
        linear_2 = torch.nn.Linear(100, 100).cuda()
        linear_3 = torch.nn.Linear(100, 100).cuda()
    work()
    work2()
    work()
    work()

def test_line_report_method():
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(100, 100).cuda()
            self.drop = torch.nn.Dropout(0.1)

        @profile_every(1)
        def forward(self, inp):
            return self.drop(self.linear(inp))

    net = Net()
    inp = torch.Tensor(50, 100).cuda()
    net(inp)

def test_line_report_profile():

    @profile
    def work():
        # comment
        linear = torch.nn.Linear(100, 100).cuda()
        linear_2 = torch.nn.Linear(100, 100).cuda()
        linear_3 = torch.nn.Linear(100, 100).cuda()

    work()
    work()

def test_line_report_profile_interrupt():

    @profile
    def work():
        # comment
        linear = torch.nn.Linear(100, 100).cuda()
        linear_2 = torch.nn.Linear(100, 100).cuda()
        linear_3 = torch.nn.Linear(100, 100).cuda()

    @profile_every(1)
    def work2():
        linear_2 = torch.nn.Linear(100, 100).cuda()
        linear_3 = torch.nn.Linear(100, 100).cuda()

    work()
    work2()
    raise KeyboardInterrupt
