import torch
from pytorch_memlab import MemReporter

import pytest


concentrate_mode = False

def test_reporter():
    linear = torch.nn.Linear(1024, 1024).cuda()
    inp = torch.Tensor(512, 1024).cuda()

    out = linear(inp).mean()
    out.backward()

    reporter = MemReporter(linear)
    reporter.report()

@pytest.mark.skipif(concentrate_mode, reason='concentrate')
def test_reporter_tie_weight():
    linear = torch.nn.Linear(1024, 1024).cuda()
    linear_2 = torch.nn.Linear(1024, 1024).cuda()
    linear_2.weight = linear.weight
    container = torch.nn.Sequential(
        linear, linear_2
    )
    reporter = MemReporter(container)
    inp = torch.Tensor(512, 1024).cuda()

    out = container(inp).mean()
    out.backward()

    reporter = MemReporter(container)
    reporter.report()

@pytest.mark.skipif(concentrate_mode, reason='concentrate')
def test_reporter_LSTM():
    lstm = torch.nn.LSTM(256, 256, num_layers=1).cuda()
    # lstm.flatten_parameters()
    inp = torch.Tensor(256, 256, 256).cuda()
    out, _ = lstm(inp)
    out.mean().backward()

    reporter = MemReporter(lstm)
    reporter.report()
