import torch
from pytorch_memlab import MemReporter

import pytest


concentrate_mode = False

def test_reporter():
    linear = torch.nn.Linear(1024, 1024)
    inp = torch.Tensor(512, 1024)
    reporter = MemReporter(linear)

    out = linear(inp*(inp+3)).mean()
    reporter.report()
    out.backward()

    reporter.report()

def test_reporter_without_model():
    linear = torch.nn.Linear(1024, 1024)
    inp = torch.Tensor(512, 1024)
    reporter = MemReporter()

    out = linear(inp*(inp+3)).mean()
    reporter.report()
    out.backward()

    reporter.report()

def test_reporter_sparse_tensor():
    emb = torch.nn.Embedding(1024, 1024, sparse=True)
    inp = torch.arange(0, 128)
    reporter = MemReporter()

    out = emb(inp).mean()
    reporter.report()
    out.backward()
    b = emb.weight.grad * 2

    reporter.report()

@pytest.mark.skipif(concentrate_mode, reason='concentrate')
def test_reporter_tie_weight():
    linear = torch.nn.Linear(1024, 1024)
    linear_2 = torch.nn.Linear(1024, 1024)
    linear_2.weight = linear.weight
    container = torch.nn.Sequential(
        linear, linear_2
    )
    reporter = MemReporter(container)
    inp = torch.Tensor(512, 1024)

    out = container(inp).mean()
    out.backward()

    reporter = MemReporter(container)
    reporter.report()

@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
@pytest.mark.skipif(concentrate_mode, reason='concentrate')
def test_reporter_LSTM():
    lstm = torch.nn.LSTM(256, 256, num_layers=1).cuda()
    # lstm.flatten_parameters()
    inp = torch.Tensor(256, 256, 256).cuda()
    out, _ = lstm(inp)
    out.mean().backward()

    reporter = MemReporter(lstm)
    reporter.report()

@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
@pytest.mark.skipif(concentrate_mode, reason='concentrate')
def test_reporter_device():
    lstm_cpu = torch.nn.LSTM(256, 256)
    lstm = torch.nn.LSTM(256, 256, num_layers=1).cuda()
    # lstm.flatten_parameters()
    inp = torch.Tensor(256, 256, 256).cuda()
    out, _ = lstm(inp)
    out.mean().backward()

    reporter = MemReporter(lstm)
    reporter.report()
    reporter.report(device=torch.device('cuda:0'))
