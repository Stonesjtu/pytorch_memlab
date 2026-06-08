import weakref

import torch
import torch.optim
from pytorch_memlab import MemReporter
from pytorch_memlab.utils import readable_size

import pytest


concentrate_mode = False

def test_readable_size_with_nan():
    assert readable_size(float('nan')) == ''

def test_readable_size_with_symbolic_byte_size(monkeypatch):
    class SymbolicSize:
        def __format__(self, spec):
            raise TypeError('unsupported format string')

        def __str__(self):
            return '4M<ByteSize amount=s0>'

    monkeypatch.setattr('pytorch_memlab.utils.calmsize', lambda num_bytes: SymbolicSize())

    assert readable_size(object()) == '4M<ByteSize amount=s0>'

def test_reporter_ignores_dead_weakrefs(monkeypatch):
    class WeakrefTarget:
        pass

    target = WeakrefTarget()
    dead_proxy = weakref.proxy(target)
    del target

    tensor = torch.empty(1)
    monkeypatch.setattr(
        'pytorch_memlab.mem_reporter.gc.get_objects',
        lambda: [dead_proxy, tensor],
    )

    reporter = MemReporter()
    reporter.report()

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

def test_reporter_with_optimizer():
    linear = torch.nn.Linear(1024, 1024)
    inp = torch.Tensor(512, 1024)
    optimizer = torch.optim.Adam(linear.parameters())
    # reporter = MemReporter(linear)

    out = linear(inp*(inp+3)*(inp+2)).mean()
    reporter = MemReporter(linear)
    reporter.report()
    out.backward()
    # reporter.report()
    optimizer.step()

    reporter.add_optimizer(optimizer)
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
