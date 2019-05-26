import torch

from pytorch_memlab import Courtesy, MemReporter

def test_reporter():
    linear = torch.nn.Linear(1024, 1024).cuda()
    inp = torch.Tensor(512, 1024).cuda()

    out = linear(inp).mean()
    out.backward()

    reporter = MemReporter(linear)
    reporter.report()
    ct = Courtesy()
    ct.yield_memory()
    print('gpu>>>>>>>>>>>>>>>>>>cpu')
    reporter.report()
    ct.restore()
    print('cpu>>>>>>>>>>>>>>>>>>gpu')
    reporter.report()

def test_courtesy_context():
    linear = torch.nn.Linear(1024, 1024).cuda()
    inp = torch.Tensor(512, 1024).cuda()

    out = linear(inp).mean()
    out.backward()

    reporter = MemReporter(linear)
    with Courtesy() as ct:
        print('gpu>>>>>>>>>>>>>>>>>>cpu')
        reporter.report()
