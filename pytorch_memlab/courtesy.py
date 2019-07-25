import gc
import torch


class Courtesy():
    """A class to yield CUDA memory at any time in the training

    The whole save/load is a bit tricky because all data transfer should
    be inplace operation and gradient agnostic
    """
    def __init__(self):
        self.loc_map = {}

    def yield_memory(self):
        """Transfer all the CUDA tensors into CPU memory"""
        tensors = [obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor)]
        for t in tensors:
            # in case tensors appear more than once
            if t not in self.loc_map:
                self.loc_map[t] = t.device

            t.data = t.data.cpu()
            # parameters have one more wrapper for .data
            if isinstance(t, torch.nn.Parameter):
                # sometimes Parameter does not have grad
                try:
                    t.grad.data = t.grad.cpu()
                finally:
                    pass
        torch.cuda.empty_cache()

    def restore(self):
        """Restore the tensors into original CUDA devices"""
        for t, device in self.loc_map.items():
            t.data = t.data.to(device)
            if isinstance(t, torch.nn.Parameter):
                # sometimes Parameter does not have grad
                try:
                    t.grad = t.grad.to(device)
                finally:
                    pass
        self.loc_map.clear()

    def __enter__(self):
        self.yield_memory()
        return self

    def __exit__(self, *args):
        self.restore()
