import gc
import torch
def courtesy():
    print('before', torch.cuda.memory_allocated())
    tensors = [obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor)]
    loc_map = {}
    for t in tensors:
        # in case tensors appear more than once

        if id(t) not in loc_map:
            loc_map[id(t)] = (t, t.device)
        else:
            print('redundant tensor encountered')

        t.data = t.data.cpu()

        # parameters have one more wrapper for .data
        if type(t) is torch.nn.Parameter:
            t.grad = t.grad.cpu()

    print('after', torch.cuda.memory_allocated())
    # from mem_report import mem_report
    # mem_report()

    for obj_id, (t, device) in loc_map.items():
        t.data = t.data.to(device)
        if type(t) is torch.nn.Parameter:
            t.grad = t.grad.to(device)

    print('restored', torch.cuda.memory_allocated())
    torch.cuda.empty_cache()

    import time
    time.sleep(50)

