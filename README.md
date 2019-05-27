pytorch memlab
======

A **CUDA** memory management laboratory for pytorch

Why I wrote this
-----

Out-Of-Memory errors in pytorch happen frequently, for new-bees and
experienced programmers. A common reason is that most people don't really
learn the underlying memory management philosophy of pytorch and GPUs.
They wrote memory in-efficient codes and complained about pytorch eating too
much CUDA memory.

In this repo, I'm going to share some useful tools to help debugging OOM, or
to inspect the underlying mechanism if anyone is interested in.


### Memory Profiler

The memory profiler is a modification of python's `line_profiler`, it gives
the memory usage info for each line of code in the specified function/method.

#### Sample:

```python
from pytorch_memlab import profile
@profile
def work():
    linear = torch.nn.Linear(100, 100).cuda()
    linear2 = torch.nn.Linear(100, 100).cuda()
    linear3 = torch.nn.Linear(100, 100).cuda()

```

After the script finishes or interrupted by keyboard, it gives the following
profiling info.

```
Function: work at line 71

Line # Max usage   Peak usage diff max diff peak  Line Contents
===============================================================
71                                               @profile
72                                               def work():
73                                                   # comment
74   885.00K        1.00M   40.00K    0.00B          linear = torch.nn.Linear(100, 100).cuda()
75   925.00K        1.00M   40.00K    0.00B          linear_2 = torch.nn.Linear(100, 100).cuda()
76   965.00K        1.00M   40.00K    0.00B          linear_3 = torch.nn.Linear(100, 100).cuda()

```

terminology:
  - `Max usage`: the max of pytorch's allocated memory (the finish memory)
  The memory usage after this line is executed.
  - `Peak usage`: the max of pytorch's cached memory (the peak memory)
  The peak memory usage during the execution of this line. (Probably) pytorch
  caches 1M CUDA memory as atomic, so the cached memory is unchanged in the
  sample above.
  - `diff max`: the `Max memory` usage difference caused by this line
  - `diff peak`: the `Peak memory` usage difference caused by this line


If you use `profile` decorator, the memory statistics are collected during
multiple runs and only the maximum one is displayed at the end.
We also provide a more flexible API called `profile_every` which prints the
memory info every *N* times of function execution. You can simply replace
`@profile` with `@profile_every(1)` to print the memory usage for each 
execution.

The `@profile` and `@profile_every` can also be mixed to gain more control
of the debugging granularity.

- You can also add the decorator in the module class:

```python
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
    @profile
    def forward(self, inp):
        #do_something
```

More samples can be found in `test/test_line_profiler.py`


### Memory Reporter

As *Memory Profiler* only gives the overall memory usage information by lines,
a more low-level memory usage information can be obtained by *Memory Reporter*.

*Memory reporter* iterates all the `Tensor` objects and gets the underlying
`Storage` object to get the actual memory usage instead of the surface
`Tensor.size`.

#### Sample

A minimal one:

```python
linear = torch.nn.Linear(1024, 1024).cuda()
reporter = MemReporter()
reporter.report()
```

```
Element type                                            Size  Used MEM
-------------------------------------------------------------------------------
Storage on cuda:0
Parameter0                                      (1024, 1024)     4.00M
Parameter1                                           (1024,)     4.00K
-------------------------------------------------------------------------------
Total Tensors: 1049600  Used Memory: 4.00M
The allocated memory on cuda:0: 4.00M
-------------------------------------------------------------------------------
```

You can also pass in a model object for automatically name inference.

```python
linear = torch.nn.Linear(1024, 1024).cuda()
inp = torch.Tensor(512, 1024).cuda()
# pass in a model to automatically infer the tensor names
reporter = MemReporter(linear)
out = linear(inp).mean()
print('========= before backward =========')
reporter.report()
out.backward()
print('========= after backward =========')
reporter.report()
```

```
========= before backward =========
Element type                                            Size  Used MEM
-------------------------------------------------------------------------------
Storage on cuda:0
weight                                          (1024, 1024)     4.00M
bias                                                 (1024,)     4.00K
Tensor0                                          (512, 1024)     2.00M
Tensor1                                                 (1,)   512.00B
-------------------------------------------------------------------------------
Total Tensors: 1573889  Used Memory: 6.00M
The allocated memory on cuda:0: 6.00M
-------------------------------------------------------------------------------
========= after backward =========
Element type                                            Size  Used MEM
-------------------------------------------------------------------------------
Storage on cuda:0
weight                                          (1024, 1024)     4.00M
weight.grad                                     (1024, 1024)     4.00M
bias                                                 (1024,)     4.00K
bias.grad                                            (1024,)     4.00K
Tensor0                                          (512, 1024)     2.00M
Tensor1                                                 (1,)   512.00B
-------------------------------------------------------------------------------
Total Tensors: 2623489  Used Memory: 10.01M
The allocated memory on cuda:0: 10.01M
-------------------------------------------------------------------------------
```


The reporter automatically deals with the sharing weights parameters:

```python
linear = torch.nn.Linear(1024, 1024).cuda()
linear2 = torch.nn.Linear(1024, 1024).cuda()
linear2.weight = linear.weight
container = torch.nn.Sequential(
    linear, linear2
)
inp = torch.Tensor(512, 1024).cuda()
# pass in a model to automatically infer the tensor names

out = container(inp).mean()
out.backward()

# verbose shows how storage is shared across multiple Tensors
reporter = MemReporter(container)
reporter.report(verbose=True)
```

- Results:
```
Element type                                            Size  Used MEM
-------------------------------------------------------------------------------
Storage on cuda:0
0.weight                                        (1024, 1024)     4.00M
0.weight.grad                                   (1024, 1024)     4.00M
0.bias                                               (1024,)     4.00K
0.bias.grad                                          (1024,)     4.00K
1.bias                                               (1024,)     4.00K
1.bias.grad                                          (1024,)     4.00K
Tensor0                                          (512, 1024)     2.00M
Tensor1                                                 (1,)   512.00B
-------------------------------------------------------------------------------
Total Tensors: 2625537  Used Memory: 10.02M
The allocated memory on cuda:0: 10.02M
-------------------------------------------------------------------------------
```

You can better understand the memory layout for more complicated module:
```python
lstm = torch.nn.LSTM(1024, 1024).cuda()
reporter = MemReporter(lstm)
reporter.report(verbose=True)
inp = torch.Tensor(10, 10, 1024).cuda()
out, _ = lstm(inp)
out.mean().backward()
reporter.report(verbose=True)
```
- As shown below, the `(->)` indicates the re-use of the same storage back-end

```
Element type                                            Size  Used MEM
-------------------------------------------------------------------------------
Storage on cuda:0
weight_ih_l0                                    (4096, 1024)    32.03M
weight_hh_l0(->weight_ih_l0)                    (4096, 1024)     0.00B
bias_ih_l0(->weight_ih_l0)                           (4096,)     0.00B
bias_hh_l0(->weight_ih_l0)                           (4096,)     0.00B
Tensor0                                       (10, 10, 1024)   400.00K
-------------------------------------------------------------------------------
Total Tensors: 8499200  Used Memory: 32.42M
The allocated memory on cuda:0: 32.52M
Memory differs due to the matrix alignment
-------------------------------------------------------------------------------
Element type                                            Size  Used MEM
-------------------------------------------------------------------------------
Storage on cuda:0
weight_ih_l0                                    (4096, 1024)    32.03M
weight_ih_l0.grad                               (4096, 1024)    32.03M
weight_hh_l0(->weight_ih_l0)                    (4096, 1024)     0.00B
weight_hh_l0.grad(->weight_ih_l0.grad)          (4096, 1024)     0.00B
bias_ih_l0(->weight_ih_l0)                           (4096,)     0.00B
bias_ih_l0.grad(->weight_ih_l0.grad)                 (4096,)     0.00B
bias_hh_l0(->weight_ih_l0)                           (4096,)     0.00B
bias_hh_l0.grad(->weight_ih_l0.grad)                 (4096,)     0.00B
Tensor0                                       (10, 10, 1024)   400.00K
Tensor1                                       (10, 10, 1024)   400.00K
Tensor2                                        (1, 10, 1024)    40.00K
Tensor3                                        (1, 10, 1024)    40.00K
-------------------------------------------------------------------------------
Total Tensors: 17018880         Used Memory: 64.92M
The allocated memory on cuda:0: 65.11M
Memory differs due to the matrix alignment
-------------------------------------------------------------------------------
```


NOTICE:
> When forwarding with `grad_mode=True`, pytorch maintains tensor buffers for
> future Back-Propagation, in C level. So these buffers are not going to be
> managed or collected by pytorch. But if you store these intermediate results
> as python variables, then they will be reported.

##### A failed example due to pytorch's C side tensor buffers

In the following example, a temp buffer is created at `inp * (inp + 2)` to 
store both `inp` and `inp + 2`, unfortunately python only knows the existence
of inp, so we have *2M* memory lost, which is the same size of Tensor `inp`.

```python
linear = torch.nn.Linear(1024, 1024).cuda()
inp = torch.Tensor(512, 1024).cuda()
# pass in a model to automatically infer the tensor names
reporter = MemReporter(linear)
out = linear(inp * (inp + 2)).mean()
reporter.report()
```

output:

```
Element type                                            Size  Used MEM
-------------------------------------------------------------------------------
Storage on cuda:0
weight                                          (1024, 1024)     4.00M
bias                                                 (1024,)     4.00K
Tensor0                                          (512, 1024)     2.00M
Tensor1                                                 (1,)   512.00B
-------------------------------------------------------------------------------
Total Tensors: 1573889  Used Memory: 6.00M
The allocated memory on cuda:0: 8.00M
Memory differs due to the matrix alignment or invisible gradient buffer tensors
-------------------------------------------------------------------------------
```


### Courtesy

### ACK

I suffered a lot debugging weird memory usage during my 3-years of developing
efficient Deep Learning models, and of course learned a lot from the great
open source community.
