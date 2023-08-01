#figure out whether the old magritte code structure (giant parallelized for loops) still works fine-ish when replacing the datastructures and ignoring the for loops
#Apparently (to noone's surprise), the looping thing is way slower. This is possibly due to torch just copying the data all the time

import torch
import torch.utils.benchmark as benchmark


N = 100000
device = "cuda"#"cpu"

vectorA = torch.randn(N, device = device)
vectorB = torch.randn(N, device = device)

vectorResLoop = torch.zeros(N, device = device)
vectorResAdd = torch.zeros(N, device = device)


def loop():
    #note: this for loop is trivial and should be parallelized automatically
    for i in range(N):
        vectorResLoop[i] = vectorA[i] + vectorB[i]

    return vectorResLoop

def add():
    vectorResAdd = vectorA + vectorB
    return vectorResAdd


# print(loop())
print(add())

num_threads = 256#torch.get_num_threads()

# t_loop = benchmark.Timer(
#     stmt='loop()',
#     setup='from __main__ import loop',
#     num_threads=num_threads,
#     globals={'vectorA': vectorA, 'vectorB': vectorB})

t_sum = benchmark.Timer(
    stmt='add()',
    setup='from __main__ import add',
    num_threads=num_threads,
    globals={'vectorA': vectorA, 'vectorB': vectorB})


# print("tloop:", t_loop.timeit(10))
print("tsum: ", t_sum.timeit(100000))