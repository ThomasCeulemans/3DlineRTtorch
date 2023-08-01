#checking how much slower random indexing is compared to normal indexing
#hmm, up to 100x slower

import torch
import torch.utils.benchmark as benchmark

N = 1000000
device = "cuda"#"cpu"

vectorA = torch.randn(N, device = device)
vectorB = torch.randn(N, device = device)

indexes = torch.randperm(N)

# vectorResLoop = torch.zeros(N, device = device)
vectorResAdd = torch.zeros(N, device = device)


# def loop():
#     #note: this for loop is trivial and should be parallelized automatically
#     for i in range(N):
#         vectorResLoop[i] = vectorA[i] + vectorB[i]

#     return vectorResLoop

def add():
    vectorResAdd = vectorA + vectorB
    return vectorResAdd

def add_random(indexes):
    vectorResAdd[indexes] = vectorA[indexes] + vectorB[indexes]
    return vectorResAdd


print(add_random(indexes))
print(add())

num_threads = torch.get_num_threads()

t_randsum = benchmark.Timer(
    stmt='add_random(indexes)',
    setup='from __main__ import add_random',
    num_threads=num_threads,
    globals={'indexes': indexes, 'vectorA': vectorA, 'vectorB': vectorB})

t_sum = benchmark.Timer(
    stmt='add()',
    setup='from __main__ import add',
    num_threads=num_threads,
    globals={'vectorA': vectorA, 'vectorB': vectorB})


print("trandsum:", t_randsum.timeit(100))
print("tsum: ", t_sum.timeit(100))