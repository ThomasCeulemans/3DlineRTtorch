import torch
import torch.utils.benchmark as benchmark

N = 100000
addlen = 50
device = 'cuda'
data = torch.randn(addlen*N, device=device)
#data that will be added every 10 indices
##scatter_add
def scatter_add_test(data):
    scatter_ind = torch.arange(N, device=device).repeat_interleave(addlen)
    scatter_result = torch.zeros(N, device=device).scatter_add(0, scatter_ind, data)
    return scatter_result

def cumsum_test(data):
    cumsum = torch.cat((torch.zeros(1, device=device),torch.cumsum(data, 0)),0)#and a zero in front
    cumsum_ind = addlen*torch.arange(N+1, device=device)
    cumsum_part = cumsum[cumsum_ind]
    cumsum_result = cumsum_part[1:]-cumsum_part[:-1]
    return cumsum_result

num_threads = torch.get_num_threads()


t_scatter = benchmark.Timer(
    stmt='scatter_add_test(data)',
    setup='from __main__ import scatter_add_test',
    num_threads=num_threads,
    globals={'data': data})

t_cumsum = benchmark.Timer(
    stmt='cumsum_test(data)',
    setup='from __main__ import cumsum_test',
    num_threads=num_threads,
    globals={'data': data})

print(t_scatter.timeit(100))
print(t_cumsum.timeit(100))

#conclusion: cumsum is faster by a factor 2
#not that it probably matters that much, as we will only use it to do partial cumsum in a portion
#I expect that computing gaussian will take way longer