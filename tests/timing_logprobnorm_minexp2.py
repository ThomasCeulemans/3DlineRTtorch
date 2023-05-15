import torch
import math
import torch.utils.benchmark as benchmark

N = 4000000
device = 'cpu'
data = torch.rand(N, device=device)
#data that will be added every 10 indices
##scatter_add
def logprob_norm_test(data):
    m = torch.distributions.Normal(torch.zeros(1, device=device), torch.ones(1, device=device))
    return m.log_prob(data).exp()

def full_formula(data):
    return 1.0/math.sqrt(2.0*torch.pi) * torch.exp(-(data**2)/2)

num_threads = torch.get_num_threads()


t_logprob = benchmark.Timer(
    stmt='logprob_norm_test(data)',
    setup='from __main__ import logprob_norm_test',
    num_threads=num_threads,
    globals={'data': data})

t_fullform = benchmark.Timer(
    stmt='full_formula(data)',
    setup='from __main__ import full_formula',
    num_threads=num_threads,
    globals={'data': data})

print(t_logprob.timeit(100))
print(t_fullform.timeit(100))
# print(data)
# print(logprob_norm_test(data))
# print(full_formula(data))
print(logprob_norm_test(data)-full_formula(data))

#conclusion: computing it yourself is faster by a factor 2-3
#probably some setup cost involved in using that probability class
#gpu computing only seems worth it starting from 4 million data points
#(cpu) in cache vs latency on gpu