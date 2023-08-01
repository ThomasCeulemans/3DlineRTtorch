import torch
import time
import torch.nn.functional as F

#for simplicity of setup, assume no doppler shifting yet
#note: this thing requires quite a bit of memory Ndatapoints*Nfreqs*Nlinefreqs
#timing-wise, it is fine, but a sparser version would be nice; (however, in worst case, every line will lie close to every freq will be needed; so memory cost still the same)
N = 500000 #number points
Nfreqs = 31
Nlinefreqs = 51
min_freq = 2.5
max_freq = 3.5
target_freqs = torch.linspace(min_freq, max_freq, steps=Nfreqs).view(-1,1)


density = torch.arange(N*Nfreqs*Nlinefreqs)

# minlinefreqs = torch.Tensor([1.0,2.0,3.0,4.0,5.0]).repeat(N).view(1,-1)
# maxlinefreqs = torch.Tensor([3.0,4.0,5.0,6.0,7.0]).repeat(N).view(1,-1)
minlinefreqs = torch.linspace(1.0, 5.0, steps = Nlinefreqs).repeat(N).view(1,-1)
maxlinefreqs = torch.linspace(3.0, 7.0, steps = Nlinefreqs).repeat(N).view(1,-1)
print(minlinefreqs)

# target_freq = 11.5* torch.ones(N*5) #so resulting indices should be: 1,2


time_condition_start = time.time()
indices_condition = torch.logical_and((minlinefreqs < target_freqs),(target_freqs < maxlinefreqs)).flatten()
print(indices_condition)
time_condition_end = time.time()
density[indices_condition]
print(indices_condition)

#output interpretation: indices_condition.type==bool: [[[for each line] for each point ] for each freq]
# so quite simple to transform density to correct structure: just .repeat(nfreqs)


# #data should be in correct format: 
# #inspired by https://discuss.pytorch.org/t/find-the-nearest-value-in-the-list/73772/6
# def pytorch_search_lower_bound(lower_bounds, searchvals):
#     dim0_size = lower_bounds.size(dim=0)
#     flip_data = lower_bounds.flip(dims=(1,))
#     tmp_feat = F.relu(searchvals-flip_data)
#     print("temp_fun:", tmp_feat)
#     mat_tmp = torch.min(tmp_feat, dim=1)
#     print(mat_tmp)
#     return mat_tmp[1]

# def pytorch_search_upper_bound(upper_bounds, searchvals):
#     tmp_feat = 1/(1+F.relu(searchvals-upper_bounds))
#     print("temp fun upper: ", tmp_feat)
#     mat_tmp = torch.max(tmp_feat, dim=1)
#     print(mat_tmp[1])
#     return mat_tmp[1]

# min_reshape = minlinefreqs.reshape(N,5)
# max_reshape = maxlinefreqs.reshape(N,5)
# target_freq_reshape = target_freq.reshape(N,5)

# time_parallel_search_start = time.time()
# print("lower bounds:",pytorch_search_lower_bound(min_reshape, target_freq_reshape))
# print("upper bounds:", pytorch_search_upper_bound(max_reshape, target_freq_reshape))
# time_parallel_search_end = time.time()


print("time condition:", time_condition_end-time_condition_start)
print(density[indices_condition])

print(indices_condition)
