#extremely simple raytracing test: every point only has a single neighbor conveniently in the right direction
# Just to test whether this could be parallelized in ideal circumstances 
import torch
import time

Raylength = 100
Nrays = 500000
neighbor_copies = 50
# device = "cpu"
device = "cuda"

neighbors= torch.arange(1, Raylength+1, device=device).type(torch.int64)
neighbors[-1] = 0
neighbors=neighbors.repeat_interleave(neighbor_copies)
self_ids = torch.arange(Raylength, device=device).type(torch.int64).repeat_interleave(neighbor_copies)
extended_neighbors = torch.stack((self_ids, neighbors), dim=1).flatten()
equilength_neighbors = extended_neighbors
equilength_neighbors[-1] = 0 #to keep in bounds
neighbors_length = 2*neighbor_copies
equilength_help = torch.arange(neighbors_length, device=device).view(1,-1)
cum_neighbors_equilength = neighbors_length*torch.arange(Raylength, device=device).type(torch.int)
print("equilength neighbors: ", cum_neighbors_equilength)
extended_neighbors = extended_neighbors[:-2]


print("extended neighbors",extended_neighbors)
nneighbors = 2*torch.ones(Raylength, device=device).type(torch.int32)
nneighbors[-1] = 1
print("nneighbors", nneighbors)
cumnneighbors = torch.cumsum(nneighbors, dim=0) - nneighbors[0]
print("cumnneighbors", cumnneighbors)

print(extended_neighbors)
distance = torch.arange(0, Raylength, device=device).type(torch.float32)
ydistance = 5.0*torch.ones(Raylength, device=device).type(torch.float32)
print(neighbors)
is_boundary_point = torch.zeros(Raylength, dtype=torch.bool, device=device)
is_boundary_point[-1]=True
start_ids = torch.randint(low=0, high=Raylength, size=(Nrays,), device=device).type(torch.int32)
start_ids_clone=start_ids.clone()
print("start", start_ids)
print("bdy", is_boundary_point)


def trace_rays_equilength_neighbors(start_ids):
    curr_ids = start_ids.type(torch.int64)
    Nraystotrace = start_ids.size()
    mask_active_rays = torch.ones(Nraystotrace, dtype=torch.bool)
    print("start ids", start_ids)
    print("start while loop", torch.all(mask_active_rays).item())
    while (torch.any(mask_active_rays).item()):
        #check neighbors using a matrix for all possibilities (possibly includes too much)
        neighbors_to_check = cum_neighbors_equilength.gather(0, curr_ids[mask_active_rays]).view(-1,1) + equilength_help
        # neighbors_to_check = cum_neighbors_equilength.gather(0, curr_ids).view(-1,1) + equilength_help
        neighbor_indices = torch.index_select(equilength_neighbors, 0, neighbors_to_check.flatten())
        reshaped_neighbors = neighbor_indices.reshape(-1, neighbors_length)

        #get distances; TODO: replace with actual computation
        distancesx = distance[reshaped_neighbors]
        distancesy = ydistance[reshaped_neighbors]

        # wrongdirection = distancesx <= distance[curr_ids].view(-1,1)
        wrongdirection = distancesx <= distance[curr_ids[mask_active_rays]].view(-1,1)
        #add penalty to wrong direction, making sure that it is never the correct point;
        #note: assumes distances are never 0
        distancesy += wrongdirection.type(torch.float) * torch.max(distancesy, 1)[0].view(-1,1)

        #get next index for each point
        selectidx = torch.min(distancesy, 1)[1]
        #and do some indexing nonsense, as we want to grab one value per point
        nextidx = reshaped_neighbors.gather(1, selectidx.unsqueeze(1)).squeeze(1)
        curr_ids[mask_active_rays] = nextidx
        # curr_ids = nextidx

        print(curr_ids)

        #FIXME: 
        # curr_lin_neighbors = torch.index_select(extended_neighbors, dim=0, index = cumnneighbors[curr_ids]:cumnneighbors[curr_ids]+nneighbors[curr_ids]]
        # next_points = torch.max(distance)
        # curr_ids[mask_active_rays] = neighbors[curr_ids[mask_active_rays]]

        mask_active_rays = torch.logical_not(is_boundary_point[curr_ids])
        # mask_inactive_rays = is_boundary_point[]

        # TODO: add data here in an arcane manner without appending
    return

def trace_rays_until_everything_stops(start_ids):
    #for all rays simultaneously, we try to find the boundary, without doing anything to 
    curr_ids = start_ids.type(torch.int64)
    next_ids = curr_ids
    Nraystotrace = start_ids.size()
    mask_active_rays = torch.ones(Nraystotrace, dtype=torch.bool)
    print("start while loop", torch.all(mask_active_rays).item())
    while (torch.any(mask_active_rays).item()):
        print(mask_active_rays)
        print()
        neighbor_indices = torch.index_select()
        # curr_lin_neighbors = torch.index_select(extended_neighbors, dim=0, index = cumnneighbors[curr_ids]:cumnneighbors[curr_ids]+nneighbors[curr_ids]]
        # next_points = torch.max(distance)
        curr_ids[mask_active_rays] = neighbors[curr_ids[mask_active_rays]]

        #can't subscript mask with itself to ?save? minor amount of time
        mask_active_rays = torch.logical_not(is_boundary_point[curr_ids])
        # mask_inactive_rays = is_boundary_point[]

        # TODO: add data here in an arcane manner without appending
    return


def trace_rays_dense_neighbors(start_ids_numpy, neighbors, nneighbors, cumnneighbors):
    #adapted from https://stackoverflow.com/questions/64004559/is-there-multi-arange-in-numpy
    #in this a[:,0] is the start, a[:,1] is the stop, a[:,2] is the step size (not necessary for me)
    # def multi_arange(a):
    #     steps = a[:,2]
    #     lens = ((a[:,1]-a[:,0]) + steps-np.sign(steps))//steps
    #     b = np.repeat(steps, lens)
    #     ends = (lens-1)*steps + a[:,0]
    #     b[0] = a[0,0]
    #     b[lens[:-1].cumsum()] = a[1:,0] - ends[:-1]
    #     return b.cumsum()
    
    def multi_arange(start, delta):
        lens = delta
        #increment will contain the increments required
        increment = torch.ones(torch.sum(delta, dim=0), device=device).type(torch.int)
        end = delta-1 + start
        increment[0] = start[0]
        increment[delta[:-1].cumsum(dim = 0)] = start[1:]-end[:-1]
        return increment.cumsum(dim = 0)
    
    # print("neighbors", neighbors,"nneighbors",  nneighbors, "cumnneighbors", cumnneighbors)
    

    curr_ids = start_ids.type(torch.int64)
    Nraystotrace = start_ids.size()
    mask_active_rays = torch.ones(Nraystotrace, dtype=torch.bool)
    print("start ids", start_ids)
    print("start while loop", torch.all(mask_active_rays).item())
    while (torch.any(mask_active_rays).item()):
        masked_curr_ids = curr_ids[mask_active_rays]
        masked_nneighbors = nneighbors[masked_curr_ids]
        mask_size = masked_curr_ids.size(dim=0)
        indices_to_check = multi_arange(torch.gather(cumnneighbors, 0, masked_curr_ids), torch.gather(nneighbors, 0, masked_curr_ids))
        neighbors_to_check = neighbors[indices_to_check]

        #get distances; TODO: replace with actual computation
        distancesx = torch.gather(distance, 0, neighbors_to_check)
        distancesy = torch.gather(ydistance, 0, neighbors_to_check)

        wrongdirection = distancesx <= distance[torch.repeat_interleave(masked_curr_ids, masked_nneighbors)]


        #add penalty to wrong direction, making sure that it is never the correct point;
        #note: assumes distances are never 0
        #but first get the maximum distance in y direction per slice?; err, is penalty; adding too high penalty doesn't matter
        # print("pre penalty dist y", distancesy)
        distancesy += wrongdirection.type(torch.float) * torch.max(distancesy, 0)[0]

        scatter_ids = torch.repeat_interleave(torch.arange(mask_size, device=device).type(torch.int64), masked_nneighbors)

        tempstuff = torch.zeros(mask_size, device=device)
        # print("temp stuff", tempstuff)
        minydists_per_point = tempstuff.scatter_reduce(0, scatter_ids, distancesy, reduce="amin", include_self=False)#TODO: technically, we can reuse x dist at this point
        #broadcast these minimal distances once more, using gather
        minydists = minydists_per_point.gather(0, scatter_ids)
        minindices = torch.nonzero(minydists == distancesy).flatten()#torch.nonzero likes to transpose the matrix for some reason
        corresp_scatter_ids = torch.gather(scatter_ids, 0, minindices)

        #searching results in finding the first index corresponding to that number; depending on the options chosen
        index_of_index = torch.searchsorted(corresp_scatter_ids, torch.arange(mask_size, device=device).type(torch.int64))
        nextidx = minindices[index_of_index]

        # #option 2: use unique_consecutive
        # __, counts = torch.unique_consecutive(corresp_scatter_ids, return_counts=True)
        # index_of_index = torch.cumsum(counts, dim=0).roll(1)
        # index_of_index[0] = 0
        # nextidx = minindices[index_of_index]

        

        curr_ids[mask_active_rays] = neighbors_to_check[nextidx]
        # print("neighbors_to_check", neighbors_to_check)
        print(curr_ids)

        mask_active_rays = torch.logical_not(is_boundary_point[curr_ids])

    return 

# start = time.time()
# trace_rays_equilength_neighbors(start_ids)
# end = time.time()
# print("total time", end-start)
# trace_rays_until_everything_stops(start_ids)

start = time.time()
# start_ids = torch.zeros(Nrays, device=device).type(torch.int)
trace_rays_dense_neighbors(start_ids_clone, equilength_neighbors, 2*neighbor_copies*torch.ones(Raylength, device=device).type(torch.int), cum_neighbors_equilength)
end = time.time()
print("total time dense", end-start)