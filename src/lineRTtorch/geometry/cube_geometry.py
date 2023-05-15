import torch

class CubeGeometry:
    def __init__(self, minxyz, maxxyz):
        self.nlines = 1
        self.N = 4
        self.x = torch.linspace(minxyz[0],maxxyz[0], self.N)
        self.y = torch.linspace(minxyz[1],maxxyz[1], self.N)
        self.z = torch.linspace(minxyz[2],maxxyz[2], self.N)
        # self.grid_x, self.grid_y, self.grid_z = torch.meshgrid(x, y, z, indexing='ij')#implicitly, we use these positions, but I have already defined the conversion from linear index to x,y,z indices
        self.v = torch.zeros([self.N**3,3])#hmm, might also use function as description instead
        self.rho = torch.zeros([self.N**3,self.nlines])
        self.turb = 1.0e-6 * torch.ones([self.N**3]) #TODO: actually compute



    def position(self, index):
        x_ind, y_ind, z_ind = self.linear_index_to_xyz(self, index)
        return self.x[x_ind], self.y[y_ind], self.z[z_ind]
    
    def velocity(self, index):
        return self.v[index, :]
    
    #returns the density per line
    def rho(self, index):
        return self.rho[index, :]
    
    #in units of the speed of light
    def turb(self, index):
        return self.turb[index]
    
    #note: very limited ray directions to start with
    # @param [in] dir:(int) 
    def map_rays_indices(self, dir):
        #creates range of range list
        #probably slow and ugly, but it works
        rrange = torch.cat([torch.arange(temp) for temp in torch.arange(1, self.N+1)])
        lengths = torch.arange(1, self.N+1)
        xyrange = torch.arange(self.N)
        #TODO: later support for domain decomposition: add option to only use a part of the xygrid
        x_grid, y_grid = torch.meshgrid(xyrange, xyrange, indexing='ij')
        interleaved_x_grid = torch.flatten(x_grid).repeat_interleave(rrange.nelement())
        interleaved_y_grid = torch.flatten(y_grid).repeat_interleave(rrange.nelement())
        repeated_rrange = rrange.repeat(x_grid.nelement())
        repeated_lengths = lengths.repeat(x_grid.nelement())
        # print(repeated_rrange)
        print(rrange)
        print(xyrange)
        print(x_grid)
        print(repeated_rrange)
        print(interleaved_x_grid)
        print(interleaved_y_grid)
        match dir:
            case 1:
                return (self.xyz_to_linear_index(repeated_rrange, interleaved_x_grid, interleaved_y_grid), repeated_lengths)
            case 2:
                return (self.xyz_to_linear_index(self.N-1-repeated_rrange, interleaved_x_grid, interleaved_y_grid), repeated_lengths)
            case 3:
                return (self.xyz_to_linear_index(interleaved_y_grid, repeated_rrange, interleaved_x_grid), repeated_lengths)
            case 4:
                return (self.xyz_to_linear_index(interleaved_y_grid, self.N-1-repeated_rrange, interleaved_x_grid), repeated_lengths)
            case 5:
                return (self.xyz_to_linear_index(interleaved_x_grid, interleaved_y_grid, repeated_rrange), repeated_lengths)
            case 6:
                return (self.xyz_to_linear_index(interleaved_x_grid, interleaved_y_grid, self.N-1-repeated_rrange), repeated_lengths)
            case _:
                raise RuntimeError("wrong way of defining direction") 


        #should return (per point): what the different rays are: other points on the ray O(N) and index per ray
        #thus results in N^4 storage cost; hmm, might be a bit much to return at once (assuming 8 bytes per data point, we get 8*128^4= 2Gb for a relative small cube 128^3)
        #thus I might also need to ask the max size and then compute in batches (also return bool (for when mapped entire and int for what the next index to map would be)
        #TODO: make sure that this list is sorted based on the path length (for shortchar)
        # return indices
    

    def linear_index_to_xyz(self, index):
        x = index/self.N**2
        y = (index/self.N) % self.N
        z = index % self.N
        return (x, y, z)

    def xyz_to_linear_index(self, x, y, z):
        print(x*self.N**2 + y*self.N + z)
        return x*self.N**2 + y*self.N + z

    #TODO: if time permits later on, try out comoving style solver for more speedup
    #def map_rays_comoving
