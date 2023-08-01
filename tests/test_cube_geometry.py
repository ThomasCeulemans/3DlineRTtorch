import lineRTtorch.geometry.cube_geometry as geom
import lineRTtorch.radiative_transfer.solver as rt_solv
import torch

print(geom)
cube = geom.CubeGeometry(torch.zeros(3), torch.ones(3))

indices, lengths = cube.map_rays_indices(1)
# optdepth = rt_solv.compute_intensity(cube.rho[indices,0], torch.zeros(indices.size()), 0, lengths)
optdepth = rt_solv.compute_intensity(cube, indices, lengths)
print(optdepth)
