import lineRTtorch.geometry.cube_geometry as geom
import torch

print(geom)
cube = geom.CubeGeometry(torch.zeros(3), torch.ones(3))

print(cube.map_rays_indices(6))
