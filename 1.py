# from utils.plots import plot_results
#
# plot_results(file="", dir="runs")

# import math
# print(-math.log((1 - 1e-5) / 1e-5))

import torch
fg_mask = torch.zeros((2, 5))
fg_mask[0, 0] = 1
fg_mask[1, 0] = 1
print(fg_mask)
bbox = torch.randn(2, 4)
print(bbox)

target = torch.where(fg_mask[..., None].repeat(1, 1, 4) == 1, bbox[:, None], 
                    torch.zeros((1, 1, 4), dtype=fg_mask.dtype, device=fg_mask.device))
print(target)
print(target.shape)
