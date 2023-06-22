import numpy as np
import torch

def create_corrupt_2d_circular(h, w, radius=3):
    def corrupt_2d_circular(seed):
        # The center can be anywhere in the grid, but we minus the radius + 2 because we do not want to cross the border (which can cause indexing issue)
        with torch.no_grad():
            seed = torch.clone(seed)
            center = (
                np.random.randint(radius + 2, w - (radius + 2)),
                np.random.randint(radius + 2, h - (radius + 2)),
            )

            Y, X = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

            mask = dist_from_center <= radius
            seed[:, mask] *= 0.0
            return seed
    
    return corrupt_2d_circular