import warnings
from einops import rearrange
import matplotlib.pyplot as plt
import torch
import numpy as np

class NCAGrid(object):
  def __init__(self,
               num_target_channels,
               num_static_channels,
               num_hidden_channels,
               grid_size,
               seed=None,
               model=None):
    # Check if the seed has valid dimension
    # Seed has dimension of Channel x H x W (Pytorch standard)

    if num_target_channels > 3:
      warnings.warn("Visualization may not work well for target channel larger than 3")

    if len(grid_size) != 2:
      raise ValueError("NCAGrid is only applicable to 2D grid.")

    if seed is None:
      self.seed = self.generate_initial_seed(
        grid_size=grid_size,
        num_hidden_channels=num_hidden_channels,
        num_target_channels=num_target_channels,
        num_static_channels=num_static_channels
      )

    else:
      self.seed = seed

    self.device = model.device
    self.seed = self.seed.to(self.device)
    self.grid_size = grid_size
    self.model = model

  def visualize_target_channel(self):
    # Only works for RGB target channel
    with torch.no_grad():
      # img = rearrange(self.model.get_target_channel(self.seed.unsqueeze(dim=0)).squeeze(), 'c h w -> w h c')
      img = rearrange(self.model.get_target_channel(self.seed.unsqueeze(dim=0)).squeeze(), 'c h w -> h w c')
      img = self._clip(img)
      img = img.cpu().detach().numpy()
    return plt.imshow(img)

  def visualize_alive_channel(self):
    with torch.no_grad():
      img = self.model.get_alive_channel(self.seed.unsqueeze(dim=0)).squeeze()
      img = img.cpu().detach().numpy()
    return plt.imshow(img, cmap='gray')

  def simulate_n_steps(self,
                       n_steps=64,
                       make_video=True,
                       return_alive_channel=True,
                       noise_schedule=[],
                       noise_radius=3,
                       **kwargs):
    if self.model is None:
      raise ValueError("Model not found")

    if make_video:
      with torch.no_grad():
        vid_frame = []
        for i in range(n_steps):
          if i in noise_schedule:
            self.add_circular_noise(radius=noise_radius)
          self.seed = self.model(self.seed.unsqueeze(dim=0)).squeeze()
          frame = rearrange(self.model.get_target_channel(self.seed.unsqueeze(dim=0)).squeeze(), "c h w -> h w c")
          if return_alive_channel:
            alive_channel = rearrange(self.model.get_alive_channel(self.seed.unsqueeze(dim=0))[0, 0:1, :, :], "c h w -> h w c")
            frame = torch.concat([frame, alive_channel], dim=2)
          frame = (self._clip(frame) * 255).type(torch.uint8)
          vid_frame.append(frame)
        vid_frame = torch.stack(vid_frame)
        return vid_frame.to("cpu")

    else:
       self.seed = self.model.forward_n_times(self.seed.unsqueeze(dim=0), n=n_steps).squeeze()

  def _clip(self, img):
    return torch.clamp(img, min=0, max=1)

  def add_circular_noise(self, radius=4):
    w, h = self.grid_size
    with torch.no_grad():
      self.seed = torch.clone(self.seed)
      center = (
          np.random.randint(radius + 2, w - (radius + 2)),
          np.random.randint(radius + 2, h - (radius + 2)),
      )

      Y, X = np.ogrid[:h, :w]
      dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

      mask = dist_from_center <= radius
      self.seed[:, mask] *= 0.0    

  @property
  def size(self):
    return self.seed.size()

  def generate_initial_seed(self, grid_size,
                            num_target_channels,
                            num_hidden_channels,
                            num_static_channels,
                            white_background=False):
    total_channels = num_target_channels + num_hidden_channels + num_static_channels + 1
    seed = torch.zeros(
        total_channels,
        grid_size[0],
        grid_size[1])
    # Set the center seed to be alive
    seed[0, grid_size[0] // 2, grid_size[1] // 2] = 1.0  # Set alive channel to one
    seed[total_channels - num_hidden_channels:, grid_size[0] // 2, grid_size[1] // 2] = 1.0 # Set all hidden channels to 1
    if white_background:
      seed[1: num_target_channels + 1, :, :] = 1.0 # Set all hidden channels to 1
    return seed