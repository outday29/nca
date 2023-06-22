import warnings
from einops import rearrange
import matplotlib.pyplot as plt
import torch

from utils import VideoWriter

class NCAGrid(object):
  def __init__(self, seed, num_target_channels, num_static_channels, num_hidden_channels, model=None):
    # Check if the seed has valid dimension
    # Seed has dimension of Channel x H x W (Pytorch standard)

    if num_target_channels > 3:
      warnings.warn("Visualization may not work well for target channel larger than 3")

    self.seed = seed
    self.alive_channel = self.seed[0, :, :]
    self.alive_channel = torch.unsqueeze(self.alive_channel, dim=0)
    self.target_channel = self.seed[1: (num_target_channels + 1), :, :]
    
    if num_static_channels != 0:
      start_idx = (num_target_channels + 1)
      end_idx = start_idx + num_static_channels
      self.static_channel = self.seed[start_idx:end_idx, :, :]
      self.hidden_channel = self.seed[end_idx:, :, :]
    
    else:
      self.hidden_channel = self.seed[(num_target_channels + 1), :, :]
    
    self.model = model
  
  def set_static_channel(self, value):
    if value.size() == self.static_channel.size():
      self.static_channel = value
    
    else:
      raise ValueError("value must have the same dimension as the current static_channel")
  
  def set_hidden_channel(self, value):
    if value.size() == self.hidden_channel.size():
      self.static_channel = value
    
    else:
      raise ValueError("value must have the same dimension as the current hidden_channel") 
  
  def set_alive_channel(self, value):
    if value.size() == self.alive_channel.size():
      self.alive_channel = value
    
    else:
      raise ValueError("value must have the same dimension as the current alive_channel")

  def visualize_target_channel(self):
    # Only works for RGB target channel
    with torch.no_grad():
      img = rearrange(self.target_channel, 'c h w -> w h c')
      img = self._clip(img)
      img = img.detach().numpy()
    return plt.imshow(img)

  def visualize_alive_channel(self):
    with torch.no_grad():
      img = self.alive_channel.squeeze()
      img = rearrange(img, 'h w -> w h')
      img = img.detach().numpy()
    return plt.imshow(img, cmap='gray')
  
  def simulate_n_steps(self, n_steps=64, make_video=True, frame_size=(60, 60), **kwargs):
    if self.model is None:
      raise ValueError("Model not found")
    
    if make_video:
      with VideoWriter(**kwargs) as vid:
          with torch.no_grad():
            for i in range(n_steps):
              self.seed = self.model(self.seed.unsqueeze(dim=0)).squeeze()
              frame = einops.rearrange(self.seed, "c h w -> h w c")
              vid.add(frame)

    else:
       self.seed = self.model.forward_n_steps(self.seed, n=n_steps)
  
  def _clip(self, img):
    return torch.clamp(img, min=0, max=1)

  @property
  def size(self):
    return self.seed.size()
  
  @staticmethod
  def generate_initial_seed(grid_size, num_target_channels, num_hidden_channels, num_static_channels):
    # The user may want to define initial seeds, such as in https://colab.research.google.com/drive/1vG7yjOHxejdk_YfvKhASanNs0YvKDO5-#scrollTo=aQqUWJwwAK5r
    assert len(grid_size) == 2

    total_channels = num_target_channels + num_hidden_channels + num_static_channels + 1
    seed = torch.zeros(
        total_channels,
        grid_size[0],
        grid_size[1])
    # Set the center seed to be alive
    seed[0, grid_size[0] // 2, grid_size[1] // 2] = 1.0  # Set alive channel to one
    seed[(total_channels - num_hidden_channels):, grid_size[0] // 2, grid_size[1] // 2] = 1.0 # Still need to investigate this
    return seed