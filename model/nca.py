# References:
# https://github.com/shyamsn97/controllable-ncas/blob/master/controllable_nca/nca.py
# https://github.com/chenmingxiang110/Growing-Neural-Cellular-Automata/blob/master/lib/CAModel.py
# https://github.com/chenmingxiang110/Growing-Neural-Cellular-Automata/blob/master/lib/CAModel.py
# https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/adversarial_reprogramming_ca/adversarial_growing_ca.ipynb#scrollTo=ByHbsY0EuyqB

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from matplotlib import pyplot as plt

ALIVE_CHANNEL_IDX = 0

class Updater(nn.Module):
    """
    Takes in the input from the perceiver, and output the state changes for each of the cell in the grid.
    """
    def __init__(
                self,
                in_channels: int,
                out_channels: int,
                zero_bias: bool = True):
        """
        in_channels are the perceiver's output for each of the grid cell.
        out_channels are the updater's output for each of the grid cell.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out = nn.Sequential(
                torch.nn.Conv2d(self.in_channels, 64, 1), # Must only have 1, because of locality.
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, 1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, self.out_channels, 1, bias=False),
        )

        def init_weights(m):
            if isinstance(m, torch.nn.Conv2d):
                if getattr(m, "bias", None) is not None:
                    if zero_bias:
                        # So that the initial residual is stablized
                        torch.nn.init.zeros_(m.bias)

        with torch.no_grad():
            self.apply(init_weights)

    def forward(self, x):
        return self.out(x)

class Perceiver(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model = nn.Conv2d(
                                self.in_channels,
                                self.out_channels,
                                3,
                                stride=1,
                                padding=1,
                                groups=groups,
                                bias=False,
                            )
        
    def forward(self, x):
        # Perceiver will take the raw input
        return self.model(x)

class NCA(nn.Module):
    def __init__(self, 
                num_hidden_channels,
                num_target_channels,
                perceiver,
                updater,
                cell_fire_rate = 0.5,
                clip_value = [-10, 10],
                alive_threshold = 0.1,
                num_static_channels=0,
                use_alive_channel=True,
                stochastic=True,
                device="cuda"
                ):
        
        """
        Note:
        - Static channels are not changeable by the updater, but can be set manually. This is useful in cases such as GoalNCA, where static channels are basically external signals
        - Target channels are the channels that we are optimizing upon.
        - Hidden channels are the channels that are internal signals used by the grid cells.
        
        Arguments:
        
        Internally, the dimensions of the tensor is organized as below:
        - 1 alive_channel
        - N target_channels
        - N static channels
        - N hidden channels
        """
        super().__init__()
        self.total_channels = num_hidden_channels + num_static_channels + num_target_channels + 1 # 1 is for alive_channels
        self.num_hidden_channels = num_hidden_channels
        self.num_static_channels = num_static_channels
        self.num_target_channels = num_target_channels
        self.perceiver = perceiver
        self.updater = updater
        self.cell_fire_rate = cell_fire_rate,
        self.alive_threshold = alive_threshold # Above the threshold, the cell is considered alive.
        self.clip_value = clip_value # To clamp the value of the cell states for all dimensions. Useful for stability in the beginning 
        self.stochastic = stochastic
        self.use_alive_channel = use_alive_channel
        self.device = device

    def get_stochastic_update_mask(self, x):
        # Return a stochastic update mask based on a 
        """
        Return stochastic update mask
        Args:
                x ([type]): [description]
        """
        return (
                torch.clamp(torch.rand_like(x[:, 0:1], device=x.device), 0.0, 1.0).float()
                < torch.tensor(self.cell_fire_rate).to(self.device)
        )
    
    def get_alive_mask(self, x):
        # x dimension is (batch_size, num_channels, width, height)
        # A cell is only alive if itself or its surrounding has living cells
        if not self.use_alive_channel:
                return torch.ones_like(x, dtype=torch.bool, device=x.device)
        
        return (
                F.max_pool2d(
                        x[:, ALIVE_CHANNEL_IDX: ALIVE_CHANNEL_IDX + 1, :, :], # To have the unsqueeze effect
                        kernel_size=3,
                        stride=1,
                        padding=1,
                )
                > self.alive_threshold
        )

    def get_alive_channel(self, x):
        alive_channel = x[:, 0:1, :, :]
        return alive_channel
    
    def get_static_channel(self, x):
        # The first dimension is the batch size
        if self.num_static_channels != 0:
            start_idx = (self.num_target_channels + 1)
            end_idx = start_idx + self.num_static_channels
            static_channel = x[:, start_idx:end_idx, :, :]
            return static_channel
        
        else:
            return None
    
    def get_target_channel(self, x, return_alive_channel = False):
        target_channel = None
        if return_alive_channel:
            target_channel = x[:, :(self.num_target_channels + 1), :, :]
            permutation_indices = [1, 2, 3, 0]
            target_channel = target_channel[:, permutation_indices, :, :]
        else:
            target_channel = x[:, 1: (self.num_target_channels + 1), :, :]
        return target_channel
    
    def get_hidden_channel(self, x):
        if self.num_static_channels != 0:
            start_idx = (self.num_target_channels + 1)
            end_idx = start_idx + self.num_static_channels
            hidden_channel = x[:, end_idx:, :, :]
            return hidden_channel
        
        else:
            hidden_channel = x[:, (self.num_target_channels + 1), :, :]
            return hidden_channel
    
    def get_channels(self, x, alive=True, target=True, static=True, hidden=True):
        to_be_added = []
        if alive:
            to_be_added.append(self.get_alive_channel)

        if target:
            to_be_added.append(self.get_target_channel)

        if static:
            to_be_added.append(self.get_static_channel)

        if hidden:
            to_be_added.append(self.get_hidden_channel)

        result = None
        for i in to_be_added:
            if result is None:
                result = i(x)
            else:
                # Dim 0 is batch_size, 1 is all the channels
                result = torch.cat([result, i(x)], dim=1)
        return result
    
    def visualize_target_channel(self, x):
        # Only works for RGB target channel
        with torch.no_grad():
            img = rearrange(x, 'c h w -> w h c')
            img = self._clip(img)
        return plt.imshow(img)

    def visualize_alive_channel(self, x):
        with torch.no_grad():
            img = x.unsqueeze()
            img = rearrange(img, 'h w', 'w h')
        return plt.imshow(img, cmap='gray')
    
    def _clip(self, img):
        return torch.clamp(img, min=0, max=1)
    
    def forward(self, x):
        perception = self.perceiver(x)
        delta = self.updater(perception)
        alive_mask = self.get_alive_mask(x)
        stochastic_mask = self.get_stochastic_update_mask(x)
        mask = torch.logical_and(alive_mask, stochastic_mask)
        updated_x = torch.where(mask, x + delta, x) # If it is true, update it, else, do not update it
        updated_x = torch.clamp(updated_x, -10.0, 10.0)
        return updated_x
    
    def forward_n_times(self, x, n):
        for i in range(n):
            x = self.forward(x)
        return x

class GoalEncoder(nn.Module):
    def __init__(self, num_goals: int, embed_size: int):
        super().__init__()
        self.embed = nn.Embedding(num_goals, embed_size)
    
    def forward(self, goal_encoding):
        embedding = self.embed(goal_encoding)
        return embedding

class GoalNCA(NCA):
    """
    Similar to NCA, but can receive goal signals through static channels to know what shape to form.
    """
    def __init__(self, num_goals, **kwargs):
        super().__init__(**kwargs)
        self.num_goals = num_goals
        self.num_output_channels = self.total_channels - self.num_static_channels
        self.goal_encoder = GoalEncoder(num_goals=num_goals, embed_size=self.num_output_channels)
    
    def get_goal_encodings(self, goal_idx, x):
        goal_idx = goal_idx.long()
        goal_idx = rearrange(goal_idx, 'b e h w -> b h w e')
        goal_encoding = self.goal_encoder(goal_idx)
        alive_mask = self.get_alive_mask(x)
        alive_mask = rearrange(alive_mask, 'b e h w -> b h w e')
        alive_mask = alive_mask.unsqueeze(dim=-1)
        return goal_encoding * alive_mask

    def forward(self, x):
        x_without_static = self.get_channels(x, static=False)
        perception = self.perceiver(x_without_static)
        goal_idx = self.get_static_channel(x)
        goal_encodings = self.get_goal_encodings(goal_idx, x_without_static).squeeze(dim=3)
        goal_encodings = rearrange(goal_encodings, 'b h w e -> b e h w')
        delta = self.updater(perception) + goal_encodings
        alive_mask = self.get_alive_mask(x)
        stochastic_mask = self.get_stochastic_update_mask(x)
        mask = torch.logical_and(alive_mask, stochastic_mask)
        delta = torch.where(mask, delta, 0)
        delta = torch.cat([delta[:, 0:self.num_target_channels + 1, ...], torch.zeros_like(self.get_static_channel(x)), delta[:, self.num_target_channels + 1:, ...]], dim=1)
        x = x + delta
        return x

    def forward_n_times(self, x, n):
        for i in range(n):
            x = self.forward(x)
        return x