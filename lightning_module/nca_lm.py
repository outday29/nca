from torch import nn
import pytorch_lightning as pl
import itertools
import torch

class NCALightningModule(pl.LightningModule):
  def __init__(self, model, train_step, seed_cache_dir, seed_cache_size):
    super().__init__()
    self.model = model # The model must be stored in self.model, callbacks depend on that
    self.loss = nn.MSELoss()
    self.train_step = train_step
    self.seed_cache_dir = seed_cache_dir
    self.seed_cache_size = seed_cache_size
    self.num_gen = iter(self.get_num_generator())
    self.save_hyperparameters()

  def training_step(self, batch, batch_idx):
    seeds, images = batch
    final_seeds = self.model.forward_n_times(seeds, self.train_step)
    final_seeds_target_channels = self.model.get_target_channel(final_seeds, return_alive_channel=True)
    loss = self.loss(final_seeds_target_channels, images)
    loss_per_seed = self.get_loss_per_seed(final_seeds_target_channels, images)
    return {"loss": loss, "loss_per_seed": loss_per_seed, "seeds": final_seeds}

  def validation_step(self):
    # Not applicable
    pass
  
  def predict_step(self):
    # Not applicable
    pass
  
  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=0.02)
  
  def get_num_generator(self):
    for i in itertools.count():
      yield i % self.seed_cache_size
    
  def get_loss_per_seed(self, seeds_target_channels, target_channel):
    loss_per_seed = []
    with torch.no_grad():
      for i in range(seeds_target_channels.shape[0]):
        loss_per_seed.append(self.loss(seeds_target_channels[0:1], target_channel[0:1]))
    loss_per_seed = torch.tensor(loss_per_seed)
    return loss_per_seed
  
  def get_best_seed_idx(self, seeds_target_channels, target_channel):
    # The best solution we have now
    loss_per_example = []
    with torch.no_grad():
      for i in range(seeds_target_channels.shape[0]):
        loss_per_example.append(self.loss(seeds_target_channels[0:1], target_channel[0:1]))
      loss_per_example = torch.tensor(loss_per_example)
      return torch.argmin(loss_per_example).tolist()