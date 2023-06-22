from pytorch_lightning.callbacks import Callback
import itertools
import random
import torch

# For saving video https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_video

def get_num_generator(max_num):
    for i in itertools.count():
      yield i % max_num

class CacheBestSeedBase(Callback):
    # Cache the best seed for each batch or epoch
    def __init__(self, cache_dir, num_generator, top_k=1):
        self.cache_dir = cache_dir
        self.num_gen = num_generator
        self.top_k = top_k
        self.seed_list = None
        self.loss_per_seed_list = None
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.seed_list is None:
            self.seed_list = outputs["seeds"]
            self.loss_per_seed_list = outputs["loss_per_seed"]
        
        else:
            self.seed_list = torch.concat([self.seed_list, outputs["seeds"]], dim=0)
            self.loss_per_seed_list = torch.concat([self.loss_per_seed_list, outputs["loss_per_seed"]], dim=0)
    
    def on_train_epoch_end(self, trainer, pl_module):
        best_seed_idx = torch.argmin(self.loss_per_seed_list).tolist()
        best_seed = self.seed_list[best_seed_idx]
        self.cache_seed(trainer, pl_module, best_seed)
        self.reset()
    
    def cache_seed(self, trainer, pl_module, seed):
        raise NotImplementedError
    
    def reset(self):
        self.seed_list = None
        self.loss_per_seed_list = None

class CacheBestSeed(CacheBestSeedBase):
    def cache_seed(self, trainer, pl_module, seed):
        torch.save(seed, self.cache_dir / ("seed_" + str(next(self.num_gen)) + ".pt"))

class GoalNCACacheBestSeed(CacheBestSeedBase):
    # Cache the best seed for each batch or epoch
    def cache_seed(self, trainer, pl_module, seed):
        goal_idx = pl_module.model.get_static_channel(seed.unsqueeze(dim=0))[0, 0, 0, 0].tolist() # Just pick any position, since it is the same anyways
        goal_idx = int(goal_idx)
        torch.save(seed, self.cache_dir / f"goal_{goal_idx}" / ("seed_" + str(next(self.num_gen)) + ".pt"))

class CacheCorruptedSeedBase(Callback):
    def __init__(self, cache_dir, num_generator, loss_threshold, corrupt_func, chance=0.1):
        self.cache_dir = cache_dir
        self.num_gen = num_generator
        self.loss_threshold = loss_threshold # If below the loss threshold, apply the corruption
        self.corrupt_func = corrupt_func # Takes a seed argument
        self.chance = chance # Chance of applying corrupt function
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        seeds = outputs['seeds']
        loss_per_seed = outputs['loss_per_seed']
        mask = loss_per_seed < self.loss_threshold
        target_seeds = torch.clone(seeds[mask])
        for i in target_seeds:
            if random.random() < self.chance:
                cur_seed = self.corrupt_func(i)
                torch.save(cur_seed, self.cache_dir / ("seed_" + str(next(self.num_gen)) + ".pt"))
    
    def cache_seed(self, trainer, pl_module, seed):
        torch.save(seed, self.cache_dir / ("seed_" + str(next(self.num_gen)) + ".pt"))

class CacheCorruptedSeed(Callback):
    def __init__(self, cache_dir, num_generator, loss_threshold, corrupt_func, chance=0.1):
        self.cache_dir = cache_dir
        self.num_gen = num_generator
        self.loss_threshold = loss_threshold # If below the loss threshold, apply the corruption
        self.corrupt_func = corrupt_func # Takes a seed argument
        self.chance = chance # Chance of applying corrupt function
    
    # Apply corruption before caching it
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        seeds = outputs['seeds']
        loss_per_seed = outputs['loss_per_seed']
        mask = loss_per_seed < self.loss_threshold
        target_seeds = torch.clone(seeds[mask])
        for i in target_seeds:
            if random.random() < self.chance:
                cur_seed = self.corrupt_func(i)
                torch.save(cur_seed, self.cache_dir / ("seed_" + str(next(self.num_gen)) + ".pt"))
                
class VisualizeBestSeed(Callback):
    def __init__(self, details="Best seeds"):
        self.best_seed = None
        self.best_loss = None
        self.target = None
        self.details = details
    
    def setup(self, trainer, pl_module, stage):
        self.tensorboard = pl_module.logger.experiment
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        best_seed_idx = torch.argmin(outputs["loss_per_seed"]).tolist()
        cur_best_seed = outputs["seeds"][best_seed_idx]
        cur_best_loss = outputs["loss_per_seed"][best_seed_idx].tolist()
        if self.best_seed is None:
            self.best_seed = cur_best_seed
            self.best_loss = cur_best_loss
            self.target = batch[1][best_seed_idx]
        
        else:
            if cur_best_loss < self.best_loss:
                self.best_seed = cur_best_seed
                self.best_loss = cur_best_loss
                self.target = batch[1][best_seed_idx]
            
    def on_train_epoch_end(self, trainer, pl_module):
        best_seed_target_channel = pl_module.model.get_target_channel(self.best_seed.unsqueeze(dim=0)).squeeze()
        # Target now has an additional alpha channel, we do not need that, alpha channel is the last channel
        target = self.target[0:3, :, :]
        img = torch.cat([best_seed_target_channel, target], dim=2)
        # img = einops.rearrange(img, 'c h w -> w h c')
        img = torch.clamp(img, min=0, max=1)
        img = img.unsqueeze(dim=0)
        self.tensorboard.add_images(self.details, img, global_step=pl_module.current_epoch)
        self.reset()
    
    def reset(self):
        self.best_seed = None
        self.best_loss = None
        self.target = None
        
class VisualizeRun(Callback):
    # Simulate run on a model, save a video with Tensorboard
    # Randomly choose and grow a seed from the batch.
    def __init__(self, interval, simulate_step, fps=4, tag="Test run"):
        # We cannot choose the best seed because we might actually choose the seed that already reaches the target state, 
        # meaning we cannot see the growth.
        self.interval = interval
        self.simulate_step = simulate_step
        self.fps = fps
        self.tag = tag
        self.test_seed = None
            
    def setup(self, trainer, pl_module, stage):
        self.tensorboard = pl_module.logger.experiment
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        seeds, images = batch
        self.test_seed = torch.clone(random.choice(seeds))
    
    def on_train_epoch_end(self, trainer, pl_module):
        # current_epoch starts from 0
        if pl_module.current_epoch % self.interval == 0:
            # May want to write multiple video
            model = pl_module.model
            vid_tensor = self.simulate(model, self.test_seed)
            # Vid_tensor is N, T, C, H, W
            self.tensorboard.add_video(
                tag=self.tag,
                vid_tensor=vid_tensor,
                global_step=pl_module.current_epoch,
                fps = self.fps
            )
    
    def simulate(self, model, seed):
        seed = seed.unsqueeze(dim=0)
        vid_tensor = seed # The first T
        for i in range(self.simulate_step):
            seed = model(seed)
            vid_tensor = torch.concat([vid_tensor, seed], dim=0)
        vid_tensor = model.get_target_channel(vid_tensor)
        vid_tensor = vid_tensor.unsqueeze(dim=0) # For the N dimension
        return vid_tensor