import pytorch_lightning as pl
import torchvision.transforms as T
from urllib.request import urlopen
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from PIL import Image
import shutil
import pickle
import random
import torch

from .utils import generate_initial_seed
from ..utils import load_image


class NCADatasetBase(Dataset):
    def __init__(
        self,
        target_image_path,
        seed_cache_dir,
        grid_size,
        num_hidden_channels,
        num_static_channels,
        num_target_channels,
        dataset_size=64,
        clear_cache=False,
        device="cuda",
    ):
        self.target_image_path = target_image_path
        self.seed_cache_dir = seed_cache_dir
        self.num_hidden_channels = num_hidden_channels
        self.num_static_channels = num_static_channels
        self.num_target_channels = num_target_channels
        self.grid_size = grid_size
        self.dataset_size = dataset_size
        self.device = device

        if clear_cache:
            self.clear_cache

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self):
        raise NotImplementedError

    def clear_cache(self):
        for path in self.seed_cache_dir.iterdir():
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)

    def cache_is_empty(self):
        for p in self.seed_cache_dir.iterdir():
            if p.is_file() and (str(p).endswith(".pt")):
                return False

        return True

    def get_random_seed_cache(self):
        cached_seed = list(self.seed_cache_dir.iterdir())
        cached_seed = list(
            filter(lambda x: x.is_file() and str(x).endswith(".pt"), cached_seed)
        )

        if len(cached_seed) == 0:
            raise Exception("No cache?")

        seed_path = random.choice(cached_seed)
        seed = torch.load(seed_path)
        return seed.to(self.device)

    def get_alive_mask(self, x):
        # x dimension is (batch_size, num_channels, width, height)
        # A cell is only alive if itself or its surrounding has living cells
        if not self.use_alive_channel:
            return torch.ones_like(x, dtype=torch.bool, device=x.device)

        return (
            F.max_pool2d(
                x[
                    :, self.alive_channel_idx : self.alive_channel_idx + 1, :, :
                ],  # To have the unsqueeze effect
                kernel_size=3,
                stride=1,
                padding=1,
            )
            > self.alive_threshold
        )

    def get_alive_channel(self, x):
        alive_channel = x[0, :, :]
        return alive_channel

    def get_static_channel(self, x, return_mask=False):
        # The first dimension is the batch size
        if self.num_static_channels != 0:
            start_idx = self.num_target_channels + 1
            end_idx = start_idx + self.num_static_channels
            if return_mask:
                static_channel_mask = torch.zeros_like(x).bool()
                static_channel_mask[start_idx:end_idx, :, :] = True
                return static_channel_mask

            else:
                static_channel = x[start_idx:end_idx, :, :]
                return static_channel

        else:
            return None

    def get_target_channel(self, x, return_alive_channel=False):
        target_channel = None
        if return_alive_channel:
            target_channel = x[: (self.num_target_channels + 1), :, :]
            permutation_indices = [1, 2, 3, 0]
            target_channel = target_channel[:, permutation_indices, :, :]
        else:
            target_channel = x[1 : (self.num_target_channels + 1), :, :]
        return target_channel

    def get_hidden_channel(self, x):
        if self.num_static_channels != 0:
            start_idx = self.num_target_channels + 1
            end_idx = start_idx + self.num_static_channels
            hidden_channel = x[end_idx:, :, :]
            return hidden_channel

        else:
            hidden_channel = x[(self.num_target_channels + 1), :, :]
            return hidden_channel

    def set_static_channel(self, x, value):
        # In the future, we will have to resort to indexing manually instead of using boolean mask.
        x_static_channel_mask = self.get_static_channel(x, return_mask=True)
        x[x_static_channel_mask] = value

        # else:
        #   raise ValueError("value must have the same dimension as the current static_channel")

    # def set_hidden_channel(self, x, value):
    #   # if value.size() == self.num_hidden_channels:
    #   x_hidden_channel = self.get_hidden_channel(x)
    #   x_hidden_channel.fill_(value)

    #   # else:
    #   #   raise ValueError("value must have the same dimension as the current hidden_channel")

    # def set_alive_channel(self, x, value):
    #   x_alive_channel = self.get_alive_channel(x)
    #   x_alive_channel.fill_(value)

    # def preprocess_target_image(self, image_path):
    #   target_image = Image.open(image_path).convert("RGB")
    #   # Need to check whether the size is paddable
    #   target_image = T.ToTensor()(target_image)
    #   target_image = T.Pad(padding=4)(target_image)
    #   return target_image


class NCADataset(NCADatasetBase):
    def __init__(
        self,
        target_image_path,
        seed_cache_dir,
        grid_size,
        num_hidden_channels,
        num_static_channels,
        num_target_channels,
        thumbnail_size,
        dataset_size=64,
        clear_cache=False,
        device="cuda",
        use_cache_p=0.8,
    ):
        super().__init__(
            target_image_path=target_image_path,
            seed_cache_dir=seed_cache_dir,
            grid_size=grid_size,
            num_hidden_channels=num_hidden_channels,
            num_static_channels=num_static_channels,
            num_target_channels=num_target_channels,
            dataset_size=dataset_size,
            clear_cache=clear_cache,
            device=device,
        )
        self.thumbnail_size = thumbnail_size
        self.use_cache_p = use_cache_p
        self.target_image_processed = load_image(
            self.target_image_path,
            size=self.grid_size[0],
            thumbnail_size=thumbnail_size,
        )

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # Since this is only 1 image, we do not need to organize into folder

        if random.random() < self.use_cache_p:
            if self.cache_is_empty():
                new_seed = generate_initial_seed(
                    grid_size=self.grid_size,
                    num_hidden_channels=self.num_hidden_channels,
                    num_static_channels=self.num_static_channels,
                    num_target_channels=self.num_target_channels,
                ).to(self.device)

                return (new_seed, self.target_image_processed.to(self.device))

            else:
                return (
                    self.get_random_seed_cache().to(self.device),
                    self.target_image_processed.to(self.device),
                )

        else:
            new_seed = generate_initial_seed(
                grid_size=self.grid_size,
                num_hidden_channels=self.num_hidden_channels,
                num_static_channels=self.num_static_channels,
                num_target_channels=self.num_target_channels,
            ).to(self.device)

            return (new_seed, self.target_image_processed.to(self.device))


class GoalNCADataset(NCADatasetBase):
    def __init__(
        self,
        target_image_path,
        seed_cache_dir,
        grid_size,
        num_hidden_channels,
        num_static_channels,
        num_target_channels,
        thumbnail_size,
        dataset_size=64,
        clear_cache=False,
        device="cuda",
        use_cache_p=0.8,
    ):
        super().__init__(
            target_image_path=target_image_path,
            seed_cache_dir=seed_cache_dir,
            grid_size=grid_size,
            num_hidden_channels=num_hidden_channels,
            num_static_channels=num_static_channels,
            num_target_channels=num_target_channels,
            dataset_size=dataset_size,
            clear_cache=clear_cache,
            device=device,
        )

        self.num_goal = len(self.target_image_path)
        self.thumbnail_size = thumbnail_size
        self.use_cache_p = use_cache_p
        self.create_cache_dir()
        self.preprocess()

    def preprocess(self):
        target_image_processed = []
        for i in self.target_image_path:
            target_image_processed.append(
                load_image(
                    i, size=self.grid_size[0], thumbnail_size=self.thumbnail_size
                )
            )
        self.target_image_processed = target_image_processed

    def cache_is_empty(self, goal_idx):
        seed_cache_dir = self.seed_cache_dir / f"goal_{goal_idx}"
        for p in seed_cache_dir.iterdir():
            if p.is_file() and (str(p).endswith(".pt")):
                return False

        return True

    def create_cache_dir(self):
        for i in range(self.num_goal):
            seed_cache_path = self.seed_cache_dir / f"goal_{i}"
            seed_cache_path.mkdir(exist_ok=True, parents=True)

    def get_random_seed_cache(self, goal_idx):
        seed_cache_dir = self.seed_cache_dir / f"goal_{goal_idx}"
        cached_seed = list(seed_cache_dir.iterdir())
        cached_seed = list(
            filter(lambda x: x.is_file() and str(x).endswith(".pt"), cached_seed)
        )

        if len(cached_seed) == 0:
            raise Exception("No cache?")

        seed_path = random.choice(cached_seed)
        seed = torch.load(seed_path)
        return seed.to(self.device)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # First select goal
        goal_idx = random.randint(0, self.num_goal - 1)
        # The cache dir will be organized like self.cache_dir/goal_{goal_idx}
        if random.random() < self.use_cache_p:
            if self.cache_is_empty(goal_idx):
                new_seed = generate_initial_seed(
                    grid_size=self.grid_size,
                    num_hidden_channels=self.num_hidden_channels,
                    num_static_channels=self.num_static_channels,
                    num_target_channels=self.num_target_channels,
                )
                new_seed = new_seed.to(self.device)
                # goal_idx_channel = torch.ones(self.num_static_channels, self.grid_size[0], self.grid_size[1]) * torch.tensor(goal_idx)
                self.set_static_channel(new_seed, goal_idx)
                target_image = self.processed_target_image[goal_idx].to(self.device)
                return (new_seed, target_image)

            else:
                seed = self.get_random_seed_cache(goal_idx)
                target_image = self.processed_target_image[goal_idx].to(self.device)
                return (seed, target_image)

        else:
            new_seed = generate_initial_seed(
                grid_size=self.grid_size,
                num_hidden_channels=self.num_hidden_channels,
                num_static_channels=self.num_static_channels,
                num_target_channels=self.num_target_channels,
            )
            new_seed = new_seed.to(self.device)
            # goal_idx_channel = torch.ones(self.num_static_channels, self.grid_size[0], self.grid_size[1]) * torch.tensor(goal_idx)
            self.set_static_channel(new_seed, goal_idx)
            target_image = self.processed_target_image[goal_idx].to(self.device)
            return (new_seed, target_image)


class NCADataModule(pl.LightningDataModule):
    def __init__(
        self,
        seed_cache_dir,
        grid_size,
        num_hidden_channels,
        num_target_channels,
        num_static_channels,
        target_image_path,
        batch_size,
        thumbnail_size,
        dataset_size,
        device="cuda",
        use_cache_p=0.8,
        clear_cache=False,
    ):
        super().__init__()
        self.target_image_path = target_image_path
        self.batch_size = batch_size
        self.dataset = NCADataset(
            target_image_path=target_image_path,
            seed_cache_dir=seed_cache_dir,
            grid_size=grid_size,
            num_hidden_channels=num_hidden_channels,
            num_target_channels=num_target_channels,
            num_static_channels=num_static_channels,
            thumbnail_size=thumbnail_size,
            dataset_size=dataset_size,
            clear_cache=clear_cache,
            device=device,
            use_cache_p=use_cache_p,
        )

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)


class GoalNCADataModule(pl.LightningDataModule):
    def __init__(
        self,
        seed_cache_dir,
        grid_size,
        num_hidden_channels,
        num_target_channels,
        num_static_channels,
        target_image_path,
        batch_size,
        thumbnail_size,
        dataset_size,
        use_cache_p=0.8,
        clear_cache=False,
    ):
        super().__init__()
        self.target_image_path = target_image_path
        self.batch_size = batch_size
        self.dataset = GoalNCADataset(
            target_image_path=target_image_path,
            seed_cache_dir=seed_cache_dir,
            grid_size=grid_size,
            num_hidden_channels=num_hidden_channels,
            num_target_channels=num_target_channels,
            num_static_channels=num_static_channels,
            thumbnail_size=thumbnail_size,
            dataset_size=dataset_size,
            use_cache_p=use_cache_p,
            clear_cache=clear_cache,
        )

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
