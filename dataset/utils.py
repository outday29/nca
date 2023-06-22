import torch

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