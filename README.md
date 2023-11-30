## Overview

This is a reproduction of [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) by Mordvintsev et al., 2020 and [Goal-Guided Neural Celluar Automata: Learning to Control Self-Organising Systems](https://arxiv.org/abs/2205.06806) by Sudhakaran et al., 2022 using Pytorch and Pytorch Lightning.

### Target channels visualization
<div style="display: flex; flex-direction: row;">
  <img src="data/city_demo.gif" width="100" height="100">
  <img src="data/dice_demo.gif" width="100" height="100">
  <img src="data/piano_demo.gif" width="100" height="100">
  <img src="data/rugby_demo.gif" width="100" height="100">
</div>

### Alive channel visualization

<div style="display: flex; flex-direction: row;">
  <img src="data/city_demo_alive.gif" width="100" height="100">
  <img src="data/dice_demo_alive.gif" width="100" height="100">
  <img src="data/piano_demo_alive.gif" width="100" height="100">
  <img src="data/rugby_demo_alive.gif" width="100" height="100">
</div>

### Monitoring training progress via Tensorboard

You can visualize the best seed (seed with the best loss or most closely resembles the target image) for each step as well as test run of a random seed in Tensorboard.

<img src="data/nca_tensorboard.gif">

## Usage

### Setup

First, clone this repository using `git`:

```bash
git clone https://github.com/outday29/nca
```

After that, install the required dependencies in the `requirements.txt`. You are encouraged to create a virtual environment before doing so.

```bash
pip install -r requirements.txt
```

Then download the emoji pictures for the NCA to grow by running `download_dataset.py`,

```bash
python download_dataset.py
```

### Examples

Below are the notebooks that demonstrate how to train different neural cellular automata:

- `growing_nca.ipynb` for NCA found in Growing Neural Cellular Automata (Mordvintsev et al., 2020). If you are running on Google Colab, you may refer to [here](https://colab.research.google.com/drive/1PjZvqySmcU-mzy9zenPnBFFhJqa6O-oO?usp=sharing).
- (In progress) `goal_nca.ipynb` for NCA found in Goal-Guided NCA (Sudhakaran et al., 2022)

### References
- https://github.com/shyamsn97/controllable-ncas
- https://github.com/chenmingxiang110/Growing-Neural-Cellular-Automata
- https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/adversarial_reprogramming_ca/adversarial_growing_ca.ipynb#scrollTo=ByHbsY0EuyqB
