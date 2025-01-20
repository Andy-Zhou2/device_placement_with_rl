# Device Placement for Large Language Models with Reinforcement Learning

This repository contains implementations and training scripts for optimizing device placements for large language models such as OPT-125m.
This project is a part of the R244 course at University of Cambridge. 

Follow the instructions below to set up the environment and run the training scripts.

## Environment Setup

### Step 1: Create a Conda Environment
Create a new Conda environment named `device_rl`:
```bash
conda create -n device_rl python=3.11
conda activate device_rl
```

### Step 2: Install TensorFlow with CUDA
Install TensorFlow with GPU support: following the instructions on the [TensorFlow website](https://www.tensorflow.org/install/pip).


### Step 3: Install PyTorch (CPU-only)
Install the CPU-only version of PyTorch: following the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/).


### Step 4: Install the Local Transformers Package
Install the local transformers package in editable mode:
```bash
pip install -e ./transformers
```


## Training Scripts
### Toy Example
To run the toy training example:
```bash
python toy_example/tf_device_training.py
```

To run the OPT example:
```bash
python opt/training.py
```

