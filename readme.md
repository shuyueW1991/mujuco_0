# Introduction

## Installation

### General

``` bash
pip install -r requirements.txt
```

### Advanced

- For train

``` bash
pip install tensorboard
pip install stable-baselines3[extra]
```

- For inference with GUI

``` bash
pip install opencv-python
```

## Quickstart

### Train

- PPO (GPU 0)
  
``` bash
python train.py --algo PPO --total-timesteps 500000 --gpu-id 0 --mjcf robot.xml --save-path ppo_robot.zip
```

- TD3 (GPU 1)

``` bash
python train.py --algo TD3 --total-timesteps 500000 --gpu-id 1 --mjcf robot.xml --save-path td3_robot.zip
```

- DDPG (CPU)

``` bash
python train.py --algo DDPG --total-timesteps 500000 --gpu-id -1 --mjcf robot.xml --save-path ddpg_robot.zip
```

### Inference

- PPO

``` bash
python infer.py --algo PPO --gpu-id 0 --mjcf robot.xml --model ppo_robot.zip --episodes 3 --deterministic
```

- TD3

``` bash  
python infer.py --algo TD3 --gpu-id 0 --mjcf robot.xml --model td3_robot.zip --episodes 3 --deterministic
```
