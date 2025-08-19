Quickstart
1.   Train
- PPO (GPU 0):
  - python train.py --algo PPO --total-timesteps 500000 --gpu-id 0 --mjcf robot.xml --save-path ppo_robot.zip
- TD3 (GPU 1):
  - python train.py --algo TD3 --total-timesteps 500000 --gpu-id 1 --mjcf robot.xml --save-path td3_robot.zip
- DDPG (CPU):
  - python train.py --algo DDPG --total-timesteps 500000 --gpu-id -1 --mjcf robot.xml --save-path ddpg_robot.zip

2.   Inference
- PPO:
  - python infer.py --algo PPO --gpu-id 0 --mjcf robot.xml --model ppo_robot.zip --episodes 3 --deterministic
- TD3:
  - python infer.py --algo TD3 --gpu-id 0 --mjcf robot.xml --model td3_robot.zip --episodes 3 --deterministic