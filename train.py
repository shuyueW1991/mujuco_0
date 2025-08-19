import os
import argparse
import torch
from stable_baselines3 import PPO, DDPG, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from robot_env import FourWheelRobotEnv

ALGOS = {
    "PPO": PPO,
    "DDPG": DDPG,
    "TD3": TD3,
}


def make_env(model_path, max_steps, target_distance):
    return lambda: FourWheelRobotEnv(model_path=model_path, max_episode_steps=max_steps, target_distance=target_distance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=ALGOS.keys(), default="PPO", help="RL algorithm")
    parser.add_argument("--mjcf", type=str, default="robot.xml", help="MuJoCo XML model path")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--target-distance", type=float, default=1.5)
    parser.add_argument("--save-path", type=str, default="trained_model.zip")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0, help="Set to -1 for CPU, else select GPU id")
    args = parser.parse_args()

    # GPU selection
    if args.gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")

    os.makedirs(args.log_dir, exist_ok=True)

    # Vectorized env
    env = DummyVecEnv([make_env(args.mjcf, args.max_episode_steps, args.target_distance)])

    # Policy type
    policy = "MlpPolicy"

    # Create model
    Algo = ALGOS[args.algo]
    model = Algo(policy, env, verbose=1, tensorboard_log=args.log_dir, seed=args.seed, device=device)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(save_freq=50_000, save_path=args.log_dir, name_prefix=f"{args.algo.lower()}_ckpt")

    # Train
    model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback, progress_bar=True)

    # Save
    model.save(args.save_path)
    print(f"Saved model to {args.save_path}")


if __name__ == "__main__":
    main()