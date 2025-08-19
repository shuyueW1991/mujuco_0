import os
import argparse
import torch
import cv2  # 添加OpenCV导入
from stable_baselines3 import PPO, DDPG, TD3
from robot_env import FourWheelRobotEnv

ALGOS = {
    "PPO": PPO,
    "DDPG": DDPG,
    "TD3": TD3,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=ALGOS.keys(), required=True)
    parser.add_argument("--model", type=str, required=True, help="Path to trained model .zip")
    parser.add_argument("--mjcf", type=str, default="robot.xml")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--target-distance", type=float, default=1.5)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--render", action="store_true", help="Render the environment")  # 添加渲染选项
    args = parser.parse_args()

    # GPU selection
    if args.gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")

    env = FourWheelRobotEnv(model_path=args.mjcf, max_episode_steps=args.max_episode_steps, target_distance=args.target_distance)

    Algo = ALGOS[args.algo]
    model = Algo.load(args.model, device=device)

    success_count = 0

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        trunc = False
        step = 0
        ep_rew = 0.0
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done, trunc, info = env.step(action)
            ep_rew += reward
            step += 1
            
            # 如果启用渲染，则显示画面
            if args.render:
                frame = env.render()
                if frame is not None:
                    # OpenCV使用BGR格式，需要转换
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imshow('Robot Environment', frame_bgr)
                    # 检查是否按下了'q'键，如果是则退出
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        return
        
        print(f"Episode {ep+1}: steps={step}, total_reward={ep_rew:.2f}, distance={info['distance_traveled']:.3f}")
        if info['distance_traveled'] >= args.target_distance:
            success_count += 1

    if args.render:
        cv2.destroyAllWindows()
        
    print(f"Success {success_count}/{args.episodes}")


if __name__ == "__main__":
    main()