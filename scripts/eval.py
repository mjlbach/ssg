import re

from collections import defaultdict
from pathlib import Path

import cv2
import hydra
import ray
from omegaconf import OmegaConf
from ray.rllib.agents import ppo
from ray.tune.registry import register_env
import numpy as np
from ray.rllib.models import ModelCatalog
from ray.tune.logger import UnifiedLogger

import ssg
from ssg.utils.callbacks import MetricsCallback
from ssg.policies.model import ComplexInputNetwork

import train

ModelCatalog.register_custom_model("graph_extractor", ComplexInputNetwork)

@hydra.main(config_path=ssg.CONFIG_PATH, config_name="config")
def main(cfg):
    # instantiate env class
    ray.init(local_mode=cfg.debug)
    env_config = OmegaConf.to_object(cfg)
    register_env("igibson_env_creator", train.igibson_env_creator)
    checkpoint_path = Path(cfg.experiment_save_path, cfg.experiment_name)
    config = {
        "env": "igibson_env_creator",
        "model": OmegaConf.to_object(cfg.model),
        "env_config": env_config,  # config to pass to env class
        "num_workers": 0,
        "num_envs_per_worker": 1,
        "create_env_on_driver": False,
        "evaluation_interval": 0,
        "framework": "torch",
        "seed": cfg.seed,
        "lambda": cfg.gae_lambda,
        "lr": cfg.learning_rate,
        "train_batch_size": cfg.n_steps,
        "rollout_fragment_length": cfg.n_steps // cfg.num_envs,
        "num_sgd_iter": cfg.n_epochs,
        "sgd_minibatch_size": cfg.batch_size,
        "gamma": cfg.gamma,
        "num_gpus": 1,
        "callbacks": MetricsCallback,
    }

    log_path = str(checkpoint_path.joinpath("eval"))
    Path(log_path).mkdir(parents=True, exist_ok=True)
    agent = ppo.PPOTrainer(
        config,
        logger_creator=lambda x: UnifiedLogger(x, log_path), #type: ignore
    )

    if Path(checkpoint_path).exists():
        checkpoints = Path(checkpoint_path).rglob("checkpoint-*")
        checkpoints = [
            str(f) for f in checkpoints if re.search(r".*checkpoint-\d*$", str(f))
        ]
        checkpoints = sorted(checkpoints)
        if len(checkpoints) > 0:
            agent.restore(checkpoints[-1])

    env = agent.workers.local_worker().env #type: ignore
    state = env.reset()
    trials = 0
    successes = 0

    video_folder = Path(
            cfg.experiment_save_path,
            cfg.experiment_name,
            "eval"
        )
    frame = state["rgb"][:, :, :3]
    video = cv2.VideoWriter(  # type: ignore
        str(video_folder.joinpath('video.mp4')),
        cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore
        15,
        (frame.shape[1], frame.shape[0]),
    )

    for _ in range(10):
        episode_reward = 0
        done = False
        success = False

        state = env.reset()
        reward_breakdown = defaultdict(lambda: 0)
        while not done:
            action = agent.compute_single_action(state)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            frame = (state["rgb"][:, :, :3] * 255).astype(np.uint8)
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # type: ignore
            for reward, value in info["reward_breakdown"].items():
                reward_breakdown[reward] += value
            if info["success"]:
                assert done
                success = True
                successes += 1

        trials += 1
        print()
        print("Episode finished:")
        print("-" * 30)
        print(f"{'Success:': <20} {success: b}")
        print(f"{'Episode reward': <20} {episode_reward: .2f}")
        for key, value in reward_breakdown.items():
            print(f"{key: <20} {value: .2f}")
        print("-" * 30)

    print()
    print(f"Success fraction: {successes/trials}")
    video.release()


if __name__ == "__main__":
    main()
