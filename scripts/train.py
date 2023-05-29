import re
from pathlib import Path

import hydra
import numpy as np
import ray
from omegaconf import OmegaConf
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import register_env
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings

import ssg
from ssg.policies.model import ComplexInputNetwork
from ssg.utils.callbacks import DummyCallback, MetricsCallback

def igibson_env_creator(env_config):
    from ssg.envs.igibson_env import iGibsonEnv

    if "scene_ids" in env_config:
        if len(env_config["scene_ids"]) != env_config["num_envs"]:
            print(
                f"Warning: number of workers (n={env_config['num_envs']}) does not match number of scenes (n={len(env_config['scene_ids'])}) in config file"
        )
        # Parllelize environments across workers 
        env_config['scene_id'] = env_config['scene_ids'][env_config.worker_index - 1]

    return iGibsonEnv(
        config_file=env_config,
        mode=env_config["mode"],
        action_timestep=1 / 10.0,
        physics_timestep=1 / 120.0,
        # use_pb_gui = True,
        rendering_settings = MeshRendererSettings(optimized=True, enable_pbr=True, enable_shadow=True, msaa=False, hide_robot=env_config.get('hide_robot', True)),
    )


ModelCatalog.register_custom_model("graph_extractor", ComplexInputNetwork)


@hydra.main(config_path=ssg.CONFIG_PATH, config_name="config")
def main(cfg):
    ray.init(local_mode=cfg.debug)
    env_config = OmegaConf.to_object(cfg)
    register_env("igibson_env_creator", igibson_env_creator)
    checkpoint_path = Path(cfg.experiment_save_path, cfg.experiment_name)

    num_epochs = np.round(cfg.training_timesteps / cfg.n_steps).astype(int)
    save_ep_freq = np.round(
        num_epochs / (cfg.training_timesteps / cfg.save_freq)
    ).astype(int)

    config = {
        "env": "igibson_env_creator",
        "model": OmegaConf.to_object(cfg.model),
        "env_config": env_config,  # config to pass to env class
        "num_workers": cfg.num_envs,
        "framework": "torch",
        "seed": cfg.seed,
        "lambda": cfg.gae_lambda,
        "lr": cfg.learning_rate,
        "train_batch_size": cfg.n_steps,
        "rollout_fragment_length": cfg.n_steps // cfg.num_envs,
        "num_sgd_iter": cfg.n_epochs,
        "sgd_minibatch_size": cfg.batch_size,
        "gamma": cfg.gamma,
        "create_env_on_driver": False,
        "num_gpus": 1,
        "callbacks": MetricsCallback,
        # "log_level": "DEBUG",
        # "_disable_preprocessor_api": False,
    }

    if cfg.eval_freq > 0 and not cfg.debug:
        eval_ep_freq = np.round(
            num_epochs / (cfg.training_timesteps / cfg.eval_freq)
        ).astype(int)
        config.update(
            {
                "evaluation_interval": eval_ep_freq,  # every n episodes evaluation episode
                "evaluation_duration": 20,
                "evaluation_duration_unit": "episodes",
                "evaluation_num_workers": 1,
                "evaluation_parallel_to_training": True,
                "evaluation_config": {
                    "callbacks": DummyCallback,
                    "record_env": True,
                },
            }
        )

    log_path = str(checkpoint_path.joinpath("log"))
    Path(log_path).mkdir(parents=True, exist_ok=True)
    trainer = ppo.PPOTrainer(
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
            trainer.restore(checkpoints[-1])

    for i in range(num_epochs):
        # Perform one iteration of training the policy with PPO
        trainer.train()

        if (i % save_ep_freq) == 0:
            checkpoint = trainer.save(checkpoint_path)
            print("checkpoint saved at", checkpoint)


if __name__ == "__main__":
    main()
