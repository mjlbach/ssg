from ssg.utils.floor_sampler import sample_on_floor

import hydra
import numpy as np
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from omegaconf import OmegaConf

import ssg
from ssg.envs.igibson_env import iGibsonEnv

@hydra.main(config_path=ssg.CONFIG_PATH, config_name="config")
def main(cfg):

    cfg.image_width = 512
    cfg.image_height = 512

    cfg.mode = "gui_non_interactive"

    env = iGibsonEnv(
        config_file=OmegaConf.to_object(cfg),
        mode=cfg.mode,
        action_timestep=1 / 10.0,
        physics_timestep=1 / 120.0,
        rendering_settings=MeshRendererSettings(
            enable_pbr=True, enable_shadow=True, msaa=False, hide_robot=cfg.hide_robot
        ),
        # use_pb_gui=True,
    )
    env.reset()
    env.slam.load()
    env.step(3)
    breakpoint()
    env.task.target_obj.set_position([100, 100, 100])
    for room in env.simulator.scene.room_ins_name_to_ins_id.keys():
        for _ in range(5):
            sample_on_floor(env.robots[0], env.simulator.scene, room)
            for _ in range(12):
                env.step(3)
    # env.slam.dump()

if __name__ == "__main__":
    main()
