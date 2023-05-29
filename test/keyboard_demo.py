import os
from collections import defaultdict

import cv2
import graph_visualization
import hydra
import numpy as np
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from omegaconf import OmegaConf

import ssg
from ssg.envs.igibson_env import iGibsonEnv

key_to_action = {
    119: 0,
    115: 1,
    100: 2,
    97: 3,
    120: 4,
    # 102: 5,
    # 103: 6
}


@hydra.main(config_path=ssg.CONFIG_PATH, config_name="config")
def main(cfg):

    cfg.image_width = 512
    cfg.image_height = 512

    cfg.mode = "gui_interactive"

    env = iGibsonEnv(
        config_file=OmegaConf.to_object(cfg),
        mode=cfg.mode,
        action_timestep=1 / 10.0,
        physics_timestep=1 / 120.0,
        rendering_settings = MeshRendererSettings(optimized=True, enable_pbr=True, enable_shadow=True, msaa=False, hide_robot=cfg.get('hide_robot', True), blend_highlight=True)
        # use_pb_gui=True,
    )

    video = None
    for episode in range(10000):
        episode_reward = 0
        # env.task.target_obj.unhighlight()
        state = env.reset()
        # env.task.target_obj.highlight()
        reward = 0
        info = {}

        reward_breakdown = defaultdict(lambda: 0)

        for idx in range(500):
            key = cv2.waitKey()  # type: ignore
            if key in key_to_action:  # wasdx
                action = key_to_action[key]
            elif key == 113:  # q
                if video is not None:
                    video.release()
                env.close()
                quit()
            elif key == 112:  # p
                print(f"Episode reward: {episode_reward}")
                print(f"Timestep: {idx}")
                print(f"Timestep reward: {reward}")
                print(f"Episode info: {info}")
                continue
            elif key == 121:  # y
                if video is None:
                    frame = state["rgb"][:, :, :3]
                    video_path = os.path.expanduser("~/eval_episodes.mp4")
                    video = cv2.VideoWriter(  # type: ignore
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore
                        15,
                        (frame.shape[1], frame.shape[0]),
                    )
                    print("Recording!")
                continue
            elif key == 114:  # r
                break
            elif key == 103:
                if hasattr(env, "scene_graph"):
                    graph_visualization.generate_pyvis_graph(env)
                else:
                    print("No scene graph modality")
                continue
            else:
                print(f"{key} not mapped!")
                continue

            state, reward, done, info = env.step(action)

            if video is not None:
                frame = (state["rgb"][:, :, :3] * 255).astype(np.uint8)
                video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # type: ignore

            episode_reward += reward
            print("Reward breakdown:")
            for reward, value in info["reward_breakdown"].items():
                print(f"{reward}: {value:.2f}")
                reward_breakdown[reward] += value
            print()

            if done:
                break


            # xy = env.robots[0].get_position()[0:2]
            # room_instance = env.simulator.scene.get_room_instance_by_point(xy)
            # print(room_instance)

        print(
            "Episode {} finished after {} timesteps.".format(episode, env.current_step)
        )
        print("Episode reward: ", episode_reward)
        for key, value in reward_breakdown.items():
            print(f"{key}: {value}")
        print()
    env.close()


if __name__ == "__main__":
    main()
