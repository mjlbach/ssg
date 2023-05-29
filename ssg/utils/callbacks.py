from pathlib import Path

import cv2
from ray.rllib.agents.callbacks import DefaultCallbacks


class MetricsCallback(DefaultCallbacks):
    def on_episode_step(self, *, worker, base_env, policies, episode, **kwargs):
        reward_breakdown = episode.last_info_for()["reward_breakdown"]
        if "reward_breakdown" not in episode.user_data:
            episode.user_data["reward_breakdown"] = {}
            for key, value in reward_breakdown.items():
                episode.user_data["reward_breakdown"][key] = value
        else:
            for key, value in reward_breakdown.items():
                episode.user_data["reward_breakdown"][key] += value

    def on_episode_end(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        # Make sure this episode is really done.
        assert episode.batch_builder.policy_collectors["default_policy"].batches[-1][
            "dones"
        ][-1], (
            "ERROR: `on_episode_end()` should only be called " "after episode is done!"
        )
        reward_breakdown = episode.user_data["reward_breakdown"]
        for key, value in reward_breakdown.items():
            if key == "total":
                continue
            episode.custom_metrics[key] = value

        for key, value in reward_breakdown.items():
            if key == "total":
                continue
            if reward_breakdown["total"] == 0:
                episode.custom_metrics[key + "_fraction"] = 0
            else:
                episode.custom_metrics[key + "_fraction"] = (
                    value / reward_breakdown["total"]
                )
        episode.custom_metrics["success"] = int(episode.last_info_for()["success"])


class DummyCallback(DefaultCallbacks):
    def on_episode_step(self, *, worker, base_env, policies, episode, **kwargs):
        pass

    def on_episode_end(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        pass


class VideoCallback(DefaultCallbacks):
    def on_episode_step(self, *, worker, base_env, policies, episode, **kwargs):
        if "video_frames" not in episode.user_data:
            episode.user_data["video_frames"] = []
        episode.user_data["video_frames"].append(base_env.try_render())

    def on_episode_end(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        frames = episode.user_data["video_frames"]
        save_path = worker.env_context["experiment_save_path"]
        name = worker.env_context["experiment_name"]
        video_folder = Path(save_path, name, "videos")
        video_folder.mkdir(parents=True, exist_ok=True)
        path = str(video_folder)
        # video_path = Path(self.video_path)
        video_path = f"{path}/episode-{episode.batch_builder.env_steps}.mp4"
        video = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, frames[0].shape[:2]
        )

        for frame in frames:
            screen = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(screen)
        video.release()
