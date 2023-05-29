from igibson.utils.motion_planning_wrapper import MotionPlanningWrapper

from ssg.envs.igibson_env import iGibsonEnv


def main():

    config = {
        "build_graph": True,
        "clutter": False,
        "debug": False,
        "depth_high": 10.0,
        "depth_low": 0.0,
        "depth_noise_rate": 0.0,
        "discount_factor": 0.99,
        "dist_tol": 1,
        "eval_freq": 20000,
        "fisheye": False,
        "image_height": 512,
        "image_width": 512,
        "load_object_categories": ["agent", "floor"],
        "load_texture": True,
        "max_step": 500,
        "mode": "headless",
        "not_load_object_categories": ["ceilings"],
        "num_envs": 8,
        "object_randomization_freq": None,
        "output": ["rgb", "ins_seg", "depth", "sem_map", "occupancy_grid"],
        "pixel_reward_scaling": 350.0,
        "potential_reward_scaling": 1.0,
        "pybullet_load_texture": False,
        "resume": True,
        "reward_functions": ["pixel", "search", "visitation"],
        "robot": {
            "action_normalize": True,
            "action_type": "continuous",
            "base_name": None,
            "controller_config": {"base": {"name": "DifferentialDriveController"}},
            "name": "Turtlebot",
            "proprio_obs": ["dd_base_lin_vel", "dd_base_ang_vel"],
            "rendering_params": None,
            "scale": 1.0,
            "self_collision": False,
        },
        "save_freq": 200000,
        "scan_noise_rate": 0.0,
        "scene": "igibson",
        "scene_id": "Rs_int",
        "should_open_all_doors": False,
        "stack_frames": 0,
        "success_reward_scaling": 10.0,
        "texture_randomization_freq": None,
        "training_timesteps": 5000000,
        "urdf_file": "Rs_int",
        "vertical_fov": 120,
        "visitation_reward_scaling": 750.0,
    }

    env = iGibsonEnv(
        config_file=config,
        mode="gui_interactive",
        action_timestep=1 / 10.0,
        physics_timestep=1 / 40.0,
    )

    motion_planner = MotionPlanningWrapper(env)

    for _ in range(10):
        env.reset()
        plan = None
        itr = 0
        # x, y, z = env.task.target_obj.get_position()
        import pdb

        pdb.set_trace()
        while plan is None and itr < 10:
            plan = motion_planner.plan_base_motion([1.0, -1.5, 0])
            itr += 1

        motion_planner.dry_run_base_plan(plan)

    env.close()

    # Batch.from_data_list([data, data])


if __name__ == "__main__":
    main()
