import cv2
import numpy as np
from igibson.object_states.robot_related_states import ObjectsInFOVOfRobot
from igibson.reward_functions.reward_function_base import BaseRewardFunction
from igibson.utils.utils import l2_distance


class SearchReward(BaseRewardFunction):
    """
    Point goal reward
    Success reward for reaching the goal with the robot's base
    """

    def __init__(self, config):
        super(SearchReward, self).__init__(config)
        self.success_reward = self.config["success_reward_scaling"]
        self.dist_tol = self.config.get("dist_tol", 0.5)
        self.name = "search_reward"

    def get_reward(self, task, env):
        """
        Check if the distance between the robot's base and the goal
        is below the distance threshold

        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        success = (
            l2_distance(
                env.robots[0].get_position()[:2], task.target_obj.get_position()[:2]
            )
            < self.dist_tol
        )

        reward = self.success_reward if success else 0.0
        return reward


class VisitationReward(BaseRewardFunction):
    """
    Point goal reward
    Success reward for reaching the goal with the robot's base
    """

    def __init__(self, config, debug=False):
        super(VisitationReward, self).__init__(config)
        self.debug = debug


        self.visitation_reward_weight = self.config["visitation_reward_scaling"]
        self.visited_delta = self.config.get("visited_delta", True)
        self.name = "visitation_reward"
        if self.debug:
            cv2.namedWindow("Visitation")

    def reset(self, task, env):
        self.visited_map = np.zeros([100, 100], dtype=np.int8)
        self.last_reward = 0

    def get_reward(self, task, env):
        """
        Check if the distance between the robot's base and the goal
        is below the distance threshold

        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        x, y, _ = env.robots[0].get_position()
        scene = env.simulator.scene
        coords = np.flip(
            (np.array([x, y]) / scene.seg_map_resolution + scene.seg_map_size / 2.0)
        ).astype(int)
        coords = np.clip(coords, a_min=0, a_max=99)
        if env.simulator.scene.get_room_type_by_point(np.array([x, y])) is not None:
            self.visited_map[coords[0], coords[1]] = 1
        if self.debug:
            print(coords)
            new_map = (
                np.zeros([self.visited_map.shape[0], self.visited_map.shape[1], 3]) + 1
            )
            new_map[self.visited_map == 1] = [0, 0, 0]
            x, y, _ = task.target_obj.get_position()
            coords = np.flip(
                (np.array([y, x]) / scene.seg_map_resolution + scene.seg_map_size / 2.0)
            ).astype(int)
            cv2.circle(new_map, coords, 2, [255, 0, 0], 1) #type: ignore
            new_map = cv2.resize(new_map, [512, 512]) #type: ignore
            cv2.imshow("Visitation", new_map) #type: ignore
        reward = np.mean(self.visited_map) * self.visitation_reward_weight
        if self.visited_delta:
            temp_reward = reward.copy()
            reward = reward - self.last_reward
            self.last_reward = temp_reward
        return reward


class ObservationReward(BaseRewardFunction):
    """
    Point goal reward
    Success reward for reaching the goal with the robot's base
    """

    def __init__(self, config):
        super(ObservationReward, self).__init__(config)
        self.name = "observation_reward"
        self.observation_reward_weight = self.config["observation_reward_scaling"]
        self.last_reward = 0

    def reset(self, task, env):
        self.last_reward = 0

    def get_reward(self, task, env):
        """
        Check if the distance between the robot's base and the goal
        is below the distance threshold

        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        sem_map = env.last_state["ego_sem_map"]
        last_nonzero = np.count_nonzero(sem_map)
        reward = last_nonzero / sem_map.size * self.observation_reward_weight

        # Penalize agent for falling off
        x, y, z = env.robots[0].get_position()
        if env.simulator.scene.get_room_type_by_point(np.array([x, y])) is None:
            reward = 0

        temp_reward = reward
        reward = reward - self.last_reward
        self.last_reward = temp_reward

        # Do not reward for first timestep
        return reward


class PixelReward(BaseRewardFunction):
    """
    Pixel reward
    """

    def __init__(self, config):
        super(PixelReward, self).__init__(config)
        self.pixel_reward_scaling = self.config["pixel_reward_scaling"]
        self.pixel_delta = self.config.get("pixel_delta", True)
        self.last_reward = 0
        self.name = "pixel_reward"

    def reset(self, task, env):
        self.last_reward = 0

    def get_reward(self, task, env):
        # seg = np.round(seg * MAX_INSTANCE_COUNT).astype(int)
        body_ids = env.simulator.renderer.get_pb_ids_for_instance_ids(
            env.last_state["ins_seg"]
        )
        reward = (
            np.mean(body_ids == task.target_obj.get_body_ids()[0])
            * self.pixel_reward_scaling
        )

        if self.pixel_delta:
            temp_reward = reward.copy()
            reward = reward - self.last_reward
            self.last_reward = temp_reward

        # Do not reward for first timestep
        if env.current_step == 1:
            return 0

        return reward


class PotentialReward(BaseRewardFunction):
    """
    Potential reward
    Assume task has get_potential implemented; Low potential is preferred
    (e.g. a common potential for goal-directed task is the distance to goal)
    """

    def __init__(self, config):
        super(PotentialReward, self).__init__(config)
        self.potential_reward_weight = self.config["potential_reward_scaling"]
        self.name = "potential_reward"

    def reset(self, task, env):
        """
        Compute the initial potential after episode reset

        :param task: task instance
        :param env: environment instance
        """
        self.potential = self.get_shortest_path(env, task)

    def get_shortest_path(self, env, task, entire_path=False):
        """
        Get the shortest path and geodesic distance from the robot or the initial position to the target position

        :param env: environment instance
        :param from_initial_pos: whether source is initial position rather than current position
        :param entire_path: whether to return the entire shortest path
        :return: shortest path and geodesic distance to the target position
        """
        source = env.robots[0].get_position()[:2]
        target = task.target_obj.get_position()[:2]
        _, geodesic_distance = env.scene.get_shortest_path(0, source, target, entire_path=entire_path)
        return geodesic_distance

    def get_reward(self, task, env):
        """
        Reward is proportional to the potential difference between
        the current and previous timestep

        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        new_potential = self.get_shortest_path(env, task)
        reward = self.potential - new_potential
        reward *= self.potential_reward_weight
        self.potential = new_potential
        return reward


class TriggeredPotentialReward(BaseRewardFunction):
    """
    Potential reward
    Assume task has get_potential implemented; Low potential is preferred
    (e.g. a common potential for goal-directed task is the distance to goal)
    """

    def __init__(self, config):
        super().__init__(config)
        self.potential_reward_weight = self.config["potential_reward_scaling"]
        self.name = "potential_reward"

    def reset(self, task, env):
        """
        Compute the initial potential after episode reset

        :param task: task instance
        :param env: environment instance
        """
        self.potential = self.get_shortest_path(env, task)
        task.has_seen_goal = False

    def get_shortest_path(self, env, task, entire_path=False):
        """
        Get the shortest path and geodesic distance from the robot or the initial position to the target position

        :param env: environment instance
        :param from_initial_pos: whether source is initial position rather than current position
        :param entire_path: whether to return the entire shortest path
        :return: shortest path and geodesic distance to the target position
        """
        # source = env.robots[0].get_position()[:2]
        # target = task.source.get_position()[:2]
        source = env.robots[0].get_position()[:2]
        target = task.target_obj.get_position()[:2]
        _, geodesic_distance = np.array(
            env.scene.get_shortest_path(0, source, target, entire_path=entire_path)
        )
        return geodesic_distance

    def get_reward(self, task, env):
        """
        Reward is proportional to the potential difference between
        the current and previous timestep

        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        robot = env.robots[0]
        obj_ids = robot.states[ObjectsInFOVOfRobot].get_value()
        # if robot.inventory == None:
        for obj_id in obj_ids:
            try:
                obj = robot.simulator.scene.objects_by_id[obj_id]
                if obj.category == "microwave":
                    task.has_seen_goal = True
            except:
                pass

        new_potential = self.get_shortest_path(env, task)
        reward = self.potential - new_potential
        reward *= self.potential_reward_weight
        self.potential = new_potential

        if task.has_seen_goal:
            return reward
        else:
            return 0
        # else:
        #     return 0
