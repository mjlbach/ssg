import cv2
import numpy as np
from igibson.object_states.robot_related_states import ObjectsInFOVOfRobot
from igibson.reward_functions.reward_function_base import BaseRewardFunction
from igibson.utils.utils import l2_distance


class PickReward(BaseRewardFunction):
    """
    Point goal reward
    Success reward for reaching the goal with the robot's base
    """

    def __init__(self, config):
        super(PickReward, self).__init__(config)
        self.success_reward_scale = self.config["pick_reward_scaling"]
        self.dist_tol = 0.2
        self.name = "pick_reward"

    def reset(self, task, env):
        self.last_reward = 0
        self.last_goals_satisfied = 0

    def get_reward(self, task, env):
        """
        Check if the distance between the robot's base and the goal
        is below the distance threshold

        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        num_goals_satisfied = 0
        robot = env.robots[0]
        if robot.inventory != None:
            reward = self.success_reward_scale  # * (count - num_goals_satisfied)
        else:
            reward = 0
        temp_reward = reward
        reward = reward - self.last_reward
        self.last_reward = temp_reward
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
        x, y, z = env.robots[0].get_position()
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
            x, y, z = task.target_obj.get_position()
            coords = np.flip(
                (np.array([y, x]) / scene.seg_map_resolution + scene.seg_map_size / 2.0)
            ).astype(int)
            cv2.circle(new_map, coords, 2, [255, 0, 0], 1)
            new_map = cv2.resize(new_map, [512, 512])
            cv2.imshow("Visitation", new_map)
        reward = np.mean(self.visited_map) * self.visitation_reward_weight
        if self.visited_delta:
            temp_reward = reward.copy()
            reward = reward - self.last_reward
            self.last_reward = temp_reward
        return reward


class AdjacencyReward(BaseRewardFunction):
    """
    Point goal reward
    Adjacency reward for reaching the goal with the robot's base
    """

    def __init__(self, config):
        super(AdjacencyReward, self).__init__(config)
        self.success_reward = self.config["adjacency_reward_scaling"]
        self.dist_tol = 1.0
        self.name = "adjacency_reward"

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
        robot = env.robots[0]
        obj_ids = robot.states[ObjectsInFOVOfRobot].get_value()
        in_view = False
        if robot.inventory == None:
            for obj_id in obj_ids:
                obj = robot.simulator.scene.objects_by_id[obj_id]
                if obj.category == "notebook":
                    in_view = True
            success = (
                l2_distance(robot.get_position()[:2], task.source.get_position()[:2])
                < self.dist_tol
            ) and in_view
            reward = self.success_reward if success else 0.0
        else:
            reward = self.success_reward

        temp_reward = reward
        reward = reward - self.last_reward
        self.last_reward = temp_reward
        return reward


class ViewReward(BaseRewardFunction):
    """
    Point goal reward
    Adjacency reward for reaching the goal with the robot's base
    """

    def __init__(self, config):
        super(ViewReward, self).__init__(config)
        self.success_reward = self.config["view_reward_scaling"]
        self.name = "view_reward"

    def reset(self, task, env):
        self.last_reward = 0
        task.has_seen_goal = False

    def get_reward(self, task, env):
        """
        Check if the distance between the robot's base and the goal
        is below the distance threshold

        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        robot = env.robots[0]
        obj_ids = robot.states[ObjectsInFOVOfRobot].get_value()
        in_view = False
        if robot.inventory == None:
            for obj_id in obj_ids:
                obj = robot.simulator.scene.objects_by_id[obj_id]
                if obj.category == "notebook":
                    in_view = True
                    task.has_seen_goal = True
            success = in_view
            reward = self.success_reward if success else 0.0
        else:
            reward = self.success_reward

        temp_reward = reward
        reward = reward - self.last_reward
        self.last_reward = temp_reward
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
        task.has_seen_goal = False

    def get_shortest_path(self, env, task, entire_path=False):
        """
        Get the shortest path and geodesic distance from the robot or the initial position to the target position

        :param env: environment instance
        :param from_initial_pos: whether source is initial position rather than current position
        :param entire_path: whether to return the entire shortest path
        :return: shortest path and geodesic distance to the target position
        """
        source = env.robots[0].get_position()[:2]
        target = task.source.get_position()[:2]
        _, geodesic_distance = env.scene.get_shortest_path(
            0, source, target, entire_path=entire_path
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
        new_potential = self.get_shortest_path(env, task)
        reward = self.potential - new_potential
        reward *= self.potential_reward_weight
        self.potential = new_potential
        return reward
