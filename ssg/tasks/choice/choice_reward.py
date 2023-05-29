from igibson.reward_functions.reward_function_base import BaseRewardFunction
from igibson.utils.utils import l2_distance


class ChoiceReward(BaseRewardFunction):
    """
    Point goal reward
    Success reward for reaching the goal with the robot's base
    """

    def __init__(self, config):
        super(ChoiceReward, self).__init__(config)
        self.choice_reward = self.config["choice_scaling"]
        self.dist_tol = self.config.get("dist_tol", 0.5)
        self.name = "choice_reward"

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

        if success:
            reward = self.choice_reward
        else:
            reward = 0

        return reward


class FailurePenalty(BaseRewardFunction):
    """
    Point goal reward
    Success reward for reaching the goal with the robot's base
    """

    def __init__(self, config):
        super().__init__(config)
        self.scale = self.config["failure_scaling"]
        self.dist_tol = self.config.get("dist_tol", 0.5)
        self.name = "failure_penalty"

    def reset(self, task, env):
        self.failure_triggered = False

    def get_reward(self, task, env):
        """
        Check if the distance between the robot's base and the goal
        is below the distance threshold

        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        failure = (
            l2_distance(
                env.robots[0].get_position()[:2], task.distractor_obj.get_position()[:2]
            )
            < self.dist_tol
        )

        if failure:
            reward = -self.scale
        else:
            reward = 0

        return reward


class TimePenalty(BaseRewardFunction):
    """
    Point goal reward
    Success reward for reaching the goal with the robot's base
    """

    def __init__(self, config):
        super().__init__(config)
        self.scale = self.config.get("time_penalty", 1)
        self.name = "time_penalty"

    def get_reward(self, task, env):
        return -self.scale


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
