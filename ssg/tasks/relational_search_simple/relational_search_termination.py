from igibson.termination_conditions.termination_condition_base import (
    BaseTerminationCondition,
)
from igibson.utils.utils import l2_distance


class SearchTermination(BaseTerminationCondition):
    """
    PointGoal used for PointNavFixed/RandomTask
    Episode terminates if point goal is reached
    """

    def __init__(self, config):
        super(SearchTermination, self).__init__(config)
        self.dist_tol = self.config.get("dist_tol", 1.0)

    def get_termination(self, task, env):
        """
        Return whether the episode should terminate.
        Terminate if point goal is reached (distance below threshold)

        :param task: task instance
        :param env: environment instance
        :return: done, info
        """
        done = (
            l2_distance(
                env.robots[0].get_position()[:2], task.target_obj.get_position()[:2]
            )
            < self.dist_tol
        )
        success = done
        return done, success


class FallTermination(BaseTerminationCondition):
    """
    PointGoal used for PointNavFixed/RandomTask
    Episode terminates if point goal is reached
    """

    def __init__(self, config):
        super().__init__(config)
        self.dist_tol = self.config.get("dist_tol", 1.0)

    def get_termination(self, task, env):
        _, _, z = env.robots[0].links["eyes"].get_position()
        if z < 0.7:
            return True, False
        else:
            return False, False
