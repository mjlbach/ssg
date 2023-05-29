import igibson
import numpy as np
from igibson.object_states import OnTop
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
        body_ids = env.simulator.renderer.get_pb_ids_for_instance_ids(
            env.last_state["ins_seg"]
        )
        in_view = np.count_nonzero(body_ids == task.target_obj.get_body_ids()[0]) > 100
        done = done and in_view
        success = done
        return done, success


class ReshelveTermination(BaseTerminationCondition):
    """
    PointGoal used for PointNavFixed/RandomTask
    Episode terminates if point goal is reached
    """

    def __init__(self, config):
        super(ReshelveTermination, self).__init__(config)
        self.dist_tol = 0.2

    def get_termination(self, task, env):
        """
        Return whether the episode should terminate.
        Terminate if point goal is reached (distance below threshold)

        :param task: task instance
        :param env: environment instance
        :return: done, info
        """
        done = True
        print(task.books)
        print("starting")
        for obj in task.books:
            inside = obj.states[igibson.object_states.Inside].get_value(task.target)
            print(inside)
            done = done and inside
        print("stopping")

        success = done
        return done, success


class TransportTermination(BaseTerminationCondition):
    """
    PointGoal used for PointNavFixed/RandomTask
    Episode terminates if point goal is reached
    """

    def __init__(self, config):
        super(TransportTermination, self).__init__(config)
        self.dist_tol = 0.2

    def get_termination(self, task, env):
        """
        Return whether the episode should terminate.
        Terminate if point goal is reached (distance below threshold)

        :param task: task instance
        :param env: environment instance
        :return: done, info
        """
        done = True
        for item in task.items:
            done = done and item.states[OnTop].get_value(task.target)

        success = done
        return done, success


class PickTermination(BaseTerminationCondition):
    """
    PointGoal used for PointNavFixed/RandomTask
    Episode terminates if point goal is reached
    """

    def __init__(self, config):
        super(PickTermination, self).__init__(config)

    def get_termination(self, task, env):
        """
        Return whether the episode should terminate.
        Terminate if point goal is reached (distance below threshold)

        :param task: task instance
        :param env: environment instance
        :return: done, info
        """
        done = success = env.robots[0].inventory != None
        return done, success
