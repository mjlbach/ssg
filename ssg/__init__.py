import os

from igibson.robots.robot_base import REGISTERED_ROBOTS

from ssg.robots.fetch_magic import FetchMagic
from ssg.robots.fetch_reshelve import FetchReshelve
from ssg.robots.fetch_simple import Fetch
from ssg.robots.fetch_transport import FetchTransport
from ssg.robots.turtlebot_discrete import Turtlebot
from ssg.robots.turtlebot_discrete_transport import Turtlebot as TurtleTransport

ROOT_PATH = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(ROOT_PATH, "..", "configs")

# TODO: Remove once BehaviorRobot extends from proper BaseRobot class
REGISTERED_ROBOTS["Turtlefast"] = Turtlebot
REGISTERED_ROBOTS["TurtleTransport"] = TurtleTransport
REGISTERED_ROBOTS["Simplefetch"] = Fetch
REGISTERED_ROBOTS["FetchTransport"] = FetchTransport
REGISTERED_ROBOTS["FetchReshelve"] = FetchReshelve
REGISTERED_ROBOTS["FetchMagic"] = FetchMagic

from ssg.tasks.choice.choice_task import ChoiceTask
from ssg.tasks.binary_choice.choice_task import BinaryChoiceTask
from ssg.tasks.relational_search.relational_search_task import RelationalSearchTask
from ssg.tasks.search.search_task import SearchTask
from ssg.tasks.relational_search_simple.relational_search_task import RelationalSimpleSearchTask

REGISTERED_TASKS = {}
REGISTERED_TASKS["search"] = SearchTask
REGISTERED_TASKS["relational_search"] = RelationalSearchTask
REGISTERED_TASKS["relational_search_simple"] = RelationalSimpleSearchTask
REGISTERED_TASKS["choice"] = ChoiceTask
REGISTERED_TASKS["binary_choice"] = BinaryChoiceTask
