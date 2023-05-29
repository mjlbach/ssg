from itertools import product
from pathlib import Path

import yaml

# @package _global_
experiment_sweep = {
    "normalize_env": [True, False],
    "randomize_reset": [True, False],
    "stack_frames": [0],
    "output": [["rgb", "ins_seg", "depth", "task_obs"], ["rgb", "ins_seg", "depth"]],
}

experiments = [
    dict(zip(experiment_sweep, v)) for v in product(*experiment_sweep.values())
]

common = {
    "experiment_name": "simple_search_unnormalized",
}


def bool_to_str(boolean):
    if boolean:
        return "T"
    else:
        return "F"


def first_letter(encoding):
    letters = []
    for element in encoding:
        letters.append(element[0])
    return "".join(letters)


for exp_id, experiment in enumerate(experiments):
    rand_res = bool_to_str(experiment["randomize_reset"])
    norm_env = bool_to_str(experiment["normalize_env"])
    stack_frames = experiment["stack_frames"]
    output = first_letter(experiment["output"])

    experiment_name = f"rr_{rand_res}_ne_{norm_env}_sf_{stack_frames}_obs_{output}"
    experiment["experiment_name"] = experiment_name
    if experiment["stack_frames"] > 5:
        experiment["num_envs"] = 4
    Path("./experiment").mkdir(parents=True, exist_ok=True)
    with open(f"./experiment/{experiment_name}.yaml", "w") as file:
        documents = yaml.dump(experiment, file)

    with open(f"./experiment/{experiment_name}.yaml", "r") as file:
        lines = file.readlines()
        lines = ["# @package _global_\n"] + lines

    with open(f"./experiment/{experiment_name}.yaml", "w") as file:
        lines = file.writelines(lines)
