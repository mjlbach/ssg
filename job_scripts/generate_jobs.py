#!/usr/bin/env/python
import datetime
import os
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

import ssg

now = datetime.datetime.now().strftime("%b-%d_%H-%M-%S")
# Capture our current directory
experiment_dir = os.path.join(ssg.CONFIG_PATH, "experiment")
job_scripts = os.path.join(ssg.CONFIG_PATH, "..", "job_scripts")
out_dir = os.path.join(ssg.CONFIG_PATH, "..", "job_scripts", now)
template_dir = os.path.join(ssg.CONFIG_PATH, "..", "job_scripts", "templates")


def generate_experiments():
    # Create the jinja2 environment.
    # Notice the use of trim_blocks, which greatly helps control whitespace.
    j2_env = Environment(loader=FileSystemLoader(template_dir), trim_blocks=True)
    file_lists = []
    for job_id, experiment in enumerate(os.listdir(experiment_dir)):
        basename, _ = os.path.splitext(experiment)
        templated_file = j2_env.get_template("job_template.sh.template").render(
            experiment=basename,
            job_id=basename,
        )
        templated_file_path = os.path.join(out_dir, f"{basename}.sh")
        filename = Path(templated_file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(templated_file_path, "w") as f:
            f.write(templated_file)
        filename = Path(templated_file_path).resolve().relative_to(Path.cwd())
        file_lists.append(filename)

    return file_lists


def generate_manifest(job_list):
    # Create the jinja2 environment.
    # Notice the use of trim_blocks, which greatly helps control whitespace.
    j2_env = Environment(loader=FileSystemLoader(template_dir), trim_blocks=True)
    templated_file = j2_env.get_template("job_manifest.sh.template").render(
        job_list=job_list
    )
    templated_file_path = os.path.join(out_dir, f"start_jobs.sh")
    with open(templated_file_path, "w") as f:
        f.write(templated_file)


if __name__ == "__main__":
    file_lists = generate_experiments()
    generate_manifest(file_lists)
