# Task-Driven Graph attention for Hierarchical Object Navigation

## Installation 

Note: You may use conda instead of mamba

```bash
git clone --recursive git@github.com:mjlbach/ssg
cd ssg
conda env create -f environment.yml
conda activate ssg
bash install.sh
source .env
dvc pull
```

## Running

Before running, you will want to specify the path to ig_assets and ig_dataset:

```bash
export GIBSON_ASSETS_PATH=/home/michael/Documents/ig_data/assets
export IGIBSON_DATASET_PATH=/home/michael/Documents/ig_data/ig_dataset
```

This can be done automatically via a `.envrc` file and [direnv](https://direnv.net/).

Uses hydra, select the experiment with +experiment=path

```bash
python scripts/train.py +experiment=search ++experiment_save_path=/svl/u/mjlbach/ray_results ++experiment_name=search_test
```

Note: `++eval_frequency=0` and `++num_envs=1` will help when running locally

## Updating dependencies

Requires git 1.8.2 or above:

```
git submodule update --recursive --remote
```

If you did not clone the repo with `--recursive`, you will need to run this first:
```
git submodule update --init --recursive
```

## Updating

```bash
conda env update --file environment.yml --prune
```

## Technical debt

## Notes
* Should use ~ 5 gb RAM per worker on full Rs_int scene
* Needs about 50 gb RAM for 8 workers + driver
