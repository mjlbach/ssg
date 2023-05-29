echo "No distractors, RGB"
python scripts/eval.py +experiment=icra/directed_search/rgb ++experiment_name=directed_search_icra_rgb_seed_0 ++seed=0
python scripts/eval.py +experiment=icra/directed_search/rgb ++experiment_name=directed_search_icra_rgb_seed_1 ++seed=1
python scripts/eval.py +experiment=icra/directed_search/rgb ++experiment_name=directed_search_icra_rgb_seed_2 ++seed=2

echo "Yes distractors, RGB"
python scripts/eval.py +experiment=icra/directed_search_distractors/rgb ++experiment_name=directed_search_icra_rgb_distractors_seed_0 ++seed=0
python scripts/eval.py +experiment=icra/directed_search_distractors/rgb ++experiment_name=directed_search_icra_rgb_distractors_seed_1 ++seed=1
python scripts/eval.py +experiment=icra/directed_search_distractors/rgb ++experiment_name=directed_search_icra_rgb_distractors_seed_2 ++seed=2

echo "No distractors, RGB + SG"
python scripts/eval.py +experiment=icra/directed_search/hgt ++experiment_name=directed_search_hgt_icra_seed_0 ++seed=0
python scripts/eval.py +experiment=icra/directed_search/hgt ++experiment_name=directed_search_hgt_icra_seed_1 ++seed=1
python scripts/eval.py +experiment=icra/directed_search/hgt ++experiment_name=directed_search_hgt_icra_seed_2 ++seed=2

echo "Yes distractors, RGB + SG"
python scripts/eval.py +experiment=icra/directed_search_distractors/hgt ++experiment_name=directed_search_hgt_distractors_icra_seed_0 ++seed=0
python scripts/eval.py +experiment=icra/directed_search_distractors/hgt ++experiment_name=directed_search_hgt_distractors_seed_1 ++seed=1
python scripts/eval.py +experiment=icra/directed_search_distractors/hgt ++experiment_name=directed_search_hgt_distractors_icra_seed_1 ++seed=2

echo "No distractors, RGB + SG + TD ATTN"
python scripts/eval.py +experiment=icra/directed_search/hfam ++experiment_name=directed_search_hfam_icra_seed_0 ++seed=0
python scripts/eval.py +experiment=icra/directed_search/hfam ++experiment_name=directed_search_hfam_icra_seed_1 ++seed=1
python scripts/eval.py +experiment=icra/directed_search/hfam ++experiment_name=directed_search_hfam_icra_seed_2 ++seed=2

echo "Yes distractors, RGB + SG + TD ATTN"
python scripts/eval.py +experiment=icra/directed_search_distractors/hfam ++experiment_name=directed_search_hfam_distractors_icra_seed_0 ++seed=0
python scripts/eval.py +experiment=icra/directed_search_distractors/hfam ++experiment_name=directed_search_hfam_distractors_icra_seed_1 ++seed=1
python scripts/eval.py +experiment=icra/directed_search_distractors/hfam ++experiment_name=directed_search_hfam_distractors_icra_seed_2 ++seed=2


