enroot start --root \
	--mount $IGIBSON_DATASET_PATH:$IGIBSON_DATASET_PATH \
	--mount $GIBSON_ASSETS_PATH:$GIBSON_ASSETS_PATH \
	--mount $HOME/ray_results:/ray_results \
	ssg_sha_exp
