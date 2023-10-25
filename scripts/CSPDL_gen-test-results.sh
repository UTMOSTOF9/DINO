export log_folder=$1

python gen_results.py \
	--config-path $log_folder/config_cfg.py \
	--ckpt-path $log_folder/checkpoint_best_regular.pth  \
	--data-folder data/test \
	--output-folder $log_folder/test-results