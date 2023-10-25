export log_folder=$1

python gen_results.py \
	--config-path $log_folder/config_cfg.py \
	--ckpt-path $log_folder/checkpoint_best_regular.pth \
	--data-folder data/valid \
	--output-folder $log_folder/valid-results


python scripts/evaluate.py \
	--pred_path $log_folder/valid-results/results.json \
	--target_path data/annotations/val.json