export experiment=$1

python gen_results.py \
	--config-path $experiment/config_cfg.py \
	--ckpt-path $experiment/checkpoint_0004.pth \
	--data-folder data/valid \
	--output-folder $experiment/valid-results


python scripts/evaluate.py \
	$experiment/valid-results/results.json \
	data/annotations/val.json