python main.py \
	--output_dir logs/DINO/SWIN-5scale-2 -c config/DINO/CSPDL_DINO_5scale_swin_2.py --coco_path data \
	--dataset_file CSPDL --pretrain_model_path logs/DINO/SWIN-5scale-1/checkpoint_best_regular.pth --amp \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0
