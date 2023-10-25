python main.py \
	--output_dir logs/DINO/SWIN-5scale-1 -c config/DINO/CSPDL_DINO_5scale_swin_1.py --coco_path data \
	--dataset_file CSPDL --pretrain_model_path backbone/checkpoint0027_5scale_swin.pth --amp \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0
