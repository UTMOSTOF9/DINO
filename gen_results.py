import json
from pathlib import Path

import numpy as np
import torch
from fire import Fire
from PIL import Image

import datasets.transforms as T
from main import build_model_main
from util import box_ops
from util.slconfig import SLConfig
from util.visualizer import COCOVisualizer

ID2Name = {
    1: "fish",
    2: "jellyfish",
    3: "penguin",
    4: "puffin",
    5: "shark",
    6: "starfish",
    7: "stingray"
}


def main(config_path: str, ckpt_path: str, data_folder: str, device: str = 'cuda', out_folder: str = 'results'):
    args = SLConfig.fromfile(config_path)
    args.device = 'cuda'
    model, _, postprocessors = build_model_main(args)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    _ = model.eval().to(device=device)

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    files = Path(data_folder).glob('*.jpg')
    out_folder = Path(out_folder)
    if not out_folder.exists():
        out_folder.mkdir(parents=True, exist_ok=True)

    results_json = {}

    for i, file in enumerate(files):
        image = Image.open(file).convert("RGB")  # load image
        blob, _ = transform(image, None)
        blob = blob.to(device=device)
        output = model(blob[None])
        output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).to(device=device))[0]

        # visualize outputs
        thershold = 0.3  # set a thershold

        vslzr = COCOVisualizer()

        scores = output['scores']
        labels = output['labels']
        boxes = output['boxes']
        boxes_to_plot = box_ops.box_xyxy_to_cxcywh(boxes)
        select_mask = scores > thershold

        box_label = [ID2Name[int(item)] for item in labels[select_mask]]
        pred_dict = {
            'boxes': boxes_to_plot[select_mask],
            'size': torch.Tensor([blob.shape[1], blob.shape[2]]),
            'box_label': box_label
        }
        vslzr.visualize(blob.cpu(), pred_dict, savepath=out_folder / f'{i:04d}.jpg', dpi=200)
        img_h, img_w = np.array(image).shape[:2]
        boxes = boxes.cpu().numpy().astype(float).round(5) * (img_w, img_h, img_w, img_h)
        scores = scores.cpu().numpy().astype(float).round(5)
        labels = labels.cpu().numpy().astype(int)
        results_json[file.name] = {
            'boxes': boxes.tolist(),
            'scores': scores.tolist(),
            'labels': labels.tolist(),
        }

    with open(out_folder / 'results.json', 'w') as f:
        json.dump(results_json, f, indent=4)


Fire(main)
