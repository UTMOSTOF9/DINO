from pathlib import Path

import datasets.transforms as T
import torch
from fire import Fire
from main import build_model_main
from PIL import Image
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


def main(config_path, ckpt_path, data_folder):
    args = SLConfig.fromfile(config_path)
    args.device = 'cuda'
    model, criterion, postprocessors = build_model_main(args)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    _ = model.eval().cuda()

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    files = Path(data_folder).glob('*.jpg')
    for file in files:
        image = Image.open(file).convert("RGB")  # load image
        blob, _ = transform(image, None)
        blob = blob.cuda()[None]
        output = model(blob)
        output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]

        # visualize outputs
        thershold = 0.3  # set a thershold

        vslzr = COCOVisualizer()

        scores = output['scores']
        labels = output['labels']
        boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
        select_mask = scores > thershold

        box_label = [ID2Name[int(item)] for item in labels[select_mask]]
        pred_dict = {
            'boxes': boxes[select_mask],
            'size': torch.Tensor([image.shape[1], image.shape[2]]),
            'box_label': box_label
        }
        vslzr.visualize(image, pred_dict, savedir=None, dpi=100)
        breakpoint()


Fire(main)
