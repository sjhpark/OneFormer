# ------------------------------------------------------------------------------
# Reference: https://colab.research.google.com/github/SHI-Labs/OneFormer/blob/main/colab/oneformer_colab.ipynb
# Modified and Refactored by Sam Park (https://github.com/sjhpark)
# ------------------------------------------------------------------------------

"""
<Example: How to run this file>
python demo.py --config-file ../configs/coco/dinat/oneformer_dinat_large_bs16_100ep.yaml\
  --input ../sample_imgs/frame6250.jpg \
  --output ../semantic_inference/frame6250_semantic_coco_dinat   \
  --task semantic \
  --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS ../checkpoints/150_16_dinat_l_oneformer_coco_100ep.pth

python demo.py --config-file ../configs/coco/swin/oneformer_swin_large_bs16_100ep.yaml\
  --input ../sample_imgs/frame6250.jpg \
  --output ../semantic_inference/frame6250_semantic_coco_swin   \
  --task semantic \
  --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS ../checkpoints/150_16_swin_l_oneformer_coco_100ep.pth

python demo.py --config-file ../configs/cityscapes/dinat/oneformer_dinat_large_bs16_90k.yaml\
  --input ../sample_imgs/frame6250.jpg \
  --output ../semantic_inference/frame6250_semantic_cityscapes_dinat   \
  --task semantic \
  --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS ../checkpoints/250_16_dinat_l_oneformer_cityscapes_90k.pth
"""

import argparse
import multiprocessing as mp
import os
import torch
import random

from defaults import DefaultPredictor
from visualizer import Visualizer, ColorMode

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

# Import libraries
import numpy as np
import cv2
import torch

sys.path.insert(2, os.path.join(sys.path[1], 'detectron2/'))
# Import detectron2 utilities
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog


# import OneFormer Project
from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)

device = torch.device("cuda")
SWIN_CFG_DICT = {"cityscapes": "configs/cityscapes/oneformer_swin_large_IN21k_384_bs16_90k.yaml",
            "coco": "../configs/coco/swin/oneformer_swin_large_IN21k_384_bs16_100ep.yaml",
            "ade20k": "configs/ade20k/oneformer_swin_large_IN21k_384_bs16_160k.yaml",}

DINAT_CFG_DICT = {"cityscapes": "configs/cityscapes/oneformer_dinat_large_bs16_90k.yaml",
            "coco": "../configs/coco/dinat/oneformer_dinat_large_bs16_100ep.yaml",
            "ade20k": "configs/ade20k/oneformer_dinat_large_IN21k_384_bs16_160k.yaml",}

def setup_cfg(dataset, model_path, use_swin):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    add_dinat_config(cfg)
    add_convnext_config(cfg)
    add_oneformer_config(cfg)
    if use_swin:
      cfg_path = SWIN_CFG_DICT[dataset]
    else:
      cfg_path = DINAT_CFG_DICT[dataset]
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.DEVICE = 'cuda'
    cfg.MODEL.WEIGHTS = model_path
    cfg.freeze()
    return cfg

def setup_modules(dataset, model_path, use_swin):
    cfg = setup_cfg(dataset, model_path, use_swin)
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(
        cfg.DATASETS.TEST_PANOPTIC[0] if len(cfg.DATASETS.TEST_PANOPTIC) else "__unused"
    )
    return predictor, metadata

def panoptic_run(img, predictor, metadata):
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
    predictions = predictor(img, "panoptic")
    panoptic_seg, segments_info = predictions["panoptic_seg"]
    out = visualizer.draw_panoptic_seg_predictions(
    panoptic_seg.to('cpu'), segments_info, alpha=0.5
)
    return out

def instance_run(img, predictor, metadata):
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
    predictions = predictor(img, "instance")
    instances = predictions["instances"].to('cpu')
    out = visualizer.draw_instance_predictions(predictions=instances, alpha=0.5)
    return out

def semantic_run(img, predictor, metadata):
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
    predictions = predictor(img, "semantic")
    out = visualizer.draw_sem_seg(
        predictions["sem_seg"].argmax(dim=0).to('cpu'), alpha=0.5
    )
    return out

TASK_INFER = {"panoptic": panoptic_run, 
              "instance": instance_run, 
              "semantic": semantic_run}

predictor, metadata = setup_modules("coco", "../checkpoints/150_16_dinat_l_oneformer_coco_100ep.pth", use_swin=False)

img_name = 'ski.png'
img = f'../sample_imgs/{img_name}'
img = cv2.imread(img)
task = "semantic" #@param
print(f"Running inference...")
out = TASK_INFER[task](img, predictor, metadata).get_image()
out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
# save the output image
out_name = f'../demo/semantic_inference/{img_name[:-4]}_semantic_coco_dinat_colab_demo.png'
cv2.imwrite(out_name, out)
print(f"Output image saved as {out_name}")
