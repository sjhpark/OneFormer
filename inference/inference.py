# ------------------------------------------------------------------------------
# Reference: https://colab.research.google.com/github/SHI-Labs/OneFormer/blob/main/colab/oneformer_colab.ipynb
# Modified and Refactored by Sam Park (https://github.com/sjhpark)
# ------------------------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore") # Turn off all warnings

import logging
logging.disable(logging.CRITICAL) # Disable all logging
logging.getLogger("tqdm").setLevel(logging.INFO) # Enable tqdm logging

import os
import sys
import cv2
import time
import argparse
from tqdm import tqdm
from natsort import natsorted
import pandas as pd

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from demo.defaults import DefaultPredictor
from demo.visualizer import Visualizer, ColorMode

# import OneFormer Project
from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
  )

SWIN_CFG_DICT = {"cityscapes": "../configs/cityscapes/swin/oneformer_swin_large_bs16_90k.yaml",
                "coco": "../configs/coco/swin/oneformer_swin_large_IN21k_384_bs16_100ep.yaml",
                "ade20k": "../configs/ade20k/swin/oneformer_swin_large_bs16_160k.yaml",}

DINAT_CFG_DICT = {"cityscapes": "../configs/cityscapes/dinat/oneformer_dinat_large_bs16_90k.yaml",
                "coco": "../configs/coco/dinat/oneformer_dinat_large_bs16_100ep.yaml",
                "ade20k": "../configs/ade20k/dinat/oneformer_dinat_large_bs16_160k.yaml",}

SWIN_CHECKPOINT_DICT = {"cityscapes": "../checkpoints/250_16_swin_l_oneformer_ade20k_160k.pth",
                        "coco": "../checkpoints/150_16_swin_l_oneformer_coco_100ep.pth",
                        "ade20k": "../checkpoints/250_16_swin_l_oneformer_ade20k_160k.pth",}

DINAT_CHECKPOINT_DICT = {"cityscapes": "../checkpoints/250_16_dinat_l_oneformer_cityscapes_90k.pth",
                        "coco": "../checkpoints/150_16_dinat_l_oneformer_coco_100ep.pth",
                        "ade20k": "../checkpoints/250_16_dinat_l_oneformer_ade20k_160k.pth",}

sys.path.insert(2, os.path.join(sys.path[1], 'detectron2/'))
# Import detectron2 utilities
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog


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
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST_PANOPTIC[0] if len(cfg.DATASETS.TEST_PANOPTIC) else "__unused")
    # # Uncomment the line below to see the list of stuff classes
    # print(metadata.stuff_classes)
    return predictor, metadata

def panoptic_run(img, predictor, metadata):
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
    predictions = predictor(img, "panoptic")
    panoptic_seg, segments_info = predictions["panoptic_seg"]
    out = visualizer.draw_panoptic_seg_predictions(panoptic_seg.to('cpu'), segments_info, alpha=0.5)
    return out

def instance_run(img, predictor, metadata):
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
    predictions = predictor(img, "instance")
    instances = predictions["instances"].to('cpu')
    out = visualizer.draw_instance_predictions(predictions=instances, alpha=0.5)
    return out

def semantic_run(img, predictor, gaze_loc):
    predictions = predictor(img, "semantic")

    '''
    predictions["sem_seg"].argmax(dim=0) is a tensor of shape (H, W) with values from metadata.stuff_classes
    - metadata.stuff_classes (stuff classes = 80 thing classes + 53 extra classes)
    '''
    pixel_classes = predictions["sem_seg"].argmax(dim=0).to('cpu')
    H, W = pixel_classes.shape
    
    x_in_scene, y_in_scene = gaze_loc
    if not 0 < int(round(x_in_scene * W)) < W or not 0 < int(round(y_in_scene * H)) < H: # x or y > 1 usually means the gaze location was outside of the scene (or screen)
      pixel_class = -1
    else:
      x_in_img = int(round(x_in_scene * W))
      y_in_img = int(round(y_in_scene * H))
      pixel_class = pixel_classes[y_in_img, x_in_img].item()
    
    return pixel_class

TASK_INFER = {"panoptic": panoptic_run, 
              "instance": instance_run, 
              "semantic": semantic_run}

if __name__ == "__main__":
  """
  This script runs task (semantic) inference on extracted OBS frames and saves semantic pixel class per frame in a csv file.
  Currently, not applied for panoptic and instance tasks.
  """
  parser = argparse.ArgumentParser(description="oneformer demo for builtin configs")
  parser.add_argument("--in_dir", type=str, help="Directory of where the input images are", required=True)
  parser.add_argument("--gaze_path", type=str, help="Path to the gaze data (e.g. gaze_data/gaze_projection)", required=True)
  parser.add_argument("--model", type=str, default="dinat", help="Model (swin or dinat)")
  parser.add_argument("--prior", type=str, default="coco", help="Pretrained weights (coco, cityscapes, or ade20k)")
  parser.add_argument("--task", type=str, default="semantic", help="Task type")
  args = parser.parse_args()

  task = args.task
  model = args.model
  prior = args.prior

  if model == "swin":
    use_swin = True
    checkpoint = SWIN_CHECKPOINT_DICT[prior]
  elif model == "dinat":
    use_swin = False
    checkpoint = DINAT_CHECKPOINT_DICT[prior]
  predictor, metadata = setup_modules(prior, checkpoint, use_swin)
  
  # gaze projection data (contains scene gaze location for each frame)
  df = pd.read_csv(args.gaze_path, sep=",")

  # run segmentation inference on each frame
  images = natsorted(os.listdir(args.in_dir))
  for img_name in tqdm(images, desc="Running inference..."):
    img_num = int(img_name.split("_")[1].split(".")[0]) # image name is in the format of "frame_000000.png"; image_num helps you track fps (e.g. #10 means 10*1/60 seconds in the 60fps video)
    gaze_loc = (df.iloc[img_num].x_in_scene, df.iloc[img_num].y_in_scene) # scene gaze location as a tuple of (x_in_scene, y_in_scene)
    if gaze_loc == (-1, -1): # if no scene gaze data is available
      pixel_class = -1
    else:
      img = f'{args.in_dir}/{img_name}'
      img = cv2.imread(img)
      pixel_class = TASK_INFER[task](img, predictor, gaze_loc)
      df.loc[img_num, "pixel_class"] = pixel_class

  df.iloc[:, -1] = df.iloc[:, -1].fillna(-1) # replace NaN in the pixel class column with -1
  df.to_csv(f"{args.gaze_path[:-4]}_{task}.csv", index=False) # save the dataframe as a csv file
  print(f"Inference is complete. All the pixel classes of the frames are saved in {args.gaze_path[:-4]}_{task}.csv")

  logging.disable(logging.NOTSET) # Re-enable all logging at the end of your script if needed