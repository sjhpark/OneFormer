# ------------------------------------------------------------------------------
# This script is modified from the work done by github.com/ldcWV
# ------------------------------------------------------------------------------
import os
import sys
import cv2
import glob
import natsort
import argparse
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from termcolor import colored
import gc

import warnings
warnings.filterwarnings("ignore") # Turn off all warnings

import logging
logging.disable(logging.CRITICAL) # Disable all logging
logging.getLogger("tqdm").setLevel(logging.INFO) # Enable tqdm logging

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from demo.defaults import DefaultPredictor

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

    classes = {}
    for i, stuff_class in enumerate(metadata.stuff_classes):
        classes[i] = stuff_class
    print(f"Semantic classes:\n{classes}\n")

    return predictor, metadata

def process_frame(img, predictor):
    predictions = predictor(img, "semantic")
    pixel_classes = predictions["sem_seg"].argmax(dim=0).to('cpu') # 2D image of pixel classes
    return np.stack([pixel_classes, pixel_classes, pixel_classes], axis=2).astype(np.uint8)
    
def process_frames(input_chunk, predictor):
    # Get capture
    pid = input_chunk[0][0][0]
    fname = os.path.join('obs_videos', f"{pid}_obs.mkv")
    cap = cv2.VideoCapture(fname)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Process each frame
    res = []
    for inp in input_chunk[0]:
        _, frame = inp
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        img = cap.read()[1]
        res.append(process_frame(img, predictor))
    
    cap.release()
    
    # Create mkv of result
    start_frame = input_chunk[0][0]
    fname = os.path.join('obs_videos_segmented', f"{pid}_segmented_{start_frame}.mkv")
    height , width , channel =  res[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'FFV1') # codec for lossless compression (pixel values will be preserved)
    video = cv2.VideoWriter(fname, fourcc, fps, (width,height))
    for i in range(len(res)):
        video.write(res[i])
    video.release()
    gc.collect() # empty unused memory
    return fname

def process_video(pid, data_df, predictor):
    inputs = [(pid, data_df['obs_frame_num'].iloc[i]) for i in range(len(data_df))]
    CHUNK_SIZE = 500
    input_chunks = [[inputs[i:i+CHUNK_SIZE]] for i in range(0, len(inputs), CHUNK_SIZE)]

    outputs = list(tqdm(itertools.starmap(process_frames, zip(input_chunks, itertools.repeat(predictor))), desc="Processing frames", total=len(input_chunks)))
    
    fps = cv2.VideoCapture(outputs[0]).get(cv2.CAP_PROP_FPS)
    height , width , channel =  cv2.VideoCapture(outputs[0]).read()[1].shape
    segmented_fname = os.path.join('obs_videos_segmented', f"{pid}_segmented.mkv")
    fourcc = cv2.VideoWriter_fourcc(*'FFV1') # codec for lossless compression (pixel values will be preserved)
    segmented_video = cv2.VideoWriter(segmented_fname, fourcc, fps, (width,height))
    for chunk_fname in outputs:
        vid_cap = cv2.VideoCapture(chunk_fname)
        while vid_cap.isOpened():
            res, frame = vid_cap.read()
            if not res:
                break
            segmented_video.write(frame)
        vid_cap.release()
        os.remove(chunk_fname) # remove video of chunk

def run_obs_segmentation(pid:str, predictor): 
    # Extract frame_corrs
    # Columns are timestamp, obs_frame_num, glasses_frame_num, glasses_gaze_x, glasses_gaze_y
    data_df = pd.read_csv(os.path.join('frame_pairs', f"{pid}_frame_corrs.csv"))

    # Run segmentation modal
    process_video(pid, data_df, predictor)

def main():
    parser = argparse.ArgumentParser(description="oneformer demo for builtin configs")
    parser.add_argument("--id", type=str, help="Participant ID  (e.g. P00)", required=True)
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
    
    predictor, _ = setup_modules(prior, checkpoint, use_swin)

    run_obs_segmentation(args.id, predictor)

if __name__ == '__main__':
    # create save dir
    if not os.path.exists("obs_videos_segmented"):
        os.makedirs("obs_videos_segmented")
    main()
