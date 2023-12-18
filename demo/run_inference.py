# ------------------------------------------------------------------------------
# Reference 1: https://github.com/facebookresearch/Mask2Former/blob/main/demo/demo.py
# Modified by Jitesh Jain (https://github.com/praeclarumjj3)
# Reference 2: https://github.com/SHI-Labs/OneFormer/blob/main/demo/demo.py
# Modified by Sam Park (https://github.com/sjhpark)
# ------------------------------------------------------------------------------
import warnings
# Turn off all warnings
warnings.filterwarnings("ignore")

import logging
# Disable all logging
logging.disable(logging.CRITICAL)
# Enable tqdm logging
logging.getLogger("tqdm").setLevel(logging.INFO)

import argparse
import multiprocessing as mp
import os
import torch
import random
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import time
import numpy as np
import tqdm

sys.path.insert(2, os.path.join(sys.path[1], 'detectron2/'))
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)
from predictor import VisualizationDemo

# constants
WINDOW_NAME = "OneFormer Demo"

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    add_dinat_config(cfg)
    add_convnext_config(cfg)
    add_oneformer_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="oneformer demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="../configs/coco/dinat/oneformer_dinat_large_bs16_100ep.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument("--task", help="Task type")

    parser.add_argument(
        "--input",
        type=str,
        default="images_sample",
        help="directory of where the input images are",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    input_fnames = os.listdir(args.input) # input file names (e.g. sample1.jpg)
    input_fpaths = [os.path.join(args.input, fname) for fname in input_fnames] # input file paths (e.g. /images/sample1.jpg)

    start_time = time.time()
    for path in tqdm.tqdm(input_fpaths):
        # use PIL, to be consistent with evaluation
        
        # Run inference
        img = read_image(path, format="BGR")
        predictions, visualized_output = demo.run_on_image(img, args.task)

        # Save inferenced image
        for k in visualized_output.keys():
            opath = os.path.join(args.input, k)    
            os.makedirs(opath, exist_ok=True)
            out_filename = os.path.join(opath, os.path.basename(path))
            visualized_output[k].save(out_filename)    
    
    print(f"Total time elapsed: {(time.time() - start_time):.3f}s")
    print(f"All the inferenced images are saved in {os.path.join(args.input, k)}")

    # Re-enable all logging at the end of your script if needed
    logging.disable(logging.NOTSET)