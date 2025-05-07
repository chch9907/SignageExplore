import os
import cv2
from easydict import EasyDict as edict
import argparse
import numpy as np

from scene_understand.perceptor import Perceptor
from scene_understand.ocr_detector import OCR
from scene_understand.ests import ESTS
from graph.topo_graph import Graph
from utils.utils import read_yaml
from arguments import get_args

def main(cfg):
    print("init ESTS")
    ESTS_model = ESTS(cfg)
    
    print('init perceptor...')
    perceptor = Perceptor(cfg, ESTS_model)
    
    print('offline process text diffusion...')
    
    
if __name__ == '__main__':
    args = get_args()
    if args.scene == '1':
        config_path = './config/scene1.yaml'
    elif args.scene == '2':
        config_path = './config/scene2.yaml'
    else:
        raise ValueError(args.scene)
    print('scene:', args.scene)
    cfg = edict(read_yaml(config_path))
    cfg.debug = args.debug
    main(cfg)
