import cv2
import numpy as np
import os, sys
from PIL import Image
import yaml
sys.path.append(os.path.dirname(__file__))  # ./
import matplotlib
matplotlib.use('TkAgg')
from arguments import get_args
from scene_understand.ocr_detector import OCR
from graph.topo_graph import Graph
from easydict import EasyDict as edict


if __name__ == '__main__':
    args = get_args()
    if args.scene == '1':
        config_path = './config/scene1.yaml'
    elif args.scene == '2':
        config_path = './config/scene2.yaml'
    else:
        raise ValueError(args.scene)
    print('scene:', args.scene)
    args_dict = vars(args)
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg.update(args_dict)
    cfg = edict(cfg)
    cfg.ocr_type = 'cnocr'
    OCRdet = OCR(cfg)

    _map = cv2.imread(cfg.map_path)
    map_path = './materials/guide_maps/scene1.png'
    _map = cv2.imread(map_path)
    
    map_landmarks = OCRdet(_map, batch_crop=True, batch_num=6, plot=False)
    print('before:', [item['text'][0] for item in map_landmarks])
    
    correct_map = {}
    remove_idxs = []
    for i, item in enumerate(map_landmarks):
        if item['text'][0] in correct_map.keys():
            map_landmarks[i]['text'][0] = correct_map[item['text'][0]]
        if '®' in item['text'][0]:
            remove_idxs.append(i)
    map_landmarks = [map_landmarks[i] for i in range(len(map_landmarks)) if i not in remove_idxs]
    print('after correction:', [item['text'][0] for item in map_landmarks])
    map_graph = Graph(_map, cfg, map_landmarks, is_global_map=True)
    map_graph.offline_process()
    map_graph.draw_on_image() 
    
    # map_graph.plot_graph_coord(text_list)
    
    # map_graph.draw_voronoi_on_image(_map)
    
    # route = map_graph.search_route(1, 20)


