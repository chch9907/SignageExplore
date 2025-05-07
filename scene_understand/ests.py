# from datasets import transforms as T 
from torchvision import transforms as T
import pickle
import numpy as np
import time
import torch
from scene_understand.ESTextSpotter.models.registry import MODULE_BUILD_FUNCS
from scene_understand.ESTextSpotter.util.slconfig import SLConfig
from utils.visualizer import COCOVisualizer
from utils.utils import to_numpy

transform = T.Compose([
    # T.RandomResize([(1000, 1000)], max_size=1100),
    T.ToTensor(),
    T.Resize([1000]),  # large image size results in high performance but low speed on edge devides
    T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])]
)

with open('./scene_understand/ESTextSpotter/chn_cls_list.txt', 'rb') as fp:
    CTLABELS = pickle.load(fp)
    
def _decode_recognition(rec):
    s = ''
    for c in rec:
        c = int(c)
        if c < 5461:
            s += str(chr(CTLABELS[c]))
        elif c == 5462:
            s += u''
    return s

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    args.device = 'cuda'
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors
    
class ESTS:
    def __init__(self, cfg):
        self._model_init(cfg.ESTS_model_path, cfg.ESTS_config_path)
        self.prob_thred = cfg.prob_thred
        
    def _model_init(self, model_path, model_config_path):
        # change the paths of the model and checkpoint in config file
        print('init ESTS:', model_path)
        args = SLConfig.fromfile(model_config_path) 
        self.ocr_model, criterion, self.postprocessors = build_model_main(args)
        checkpoint = torch.load(model_path, map_location='cpu')
        self.ocr_model.load_state_dict(checkpoint['model'])
        self.ocr_model.eval()
        self.ocr_model.cuda()
    
    def single_inference(self, ori_image, encode, show):
        ori_h, ori_w = ori_image.shape[:2]
        image = transform(ori_image)
        st = time.time()
        with torch.no_grad():
            raw_output = self.ocr_model(image[None].cuda(), encode=encode)
            print('ests time:', time.time() - st)
            
        h, w = image.shape[1:]
        img_shape = torch.Tensor([[h, w]])  # h, w
        output = self.postprocessors['bbox'](raw_output, torch.Tensor([[1.0, 1.0]]))[0]
        rec = [_decode_recognition(i) for i in output['rec']]
        scores = to_numpy(output['scores'])
        labels = to_numpy(output['labels'])
        boxes = to_numpy(output['boxes'])
        select_mask = scores > self.prob_thred 
        recs = []
        for mask, r in zip(select_mask,rec):
            if mask:
                recs.append(r)
        boxes = boxes[select_mask]
        resized_boxes = boxes.copy()
        
        resized_boxes[:, 0] *= ori_w
        resized_boxes[:, 2] *= ori_w
        resized_boxes[:, 1] *= ori_h
        resized_boxes[:, 3] *= ori_h
        scores = scores[select_mask]
        pred_dict = {
            'boxes': resized_boxes,
            'size': [image.shape[1], image.shape[2]],
            'text': recs,
            'scores': scores,
            'batch_idxs': [0] * len(recs),
        }
        
        ### select embedding
        if encode:
            pred_dict['features'] = raw_output['outputs_class_neck']
        else:
            pred_dict['features'] = [[] for _ in range(len(recs))]
        if show:
            boxes = box_xyxy_to_cxcywh(boxes)
            vslzr = COCOVisualizer()
            vslzr.visualize(image, pred_dict, savedir='vis_fin')
        return pred_dict
    
    
    def batch_inference(self, image_list, encode, show, topk_areas=2):
        raw_image_shape = []
        batch_images = []
        batch_target_sizes = []
        for image in image_list:
            raw_image_shape.append(image.shape[:2])
            image = transform(image)
            batch_images.append(image)
            batch_target_sizes.append([1.0, 1.0])
        batch_images = torch.stack(batch_images, dim=0)
        batch_target_sizes = torch.tensor(batch_target_sizes).cuda()
        st = time.time()
        with torch.no_grad():
            raw_output = self.ocr_model(batch_images.cuda(), encode=encode)
        

        batch_outputs = self.postprocessors['bbox'](raw_output, batch_target_sizes)
        batch_boxes = []
        batch_texts = []
        batch_scores = []
        batch_idxs = []
        for idx, (output, raw_shape) in enumerate(zip(batch_outputs, raw_image_shape)):
            rec = [_decode_recognition(i) for i in output['rec']]
            
            scores = to_numpy(output['scores'])
            labels = to_numpy(output['labels'])
            boxes = to_numpy(output['boxes'])
            select_mask = scores > self.prob_thred
            recs = []
            
            for mask, r in zip(select_mask,rec):
                if mask:
                    recs.append(r)
            # print('recs:', recs)
            scores = scores[select_mask]
            boxes = boxes[select_mask]
            areas = (boxes[:, 0] - boxes[:, 2]) * (boxes[:, 1] - boxes[:, 3])
            if len(recs):
                topk_idxs = np.argsort(areas)[::-1][:topk_areas]
                resized_boxes = boxes.copy()[topk_idxs]
                ori_h, ori_w = raw_shape
                resized_boxes[:, 0] *= ori_w
                resized_boxes[:, 2] *= ori_w
                resized_boxes[:, 1] *= ori_h
                resized_boxes[:, 3] *= ori_h

                scores = scores[topk_idxs]
                recs = [recs[j] for j in topk_idxs]
                batch_boxes.extend([boxes for boxes in resized_boxes])
                batch_scores.extend(scores)
                batch_texts.extend(recs)
                batch_idxs.extend([idx] * len(recs))
            
        pred_dict = {
            'boxes': batch_boxes,
            'size': batch_target_sizes,
            'text': batch_texts,
            'scores': batch_scores,
            'batch_idxs': batch_idxs,
        }
        
        ### select embedding
        if encode:
            pred_dict['features'] = raw_output['outputs_class_neck']
        else:
            pred_dict['features'] = [[] for _ in range(len(batch_texts))]
        if show:
            boxes = box_xyxy_to_cxcywh(boxes)
            vslzr = COCOVisualizer()
            vslzr.visualize(image, pred_dict, savedir='vis_fin')
        return pred_dict
    
    def __call__(self, image, show=False, encode=False):
        if isinstance(image, list):
            pred_dict = self.batch_inference(image, encode, show)
        else:
            pred_dict = self.single_inference(image, encode, show)
            

        return  pred_dict
