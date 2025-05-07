import os
from typing import Any
import cv2
import numpy as np
from typing import List
import matplotlib
# matplotlib.use('TkAgg')  # necessary for plotting results when using cnocr
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import torch

from utils.utils import OCR_filter, bb_intersection_over_union, union, random_crop, get_center, join_priority, mid_perpendicular_lines, merge_string, factorize, non_maximum_suppress


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __call__(self):
        return np.array([self.x, self.y])
    
class Reader:
    def __init__(self, ocr_type, use_gpu, use_reader_config, reader_config, min_size, text_nms, ESTS_model=None):
        self.ocr_type = ocr_type
        self.min_size = min_size
        self.text_nms = text_nms
        if ocr_type == 'ESTS':
            assert ESTS_model is not None
            self.reader = ESTS_model
        elif ocr_type == 'cnocr':

            from cnocr import CnOcr
            # ocr = CnOcr(rec_model_name='ch_PP-OCRv3') 
            if use_reader_config:
                self.reader = CnOcr(**reader_config)
            else:
                self.reader = CnOcr()
            # from cnstd import CnStd
            # self.std = CnStd(rotated_bbox=False, model_name='db_shufflenet_v2')  # default: onnx, , model_name='db_shufflenet_v2' , 'ch_PP-OCRv3_det', 'ch_PP-OCRv4'
        elif ocr_type == 'paddle':
            from paddleocr import PaddleOCR
            self.reader = PaddleOCR(use_angle_cls=True, lang='ch',  # chinese_cht, ch
                                    # det_algorithm='ch_PP-OCRv4_det',
                                    # rec_algorithm='chinese_cht_PP-OCRv3', 
                                    ocr_version='PP-OCRv3',
                                    use_onnx=False)
        elif ocr_type == 'easyocr':
            import easyocr
            self.reader = easyocr.Reader(['ch_tra','en'], gpu=use_gpu)
            self.blocklist = '[·’!"\#$%&\'()＃！（）*+,-./:;<=>?\@，：?￥★、…．＞【】［］《》？“”‘’\[\\]^_`{|}~]+'
            self.detail = 1
            self.easyocr_config = {
                'detail': self.detail,
                'blocklist': self.blocklist,
                'min_size': self.min_size,
            }
        else:
            raise ValueError(ocr_type)

    def __call__(self, img, encode=False):
        if self.ocr_type == 'ESTS':
            output = self.reader(img, encode=encode)
            res = [[box, text, score] for box, text, score in \
                   zip(output['boxes'], output['text'], output['scores'])]
            features = output['features']
            batch_idxs = output['batch_idxs']
            
        elif self.ocr_type == 'cnocr':
            ## for scene text detection
            res = self.reader.ocr(img)
            res = [[item['position'], item['text'], item['score']] for item in res]
            features = [[] for _ in range(len(res))] 
            batch_idxs = [0 for _ in range(len(res))] 
        elif self.ocr_type == 'paddle':
            res = []
            result = self.reader.ocr(img, cls=True)
            for info in result:
                if info is None:
                    continue
                for line in info:
                    box, (text, score) = line
                    res.append([np.array(box), text, score])
            features = [[] for _ in range(len(res))] 
            batch_idxs = [0 for _ in range(len(res))] 
        else:
            raise ValueError(self.ocr_type)

        res, features, batch_idxs = non_maximum_suppress(res, features, batch_idxs, self.text_nms)
        return res, features, batch_idxs
    
    def detect_only(self, img):
        if self.ocr_type == 'easyocr':
            res = self.reader.detect(img)
        elif self.ocr_type == 'cnocr':
            res = self.reader.detect(img)
        elif self.ocr_type == 'paddle':
            res, elapse = self.reader.text_detector(img)
        return res
    
    def recognize_only(self, cropped_img):
        if self.ocr_type == 'easyocr':
            img_grey = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
            text = self.reader.recognize(img_grey)
        elif self.ocr_type == 'cnocr':
            text, text_score = self.reader.ocr_for_single_line(cropped_img).values()
        elif self.ocr_type == 'paddle':
            img, text, elapse= self.reader.text_classifier([cropped_img])
        return text
    
    def filter_res_by_area(self, res:List, displace=100)-> List:
        filtered_res = []
        for j, (box, text, score) in enumerate(res):
            if len(box.shape) == 1:
                xl, yl, xr, yr = box
            else:
                xl, yl, xr, yr = box[0, 0], box[0, 1], box[2, 0], box[2, 1]
            area = (xr - xl) * (yr - yl)
            if area > self.min_size:
                filtered_res.append(res[j])
        return filtered_res


class OCR:
    def __init__(self, 
                 cfg,
                 ESTS_model=None
                 ):
        self.use_reader_config = cfg.cnocr_use_config
        self.reader_config = cfg.cnocr_config
        self.ocr_type = cfg.ocr_type
        self.use_gpu = cfg.use_gpu
        self.iou_thred = cfg.iou_thred
        self.prob_thred = cfg.prob_thred
        self.area_lowb = cfg.area_2d_bound
        self.detect_save_path = cfg.detect_save_path
        self.text_nms = cfg.text_nms
        print('ocr_type:', self.ocr_type)
        self.reader = Reader(self.ocr_type, self.use_gpu, 
                             self.use_reader_config, self.reader_config,
                             self.area_lowb,
                             self.text_nms,
                             ESTS_model)
    
    def detect(self, img):
        return self.reader.detect(img)
    
    def recognize(self, cropped_img):
        return self.reader.recognize(cropped_img)
    
    
    def batch_crop(self, image, batch_num=6, uniform=True):
        '''crop function'''
        batch_crop_imgs = [image]
        batch_rand_points = [[0, 0]]
        
        # uniform division can improve ocr detection results
        h, w = image.shape[:2]
        factors = factorize(batch_num)
        factor1, factor2 = factors if len(factors) > 1 else (1, factors[0])
        max_factor = max(factor1, factor2)
        min_factor = min(factor1, factor2)
        if h > w:
            stride_h = int(h / max_factor)
            stride_w = int(w / min_factor)
            h_len = max_factor
            w_len = min_factor
        else:
            stride_h = int(h / min_factor)
            stride_w = int(w / max_factor)
            h_len = min_factor
            w_len = max_factor
        
        crop_points = [(i * stride_h, j * stride_w) for i in range(h_len) for j in range(w_len)]
        for k in range(batch_num):
            if uniform:
                start_h, start_w = rand_points = crop_points[k]
                crop_img = image[start_h: start_h + stride_h, start_w: start_w + stride_h]
            else:
                crop_img, rand_points = random_crop(image)
            batch_crop_imgs.append(crop_img)
            batch_rand_points.append(rand_points)
        return batch_crop_imgs, batch_rand_points
    
    
    def batch_detect(self, batch_crop_imgs, batch_rand_points, encode, mask_words):
        batch_results = []
        prob_list = []
        batch_features = []
        
        def _if_mask(text):
            for mask in mask_words:
                if mask.lower() in text:
                    return True
            else:
                return False
        
        for i, crop in enumerate(batch_crop_imgs):
            result, feature, batch_idxs = self.reader(crop, encode)
            init_y, init_x = batch_rand_points[i]
            new_result = []
            feature_list = []
            for j, per_res in enumerate(result):
                filtered_res = OCR_filter(list(per_res), self.prob_thred, self.area_lowb)  # output: [bbox, text, prob]
                if filtered_res is None:
                    continue
                prob_list.append(filtered_res[2])
                bbox = np.array(filtered_res[0])
                bbox[:, 0] = (bbox[:, 0] + init_x)
                bbox[:, 1] = (bbox[:, 1] + init_y)
                filtered_res[0] = bbox
                
                new_result.append(filtered_res)
                feature_list.append(feature)
            batch_results.extend(new_result)
            batch_features.extend(feature_list)
        return batch_results, feature_list


    def merge_results(self, batch_results):
        Points_list = [[Point(*item[0][0]), Point(*item[0][2])] for item in batch_results]  # left top, right button
        num = len(Points_list)
        matrix = np.zeros((num, num), dtype=np.float32)
        cluster_k = 0
        merge_result_dict = {}
        idx_to_cluster = {}
        for i, Points1 in enumerate(Points_list):
            topleft1, rightbutten1 = Points1
            boxA = list(topleft1()) + list(rightbutten1())
            if i not in idx_to_cluster.keys():
                merge_result = batch_results[i].copy()
                merge_result[0] = [merge_result[0][0, 0], merge_result[0][0, 1],  # bbox 
                                    merge_result[0][2, 0], merge_result[0][2, 1]]
                merge_result[1] = [merge_result[1].lower()] * 2  # text
                merge_result[2] = [merge_result[2]]  # prob
                cluster_k = len(merge_result_dict)
                idx_to_cluster[i] = cluster_k
            else:
                cluster_k = idx_to_cluster[i]
                merge_result = merge_result_dict[cluster_k]

            for j, Points2 in enumerate(Points_list):
                if j == i:
                    continue
                topleft2, rightbutten2 = Points2
                boxB = list(topleft2()) + list(rightbutten2())
                iou = bb_intersection_over_union(boxA, boxB)
                
                iou_flag = int(iou > self.iou_thred)
                matrix[i][j] = iou_flag
                if  iou_flag and matrix[j][i] == 0:
                    center_ori = get_center(merge_result[0])
                    center_new = get_center(boxB)
                    merge_result[0] = union(merge_result[0], boxB)  # bbox
                    
                    idx_to_cluster[j] = cluster_k
                    if batch_results[j][1].lower() not in merge_result[1]:
                        # maintain the first place for joined text
                        priority = join_priority(center_ori, center_new)   # top left > right bottom
                        merge_result[1][0] = merge_string(merge_result[1][0], 
                                                          batch_results[j][1].lower(), 
                                                          priority)
                        merge_result[1].append(batch_results[j][1].lower())  # text
                        merge_result[2].append(batch_results[j][2])  # prob

            merge_result_dict[cluster_k] = merge_result

        return merge_result_dict
    


    def plot_results(self, img, merge_result_dict):
        img_plt = img[:, :, ::-1]  #! cv2 -> plt
        plt.imshow(img_plt)
        ax = plt.gca()
        for k, v in merge_result_dict.items():
            [lx, ly, rx, ry] = v[0]  # bbox
            text = v[1]
            rect = Rectangle((lx, ly),  rx - lx, ry - ly, linewidth=1, edgecolor='r', facecolor='none')
            ax.text(lx, ly, text)
            ax.add_patch(rect)
        plt.show()
        plt.savefig('./materials/exp_figs/test1.png')
        plt.clf()
    
    
    def __call__(self, img, 
                 timestep='', 
                 batch_crop=False,
                 batch_num=6, 
                 plot=False, 
                 save_res=False, 
                 encode=False, 
                 mask_words=[]):
        ## img shape: (h, w, 3)
        if isinstance(img, dict):
            rgb = img['rgb']
        else:
            rgb = img
        if batch_crop:
            batch_crop_imgs, batch_rand_points = self.batch_crop(rgb, batch_num)
            results, features = self.batch_detect(batch_crop_imgs, batch_rand_points, encode, mask_words)
            merge_result_dict = self.merge_results(results)
        else:
            # batch_crop_imgs, batch_rand_points = [rgb], [[0, 0]]
            results, features, batch_idxs = self.reader(rgb, encode)
            merge_result_dict = {}
            filter_features = []
            filter_batch_idxs = []
            for i in range(len(results)):
                res = OCR_filter(results[i], self.prob_thred, self.area_lowb) 
                if res is not None:
                    merge_result_dict[i] = res
                    filter_features.append(features[i])
                    filter_batch_idxs.append(batch_idxs[i])

        if plot:
            self.plot_results(rgb, merge_result_dict)

        results_list = []
        for j, value in enumerate(merge_result_dict.values()):
            if value is None:
                continue
            bbox, text, prob = value
            res = {
                'bbox': bbox,  # [xl, yl, xr, yr]
                'text': text,
                'prob': prob,
                'feature': filter_features[j] if not batch_crop else [],
                'batch_idx': filter_batch_idxs[j] if not batch_crop else [],
            }
            if save_res:
                name = f'video2_timestep{timestep}_idx{j}.png'
                file_path = os.path.join(self.detect_save_path, name)
                xl, yl, xr, yr = np.array(bbox, dtype=np.int64)
                cv2.imwrite(file_path, img[yl: yr, xl: xr])
                print(f'save detect object: timestep{timestep}_idx{j}')
            results_list.append(res)
        return results_list # bbox, landmark, prob, feature
    
