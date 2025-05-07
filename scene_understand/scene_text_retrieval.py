import random
import numpy as np
import os
import cv2
import torch
from torch.nn.functional import cosine_similarity
from torchmetrics.functional import pairwise_cosine_similarity
import pickle
from collections import OrderedDict
from tqdm import tqdm
from utils.utils import non_maximum_suppress, to_numpy, _sim_one_to_multiple

    
class STRetrieval:
    def __init__(self, cfg, ESTS_model=None):
        self.text_img_path = cfg.text_img_path
        self._map_landmarks = []
        self.loc_thred = cfg.loc_thred
        self.topk_matching = cfg.topk_matching
        self.map_name = cfg.map_path.split('/')[-1].replace('.png', '')
        self.embedding_dict = OrderedDict()
        self.ocr_model = ESTS_model

    @torch.no_grad()
    def offline_generate_text_images(self, ):
        prompt_base = ''
        for map_lm in tqdm(self._map_landmarks):
            prompt = prompt_base + map_lm
            text_img = self.diffusion_generator(map_lm, prompt)
            cv2.imwrite(os.path.join(self.text_img_path, map_lm + '.png'), text_img)
        
    @torch.no_grad()
    def offline_process(self, show=False, encode=True):
        pickle_path = os.path.join(self.text_img_path, self.map_name + '.pkl')
        print('pickle_path:', pickle_path)
        if os.path.exists(pickle_path):
            print('load offline encoded image embeddings:', pickle_path)
            with open(pickle_path, 'rb') as f:
                self.embedding_dict = pickle.load(f)
            for k in tqdm(self.embedding_dict.keys()):
                self.embedding_dict[k] = self.embedding_dict[k].cuda()
        else:
            print('offline processing text images')
            assert self.ocr_model is not None, 'offline processing requires loading ocr model'
            for img_file in tqdm(os.listdir(self.text_img_path)):
                if img_file.endswith('.png'):
                    img = cv2.imread(os.path.join(self.text_img_path, img_file))
                    embedding = self.ocr_model(img, show, encode)['features']
                    self.embedding_dict[img_file.replace('.png', '')] = embedding[0]
                    print('emb shape:', len(embedding), embedding[0].shape)
            with open(pickle_path, 'wb') as f:
                pickle.dump(self.embedding_dict, f)
        print('embedding number:', len(self.embedding_dict))
                
    @torch.no_grad()
    def visual_text_matching(self, obs_text_features, neighbor_lm_list):
        lm_text_features = torch.stack(list(self.embedding_dict.values())).squeeze(1)
        similarity_matrix = pairwise_cosine_similarity(obs_text_features.reshape(-1, obs_text_features.shape[-1]), lm_text_features)
        print("similarity_matrix:", similarity_matrix)
        
        ## match by threshold
        is_matched = torch.nonzero(torch.max(similarity_matrix, dim=1)[0] >= 0)
        is_matched = is_matched.squeeze(1).cpu().numpy()
        similarity_matrix = similarity_matrix[is_matched]
        matched_idxs = torch.argsort(similarity_matrix, dim=1, descending=True)[:, :self.topk_matching]
        matched_idxs = matched_idxs.cpu().numpy()
        neighbor_lm_list = list(self.embedding_dict.keys())
        loc_texts = [neighbor_lm_list[topk_idx[0]] for topk_idx in matched_idxs]
        match_scores = [similarity_matrix[j, topk_idx[0]] for j, topk_idx in enumerate(matched_idxs)]
        print('!matcheds text:', loc_texts, 'score:', match_scores)
        return is_matched.tolist(), loc_texts, match_scores
    
    
    @staticmethod
    def edit_distance_matching(OCR_list, _map_landmarks, loc_thred=0.4):
        if not len(OCR_list):
            return [], [], []
        scores_list = []
        max_idx_list = []
        ## string matching to calculate the similarity
        for local_lm in OCR_list:
            sc_list = _sim_one_to_multiple(local_lm, _map_landmarks)
            scores_list.append(max(sc_list))
            max_idx_list.append(np.argmax(sc_list))

        pair_list = []
        indx_labels_probs = []
        for idx, (score, text, argmax_idx) in enumerate(zip(scores_list, OCR_list, max_idx_list)):
            if score >= loc_thred:
                pair_list.append((text, _map_landmarks[argmax_idx]))
                indx_labels_probs.append((idx, _map_landmarks[argmax_idx], score))
        
        indx_labels_probs = non_maximum_suppress(indx_labels_probs) 
        if not len(indx_labels_probs):
            return [], [], []
        return [item[0] for item in indx_labels_probs], [item[1] for item in indx_labels_probs],\
            [item[2] for item in indx_labels_probs]
                
                
    def __call__(self, obs_text_features, neighbor_lm_list, OCR_list=None, _type='visual-text-matching'):
        if _type == 'visual-text-matching':
            matched_idxs, loc_texts, match_scores = self.visual_text_matching(obs_text_features, neighbor_lm_list)
            return matched_idxs, loc_texts, match_scores
        elif _type == 'edit-distance':
            return self.edit_distance_matching(OCR_list, self._map_landmarks, self.loc_thred)
        else:
            raise ValueError("_type should be within ['visual-text-matching', 'edit-distance']")