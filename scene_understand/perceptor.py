from typing import Any
import numpy as np
from numpy.linalg import norm
from typing import Dict
import cv2
from collections import deque
from itertools import chain
from copy import copy
import torch
from torch.nn.functional import cosine_similarity

# from sklearn.metrics.pairwise import cosine_similarity
from torchmetrics.functional import pairwise_cosine_similarity
from scipy.spatial.distance import cdist

# from utils.utils import get_center_dist
from scene_understand.ocr_detector import OCR
from scene_understand.ests import ESTS
from scene_understand.scene_text_retrieval import STRetrieval
from utils.utils import get_elements_by_idxs


def find_connected_components(adj_matrix):
    if torch.is_tensor(adj_matrix):
        adj_matrix = adj_matrix.detach().cpu().numpy()
    assert (
        adj_matrix.shape[0] == adj_matrix.shape[1]
    ), "adjacency matrix should be a square matrix"

    N = adj_matrix.shape[0]
    clusters = []
    visited = np.zeros(N, dtype=np.bool_)
    for i in range(N):
        if visited[i]:
            continue
        cluster = []
        queue = deque([i])  # dfs
        visited[i] = True
        while queue:
            j = queue.popleft()
            cluster.append(j)
            for k in np.nonzero(adj_matrix[j])[0]:
                if not visited[k]:
                    queue.append(k)
                    visited[k] = True
        clusters.append(cluster)
    return clusters


## 3D scene text retrieval with fusion
class Perceptor:

    def __init__(self, cfg, ESTS_model=None):
        self.cfg = cfg
        print("init perceptor:scene_text_retrieval")
        self.scene_text_retrieval = STRetrieval(cfg, ESTS_model)

        self.instance_poses_list = []
        self.instance_poses_map_list = []
        self.instance_feats_list = []
        self.instance_counts = []
        self.instance_fuse_poses = []
        self.instance_fuse_poses_map = []
        self.instance_fuse_feats = []

        self.merge_dist_thred = cfg.merge_dist_thred  # 3
        self.merge_height_thred = cfg.merge_height_thred
        self.sim_thred = cfg.sim_thred  # 0.75
        self.merge_freq = cfg.merge_freq
        self.min_filter_count = cfg.min_filter_count
        self.min_instance_count = cfg.min_instance_count
        self.step = 0
        if not cfg.debug:
            self.init_scene_text_retrieval()

    def __len__(self,):
        return len(self.instance_fuse_feats)

    def init_scene_text_retrieval(self,):
        self.scene_text_retrieval.offline_process()
    
    def instance_merge_filter(
        self,
        instance_poses,
        instance_fuse_poses,
        instance_feats,
        instance_fuse_feats,
        instance_counts,
    ):
        count_filter_idx = [j for j in range(len(instance_counts))
                            if instance_counts[j] >= self.min_filter_count]
        before_num = len(instance_counts)
        if len(count_filter_idx) < len(instance_counts):
            instance_poses = get_elements_by_idxs(instance_poses, count_filter_idx)
            instance_fuse_poses = get_elements_by_idxs(instance_fuse_poses, count_filter_idx)
            instance_feats = get_elements_by_idxs(instance_feats, count_filter_idx)
            instance_fuse_feats = get_elements_by_idxs(instance_fuse_feats, count_filter_idx)
            instance_counts = get_elements_by_idxs(instance_counts, count_filter_idx)
        
        if not len(instance_counts):
            self.instance_poses_list = []
            self.instance_feats_list = []
            self.instance_fuse_poses = []
            self.instance_fuse_feats = []
            self.instance_counts = []
            print('before filter:', before_num, 'after:', 0)
            return 
        
        dist_matrix = cdist(
            np.array(instance_fuse_poses), np.array(instance_fuse_poses)
        )
        similarity_matrix = pairwise_cosine_similarity(
            torch.stack(instance_fuse_feats, dim=0), 
            torch.stack(instance_fuse_feats, dim=0)
        ).cpu().numpy()
        adjacency_matrix = (dist_matrix < self.merge_dist_thred) & (
            similarity_matrix > self.sim_thred
        )
        adjacency_matrix = adjacency_matrix | adjacency_matrix.T

        # merge instances based on the adjacency matrix
        connected_components = find_connected_components(adjacency_matrix)

        instance_counts_tensor = torch.from_numpy(np.array(instance_counts))
        merged_instance_count = []
        merged_instance_feats = []
        merged_instance_poses = []
        fused_instance_poses = []
        fused_instance_feats = []

        for i, cluster in enumerate(connected_components):
            merged_count = instance_counts_tensor[cluster].sum()
            merged_instance_count.append(merged_count.cpu().numpy().tolist())

            merged_poses = list(chain.from_iterable([instance_poses[j] for j in cluster]))
            merged_instance_poses.append(copy(merged_poses))
            fused_instance_poses.append(np.mean(merged_poses, axis=0))

            merged_feats = list(chain.from_iterable([instance_feats[j] for j in cluster]))
            merged_instance_feats.append(copy(merged_feats))
            fused_instance_feats.append(torch.mean(torch.stack(merged_feats, dim=0), 
                                                   dim=0))
        print('before filter:', before_num, 'after:', len(merged_instance_count))
        self.instance_poses_list = merged_instance_poses
        self.instance_feats_list = merged_instance_feats
        self.instance_fuse_poses = fused_instance_poses
        self.instance_fuse_feats = fused_instance_feats
        self.instance_counts = merged_instance_count

    def get_fuse_poses_map(self, idxs):
        return [self.instance_fuse_poses_map[idx] for idx in idxs]

    def get_fuse_poses(self, idxs):
        return [self.instance_fuse_poses[idx] for idx in idxs]

    def get_verified_fuse_poses(self,):
        return [self.instance_fuse_poses[idx] for idx in len(range(self.instance_fuse_poses))
            if self.instance_counts[idx] >= self.min_instance_count
        ]

    def semantic_fusion(self, instance_poses, instance_poses_map, ocr_results):
        if self.step % self.merge_freq == self.merge_freq - 1:
            print('periodic merge and filter at step', self.step)
            self.instance_merge_filter(
                self.instance_poses_list,
                self.instance_fuse_poses,
                self.instance_feats_list,
                self.instance_fuse_feats,
                self.instance_counts,
            )
        
        first = False
        register = False
        instance_feats = [item["feature"] for item in ocr_results]
        if not len(self.instance_fuse_feats):
            first = True
        else:
            similarity_matrix = pairwise_cosine_similarity(
                torch.stack(instance_feats).cuda(), torch.stack(self.instance_fuse_feats).cuda()
            ).cuda()
        new_instance = []
        new_instance_idx = []
        fuse_idx = []
        exist_instance_poses = copy(self.instance_poses_list)
        for i in range(len(instance_poses)):
            register = False
            if not first:
                dist_matrix = [
                    np.min(norm(instance_poses[i][:2] - np.array(exist_poses)[:, :2], axis=1))
                    for exist_poses in exist_instance_poses
                ]  # min dist to pointset
                height_matrix = [
                    np.min(np.abs(instance_poses[i][2] - np.array(exist_poses)[:, 2]))
                    for exist_poses in exist_instance_poses
                ]

                for j in np.argsort(
                    dist_matrix
                ):  # sort key segments by distance (ascending order)
                    if (  ## fusion condition
                        dist_matrix[j] < self.merge_dist_thred and 
                        height_matrix[j] < self.merge_height_thred and
                        similarity_matrix[i, j] > self.sim_thred
                    ):
                        print('fuse to', j, 'dist:', round(dist_matrix[j], 2), 'sim:', round(similarity_matrix[i, j].item(), 2), \
                            'height:', height_matrix[j])
                        self.instance_poses_list[j].append(instance_poses[i])
                        self.instance_fuse_poses[j] = np.mean(
                            self.instance_poses_list[j], axis=0
                        )
                        self.instance_feats_list[j].append(instance_feats[i])
                        self.instance_fuse_feats[j] = torch.mean(
                            torch.stack(self.instance_feats_list[j], dim=0),
                            dim=0,
                        )
                        self.instance_counts[j] += 1
                        register = True
                        fuse_idx.append(j)
                        new_instance.append(ocr_results[i])
                        break

            if not register:
                fuse_idx.append(len(self))
                new_instance_idx.append(i)
                new_instance.append(ocr_results[i])
                self.instance_poses_list.append([instance_poses[i]])
                self.instance_feats_list.append([instance_feats[i]])
                self.instance_fuse_poses.append(instance_poses[i])
                self.instance_fuse_feats.append(instance_feats[i])
                self.instance_counts.append(1)
                
        self.step += 1
        return new_instance, new_instance_idx, fuse_idx


    def retrieval(self, match_idxs, neighbor_lm_list, _type):
        if not len(match_idxs):
            return [], [], []
        query_features = torch.stack(
            [self.instance_fuse_feats[j] for j in match_idxs]
        ).cuda()
        matched_idxs, loc_texts, match_scores = self.scene_text_retrieval(
            query_features, neighbor_lm_list, _type
        )
        return matched_idxs, loc_texts, match_scores
    

