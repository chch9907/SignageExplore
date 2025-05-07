#!/usr/bin/env python
import os
import cv2
from std_srvs.srv import Empty
import rospy
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from nav_msgs.msg import OccupancyGrid, MapMetaData
from std_msgs.msg import Int8MultiArray
import tf
from itertools import chain
from geometry_msgs.msg import Twist, Point, PoseArray
from tf.transformations import euler_from_quaternion
from copy import copy
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from utils.utils import _normalize_heading, read_map_pgm, _world_to_map, _world_to_map2, UNKNOWN, OBSTACLE, get_elements_by_idxs
from .contour_detector import frontier_detection, find_contours_and_cluster

color_list = ['magenta', 'pink', 'cyan',  'lightgreen',  'mediumpurple', 'skyblue', 'lime', 'olive', 'azure', 'blue']  # 'yellow', '#F39C12', 'brown', 

class Mapper:
    def __init__(self, ros, cfg):
        self.ros = ros
        self.cur_pose_world = np.array([0, 0, 0])
        self.cur_pose_map = [0, 0]
        self.prev_pose_world = np.array([0, 0, 0])
        self.prev_pose_map = [0, 0]
        self.first_state = True
        self.crop_dist = cfg.crop_dist
        self.cluster_dist = cfg.contour_cluster_dist
        self.filter_contours = cfg.filter_contours
        self.near_dist = cfg.near_dist
        self.movement_satisfy = True
        self.clustered_contours = []
        self.semantic_map_dict = OrderedDict()
        self.step = 0
        self.render_size = cfg.render_size
        self.output_dir = cfg.output_dir
    
    
    def check_movement_satisfy(self, ):
        self.movement_satisfy = self.first_state or \
            np.linalg.norm((self.cur_pose_world - self.prev_pose_world)[:2]) > self.update_movement
        if self.movement_satisfy:
            self.prev_pose_map = self.cur_pose_map
            if self.first_state:
                self.first_state = False

    @staticmethod
    def read_disk_occupancy_map(prefix='./utils/jueying'):
        map_array = read_map_pgm(prefix, vis=False)
        return map_array
    
    def get_clustered_contours(self, cur_pose=None, step=1, show_contours=False, show_clusters=False):
        if cur_pose is None:
            cur_pose = self.ros.get_tf()
        cur_pose_map = self.ros.get_pose_map(cur_pose)
        self.clustered_contours = find_contours_and_cluster(self.ros.map_array, cur_pose_map, 
                                                    crop_dist=None,
                                                    cluster_dist=self.cluster_dist,
                                                    step=step,
                                                    show_contours=show_contours,
                                                    show_clusters=show_clusters)
        return self.clustered_contours
    
    
    def project_semantics(self, registered_poses, registered_texts, contours, viewpoints_world, predicted_lmpose_world, 
                          next_subgoal_world, next_subgoal_idx, cur_pose, step, exp_id='', max_contour_length=120, 
                          plot=True, save=False, is_nbv=False):
        self.semantic_map_dict = [] 
        loc_texts = registered_texts
        fuse_poses = registered_poses 
        indice_list = [0]
        for ctr in contours:
            indice_list.append(len(ctr) + indice_list[-1])

        stacked_contours = np.vstack(contours)  # coord: (x, y)
        
        argmin_list = []
        register_ctr_idxs = []
        argmin_list = []
        filter_contour_idxs = []
        fuse_poses_map = np.array([_world_to_map(self.ros.mapData, point) # (x, y)
                          for point in fuse_poses])  #! fuse_poses_map can be None
        for per_pose in fuse_poses_map:
            dist_list = np.linalg.norm(per_pose - stacked_contours, ## map coord
                                                   axis=1)
            argmin = np.argmin(dist_list)
            argmin_list.append(argmin)
            ## register to the nearest contour
            j = 0
            while argmin >= indice_list[j]:
                j += 1
            register_ctr_idxs.append(j - 1)
            
            dist_contour = dist_list[indice_list[j - 1]: indice_list[min(j, len(indice_list) - 1)]]
            
            filter_contour_idx = list(np.where(np.array(dist_contour) <= self.near_dist)[0])
            filter_contour_idxs.append(filter_contour_idx)
            
        ## subdivide the contour(polyline) if multiple landmarks register to the same one 
        seen = []
        register_ctr_idxs = np.array(register_ctr_idxs)
        register_ctr_points = [[] for _ in range(len(register_ctr_idxs))]
        mid_points = []
        
        for i, ctr_idx in enumerate(register_ctr_idxs):
            occur_idxs = list(np.where(register_ctr_idxs == ctr_idx)[0])
            hit_contour = copy(contours[ctr_idx])
            if self.filter_contours:
                hit_contour = get_elements_by_idxs(hit_contour, filter_contour_idxs[i])

            if ctr_idx not in seen:
                seen.append(ctr_idx)
                occur_idxs = list(np.where(register_ctr_idxs == ctr_idx)[0])
                hit_contour = copy(contours[ctr_idx])
                
                if len(occur_idxs) > 1:  # subdivide contours, except their labels are the same but not fused
                    subdivide_texts = [loc_texts[occ_idx] for occ_idx in occur_idxs]
                    print('subdivide contours:', subdivide_texts)
                    argmin_list_local = [(argmin_list[occ_idx] - indice_list[ctr_idx], occ_idx)
                                         for occ_idx in occur_idxs]
                    sorted_argmin_list_local = sorted(argmin_list_local, key=lambda x: x[0])
                    pre_mid_idx = 0
                    for j in range(len(sorted_argmin_list_local) - 1):
                        mid_idx = (sorted_argmin_list_local[j][0] + sorted_argmin_list_local[j + 1][0]) // 2
                        cur_loc_text = loc_texts[sorted_argmin_list_local[j][1]]
                        self.semantic_map_dict.append([cur_loc_text, hit_contour[pre_mid_idx:mid_idx]])

                        pre_mid_idx = mid_idx
                        mid_points.append(hit_contour[mid_idx])

                    cur_loc_text = loc_texts[sorted_argmin_list_local[j][1]]
                    self.semantic_map_dict.append([cur_loc_text, hit_contour[pre_mid_idx:]])
                else:
                    self.semantic_map_dict.append([loc_texts[occur_idxs[0]], hit_contour]) 

        if plot:
            self.render_map(fuse_poses_map, cur_pose, viewpoints_world, predicted_lmpose_world, next_subgoal_world, next_subgoal_idx,
                            np.array(mid_points), exp_id, step, save=True, is_nbv=is_nbv)
        
        
    def render_map(self, fuse_poses_map, cur_pose, viewpoints_world, predicted_lmpose_world, next_subgoal_world,
                   next_subgoal_idx, mid_points, exp_id, step, fontsize=5, save=False, is_nbv=False):
        '''plot coordinate: (x, y)'''
        map_array = copy(self.ros.map_array)  # map can update
        obstacles = np.where(map_array == OBSTACLE)
        unknowns = np.where(map_array == UNKNOWN)
        mapData = copy(self.ros.mapData)
        cur_pose_map = _world_to_map2(mapData, cur_pose) 
        plt.scatter(cur_pose_map[1], cur_pose_map[0], marker='*')
        plt.scatter(unknowns[1], unknowns[0], s=self.render_size, color='darkgray', alpha=0.3)

        if len(self.semantic_map_dict):
            semantic_contours = list(chain([ctr[1].tolist() for ctr in 
                                            self.semantic_map_dict]))[0]
        else:
            semantic_contours = []
        
        obstacles_list = np.vstack((obstacles[1], obstacles[0])).T.tolist()
        remain_obstacles = np.array([point for point in obstacles_list 
                                     if point not in semantic_contours])
        plt.scatter(remain_obstacles[:, 0], remain_obstacles[:, 1], s=self.render_size, color='black', alpha=0.3)
        
        text_list = []
        for i, (text, contour) in enumerate(self.semantic_map_dict):
            if text not in text_list:
                idx = len(text_list) 
                text_list.append(text)
            else:
                idx = text_list.index(text)
            
            plt.scatter(contour[:, 0], contour[:, 1], c=color_list[i], 
                        s=self.render_size*2)
        print('text on signage map:', text_list)
        if len(fuse_poses_map):
            plt.scatter(fuse_poses_map[:, 0], fuse_poses_map[:, 1], marker='o', color='yellow', s=self.render_size*4)

        ## draw current unvisited frontiers
        if len(self.ros.cur_frontiers):
            cur_frontiers = copy(self.ros.cur_frontiers)
            cur_frontiers_map = np.array([_world_to_map2(mapData, point) # (x, y)
                            for point in cur_frontiers]) 
            plt.scatter(cur_frontiers_map[:, 1], cur_frontiers_map[:, 0], marker='o', color='blue', s=self.render_size*4)
            for i in range(len(cur_frontiers_map)):
                plt.text(cur_frontiers_map[i, 1], cur_frontiers_map[i, 0], str(i), fontsize=fontsize)

        if  len(next_subgoal_world):
            predicted_lmpose_map = []
            mapData = copy(self.ros.mapData)
            for lmpose_world in predicted_lmpose_world:
                predicted_lmpose_map.append(_world_to_map2(mapData, lmpose_world, predict=True))
            predicted_lmpose_map = np.array(predicted_lmpose_map)
            plt.scatter(predicted_lmpose_map[:, 1], predicted_lmpose_map[:, 0], marker='o', color='red')
            for i in range(len(predicted_lmpose_map)):
                plt.text(predicted_lmpose_map[i, 1], predicted_lmpose_map[i, 0], str(i), fontsize=fontsize)
   

        if save:
            print('save semantic map at step:', step)
            if is_nbv:
                plt.savefig(f'{self.output_dir}/{exp_id}/nbv_step{step}.png')
            else:
                plt.savefig(f'{self.output_dir}/{exp_id}/map_step{step}.png')
        plt.clf()

