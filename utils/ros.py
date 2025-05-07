#!/usr/bin/env python3
#! This .py file must be run in ROS environment.
import math
import sys
import os
# sys.path.append('../')
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import cv2
import tf
import rospy
import message_filters
from functools import partial
from math import pi, radians, sqrt, pow, atan2
import pickle
import pyrealsense2 as rs
import numpy as np
from copy import copy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
# from scipy.spatial.transform import Rotation as R
from queue import Queue
from nav_msgs.msg import OccupancyGrid
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Twist, Point, PoseArray, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan, CameraInfo
from utils.realsense import RealSenseCamera, RealSenseCameraDepth
from utils.utils import _normalize_heading, _calc_distance, pixel_to_world, get_new_pose, _world_to_map2, get_center_dist
    
class ROS:
    def __init__(self, cfg, ros_cfg, debug, depth_intrinsics=None, isMapping=False, 
                 camera_obj=None, rgbd_topics=None,
                 buffer = 10, sync_duration = 0.5):
        # print('init ros_interface')
        self.lidar_topic = ros_cfg.topics.lidar
        self.odom_topic = ros_cfg.topics.odom
        self.vel_topic = ros_cfg.topics.vel
        self.map_topic = ros_cfg.topics.map 
        self.mapData = OccupancyGrid()
        self.map_array = np.array([])
        self.min_height = cfg.min_height
        self.max_height = cfg.max_height

        
        ## publisher and subscriber
        self.rgb_img = None
        self.depth_img = None
        self.sleep_t = ros_cfg.sleep_time
        self.duration = ros_cfg.duration
        self.odom_frame = ros_cfg.frames.odom
        self.base_frame = ros_cfg.frames.base
        self.decimal = ros_cfg.decimal
        self.tf_listener = tf.TransformListener()
        
        if debug:
            return
        rospy.sleep(self.sleep_t)  # cache tf
        try:
            self.tf_listener.waitForTransform(self.odom_frame,
                                              self.base_frame,
                                              rospy.Time(),
                                              rospy.Duration(self.duration))
            
        except (tf.Exception, tf.ConnectivityException, tf.LookupException) as e:
            rospy.loginfo("Cannot find base_frame transformed from /odom.")
            rospy.loginfo(e)
        self.rate = rospy.Rate(ros_cfg.rateHz)
        self.rgbd_topics = rgbd_topics
        
        
        if isMapping: # only for mapping
            self.depth_intrinsics = depth_intrinsics
            rospy.Subscriber(self.map_topic, OccupancyGrid, self.map_callback)
            print('semantic_mapping.mapping: wait for map')
            while not len(self.mapData.data) and not rospy.is_shutdown():
                pass
            self._pixel_to_local = partial(pixel_to_world, self.depth_intrinsics)

            ## get cur_unvisited_frontiers for drawing map figures
            frontier_pub_topic = rospy.get_param('~frontier_pub_topic','/cur_frontiers')
            rospy.Subscriber(frontier_pub_topic, PoseArray, self.ftr_callback)
            nbv_topic = '/nbv_point'
            rospy.Subscriber(nbv_topic, Point, self.nbv_callback)
            self.cur_frontiers = []
            # self.pre_next_best_pose = []
            self.nbv_cur_pose = []
            self.next_best_pose = []
            self.new_nbv = False
        else:
            if camera_obj is not None:  # online
                self.camera_id = ros_cfg.camera_id
                self.camera_D455 = camera_obj  #RealSenseCameraDepth(name=self.camera_id, fps=ros_cfg.camera_fps)
                # self.depth_intrinsics = depth_intrinsics  # self.camera_D455.depth_intrinsics
            elif rgbd_topics is not None:  # offline
                self.bridge = CvBridge()
                rgb_sub = message_filters.Subscriber(rgbd_topics['rgb'], Image)
                depth_sub = message_filters.Subscriber(rgbd_topics['depth'], Image)
                ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 
                                                                buffer, sync_duration)
                self.subs = [rgb_sub, depth_sub]
                ts.registerCallback(self.sync_rgbd_callback)
                self.depth_scale = 0.001

        
    def ftr_callback(self, msg):
        self.cur_frontiers = []
        for pose in msg.poses:
            tmp = [pose.position.x, pose.position.y]
            self.cur_frontiers.append(tmp)
        print("cur_frontiers:", self.cur_frontiers)
    
    def nbv_callback(self, msg):
        # self.pre_next_best_pose = copy(self.next_best_pose)
        self.next_best_pose = [msg.x, msg.y]
        self.nbv_cur_pose = self.get_tf()
        self.new_nbv = True
        

    def sync_rgbd_callback(self, rgb_msg, depth_msg):
        self.rgb_callback(rgb_msg)
        self.depth_callback(depth_msg)
    
    
    def rgb_callback(self, msg):
        try:
            # Convert ROS CompressedImage message to OpenCV2
            rgb_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")  # "bgr8"
            # cv2.imwrite('{}/rgb/{}.jpeg'.format(data_path, image_time.to_sec()), rgb_img)
            # rospy.loginfo("!!save rgb:{:.2f}".format(image_time.to_sec())) 
        except CvBridgeError as e:
            rospy.loginfo('rgb error:{}'.format(e))
        self.rgb_img = np.array(rgb_img)


    def depth_callback(self, msg, colorize=False):
        # encoding type:
        # passthrough: remain original dtype, 
        # 16UC1: convert depth to mm unit, 
        # 32FC1: convert depth to m unit (higher accuracy).
        encoding = '16UC1' 
        try:
            # Convert ROS CompressedImage message to OpenCV2
            depth_time = msg.header.stamp
            # depth_time = depth_time.to_sec()
            depth_img = self.bridge.imgmsg_to_cv2(msg, encoding)
            depth_array = np.array(depth_img, dtype=np.float32) * self.depth_scale
            
            if colorize:
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), 
                                                   cv2.COLORMAP_JET)
                cv2.imwrite(f"colored_depth_{depth_time.to_sec()}.jpeg", depth_colormap)
            # print(depth_array)
            # with open('{}/depth/{}.pkl'.format(data_path, depth_time.to_sec()), 'wb') as f:
            #     pickle.dump(depth_array, f)
            #     rospy.loginfo("pickling depth data:{}".format(depth_time.to_sec()))
            # cv2.imwrite('{}/depth/{}.jpeg'.format(data_path, image_time.to_sec()), depth_img)
            # rospy.loginfo("save depth image:{:.2f}".format(depth_time.to_sec())) 
        except CvBridgeError as e:
            rospy.loginfo(e)
        self.depth_img = depth_array

    
    def wrap_intrinsics(self, camera_info_topic):
        # https://medium.com/@yasuhirachiba/converting-2d-image-coordinates-to-3d-coordinates-using-ros-intel-realsense-d435-kinect-88621e8e733a
        # https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/CameraInfo.html
        cameraInfo = rospy.wait_for_message(camera_info_topic, CameraInfo, timeout=5)
        intrinsics = rs.intrinsics()
        intrinsics.width = cameraInfo.width
        intrinsics.height = cameraInfo.height
        intrinsics.ppx = cameraInfo.K[2]
        intrinsics.ppy = cameraInfo.K[5]
        intrinsics.fx = cameraInfo.K[0]
        intrinsics.fy = cameraInfo.K[4]
        #intrinsics.model = cameraInfo.distortion_model
        intrinsics.model  = rs.distortion.none     
        intrinsics.coeffs = [i for i in cameraInfo.D]
        return intrinsics


    def get_image(self):
        if hasattr(self, 'camera_D455'):
            color_img, depth_img, depth_colormap = self.camera_D455.get_image()
            return color_img, depth_img, depth_colormap
        elif self.rgbd_topics is not None:
            return self.rgb_img, self.depth_img, None
        # return {'rgb': color_img, 'depth': depth_img, 'depth_color': depth_colormap}
    
    
    def close_camera(self,):
        if hasattr(self, 'camera_D455'):
            self.camera_D455.stop()
        elif self.rgbd_topics is not None:
            for sub in self.subs:
                sub.sub.unregister()
        
        
    def get_tf(self):
        try:
            self.tf_listener.waitForTransform(self.odom_frame,
                                              self.base_frame,
                                              rospy.Time(0),
                                              rospy.Duration(self.duration))
            (trans, rot) = self.tf_listener.lookupTransform(self.odom_frame,
                                                            self.base_frame,
                                                            rospy.Time(0))
        except (tf.Exception, tf.ConnectivityException, tf.LookupException) as e:
            rospy.loginfo(f"TF exception, cannot get odom pose:{e}")
            return
        point = Point(*trans)
        _, _, yaw = euler_from_quaternion(rot)
        # return np.round([round(item, self.decimal) 
        #         for item in [point.x, point.y, _normalize_heading(yaw)]])
        # return np.round([point.x, point.y, _normalize_heading(yaw)], self.decimal)
        return np.round([point.x, point.y, yaw], self.decimal)
    
    def get_pose_map(self, cur_pose):
        return _world_to_map2(self.mapData, cur_pose)
    
    def pixel_to_map(self, pixel, depth, cur_pose_world, map_obj):
        
        world_coord_local = self._pixel_to_local(pixel, depth)
        height = world_coord_local[2]
        world_coord_local = np.array(list(world_coord_local)[:2] + [0])
        try:
            world_coord_global = get_new_pose(cur_pose_world, world_coord_local)
            world_coord_global[2] = height  # add height dim
            map_coord = _world_to_map2(map_obj, world_coord_global[:2])
            if map_coord is not None:
                map_coord = np.round(map_coord)
        except:
            print('error',pixel, depth, world_coord_local)
        return map_coord, world_coord_global  # 3d coord
    
    def get_region_depth(self, depth, ocr_results, cur_pose_world, 
                         filter_by_area_3d=True, area_3d_bound=30):
        mapData = copy(self.mapData)
        if isinstance(ocr_results, dict):
            ocr_results = [ocr_results]
        centers_world = []
        centers_map = []
        distances = []
        areas_3d = []
        filter_idxs = []
        for i, item in enumerate(ocr_results):
            xl, yl, xr, yr = item['bbox'].astype(np.int64)
            depth_region = depth[yl: yr, xl: xr].astype(float)  # array
            # distance = np.min(depth_region) if np.min(depth_region) != 0 else np.mean(depth_region)
            if not len(depth_region):
                filter_idxs.append(i)
                continue
            distance = np.median(depth_region)
            center_2d = np.array([np.floor(xl + xr) / 2, np.floor(yl + yr) / 2])
            center_map, center_world = self.pixel_to_map(center_2d, distance, cur_pose_world, mapData)
            if center_map is None:
                filter_idxs.append(i)
                continue

            dist_topleft = depth_region[0, 0]
            dist_buttomright = depth_region[-1, -1]
            _, topleft_world = self.pixel_to_map(np.array([xl, yl]), dist_topleft, cur_pose_world, mapData)
            _, buttomright_world = self.pixel_to_map(np.array([xr, yr]), dist_buttomright, cur_pose_world, mapData)
            area_3d = int(abs(topleft_world[2] - buttomright_world[2]) * get_center_dist(topleft_world[:2], buttomright_world[:2])  * 1000)
            areas_3d.append(area_3d)
            area_2d = (xr - xl) * (yr - yl)
            
            
            # print('center height:', center_world[2])
            print('text:', item['text'], 'distance:', round(distance, 2), 'area_2d:', area_2d, 'area_3d:', area_3d, 'height:', center_world[2])
            
            if filter_by_area_3d and (area_3d < area_3d_bound or center_world[2] < self.min_height or center_world[2] > self.max_height):
                if area_3d < area_3d_bound:
                    print('filter by area 3d:', area_3d, area_3d_bound)
                else:
                    print('filter by height:', center_world[2], self.min_height, self.max_height)
                filter_idxs.append(i)
                continue
            
            centers_world.append(center_world)
            centers_map.append(center_map)
            distances.append(distance)
            
        return centers_map, centers_world, distances, areas_3d, filter_idxs

    def map_callback(self, msg):
        self.mapData = msg
        self.map_array = np.asarray(msg.data).reshape((msg.info.height,
                                            msg.info.width))