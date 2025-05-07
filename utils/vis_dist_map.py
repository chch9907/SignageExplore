## This file is used to visualize the sdf map obtained from distance_map
## Move it to the directory of distance_map package when using it
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rospy
from distance_map_msgs.msg import DistanceMap
from nav_msgs.msg import OccupancyGrid
import numpy as np

dist_map = DistanceMap()
raw_map = OccupancyGrid()
j = 0
k = 0

def dist_map_callback(msg):
    global dist_map, j
    if j % 20 == 0:
        dist_map = msg
        map_data = np.asarray(msg.data).reshape((msg.info.height, msg.info.width))
        fig, ax = plt.subplots()

        ax.imshow(map_data, interpolation='nearest')
        plt.savefig(f'dist_map_{j}.png')
        ax = plt.gca()
        ax.clear()
    j += 1

def raw_map_callback(msg):
    global raw_map, j
    size = 1
    if j % 20 == 0:
        raw_map = msg
        map_array = np.asarray(msg.data).reshape((msg.info.height, msg.info.width))
        obstacles = np.where(map_array == 100)
        unknowns = np.where(map_array == -1)
        fig, ax = plt.subplots()
        ax.scatter(obstacles[1], obstacles[0], color='black', s=size, label='obstacles')
        ax.scatter(unknowns[1], unknowns[0], color='gray', s=size, label='unknowns')
        plt.savefig(f'raw_map_{j}.png')
        plt.cla()
        plt.close()
    j += 1

    
if __name__ == '__main__':
    rospy.init_node('vis_dist_map', anonymous=False)
    map_topic = '/distance_map_node/distance_field_obstacles'
    raw_topic = '/projected_map_erode'
    rospy.Subscriber(map_topic, DistanceMap, dist_map_callback)
    rospy.Subscriber(raw_topic, OccupancyGrid, raw_map_callback)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        rate.sleep()
    
