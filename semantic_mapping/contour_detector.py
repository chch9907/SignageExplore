from queue import Queue
import numpy as np
import copy
import matplotlib.pyplot as plt
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage import measure
import itertools
import time
from sklearn.cluster import AgglomerativeClustering

from utils.utils import read_map_pgm
FREE = 254
OBSTACLE = 0
UNKNOWN = 205
MAX_VALUE = WHITE = 255  # white
BLACK = 0

def _is_frontier(p, OG_map, neighbor_num):
    if OG_map[p] == FREE:  # free
        neighbors = _get_neighbor(p, OG_map, neighbor_num)
        if any([OG_map[adj_p] == UNKNOWN for adj_p in neighbors]):  # unknown
            return True
    return False

def _get_neighbor(p, OG_map, neighbor_num):
    assert neighbor_num in [4, 8]
    neighbors = []
    xmax, ymax = OG_map.shape  # width, height
    if neighbor_num == 4:
        adds = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # up, right, down, left
    elif neighbor_num == 8:
        adds = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]  # clockwise
    else:
        assert False, 'neighbor_num should be 4 or 8'
    
    for add in adds:
        if p[0] + add[0] >= 0 and p[0] + add[0] < xmax \
            and p[1] + add[1] >= 0 and p[1] + add[1] < ymax:
                
                neighbors.append((p[0] + add[0], p[1] + add[1]))
    return neighbors


def _wavefront_frontier_detection(cur_pose, OG_map, neighbor_num=4):
    que_bfs = Queue()
    que_ftr = Queue()
    Map_Open_List = []
    Map_Close_List = []
    Frontier_Open_List = []
    Frontier_Close_List = [] 
    
    que_bfs.queue.clear()
    que_bfs.put(cur_pose)
    Map_Open_List.append(cur_pose)
    frontiers_list = []
    while not que_bfs.empty():
        p = que_bfs.get()
        if p in Map_Close_List:
            continue
        if _is_frontier(p, OG_map, neighbor_num):
            frontier = []
            que_ftr.queue.clear()
            que_ftr.put(p)
            Frontier_Open_List.append(p)
            
            while not que_ftr.empty():
                q = que_ftr.get()
                if q in (Map_Close_List + Frontier_Close_List):
                    continue
                if _is_frontier(q, OG_map, neighbor_num):
                    frontier.append(q)
                    for adj_q in _get_neighbor(q, OG_map, neighbor_num):
                        if OG_map[adj_q] == FREE and adj_q not in (Frontier_Open_List + \
                            Frontier_Close_List + Map_Close_List):
                            que_ftr.put(adj_q)
                            Frontier_Open_List.append(adj_q)
                Frontier_Close_List.append(q)
                
            frontiers_list.append(frontier)
            for v in frontier:
                if v not in Map_Close_List:
                    Map_Close_List.append(v)
        
        for adj_p in _get_neighbor(p, OG_map, neighbor_num):
            if OG_map[adj_p] == FREE and adj_p not in (Map_Open_List + Map_Close_List):     
                que_bfs.put(adj_p)
                Map_Open_List.append(adj_p)
                
        if p not in Map_Close_List:
            Map_Close_List.append(p)
    return frontiers_list


def _get_connected_regions(map_array, frontiers, cur_pose, step=1, show=False, save=False, connectivity=2, thred_area=5):
    # binarization
    new_map = np.zeros_like(map_array)
    for ftr in frontiers:
        new_map[tuple(ftr)] = WHITE

    # find connected regions
    img_label, num = measure.label(new_map, connectivity=connectivity, return_num=True)
    regions = measure.regionprops(img_label) 
    
    # remove small regions
    valid_labels = []
    frontier_centroids = []
    frontier_areas = []
    width, height = map_array.shape
    for i, rg in enumerate(regions):
        if rg.area <= thred_area:
            cluster_indices = np.where(img_label == i)
            img_label[cluster_indices] = 0
        else:
            centroid = list(rg.centroid)
            centroid[0] = min(height, round(centroid[0]))
            centroid[1] = min(width, round(centroid[1]))
            assert map_array[tuple(centroid)] == FREE

            # checking if there is a wall in the cluster：
            cluster_indices = np.where(img_label == i)
            if not np.any(map_array[cluster_indices]) == 0:
                frontier_centroids.append(centroid)
                valid_labels.append(i)
                frontier_areas.append(cluster_indices)  
    print(frontier_centroids)
    print(frontier_areas)
    
    ######## plot ########
    if show or save:
        
        white_image = np.zeros_like(map_array, dtype=np.uint8)
        white_image.fill(WHITE)
        white_image[np.where(img_label > 0)] = 128
        white_image = Image.fromarray(white_image).convert('RGB')
        draw = ImageDraw.Draw(white_image)

        if show:
            cv2.imshow('frontier', white_image) 
            if cv2.waitKey(0) == ord('q'):
                pass
            cv2.destroyWindow()
        if save:
            if cur_pose is not None:
                draw.point([(cur_pose[1], cur_pose[0])], fill=MAX_VALUE)
                
            cv2.imwrite(f"./materials/exp_figs/raw_map_{step}.png", map_array)  
            white_image.save(f'./materials/exp_figs/frontier_{step}.png')
    
    
    ## method 2:
    # from sklearn.cluster import AgglomerativeClustering
    # X = np.array(frontiers)
    # cluster = AgglomerativeClustering(n_clusters=None, metric='euclidean', distance_threshold=100, compute_full_tree=True).fit_predict(X)
    # print(cluster)
    
    ## method 3:
    # import matplotlib.pyplot as plt
    # from scipy.cluster.hierarchy import dendrogram, linkage
    # # generate the linkage matrix
    # X = np.array(frontiers)
    # Z = linkage(X,
    #             method='complete',  # dissimilarity metric: max distance across all pairs of 
    #                                 # records between two clusters
    #             metric='euclidean'
    #     )                           # you can peek into the Z matrix to see how clusters are 
    #                                 # merged at each iteration of the algorithm
    # plt.figure(figsize=(30, 10))
    # dendrogram(Z)
    # plt.show()
    # # retreive clusters with `max_d`
    # from scipy.cluster.hierarchy import fcluster
    # max_d = 100       # I assume that your `Latitude` and `Longitude` columns are both in 
    #                 # units of miles
    # clusters = fcluster(Z, max_d, criterion='inconsistent')
    # print(clusters)
    
    return frontier_centroids, frontier_areas


def frontier_detection(cur_pose, 
                       map_array=None, 
                       name='', 
                       neighbor_num=4,
                       step=1,
                       show=False, 
                       save=False):
    '''cur_pose is cur_pose in map coordinate'''
    if name != '':
        map_array = read_map_pgm(name, vis=False)
    else:
        # mapping values
        map_array[np.where(map_array == 0)] = FREE 
        map_array[np.where(map_array == 100)] = OBSTACLE
        map_array[np.where(map_array == -1)] = UNKNOWN

    
    ######## detect ########
    st = time.time()
    frontiers = _wavefront_frontier_detection(cur_pose, map_array, neighbor_num)
    
    ######## cluster ########
    st = time.time()
    height = map_array.shape[0]  # (126, 98)
    frontiers = np.array(list(itertools.chain.from_iterable(frontiers)))
    centroids, areas = _get_connected_regions(map_array, frontiers, cur_pose, step, show, save)  # method 1
            
    return (centroids, areas)


def find_contours_and_cluster(OG_map, 
                              cur_pose=None, 
                              filter_num=10, 
                              filter_std=5, 
                              crop_dist=5, 
                              cluster_dist=10,
                              step=1,
                              show_contours=False, 
                              show_clusters=False):
    ## find contours
    # OG_map: [height, wdith]
    OG_map = OG_map.astype(np.uint8)  # input type must be uint8
    ret, binary = cv2.threshold(OG_map, OBSTACLE, MAX_VALUE, cv2.THRESH_BINARY)  
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    
    ## filter small contours 
    filtered_contours = []
    for ctr in contours:
        std = np.std(ctr, axis=0)
        if ctr.shape[0] > filter_num and np.linalg.norm(std) > filter_std:
            filtered_contours.append(ctr)
    flatted_contours = np.vstack(filtered_contours).squeeze(1)
    if show_contours:
        white_img = np.zeros_like(binary, dtype=np.uint8)
        white_img.fill(WHITE)  # black
        for point in flatted_contours:
            white_img[point[1], point[0]] = BLACK
        if cur_pose is not None:
            white_img[cur_pose[1], cur_pose[0]] = BLACK
        cv2.imwrite(f"./materials/exp_figs/contours_{step}.png", white_img)  
    
    
    
    ## cluster contours
    clusters = AgglomerativeClustering(n_clusters=None, metric='euclidean', 
                                       linkage='single', distance_threshold=cluster_dist, 
                                       compute_full_tree=True).fit(flatted_contours)
    # print(clusters.n_clusters_)
    labels = np.array(clusters.labels_)
    cluster_contours = [flatted_contours[np.where(labels == k)] 
                        for k in range(clusters.n_clusters_)]
    if show_clusters:
        height = OG_map.shape[0]
        plt.scatter(flatted_contours[:, 0], height - flatted_contours[:, 1], c=labels, linewidths=0.1)
        if cur_pose is not None:
            plt.scatter(cur_pose[1], cur_pose[0], marker='*')
        plt.savefig(f'./materials/exp_figs/clusters_{step}.png')
        plt.show()
    return cluster_contours 
    
if __name__ == '__main__':
    prefix = ''
    cur_pose = (1200, 400)
    clusters = frontier_detection(cur_pose, name=prefix, show=True)
    