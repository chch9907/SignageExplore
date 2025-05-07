import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import collections
import numpy as np
from typing import Dict, List
import pickle
from networkx.algorithms.components import connected_components
from scipy.spatial.distance import cdist
from PIL import ImageDraw, Image
from scipy.spatial import Voronoi, voronoi_plot_2d
import scipy.ndimage as ndimage
import os
from scene_understand.scene_text_retrieval import STRetrieval
from utils.utils import get_center_dist, nearest_neighbor

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __call__(self):
        return np.array([self.x, self.y])


class Graph(object):
    def __init__(self, obs, cfg, OCR_dict=[], map_landmarks=None, is_global_map=False):
        self.is_global_map = is_global_map
        self.prune = cfg.prune
        self.edge_thred = cfg.edge_thred
        self.neighbor_thred = cfg.neighbor_thred
        self.search_type = cfg.search_type
        self.cfg = cfg
        self.OCR_dict = OCR_dict
        self.map_landmarks = map_landmarks
        self.scores_list = []
        
        if self.is_global_map:
            self.image = obs
            self.depth = None
        else:
            self.image = obs['rgb'] 
            self.depth = obs['depth']   
        num_text = len(OCR_dict)
        self.num_text = num_text
        self.node_landmarks = []
        self.node_probs = []
        self.node_bboxs = []
        self.node_centers = []
        self.node_viewpoints = []
        self.node_neighbors = [[] for _ in range(num_text)]
        self.edges = np.zeros((num_text, num_text))
        self.graph = nx.Graph()
        self.prune = cfg.prune
        self.edge_thred = cfg.edge_thred
        self.neighbor_thred = cfg.neighbor_thred
        self.search_type = cfg.search_type
        self.scores_list = []
        self.is_global_map = is_global_map
        self.graph_path = cfg.map_path.replace('png', 'pkl')
        if self.is_global_map and os.path.exists(self.graph_path):
            self.load_graph(self.graph_path)
        else:
            self.generate_graph(OCR_dict, map_landmarks)
            # self.save_graph(self.graph_path)
        print("node_neighbors:", self.node_neighbors)
    
    def offline_process(self,):
        print('offline process topological graph.')
        # self.generate_graph(self.OCR_dict, map_landmarks)
        self.save_graph(self.graph_path)
    
    def __len__(self) -> int:
        return len(self.node_landmarks)
    
    def generate_graph(self, ocr_list: List[Dict], map_graph: List[Dict]):
        print('generate graph......')
        if map_graph is not None and not self.is_global_map:  
            '''remove unuseful text information'''
            local_OCR_list = [item['text'] for item in ocr_list]
            isloc_indexs, scores_list = \
                STRetrieval.edit_distance_matching(local_OCR_list, map_graph.node_landmarks)
            ocr_list = [res for i, res in enumerate(ocr_list) if i in isloc_indexs]
            self.scores_list = scores_list
        
        ## nodes
        for idx, merge_OCR in enumerate(ocr_list):
            bbox = merge_OCR['bbox']
            text = merge_OCR['text']
            prob = merge_OCR['prob']
            xl, yl, xr, yr = bbox    
            center = [(xl + xr) / 2, (yl + yr) / 2]
            self.graph.add_node(idx)
            self.node_bboxs.append(np.array(bbox, dtype=np.int64))
            self.node_landmarks.append(text)  # landmark_list
            self.node_probs.append(prob)
            self.node_centers.append(center)

        if not self.is_global_map:  # no need to build graph for observation
            return
        
        ## edges
        for i in range(self.num_text):
            for j in range(self.num_text):
                dist = get_center_dist(self.node_centers[i], self.node_centers[j])
                self.edges[i, j] = dist
                if dist <= self.edge_thred and i != j:
                    self.graph.add_edge(i, j, weight=1, length=dist)
                    self.node_neighbors[i].append(j)
        
        while True:
            print('merging clusters...')
            subgraphs = [self.graph.subgraph(c) for c in connected_components(self.graph)]
            n_subgraphs = len(subgraphs)
            if n_subgraphs == 1:
                break
            
            coords_sets = []
            nodes_sets = []
            for sub_G in subgraphs:
                nodes = np.array([j for j in sub_G.nodes()])
                coords = np.array([self.node_centers[j] for j in sub_G.nodes()])
                nodes_sets.append(nodes)
                coords_sets.append(coords)
            
            pre_min_array = np.zeros((n_subgraphs, n_subgraphs), dtype=np.int64)
            pre_argmin_array = np.zeros((n_subgraphs, n_subgraphs, 2), dtype=np.int64)
            for i in range(n_subgraphs):
                min_list = [pre_min_array[k, i] for k in range(i)]
                argmin_list = [tuple(reversed(pre_argmin_array[k, i])) for k in range(i)]
            
                for j in range(i, n_subgraphs):
                    if i != j:
                        dist_array = cdist(coords_sets[i], coords_sets[j]) 
                        argmin_index = np.unravel_index(dist_array.argmin(), dist_array.shape)  # tuple
                        
                        min_list.append(np.min(dist_array))
                        argmin_list.append(argmin_index)  
                        pre_min_array[i, j] = np.min(dist_array)
                        pre_argmin_array[i, j] = argmin_index
                    else:
                        min_list.append(np.inf)
                        argmin_list.append((0, 0))
                
                closet_idx = np.argmin(min_list)
                node_i = nodes_sets[i][argmin_list[closet_idx][0]]
                node_j = nodes_sets[closet_idx][argmin_list[closet_idx][1]]
                self.graph.add_edge(node_i, node_j,
                                weight=1, length=min_list[closet_idx])  # TODO:weight=1
                self.node_neighbors[node_i].append(node_j)
        # if self.prune:
        #     self.graph = nx.transitive_reduction(self.graph)  #! not implemented for undirected or cyclic graph
        
    def plot_graph_nx(self, ):
        # nx.draw_networkx_edges(self.graph, pos=nx.spring_layout(self.graph))
        nx.draw_networkx(self.graph, pos=nx.spring_layout(self.graph))
        plt.show()
    
    
    def plot_graph_coord(self, text_list, font_size=5):
        from pylab import mpl
        import matplotlib
        mpl.rcParams["font.sans-serif"] = ["FangZhengKaiTiFanTi"]
        height = self.image.shape[1]
        center_array = np.array(self.node_centers)
        # (w, h) -> (x=w, y=H-h)
        center_array[:, 1] = height - center_array[:, 1]  #! convert image coordinate to plot coordinate

        line_segs = []
        for edge in self.graph.edges:
            line = [center_array[edge[0]], center_array[edge[1]]]
            line_segs.append(line)
    
        lc = collections.LineCollection(line_segs, linewidths=1)
        fig, ax = plt.subplots()
        ax.scatter(center_array[:, 0], center_array[:, 1], color='black')
        
        for text, center in zip(text_list, center_array):
            ax.text(center[0] - font_size / 2, center[1] - font_size / 2, text, 
                    fontsize=font_size)
        ax.add_collection(lc)
        ax.autoscale()
        plt.show()
        plt.savefig('graph.png')
    
    
    def draw_on_image(self, image=None):
        if image is None:
            image = self.image
        image = image[:, :, ::-1]  #* cv2 -> plt
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.uint8(image)).convert('RGB')
        draw = ImageDraw.Draw(image)  # h, w (h > w)
        
        assert len(self.node_centers)
        center_array = np.array(self.node_centers)
        point_set = [tuple(point) for point in center_array]
        draw.point(point_set, fill='black')
        for edge in self.graph.edges:
            line = [tuple(center_array[edge[0]]), tuple(center_array[edge[1]])]
            draw.line(line, fill='purple', width=3)
        image.show()
        image.save('example.png')
    
    def draw_voronoi_on_image(self, image=None):
        if image is None:
            image = self.image
        image = image[:, :, ::-1]
        assert len(self.node_centers)
        height = self.image.shape[1]
        center_array = np.array(self.node_centers)
        center_array[:, 1] = height - center_array[:, 1]
        vor = Voronoi(center_array)
        vor.vertices[:, 1] = height - vor.vertices[:, 1]
        
        fig = plt.figure(figsize=(20,20))
        ax = plt.gca()
        
        voronoi_plot_2d(vor, point_size=10, ax=ax, show_points=False, show_vertices=False)
        ax.imshow(image)
        plt.show()

    def search_route(self, start_idx, goal_idx) -> List: 
        # dijkstra_path, astar_path
        if self.search_type == 'dijkstra':
            shortest_route = nx.dijkstra_path(self.graph, start_idx, goal_idx)
        elif self.search_type == 'astar':
            shortest_route = nx.astar_path(self.graph, start_idx, goal_idx)
        else:
            raise ValueError("self.search_type should be within ['dijkstra', 'astar']")
        return shortest_route
    
    
    def save_graph(self, save_path):
        data = {
            "node_landmarks": self.node_landmarks,
            "node_probs": self.node_probs,
            "node_centers": self.node_centers,
            "node_bboxs": self.node_bboxs,
            "node_neighbors": self.node_neighbors,
            "edges": self.edges,
            "json_graph": nx.readwrite.json_graph.node_link_data(self.graph)
        }
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        print('save graph:', save_path)
        
    def load_graph(self, load_path):
        with open(load_path, "rb") as f:
            data = pickle.load(f)
        self.node_landmarks = data['node_landmarks']
        self.node_probs = data['node_probs']
        self.node_centers = data['node_centers']
        self.node_neighbors = data['node_neighbors']
        self.node_bboxs = data['node_bboxs']
        self.edges = data['edges']
        self.num_text = len(self.node_landmarks)
        self.graph = nx.readwrite.json_graph.node_link_graph(data['json_graph'])
        print('load graph:', load_path)
        
    def clear(self):
        self.graph.clear()
        
        