import sys
import os

lib_path = os.path.abspath(os.path.join(__file__, '..'))
sys.path.insert(0, lib_path)

import glob
from math import *

import numpy as np
import torch
# import open3d as o3d
from scipy.spatial.transform import Rotation as R

from lidar_detect.pcdet.config import cfg, cfg_from_yaml_file
from lidar_detect.pcdet.datasets.dataset import DatasetTemplate
from lidar_detect.pcdet.models import build_network, load_data_to_gpu
from lidar_detect.tracker import Tracker

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA, Int16, Float32MultiArray, Float64MultiArray, Header
from custom_msgs.msg import Object, ObjectInfos, Paths
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN, OPTICS
import array
import time

rainbow_colormap = [
    [1.0, 0.0, 0.0],  #0 빨
    [1.0, 0.5, 0.0],  #1 주
    [1.0, 1.0, 0.0],  #2 노
    [0.5, 1.0, 0.0],  #3 연두
    [0.0, 1.0, 0.0],  #4 초록
    [0.0, 1.0, 0.5],  #5 민트
    [0.0, 1.0, 1.0],  #6 하늘
    [0.0, 0.5, 1.0],  #7 파랑
    [0.0, 0.0, 1.0],  #8 진파랑
    [0.5, 0.0, 1.0],  #9 보라
    [1.0, 0.0, 1.0],  #10 푸시아핑크
    [1.0, 0.0, 0.5],  #11 마젠타핑크   ##### 진한 색 ######
    [1.0, 0.8, 0.8],  #12 베이비핑크(웜톤)
    [1.0, 1.0, 0.8],  #13 레몬타르트
    [0.8, 1.0, 0.8],  #14 파스텔그린
    [0.8, 1.0, 1.0],  #15 연하늘
    [0.8, 0.8, 1.0],  #16 베리페리
    [1.0, 0.8, 1.0],  #17 라즈베리우유  
    [1.0, 1.0, 1.0]   #18 하양        ##### 파스텔 톤 #####
]

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, points, training=False, root_path=None, logger=None):

        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.points = points

    def get_points(self):
        input_dict = {
            'points': self.points
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

class detect_n_tracker(Node):
    def __init__(self, ckpt, model_cfg='kitti'):
        super().__init__('detect_tracker')

        cur_path = os.getcwd()+'/src/lidar_detect/lidar_detect'

        if model_cfg == 'kitti':
            cfg_name = '/cfgs/pointpillar_ej.yaml'
        elif model_cfg == 'waymo':
            cfg_name = '/cfgs/pointpillar_waymo.yaml'

        cfg_from_yaml_file(cur_path+cfg_name, cfg)
        self.cross_pts = np.loadtxt(cur_path+'/crosswalk/Crosswalk_for_detect.txt',dtype=np.float32,delimiter='\t')
        # self.cross_pts = np.fromfile(cur_path+'/crosswalk/Crosswalk_for_detect.txt',sep='/t')
        self.cross_idx = self.cross_pts[:,0]
        self.cross_pts = self.cross_pts[:,1:3]

        self.cross_4pts = np.loadtxt(cur_path+'/crosswalk/crosswalk_with_id.txt',dtype=np.float32,delimiter=',')
        self.cross_4idx = self.cross_4pts[:,0]
        self.cross_4pts = self.cross_4pts[:,1:3]
        self.crss_box = None

        self.cfg = cfg
        self.ckpt = cur_path+ckpt
        self.pcl_np = None
        self.cam_detect = None
        self.radius = 1.5
        self.roi_pc = None
        self.path = None

        self.total_ = None
        self.ego_info = None
        self.pcd_cross_ids = None
        self.near_crss_id = None
        self.nn_obj = NearestNeighbors(n_neighbors=100, radius=self.radius)
        self.dbs = DBSCAN(eps=0.4, min_samples=3, metric='cityblock')

        self.init = True

        self.map_ver = 0
        self.tracker = Tracker(2.5, 10, 10)
        self.rule_tracker = Tracker(0.5, 10, 4)

        self.box_tracking = self.create_publisher(Marker, '/lidar/tracking', 1)
        self.marker_all = self.create_publisher(Marker, '/lidar/raw_detect', 1)

        self.info_pub = self.create_publisher(ObjectInfos, '/lidar/detect_infos', 1)
        self.cropped_publisher = self.create_publisher(PointCloud2, '/lidar/ring_edge',1)
        
        sub = self.create_subscription(
            PointCloud2(),
            '/hesai/pandar',
            self.receive_pcl,
            1
        )
        sub

        sub_map = self.create_subscription(
            Int16(),
            '/gui/map',
            self.receive_map,
            1
        )
        sub_map

        # sub_yolo = self.create_subscription(
        #     Float32MultiArray(),
        #     'image_detection',
        #     self.receive_yolo,
        #     1
        # )
        # sub_yolo

        sub_ego = self.create_subscription(
            Float64MultiArray(),
            "/localization/ego_info",
            self.receive_ego,
            1
        )
        sub_ego

        sub_path = self.create_subscription(
            Paths(),
            '/lpp/local_path',
            self.get_roi,
            1
        )
        sub_path

        timer_period = 0.05
        self.timer = self.create_timer(timer_period, self.main)

    def receive_pcl(self,pcl):
        pcl_raw = pcl.data
        pcl_np = np.frombuffer(pcl_raw, dtype=np.float32).reshape(-1,5)  #n*5 행렬  ring data 포함
        pcl_np = pcl_np[pcl_np[:,4]<39]  #ego ring 제외

        pcl_np[:,3] = pcl_np[:,3] / 255  #hesai lidar의 경우 intensity normalize 안되어 있음
        self.pcl_np = pcl_np

        if self.init == True and self.pcl_np is not None:
            print('init')
            self.define_modes()
            self.init = False
        self.demo_dataset.points = self.pcl_np
    
    ################ Deep Learning utils ################
    def define_modes(self):
        self.demo_dataset = DemoDataset(
            dataset_cfg=self.cfg.DATA_CONFIG, class_names=self.cfg.CLASS_NAMES, points=self.pcl_np)
        self.model = build_network(model_cfg=self.cfg.MODEL, num_class=len(self.cfg.CLASS_NAMES), dataset=self.demo_dataset)
        self.model.load_params_from_file(filename=self.ckpt, logger=None, to_cpu=True)
        self.model.cuda()
        self.model.eval()

    def receive_map(self, map):
        self.map_ver = map.data
    
    ################ RuleBase utils ################
    def receive_ego(self, msg):
        ego_info_d = msg.data
        self.ego_info= np.frombuffer(ego_info_d, dtype=np.float64).reshape(-1,7)[:,:3]  #x, y, heading, 등등
        self.ego_info = self.ego_info.reshape(3,)

    def get_roi(self, msg): # y: front, x: right
        if self.pcl_np is None:
            print("self.pcl_np is None")
            return
        if self.ego_info is None:
            print("self.ego_info is None")
            return
        start = time.time()
        local_crss = self.crss_local(self.ego_info)  #nx3 (x,y,0)

        int_s = np.array(msg.s, dtype=np.int16)
        idx_50 = (int_s <= 50)
        theta = np.radians(90)
        xy_path = np.column_stack((np.array(msg.x)[idx_50]-1, np.array(msg.y)[idx_50]))
        self.path = np.copy(xy_path)
        h = np.copy(np.array(msg.h)[idx_50])

        ######### crop pointcloud according to local path #########
        pcl_np = np.copy(self.pcl_np)
        crop_filter = (pcl_np[:,0] > np.min(xy_path[:,0])-3) & (pcl_np[:,0] < np.max(xy_path[:,0])+3) & (pcl_np[:,1] > np.min(xy_path[:,1])-12) & (pcl_np[:,1] < np.max(xy_path[:,1])+12)
        cropped_pc = pcl_np[crop_filter]

        # if cropped_pc.shape[0]>0:
        self.nn_obj.fit(cropped_pc[:,:2])
        cross_idx = np.array([])
        total_idx = np.array([])

        for i, coor in enumerate(xy_path):
            if int_s[i] > 5:
                if int_s[i-1] == int_s[i]:
                    continue
                else:
                    idx = self.nn_obj.radius_neighbors(np.reshape(coor,(1,-1)), return_distance=False)
                total_idx = np.concatenate((total_idx, idx[0]), axis=None)       
        if local_crss is not None:   
            for coor in local_crss[:,:2]:
                idx = self.nn_obj.radius_neighbors(np.reshape(coor,(1,-1)), return_distance=False)
                cross_idx = np.concatenate((cross_idx, idx[0]), axis=None)
            cross_idx = np.unique(cross_idx).astype(np.int32)
            total_idx = np.unique(np.concatenate((total_idx, cross_idx))).astype(np.uint16)
        else:
            total_idx = np.unique(total_idx).astype(np.uint16)
        self.total_ = cropped_pc[total_idx]
        print('roi cut : ', (time.time()-start)*1000, 'ms')

    def check_edge(self, pcd):
        sorted_pcd = pcd[pcd[:,4].argsort()]
        rings = np.unique(sorted_pcd[:,4])
        edge_results = np.array([])

        for ring in rings:
            idx = sorted_pcd[:,4]==ring
            cur_pcd = sorted_pcd[idx]
            theta = np.arctan2(cur_pcd[:,0], cur_pcd[:,1])
            cur_pcd = cur_pcd[theta.argsort()]
            
            edge_idx = np.zeros(cur_pcd.shape[0],dtype=bool)
            diff = np.diff(cur_pcd[:,2])
            edges = np.where(np.abs(diff)>0.08)[0]
            if edges.shape[0] > 0 and diff[edges[0]] < 0:
                edge_idx[:edges[0]] = 1
                edges = np.delete(edges,0)
            if edges.shape[0] > 0 and diff[edges[-1]] > 0:
                edge_idx[edges[-1]:] = 1
                edges = np.delete(edges,-1)
            for i in range(edges.shape[0]//2):
                edge_idx[edges[i*2]:edges[i*2+1]] = 1
            
            if edge_results.shape[0] == 0:
                edge_results = cur_pcd[edge_idx]
            else:
                edge_results = np.vstack((edge_results,cur_pcd[edge_idx]))

        return edge_results    
    
    def crss_local(self, ego):
        crss_loc = np.zeros((self.cross_pts.shape[0], 3))
        crss_loc[:,:2] = self.cross_pts - ego[:2]
        head = -ego[2]
        R = np.array([[cos(head), -sin(head), 0],[sin(head), cos(head), 0], [0, 0, 1]])  #3x3
        crss_loc=np.matmul(R, crss_loc.T).T #nx3
        crss_loc_forward=crss_loc[crss_loc[:,0]>0]
        if crss_loc_forward.shape[0] == 0:
            return None
        crss_idx_forward = self.cross_idx[crss_loc[:,0]>0]
        dist = np.linalg.norm(crss_loc_forward[:,:2],axis=1)
        self.near_crss_id=crss_idx_forward[np.argmin(dist)]

        return crss_loc_forward[crss_idx_forward==self.near_crss_id]
    
    def get_crss_box(self, obj, ego): 
        head = ego[2]
        current_cross = self.cross_4pts[self.cross_4idx == self.near_crss_id]
        cross_ego_vector = current_cross - ego[:2]
        cross_ego_vector_mag = np.linalg.norm(cross_ego_vector,axis=1)
        min_idx = np.argmin(cross_ego_vector_mag)

        diff_min_vector = cross_ego_vector-cross_ego_vector[min_idx]
        diff_min_idx = np.linalg.norm(diff_min_vector, axis=1).argsort()
        diff_min_vector = diff_min_vector[diff_min_idx]
        
        crss_loc = np.zeros((current_cross.shape[0], 3))
        crss_loc[:,:2] = current_cross[diff_min_idx] - ego[:2]
        R = np.array([[cos(-head), -sin(-head), 0],[sin(-head), cos(-head), 0], [0, 0, 1]])
        crss_loc[:,:2] = (np.matmul(R, crss_loc.T).T)[:,:2]  #local 좌표 4개,z=0 (4x3)

        # print(crss_loc)
        # print(obj)
        under_0 = (crss_loc[1,0]-crss_loc[0,0])/(crss_loc[1,1]-crss_loc[0,1])*(obj[1]-crss_loc[0,1])+crss_loc[0,0]
        under_1 = (crss_loc[2,0]-crss_loc[0,0])/(crss_loc[2,1]-crss_loc[0,1])*(obj[1]-crss_loc[0,1])+crss_loc[0,0]
        upper_0 = (crss_loc[1,0]-crss_loc[3,0])/(crss_loc[1,1]-crss_loc[3,1])*(obj[1]-crss_loc[3,1])+crss_loc[3,0]
        upper_1 = (crss_loc[2,0]-crss_loc[3,0])/(crss_loc[2,1]-crss_loc[3,1])*(obj[1]-crss_loc[3,1])+crss_loc[3,0]
        under_x=max(under_0,under_1)
        upper_x=min(upper_0,upper_1)

        if under_x < obj[0] < upper_x:  #object 횡단보도에 있음
            heading_vector = diff_min_vector[1]
            width = np.linalg.norm(diff_min_vector[2])
            length = np.linalg.norm(diff_min_vector[1])

            local_ctr = (crss_loc[0]+crss_loc[3])/2
            box_heading = np.arctan2(heading_vector[0],heading_vector[1]) - head
            return local_ctr[0], local_ctr[1], 0.6, length, width, 2, box_heading

        else:
            return None

    ################ Camera Calib utils ################
    def receive_yolo(self, yolo_msg):
        detect_data = yolo_msg.data
        cam_detect = np.frombuffer(detect_data, dtype=np.float32).reshape(-1,6)  #leftup_xy, rightdown_xy, 정확도, class

        self.cam_detect = cam_detect[cam_detect[:,5]<3]  #과속방지턱 클래스 제외
        if 3.0 in cam_detect[:,5]:
            self.speed_bump = cam_detect[cam_detect[:,5] == 3]  #과속방지턱만
    
    def lidar2cam(self,pcd):
        cam2pix = np.array([[1852.666, 0, 982.862],
              [0, 1866.610, 612.790],
              [0, 0, 1]])
        R = np.array(
            [[-0.13461593, -0.99086486, -0.00808573],
            [-0.01051238,  0.00958763, -0.99989878],
            [ 0.99084209, -0.1345173,  -0.011707]])

        pcd_xyz = np.ones_like(pcd[:,:4])
        pcd_xyz[:,:3] = pcd[:,:3] - np.array([0.15253799, -1.34450982, -1.12016554])
        cam_xyz = np.matmul(R, pcd_xyz[:,:3].T)  #3xn = [3x3][3xn]

        pix_xyz = np.matmul(cam2pix, cam_xyz)  #3xn
        pix_xyz = pix_xyz.T  #nx3
        pix_s = np.expand_dims(pix_xyz[:,2],axis=1)
        pix_xyz = pix_xyz/pix_s

        return cam_xyz.T, pix_xyz[:,:2]

    ################ Visualization utils ################
    def draw_box(self,ref_boxes):
        x, y, z = ref_boxes[0:3]
        length, width, height = ref_boxes[3:6]
        yaw = ref_boxes[6]

        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)),dtype=np.float32)  #rotation matrix

        FL = np.array((length/2, width/2, z))
        FR = np.array((length/2, -width/2, z))
        RL = np.array((-length/2, width/2, z))
        RR = np.array((-length/2, -width/2, z))

        FL = np.dot(R, FL) + np.array((x,y,0))
        FR = np.dot(R, FR) + np.array((x,y,0))
        RL = np.dot(R, RL) + np.array((x,y,0))
        RR = np.dot(R, RR) + np.array((x,y,0))

        return self.wireframe(FL, FR, RL, RR, height)


    def wireframe(self, FL, FR, RL, RR, z):
        pointArr = []
        vehicleEdge = [FL,FR,RL,RR]
        # print(vehicleEdge)
        for i in range(8):
            pointArr.append(Point())

        for i, point in enumerate(pointArr):
            point.x = float(vehicleEdge[i%4][0])
            point.y = float(vehicleEdge[i%4][1])
            point.z = float(vehicleEdge[i%4][2]-int(i/4)*z)  #waymo
            # point.z = float(vehicleEdge[i%4][2]+int(i/4)*z - z/2)  #KITTI
        lineArr = [
            pointArr[0],pointArr[1],
            pointArr[1],pointArr[3],
            pointArr[3],pointArr[2],
            pointArr[2],pointArr[0],
            pointArr[4],pointArr[5],
            pointArr[5],pointArr[7],
            pointArr[7],pointArr[6],
            pointArr[6],pointArr[4],
            pointArr[0],pointArr[4],
            pointArr[1],pointArr[5],
            pointArr[3],pointArr[7],
            pointArr[2],pointArr[6],
            pointArr[5],pointArr[0],  #heading 추가
            pointArr[4],pointArr[1]
        ]

        return lineArr
    
    def rotationMatrixToEulerAngles(self, R) :
        sy = sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sy < 1e-6
        if  not singular :
            x = atan2(R[2,1] , R[2,2])
            y = atan2(-R[2,0], sy)
            z = atan2(R[1,0], R[0,0])
        else :
            x = atan2(-R[1,2], R[1,1])
            y = atan2(-R[2,0], sy)
            z = 0
        return np.array([x, y, z])

    def limit_degree(self, ang):
        if -np.pi <= ang < np.pi:
            new_ang = ang
        else:
            new_ang = ang % (2*np.pi)

            if new_ang < np.pi:
                pass
            else:
                new_ang -= 2*np.pi
        return new_ang  

    def np_to_pc2(self, np_array):

        ros_dtype = PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize

        fields = [
            PointField(name='x', offset=0*itemsize, datatype=ros_dtype, count=1),
            PointField(name='y', offset=1*itemsize, datatype=ros_dtype, count=1),
            PointField(name='z', offset=2*itemsize, datatype=ros_dtype, count=1),
            PointField(name='intensity', offset=3*itemsize, datatype=ros_dtype, count=1),
            PointField(name='ring', offset=4*itemsize, datatype=ros_dtype, count=1),
            PointField(name='label', offset=5*itemsize, datatype=ros_dtype, count=1)
    ]
        header = Header()
        header.frame_id = 'Pandar40P'

        msg_cropped_pc = PointCloud2()
        msg_cropped_pc.header = header
        msg_cropped_pc.height = 1
        msg_cropped_pc.width = np_array.shape[0]
        msg_cropped_pc.is_dense = False
        msg_cropped_pc.is_bigendian = False
        msg_cropped_pc.fields = fields
        msg_cropped_pc.point_step = (itemsize*len(np_array[0]))
        msg_cropped_pc.row_step = (itemsize*len(np_array[0]))
        pcd_byte = np_array.astype(np.float32).tobytes()
        pcd_intArr = array.array('B', pcd_byte)
        msg_cropped_pc.data = pcd_intArr
        self.cropped_publisher.publish(msg_cropped_pc)

    ################ Main callback ################
    def main(self):
        start=time.time()
        all_infos = Marker()
        all_infos.header.frame_id = 'Pandar40P'
        all_infos.type = 5  #line list
        all_infos.scale.x = 0.1

        pub_infos = ObjectInfos()
        filterted_result = None
        rule_result = []
        if self.pcl_np is not None:

            data_dict=self.demo_dataset.get_points()
            data_dict = self.demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = self.model.forward(data_dict)
            
            points=data_dict['points'][:, 1:]
            ref_boxes=pred_dicts[0]['pred_boxes'].cpu().numpy()

            for i in range(ref_boxes.shape[0]):
                if ref_boxes[i,1] < 3 and 3/2*np.pi>ref_boxes[i,6]>np.pi/2:
                    ref_boxes[i,6] -= np.pi
                ref_boxes[i,6] = np.arctan(np.tan(ref_boxes[i,6]))            

            ref_scores=pred_dicts[0]['pred_scores']
            ref_labels=pred_dicts[0]['pred_labels']

            if isinstance(points, torch.Tensor):
                points = points.cpu().numpy()
            if isinstance(ref_boxes, torch.Tensor):
                ref_boxes = ref_boxes.cpu().numpy()

            ######## raw BBox Viewer(PointPillars) #######
            ##### x, y, z, w, l, h, yaw, cls
            filterted_result = np.column_stack([ref_boxes[:,:7], ref_labels.cpu().numpy()])
            del_list = []

            if self.map_ver == 6:  #map_ver : slope
                car_thres = 0.4
            else:
                car_thres = 0.6

            for i in range(ref_boxes.shape[0]):
                score = ref_scores[i].item()
                cur_color = ColorRGBA()

                #car detect
                if (ref_labels[i] == 1 and score >= car_thres) and ref_boxes.shape[0] > 0:
                    point_list = self.draw_box(ref_boxes[i])
                    cur_color.a = 0.6
                    cur_color.r, cur_color.g, cur_color.b = rainbow_colormap[2]
                    color_list = [cur_color]*len(point_list)
                    all_infos.colors = all_infos.colors + color_list
                    all_infos.points = all_infos.points + point_list
                    # print("car", score)
                
                #pedestrian detect
                elif (ref_labels[i] == 2 and score >= 0.4) and ref_boxes.shape[0] > 0:
                    point_list = self.draw_box(ref_boxes[i])
                    cur_color.a = 0.6
                    cur_color.r, cur_color.g, cur_color.b = rainbow_colormap[6]
                    color_list = [cur_color]*len(point_list)
                    all_infos.colors = all_infos.colors + color_list
                    all_infos.points = all_infos.points + point_list
                    # print("ped", score)

                elif (ref_labels[i] == 3 and score >= 0.4) and ref_boxes.shape[0] > 0:
                    point_list = self.draw_box(ref_boxes[i])
                    cur_color.a = 0.6
                    cur_color.r, cur_color.g, cur_color.b = rainbow_colormap[9]
                    color_list = [cur_color]*len(point_list)
                    all_infos.colors = all_infos.colors + color_list
                    all_infos.points = all_infos.points + point_list
                    # print("cycle", score)

                else:
                    del_list.append(i)

            filterted_result = np.delete(filterted_result, del_list, 0)
            print('deep : ', (time.time()-start)*1000, 'ms')
            start=time.time()
            ######## LiDAR Detect RuleBase ########
            if self.total_ is not None:
                ring_edge = self.check_edge(self.total_)
                if ring_edge.shape[0] > 0:
                    self.dbs.fit(ring_edge[:,:3])

                    labels = self.dbs.labels_
                    ring_edge = np.hstack((ring_edge,labels.reshape(-1,1)))
                    self.np_to_pc2(ring_edge)

                    label_num = max(labels)
                    for lb in range(label_num+1):
                        cur_pcd = ring_edge[labels==lb]
                        mean_x, min_x, max_x = np.mean(cur_pcd[:,0]), np.min(cur_pcd[:,0]), np.max(cur_pcd[:,0])
                        mean_y, min_y, max_y = np.mean(cur_pcd[:,1]), np.min(cur_pcd[:,1]), np.max(cur_pcd[:,1])
                        mean_z, max_z, min_z = np.mean(cur_pcd[:,2]), np.max(cur_pcd[:,2]), np.min(cur_pcd[:,2])    
                        width = max_x-min_x 
                        length = max_y-min_y
                        height = max_z-min_z
                        volume = width * length * height
                        
                        if height > 0.1: #and volume > 0.01: 
                            rule_result.append([mean_x,mean_y,max_z,width, length, height,0.0,4])
                            all_infos.points = all_infos.points + self.draw_box(rule_result[-1])
                            all_infos.color.a, all_infos.color.r, all_infos.color.g, all_infos.color.b = [0.8, 1.0, 1.0, 1.0]
                            # print('ruleba 나온')

            print('rule : ', (time.time()-start)*1000, 'ms')
            start=time.time()
            ######## YOLO Detect to World ########
            
            if self.cam_detect is not None:
                pcd_cam, pcd_pix = self.lidar2cam(self.pcl_np)  #pcd_cam : nx3 / pcd_pix : nx2
                for cam_det in self.cam_detect:  #leftup_x, y, rightdown_x, y, 정확도, class
                    if cam_det[4] >= 0.55:
                        # print(pcd_cam.shape, pcd_pix.shape)
                        center_pix = np.zeros((2,))
                        center_pix[1], center_pix[0] = (cam_det[:2]+cam_det[2:4])/2

                        dists_from_center_2d = np.linalg.norm(pcd_pix - center_pix, axis=1)
                        nearest_center_point = self.pcl_np[np.argmin(dists_from_center_2d)]
                        dists_from_center = np.linalg.norm(self.pcl_np - nearest_center_point, axis=1)

                        in_box_filter = (0<=pcd_cam[:,2]) & (dists_from_center <= 1.3) & (cam_det[0]<=pcd_pix[:,1]) & (pcd_pix[:,1]<=cam_det[2]) & (cam_det[1]<=pcd_pix[:,0]) & (pcd_pix[:,0]<=cam_det[3])
                        det_pcd = self.pcl_np[in_box_filter]
                        # print(cam_det[5], det_pcd.shape)
                        
                        if det_pcd.shape[0] > 3:
                            print('카메라ㅏㅏㅏㅏㅏㅏ')
                            mean_x, min_x, max_x = np.mean(det_pcd[:,0]), np.min(det_pcd[:,0]), np.max(det_pcd[:,0])
                            mean_y, min_y, max_y = np.mean(det_pcd[:,1]), np.min(det_pcd[:,1]), np.max(det_pcd[:,1])
                            mean_z, max_z, min_z = np.mean(det_pcd[:,2]), np.max(det_pcd[:,2]), np.min(det_pcd[:,2])    
                            width = max_x-min_x 
                            length = max_y-min_y
                            height = max_z-min_z
                            volume = width * length * height
                            box_info = [mean_x,mean_y,max_z,length, width, height,0.0,6]
                            point_list = self.draw_box(box_info)

                            cur_color = ColorRGBA()
                            cur_color.a = 0.6
                            if cam_det[5] == 0:  #pedestrian
                                cur_color.r, cur_color.g, cur_color.b = rainbow_colormap[10]
                            else:  #car, truck
                                cur_color.r, cur_color.g, cur_color.b = rainbow_colormap[18]
                            color_list = [cur_color]*len(point_list)
                            all_infos.colors = all_infos.colors + color_list
                            all_infos.points = all_infos.points + point_list
                            

        self.marker_all.publish(all_infos)

        start = time.time()
            ##### tracking #####
        cur_object_rule = np.array([])
        if len(rule_result) > 0:
            self.rule_tracker.update(np.array(rule_result), v_x=0, v_y=0, dt=0.05)
            if len(self.rule_tracker.tracks) != 0:
                for i in range(len(self.rule_tracker.tracks)):
                    if (len(self.rule_tracker.tracks[i].trace) >= 1):

                        if self.rule_tracker.tracks[i].start_frames == 3:
                            x = round(self.rule_tracker.tracks[i].trace[-1][0,0],4)
                            y = round(self.rule_tracker.tracks[i].trace[-1][0,1],4)
                            z = round(self.rule_tracker.tracks[i].trace[-1][0,2],4)
                            w = round(self.rule_tracker.tracks[i].trace[-1][0,3],4)
                            l = round(self.rule_tracker.tracks[i].trace[-1][0,4],4)
                            h = round(self.rule_tracker.tracks[i].trace[-1][0,5],4)
                            yaw = round(self.rule_tracker.tracks[i].trace[-1][0,6],4)
                            cls = self.rule_tracker.tracks[i].trace[-1][0,7]
                            
                            rel_v_x = round(self.rule_tracker.tracks[i].trace[-1][0,-2],4)
                            rel_v_y = round(self.rule_tracker.tracks[i].trace[-1][0,-1],4)
                            # rel_w = round(self.rule_tracker.tracks[i].trace[-1][0,-1],4)  #상대 angular velosity

                            Id = self.rule_tracker.tracks[i].trackId
                            if cur_object_rule.shape[0] == 0:
                                cur_object_rule = np.array([Id, x, y, z, w, l, h, yaw, rel_v_x, rel_v_y, cls])

                            else:
                                cur_object_rule = np.vstack((cur_object_rule, np.array([Id, x, y, z, w, l, h, yaw, rel_v_x, rel_v_y, cls])))
        print('rule tracking : ', (time.time()-start)*1000, 'ms')
        start=time.time()
        cur_object_deep = np.array([])
        if filterted_result is not None:
            ##### filtered_result : x, y, z, w, l, h, yaw, cls = (0 : Car, 1 : Ped, 2 : Cycle, 3 : 과방턱, 4 : others(rule), 5 : crosswalk)
            self.tracker.update(filterted_result, v_x=0, v_y=0, dt=0.05)
            # print("det : ", filterted_result[:,:4])

            if len(self.tracker.tracks) != 0:
                # print(len(self.tracker.tracks))

                for i in range(len(self.tracker.tracks)):
                    if (len(self.tracker.tracks[i].trace) >= 1):

                        if self.tracker.tracks[i].start_frames == 3:
                            x = round(self.tracker.tracks[i].trace[-1][0,0],4)
                            y = round(self.tracker.tracks[i].trace[-1][0,1],4)
                            z = round(self.tracker.tracks[i].trace[-1][0,2],4)
                            w = round(self.tracker.tracks[i].trace[-1][0,3],4)
                            l = round(self.tracker.tracks[i].trace[-1][0,4],4)
                            h = round(self.tracker.tracks[i].trace[-1][0,5],4)
                            yaw = round(self.tracker.tracks[i].trace[-1][0,6],4)
                            cls = self.tracker.tracks[i].trace[-1][0,7]
                            
                            rel_v_x = round(self.tracker.tracks[i].trace[-1][0,-2],4)
                            rel_v_y = round(self.tracker.tracks[i].trace[-1][0,-1],4)
                            # rel_w = round(self.tracker.tracks[i].trace[-1][0,-1],4)  #상대 angular velosity

                            Id = self.tracker.tracks[i].trackId
                            if cur_object_deep.shape[0] == 0:
                                cur_object_deep = np.array([Id, x, y, z, w, l, h, yaw, rel_v_x, rel_v_y, cls])

                            else:
                                cur_object_deep = np.vstack((cur_object_deep, np.array([Id, x, y, z, w, l, h, yaw, rel_v_x, rel_v_y, cls])))
                            # print(cur_object.shape)
        print('deep tracking : ', (time.time()-start)*1000, 'ms')
        start=time.time()
        if cur_object_rule.shape[0] > 0 and cur_object_deep.shape[0] == 0:
            cur_object = cur_object_rule
        elif cur_object_rule.shape[0] == 0 and cur_object_deep.shape[0] > 0:
            cur_object = cur_object_deep
        elif cur_object_rule.shape[0] > 0 and cur_object_deep.shape[0] > 0:
            cur_object = np.vstack((cur_object_deep,cur_object_rule))
        else:
            cur_object = np.array([])

        # info = Object()
        tracking_boxes = Marker()
        tracking_boxes.header.frame_id = 'Pandar40P'
        tracking_boxes.type = 5  #line list
        tracking_boxes.scale.x = 0.07
        tracking_boxes.color.a = 0.6  #투명도
        if cur_object.shape[0] != 0:
            if len(cur_object.shape) == 1:
                cur_object = cur_object.reshape(1,11)
            for i, cur_ in enumerate(cur_object):
                if self.near_crss_id is not None:
                    self.crss_box = self.get_crss_box([cur_[1],cur_[2]],self.ego_info)
                if self.crss_box is not None: # crsswalk
                    cur_[1:8] = self.crss_box
                    cur_[-1] == 5  #crsswalk
                    tracking_boxes.points = tracking_boxes.points + self.draw_box(self.crss_box)
                    tracking_boxes.color.r, tracking_boxes.color.g, tracking_boxes.color.b = rainbow_colormap[3]
                else:
                    tracking_boxes.points = tracking_boxes.points + self.draw_box(cur_[1:8])
                    tracking_boxes.color.r, tracking_boxes.color.g, tracking_boxes.color.b = rainbow_colormap[3]

                info = Object()
                info.point.x, info.point.y, info.point.z = cur_[1:4]
                info.point.x += 1.0  #lidar 후륜중심 offset
                info.l, info.w, info.h = cur_[4:7]
                info.yaw = cur_[7]
                info.rel_v_x, info.rel_v_y = cur_[8:10]

                info.label = int(cur_[-1])
                pub_infos.objects.append(info)
                
        self.box_tracking.publish(tracking_boxes)
        self.info_pub.publish(pub_infos)
        print('info pub : ', (time.time()-start)*1000, 'ms')
        start=time.time()
        # print((time.time()-start)*1000, 'ms')


def main(args=None):
    rclpy.init(args=None)

    lidar = detect_n_tracker(ckpt='/trained_models/waymo_200.pth', model_cfg='waymo')
    with torch.no_grad():
        rclpy.spin(lidar)

    lidar.destroy_node()
    rclpy.shutdown()


if __name__=='__main__':
    main()
