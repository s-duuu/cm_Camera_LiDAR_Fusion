from audioop import cross
from cProfile import label
from calendar import c
from math import degrees
import math
import sys
import os
from turtle import heading

lib_path = os.path.abspath(os.path.join(__file__, '..'))
sys.path.insert(0, lib_path)
from math import *

import numpy as np
from scipy.spatial.transform import Rotation as R

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

class detect_n_tracker(Node):
    def __init__(self):
        super().__init__('detect_tracker')
        cur_path = os.getcwd()+'/src/lidar_detect/lidar_detect'
        self.cross_pts = np.loadtxt(cur_path+'/crosswalk/Crosswalk_for_detect.txt',dtype=np.float32,delimiter='\t')
        self.cross_idx = self.cross_pts[:,0]
        self.cross_pts = self.cross_pts[:,1:3]

        self.cross_4pts = np.loadtxt(cur_path+'/crosswalk/crosswalk_with_id.txt',dtype=np.float32,delimiter=',')
        self.cross_4idx = self.cross_4pts[:,0]
        self.cross_4pts = self.cross_4pts[:,1:3]
        # self.cross_ctr = (self.cross_4pts[:,0]+self.cross_4pts[:,2])/2

        # self.nn_crss_obj = NearestNeighbors(n_neighbors=1, radius=10)
        # self.nn_crss_obj.fit(self.cross_ctr)

        self.pcl_np = None
        self.cam_detect = None
        self.radius = 1.5
        self.roi_pc = None
        self.path = None
        self.total_ = None
        self.ego_info = None
        # self.cross_ids = None
        self.nn_obj = NearestNeighbors(n_neighbors=100, radius=self.radius)
        self.dbs = DBSCAN(eps=0.3, min_samples=3, metric='cityblock')

        self.init = True
        self.is_ring = False

        self.map_ver = 0
        self.cropped_publisher = self.create_publisher(PointCloud2, '/lidar/ring_edge',1)
        self.marker_all = self.create_publisher(Marker, '/lidar/raw', 1)
        
        sub = self.create_subscription(
            PointCloud2(),
            '/hesai/pandar',
            self.receive_pcl,
            1
        )
        sub

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
        # raw_idx = np.expand_dims(np.arange(pcl_np.shape[0]),axis=1)  #nx1
        # self.pcl_np = np.hstack((pcl_np,raw_idx))  #nx6

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
        for coor in local_crss[:,:2]:
            idx = self.nn_obj.radius_neighbors(np.reshape(coor,(1,-1)), return_distance=False)
            cross_idx = np.concatenate((cross_idx, idx[0]), axis=None)
        cross_idx = np.unique(cross_idx).astype(np.int32)
        total_idx = np.unique(np.concatenate((total_idx, cross_idx))).astype(np.uint16)
        self.total_ = cropped_pc[total_idx]
        # self.cross_ids = cropped_pc[cross_idx,-1]

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
            edges = np.where(np.abs(diff)>0.1)[0]
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
            return local_ctr[0], local_ctr[1], 1, length, width, 2, box_heading

        else:
            return None


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

    def main(self):
        start = time.time()
        all_infos = Marker()
        all_infos.header.frame_id = 'Pandar40P'
        all_infos.type = 5  #line list
        all_infos.scale.x = 0.05
        all_infos.color.a = 0.6
        rule_result = []
        if self.total_ is not None:
            ring_edge = self.check_edge(self.total_)
            if ring_edge.shape[0] > 0:
                self.dbs.fit(ring_edge[:,:3])

                labels = self.dbs.labels_
                ring_edge = np.hstack((ring_edge,labels.reshape(-1,1)))

                label_num = max(labels)
                for lb in range(label_num+1):
                # for lb in label_num:
                    cur_pcd = ring_edge[labels==lb]
                    mean_x, min_x, max_x = np.mean(cur_pcd[:,0]), np.min(cur_pcd[:,0]), np.max(cur_pcd[:,0])
                    mean_y, min_y, max_y = np.mean(cur_pcd[:,1]), np.min(cur_pcd[:,1]), np.max(cur_pcd[:,1])
                    mean_z, max_z, min_z = np.mean(cur_pcd[:,2]), np.max(cur_pcd[:,2]), np.min(cur_pcd[:,2])    
                    width = max_x-min_x 
                    length = max_y-min_y
                    height = max_z-min_z
                    volume = width * length * height
                    
                    if height > 0.1 and volume > 0.01: 
                        rule_result.append([mean_x,mean_y,max_z,length, width, height,0.0,4])
                        all_infos.points = all_infos.points + self.draw_box(rule_result[-1])
                        all_infos.color.r, all_infos.color.g, all_infos.color.b = [1.0, 1.0, 1.0]

                        crss_box = self.get_crss_box([mean_x,mean_y],self.ego_info)
                        if crss_box is not None:
                            all_infos.points = all_infos.points + self.draw_box(crss_box)
                            all_infos.color.r, all_infos.color.g, all_infos.color.b = [0.0, 1.0, 1.0]

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
                msg_cropped_pc.width = ring_edge.shape[0]
                msg_cropped_pc.is_dense = False
                msg_cropped_pc.is_bigendian = False
                msg_cropped_pc.fields = fields
                msg_cropped_pc.point_step = (itemsize*len(ring_edge[0]))
                msg_cropped_pc.row_step = (itemsize*len(ring_edge[0]))
                pcd_byte = ring_edge.astype(np.float32).tobytes()
                pcd_intArr = array.array('B', pcd_byte)
                msg_cropped_pc.data = pcd_intArr
                self.cropped_publisher.publish(msg_cropped_pc)   
        self.marker_all.publish(all_infos)
        # print((time.time()-start)*1000,'ms')

def main(args=None):
    rclpy.init(args=None)

    lidar = detect_n_tracker()
    rclpy.spin(lidar)

    lidar.destroy_node()
    rclpy.shutdown()


if __name__=='__main__':
    main()
