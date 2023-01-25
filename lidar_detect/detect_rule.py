import sys
import os

lib_path = os.path.abspath(os.path.join(__file__, '..'))
sys.path.insert(0, lib_path)
from math import *
from lidar_detect.detect_utils import *
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Float64MultiArray, Header
from custom_msgs.msg import Object, ObjectInfos, Paths
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, OPTICS
import array
import time

class rule_detect(Node):
    def __init__(self):
        super().__init__('detect_rule')
        cur_path = os.getcwd()+'/src/lidar_detect/lidar_detect'
        self.cross_pts = np.loadtxt(cur_path+'/crosswalk/Crosswalk_for_detect.txt',dtype=np.float32,delimiter='\t')
        self.cross_idx = self.cross_pts[:,0]
        self.cross_pts = self.cross_pts[:,1:3]

        self.pcl_np = None
        self.radius = 1.6
        self.roi_pc = None
        self.path = None
        self.total_ = None
        self.ego_info = None
        # self.near_crss_id = None
        # self.cross_ids = None
        self.nn_obj = NearestNeighbors(n_neighbors=100, radius=1.6)
        self.nn_path = NearestNeighbors(n_neighbors=1)
        # self.nn_obj_crsswalk = NearestNeighbors(n_neighbors=100, radius=1.6)
        self.dbs = DBSCAN(eps=0.4, min_samples=3, metric='cityblock')

        self.cropped_publisher = self.create_publisher(PointCloud2, '/lidar/ring_edge',1)
        self.roi_publisher = self.create_publisher(PointCloud2, '/lidar/roi',1)
        self.info_pub = self.create_publisher(ObjectInfos, '/lidar/detect_rule', 1)

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
        local_crss, crss_ids = self.crss_local(self.ego_info)  #nx3 (x,y,0)

        int_s = np.array(msg.s, dtype=np.int16)
        idx_50 = (int_s <= 50)
        theta = np.radians(90)
        xy_path = np.column_stack((np.array(msg.x)[idx_50]-1, np.array(msg.y)[idx_50]))
        self.path = np.copy(xy_path)
        h = np.copy(np.array(msg.h)[idx_50])

        ######### crop pointcloud according to local path #########
        pcl_np = np.copy(self.pcl_np)
        # crop_filter = (pcl_np[:,0] > np.min(xy_path[:,0])-3) & (pcl_np[:,0] < np.max(xy_path[:,0])+3) & (pcl_np[:,1] > np.min(xy_path[:,1])-12) & (pcl_np[:,1] < np.max(xy_path[:,1])+12)
        crop_filter = (pcl_np[:,0] > np.min(xy_path[:,0])-12) & (pcl_np[:,0] < np.max(xy_path[:,0])+12) & (pcl_np[:,1] > np.min(xy_path[:,1])-12) & (pcl_np[:,1] < np.max(xy_path[:,1])+12)
        cropped_pc = pcl_np[crop_filter]

        # if cropped_pc.shape[0]>0:
        self.nn_obj.fit(cropped_pc[:,:2])
        # self.nn_obj_crsswalk.fit(cropped_pc[:,:2])
        cross_idx = np.array([])
        total_idx = np.array([])

        crss_idx_chunk = {}
        # print(np.unique(crss_ids).astype(np.uint16))
        for id in np.unique(crss_ids).astype(np.uint16):
            crss_idx_chunk[id] = np.array([])

        for i, coor in enumerate(xy_path):
            if int_s[i] > 5:
                if int_s[i-1] == int_s[i]:
                    continue
                else:
                    idx = self.nn_obj.radius_neighbors(np.reshape(coor,(1,-1)), return_distance=False)
                total_idx = np.concatenate((total_idx, idx[0]), axis=None)  
        total_idx = np.unique(total_idx).astype(np.uint16)     
        if local_crss is not None:   
            for i, coor in enumerate(local_crss[:,:2]):
                idx = self.nn_obj.radius_neighbors(np.reshape(coor,(1,-1)), return_distance=False)
                cross_idx = np.concatenate((cross_idx, idx[0]), axis=None)
                crss_idx_chunk[int(crss_ids[i])] = np.concatenate((crss_idx_chunk[int(crss_ids[i])], idx[0]), axis=None).astype(np.uint32)
            cross_idx = np.unique(cross_idx).astype(np.uint16)
            for id in np.unique(crss_ids).astype(np.uint16):
                # print(crss_idx_chunk[id])
                crss_idx_chunk[id] = cropped_pc[crss_idx_chunk[id]]
            # total_idx = np.unique(np.concatenate((total_idx, cross_idx))).astype(np.uint16)
        # else:
        #     total_idx = np.unique(total_idx).astype(np.uint16)
        path_crss = np.vstack((self.path,cropped_pc[cross_idx,:2]))
        self.nn_path.fit(path_crss)
        roi_pcd = cropped_pc[total_idx]
        
        crss_idx_chunk['roi'] = roi_pcd
        self.total_ = crss_idx_chunk

    def check_edge(self, pcd_dict):
        results_pcd = np.array([])
        total_roi = np.array([])
        for pcd in pcd_dict.values():
            if len(pcd) > 0:
                sorted_pcd = pcd[pcd[:,4].argsort()]
                rings = np.unique(sorted_pcd[:,4])
                edge_results = np.array([])
                roi_results = np.array([])
                
                for ring in rings:
                    idx = sorted_pcd[:,4]==ring
                    cur_pcd = sorted_pcd[idx]
                    theta = np.arctan2(cur_pcd[:,1], cur_pcd[:,0])
                    cur_pcd = cur_pcd[theta.argsort()]
                    
                    edge_idx = np.zeros(cur_pcd.shape[0],dtype=bool)
                    diff = np.diff(cur_pcd[:,2])
                    edges = np.where(np.abs(diff)>0.1)[0]

                    xy_dists = np.linalg.norm(cur_pcd[:,:2],axis=1)
                    xy_diff = np.diff(xy_dists)
                    dist_edges = np.where(np.abs(xy_diff)>0.8)[0]
                    # if edges.shape[0] > 0:
                    #     print(ring, edges, diff)

                    # if edges.shape[0] == 0:
                    #     edge_idx[cur_pcd[:,2]>-1] = 1
                    # if edges.shape[0] > 0 and diff[edges[0]] < 0:
                    #     edge_idx[:edges[0]] = 1
                    #     edges = np.delete(edges,0)
                    # if edges.shape[0] > 0 and diff[edges[-1]] > 0:
                    #     edge_idx[edges[-1]:] = 1
                    #     edges = np.delete(edges,-1)
                    # for i in range(edges.shape[0]//2):
                    #     edge_idx[edges[i*2]:edges[i*2+1]] = 1

                    # if np.sum(edge_idx) > 0:
                    #     print(ring, 'edge : ', edge_idx)

                    if edges.shape[0] > 0 and diff[edges[0]] < 0:  #처음에 falling edge인 경우
                        edge_idx[:edges[0]+1] = 1
                        edges = np.delete(edges,0)
                    if edges.shape[0] > 0:
                        for i,edge in enumerate(edges):
                            # if bitween_dists[edge] < 3:
                            if diff[edge] < 0: 
                                edge_idx[edge+1:] = 0
                            else:  
                                edge_idx[edge+1:] = 1
                    
                    # if dist_edges.shape[0] > 0:
                    #     print(ring, dist_edges, xy_diff)
                        
                    # if dist_edges.shape[0] > 0 and xy_diff[dist_edges[0]] > 0:  #처음에 rising edge인 경우
                    #     edge_idx[:dist_edges[0]+1] = 1
                    #     dist_edges = np.delete(dist_edges,0)
                    # if dist_edges.shape[0] > 0:
                    #     for i,edge in enumerate(dist_edges):
                    #         if xy_diff[edge] > 0:  #rising edge
                    #             edge_idx[edge+1:] = 0
                    #         else:  #falling edge
                    #             edge_idx[edge+1:] = 1

                    # if np.sum(edge_idx) > 0:
                    #     print(ring, edge_idx)

                    if edge_results.shape[0] == 0:
                        edge_results = cur_pcd[edge_idx]
                    else:
                        edge_results = np.vstack((edge_results,cur_pcd[edge_idx]))
                    
                if results_pcd.shape[0] == 0:
                    results_pcd = edge_results
                else:
                    results_pcd = np.vstack((results_pcd,edge_results))
                
            if total_roi.shape[0] == 0:
                total_roi = pcd
            else:
                total_roi = np.vstack((total_roi,pcd))
        return results_pcd, total_roi
    
    def crss_local(self, ego):
        crss_loc = np.zeros((self.cross_pts.shape[0], 3))
        crss_loc[:,:2] = self.cross_pts - ego[:2]
        head = -ego[2]
        R = np.array([[cos(head), -sin(head), 0],[sin(head), cos(head), 0], [0, 0, 1]])  #3x3
        crss_loc=np.matmul(R, crss_loc.T).T #nx3
        # crss_loc_forward=crss_loc[crss_loc[:,0]>0]
        if crss_loc.shape[0] == 0:
            return None
        # crss_idx_forward = self.cross_idx[crss_loc[:,0]>0]
        dist = np.linalg.norm(crss_loc[:,:2],axis=1)
        # self.near_crss_id=crss_idx_forward[np.argmin(dist)]
        return crss_loc[dist<70], self.cross_idx[dist<70]

    def pub_pcd(self, pcd, publisher, num_feature):
        ros_dtype = PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize

        fields = [
            PointField(name='x', offset=0*itemsize, datatype=ros_dtype, count=1),
            PointField(name='y', offset=1*itemsize, datatype=ros_dtype, count=1),
            PointField(name='z', offset=2*itemsize, datatype=ros_dtype, count=1),
            PointField(name='intensity', offset=3*itemsize, datatype=ros_dtype, count=1),
            PointField(name='ring', offset=4*itemsize, datatype=ros_dtype, count=1),
            PointField(name='theta', offset=5*itemsize, datatype=ros_dtype, count=1),
            PointField(name='cut', offset=6*itemsize, datatype=ros_dtype, count=1),
            PointField(name='label', offset=6*itemsize, datatype=ros_dtype, count=1)
        ]
        fields = fields[:num_feature]

        header = Header()
        header.frame_id = 'Pandar40P'

        msg_cropped_pc = PointCloud2()
        msg_cropped_pc.header = header
        msg_cropped_pc.height = 1
        if pcd.shape[0] > 0:
            msg_cropped_pc.width = pcd.shape[0]
            msg_cropped_pc.is_dense = False
            msg_cropped_pc.is_bigendian = False
            msg_cropped_pc.fields = fields
            msg_cropped_pc.point_step = (itemsize*len(pcd[0]))
            msg_cropped_pc.row_step = (itemsize*len(pcd[0]))
            pcd_byte = pcd.astype(np.float32).tobytes()
            pcd_intArr = array.array('B', pcd_byte)
            msg_cropped_pc.data = pcd_intArr
        publisher.publish(msg_cropped_pc)   

    def main(self):
        start=time.time()

        pub_infos = ObjectInfos()
        rule_result = []
        ring_edge = np.array([])
        total = np.array([])
        if self.total_ is not None:
            ring_edge, total = self.check_edge(self.total_)
            # '''
            if ring_edge.shape[0] > 0:
                self.dbs.fit(ring_edge[:,:3])

                labels = self.dbs.labels_
                ring_edge = np.hstack((ring_edge,labels.reshape(-1,1)))
                
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


                    dist, idx = self.nn_path.kneighbors([[mean_x,mean_y]],1,return_distance=True)
                    print(dist, idx)

                    if dist < 1.2 and height > 0.15 and volume > 0.07 and length < 3.2: 
                        rule_result.append([mean_x,mean_y,max_z,width, length, height,0.0,4])  #ruleba class : 4
                        # print('ruleba 나온')
                
        self.pub_pcd(ring_edge, self.cropped_publisher, 8)
        self.pub_pcd(total, self.roi_publisher, 5)
                
        if len(rule_result) > 0:
            for result in rule_result:
                info = Object()
                info.point.x, info.point.y, info.point.z = result[0:3]
                info.w, info.l, info.h = result[3:6]
                info.yaw = result[6]

                info.label = int(result[-1])
                # info.nearest_crss_id = int(self.near_crss_id)
                pub_infos.objects.append(info)

        self.info_pub.publish(pub_infos)
        print('rule : ', (time.time()-start)*1000, 'ms')

def main(args=None):
    rclpy.init(args=None)

    lidar = rule_detect()
    rclpy.spin(lidar)

    lidar.destroy_node()
    rclpy.shutdown()


if __name__=='__main__':
    main()