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
        self.radius = 1.5
        self.roi_pc = None
        self.path = None
        self.total_ = None
        self.ego_info = None
        self.near_crss_id = None
        # self.cross_ids = None
        self.nn_obj = NearestNeighbors(n_neighbors=100, radius=self.radius)
        self.dbs = DBSCAN(eps=0.5, min_samples=3, metric='cityblock')

        ros_dtype = PointField.FLOAT32
        dtype = np.float32
        self.itemsize = np.dtype(dtype).itemsize

        self.fields = [
            PointField(name='x', offset=0*self.itemsize, datatype=ros_dtype, count=1),
            PointField(name='y', offset=1*self.itemsize, datatype=ros_dtype, count=1),
            PointField(name='z', offset=2*self.itemsize, datatype=ros_dtype, count=1),
            PointField(name='intensity', offset=3*self.itemsize, datatype=ros_dtype, count=1),
            PointField(name='ring', offset=4*self.itemsize, datatype=ros_dtype, count=1),
            PointField(name='theta', offset=5*self.itemsize, datatype=ros_dtype, count=1),
            PointField(name='label', offset=6*self.itemsize, datatype=ros_dtype, count=1)
        ]
        self.header = Header()
        self.header.frame_id = 'Pandar40P'

        self.cropped_publisher = self.create_publisher(PointCloud2, '/lidar/ring_edge',1)
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
        # self.cross_ids = cropped_pc[cross_idx,-1]

    def check_edge(self, pcd):
        sorted_pcd = pcd[pcd[:,4].argsort()]
        rings = np.unique(sorted_pcd[:,4])
        edge_results = np.array([])

        for ring in rings:
            idx = sorted_pcd[:,4]==ring
            cur_pcd = sorted_pcd[idx]
            theta = np.arctan2(cur_pcd[:,1], cur_pcd[:,0])
            
            cur_pcd = np.hstack((cur_pcd,theta.reshape(-1,1)))
            cur_pcd = cur_pcd[theta.argsort()]
            
            diff_bitween = np.diff(cur_pcd[:,:2],axis=0)
            diff_bitween = np.linalg.norm(diff_bitween,axis=1)
            diff_cut = np.where(diff_bitween > 1)[0]
            # print(diff_cut)
            diff_cut = np.append(diff_cut,-1)

            result_idx = np.array([],dtype=bool)
            print(ring, diff_cut)
            result_mask = np.zeros(cur_pcd.shape[0],dtype=bool)
            for cut_idx in range(diff_cut.shape[0]):
                part_idx=np.zeros(cur_pcd.shape[0],dtype=bool)
                if cut_idx == 0:
                    part_idx[:diff_cut[cut_idx]] = 1
                    # cut_pcd = cur_pcd[:diff_cut[cut_idx]]
                if diff_cut[cut_idx] == -1:
                    part_idx[diff_cut[cut_idx-1]] = 1
                    # cut_pcd = cur_pcd[diff_cut[cut_idx-1]:]
                else:
                    part_idx[diff_cut[cut_idx-1]+1:diff_cut[cut_idx]] = 1
                cut_pcd = cur_pcd[part_idx]
                    # cut_pcd = cur_pcd[diff_cut[cut_idx-1]:diff_cut[cut_idx]]            
                edge_idx = np.zeros(cut_pcd.shape[0],dtype=bool)
                diff = np.diff(cut_pcd[:,2])

                rising_edge = np.where(diff>0.08)[0]
                falling_edge = np.where(diff<-0.08)[0]
                
                edges = np.where(np.abs(diff)>0.08)[0]
                if edges.shape[0] > 0:
                    print(diff_cut[cut_idx], rising_edge,diff[rising_edge], falling_edge,diff[falling_edge])

                if edges.shape[0] > 0 and edges[0] in falling_edge:
                    edge_idx[:edges[0]] = 1
                    falling_edge = np.delete(falling_edge,falling_edge==edges[0])
                    edges = np.delete(edges,0)

                if edges.shape[0] > 0 and edges[-1] in rising_edge:
                    edge_idx[edges[-1]:] = 1
                    rising_edge = np.delete(rising_edge,rising_edge==edges[-1])
                    edges = np.delete(edges,-1)
                
                # for i in range(edges.shape[0]//2):
                #     edge_idx[edges[i*2]:edges[i*2+1]] = 1
                rise, fall = None, None
                for i in range(edges.shape[0]):
                    if rise is None and edges[i] in rising_edge:
                        rise = edges[i]
                    if fall is None and edges[i] in falling_edge:
                        fall = edges[i]
                    if rise is not None and fall is not None:
                        edge_idx[rise:fall] = 1
                        rise = None
                        fall = None
                # result_mask = np.append(result_mask,edge_idx)
                result_mask[part_idx] = edge_idx
                print(edge_idx)
            print(result_mask[result_mask==True])

            if edge_results.shape[0] == 0:
                edge_results = cur_pcd[result_mask]
            else:
                edge_results = np.vstack((edge_results,cur_pcd[result_mask]))

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
        # crss_idx_forward = self.cross_idx[crss_loc[:,0]>0]
        dist = np.linalg.norm(crss_loc_forward[:,:2],axis=1)
        # self.near_crss_id=crss_idx_forward[np.argmin(dist)]
        return crss_loc_forward[dist<70]  

    def main(self):
        start=time.time()

        pub_infos = ObjectInfos()
        msg_cropped_pc = PointCloud2()
        msg_cropped_pc.header = self.header
        rule_result = []
        ring_edge = np.array([])
        if self.total_ is not None:
            ring_edge = self.check_edge(self.total_)
            if ring_edge.shape[0] > 0:
                self.dbs.fit(ring_edge[:,:3])

                labels = self.dbs.labels_
                ring_edge = np.hstack((ring_edge,labels.reshape(-1,1)))

                label_num = max(labels)
                # crss_inter = np.intersect1d(ring_edge[:,-1],self.cross_ids)
                # print(ring_edge[:,-1])
                for lb in range(label_num+1):
                    cur_pcd = ring_edge[labels==lb]
                    mean_x, min_x, max_x = np.mean(cur_pcd[:,0]), np.min(cur_pcd[:,0]), np.max(cur_pcd[:,0])
                    mean_y, min_y, max_y = np.mean(cur_pcd[:,1]), np.min(cur_pcd[:,1]), np.max(cur_pcd[:,1])
                    mean_z, max_z, min_z = np.mean(cur_pcd[:,2]), np.max(cur_pcd[:,2]), np.min(cur_pcd[:,2])
                    width = max_x-min_x 
                    length = max_y-min_y
                    height = max_z-min_z
                    # area = width * length 
                    
                    if height > 0.15: #and area < 1: 
                        # obj_in_crss_idx = np.intersect1d(cur_pcd[:,-1],crss_inter)
                        # if obj_in_crss_idx.shape[0]>0:
                        #     rule_result.append([mean_x,mean_y,max_z,width, length, height,0.0,6])
                        # else:
                        rule_result.append([mean_x,mean_y,max_z,width, length, height,0.0,5])
                        # print('ruleba 나온')
                msg_cropped_pc.height = 1
                msg_cropped_pc.width = ring_edge.shape[0]
                msg_cropped_pc.is_dense = False
                msg_cropped_pc.is_bigendian = False
                msg_cropped_pc.fields = self.fields
                msg_cropped_pc.point_step = (self.itemsize*len(ring_edge[0]))
                msg_cropped_pc.row_step = (self.itemsize*len(ring_edge[0]))
                pcd_byte = ring_edge.astype(np.float32).tobytes()
                pcd_intArr = array.array('B', pcd_byte)
                msg_cropped_pc.data = pcd_intArr
                
        self.cropped_publisher.publish(msg_cropped_pc)
                
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
