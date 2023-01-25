import sys
import os

lib_path = os.path.abspath(os.path.join(__file__, '..'))
sys.path.insert(0, lib_path)

from math import *
import numpy as np
import torch

from lidar_detect.pcdet.config import cfg, cfg_from_yaml_file
from lidar_detect.pcdet.datasets.dataset import DatasetTemplate
from lidar_detect.pcdet.models import build_network, load_data_to_gpu
from lidar_detect.detect_utils import *

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA, Int16, Float64MultiArray, Header, Float32MultiArray
from custom_msgs.msg import Object, ObjectInfos
import time

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

class deep_detect(Node):
    def __init__(self, ckpt, model_cfg='kitti'):
        super().__init__('detect_deep')

        cur_path = os.getcwd()+'/src/lidar_detect/lidar_detect'

        if model_cfg == 'kitti':
            cfg_name = '/cfgs/pointpillar_ej.yaml'
        elif model_cfg == 'waymo':
            cfg_name = '/cfgs/pointpillar_waymo.yaml'

        cfg_from_yaml_file(cur_path+cfg_name, cfg)

        self.cfg = cfg
        self.ckpt = cur_path+ckpt
        self.pcl_np = None
        self.init = True
        self.map_ver = 0
        self.speed_bump = None

        self.info_pub = self.create_publisher(ObjectInfos, '/lidar/detect_deep', 1)
        
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

        sub_yolo = self.create_subscription(
            Float32MultiArray(),
            'image_detection',
            self.receive_yolo,
            1
        )
        sub_yolo

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

    def receive_yolo(self, yolo_msg):
        detect_data = yolo_msg.data
        cam_detect = np.frombuffer(detect_data, dtype=np.float32).reshape(-1,6)  #leftup_xy, rightdown_xy, 정확도, class
        if 3.0 in cam_detect[:,5]:
            self.speed_bump = cam_detect[(cam_detect[:,5] == 3) & (cam_detect[:,4]>0.7)]  #과속방지턱만
    
    def receive_map(self, map):
        self.map_ver = map.data

    ################ Deep Learning utils ################
    def define_modes(self):
        self.demo_dataset = DemoDataset(
            dataset_cfg=self.cfg.DATA_CONFIG, class_names=self.cfg.CLASS_NAMES, points=self.pcl_np)
        self.model = build_network(model_cfg=self.cfg.MODEL, num_class=len(self.cfg.CLASS_NAMES), dataset=self.demo_dataset)
        self.model.load_params_from_file(filename=self.ckpt, logger=None, to_cpu=True)
        self.model.cuda()
        self.model.eval()
    
    def lidar2cam(self, pcd):  #center point : x,y,z
        cam2pix = np.array([[1852.666, 0, 982.862],
              [0, 1866.610, 612.790],
              [0, 0, 1]])
        R = np.array(
            [[-0.13461593, -0.99086486, -0.00808573],
            [-0.01051238,  0.00958763, -0.99989878],
            [ 0.99084209, -0.1345173,  -0.011707]])

        pcd_xyz = pcd - np.array([0.15253799, -1.34450982, -1.12016554])  #후륜 중심으로 맞추기
        cam_xyz = np.matmul(R, pcd_xyz.reshape(3,1))  #3x1 = [3x3][3x1]

        pix_xyz = np.matmul(cam2pix, cam_xyz)  #3x1
        # pix_xyz = pix_xyz.T  #nx3
        pix_s = pix_xyz[2]
        pix_xy = pix_xyz/pix_s

        return [pix_xy[0],pix_xy[1]] #[pix_xy[1],pix_xy[0]]

    def main(self):
        start = time.time()
        pub_infos = ObjectInfos()
        filtered_result = None

        if self.pcl_np is not None:
            data_dict=self.demo_dataset.get_points()
            data_dict = self.demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = self.model.forward(data_dict)
            # print(pred_dicts)
            
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
            filtered_result = np.column_stack([ref_boxes[:,:7], ref_labels.cpu().numpy()])
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
                    continue
                #pedestrian detect
                elif (ref_labels[i] == 2 and score >= 0.37) and ref_boxes.shape[0] > 0:
                    continue
                #cyclist detect
                elif (ref_labels[i] == 3 and score >= 0.4) and ref_boxes.shape[0] > 0:
                    continue
                # elif (ref_labels[i] == 4 and score >= 0.43) and ref_boxes.shape[0] > 0:
                # # elif (ref_labels[i] == 4) and ref_boxes.shape[0] > 0:
                #     print(score)
                #     continue
                else:
                    del_list.append(i)

            filtered_result = np.delete(filtered_result, del_list, 0)

        if filtered_result is not None:
            for result in filtered_result:
                # print(result[-1])
                '''
                if int(result[-1]) == 4:  #class가 Sign일 때 speed bump(nx6) 판단
                    # pix_xy=self.lidar2cam(result[:3])
                    # # print(pix_xy)
                    pix_in_yolo = False
                    #     for sb in self.speed_bump:
                    #         print('x : ', sb[0],sb[2],pix_xy[0])
                    #         print('y : ', sb[1],sb[3],pix_xy[1])
                    #         if (sb[0]-100<pix_xy[0]<sb[2]+100) and (sb[1]-100<pix_xy[1]<sb[3]+100):
                    #             pix_in_yolo = True
                    #             # print(result[0])
                    if self.speed_bump is not None and self.speed_bump.shape[0] > 0:
                        if result[0] > 0: #and result[1] < 0:  #우측 전방
                            print(result)
                            pix_in_yolo = True
                        else:
                            pix_in_yolo = False
                    if pix_in_yolo == False:
                        continue
                        print(pix_in_yolo)
                        '''
                    
                info = Object()
                info.point.x, info.point.y, info.point.z = result[0:3]
                info.w, info.l, info.h = result[3:6]
                info.yaw = result[6]

                info.label = int(result[-1])
                pub_infos.objects.append(info)
        
        self.info_pub.publish(pub_infos)
        print('deep : ', (time.time()-start)*1000, 'ms')

def main(args=None):
    rclpy.init(args=None)

    lidar = deep_detect(ckpt='/trained_models/waymo_200.pth', model_cfg='waymo')
    with torch.no_grad():
        rclpy.spin(lidar)

    lidar.destroy_node()
    rclpy.shutdown()


if __name__=='__main__':
    main()
