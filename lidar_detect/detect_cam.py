import sys
import os

lib_path = os.path.abspath(os.path.join(__file__, '..'))
sys.path.insert(0, lib_path)

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Int16, Float32MultiArray, Float64MultiArray, Header
from custom_msgs.msg import Object, ObjectInfos

import time

class detect_cam(Node):
    def __init__(self):
        super().__init__('detect_cam')

        self.cam_detect = None

        self.info_pub = self.create_publisher(ObjectInfos, '/lidar/detect_cam', 1)

        sub = self.create_subscription(
            PointCloud2(),
            '/hesai/pandar',
            self.receive_pcl,
            1
        )
        sub

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

    def main(self):
        box_info = []
        if self.pcl_np is not None and self.cam_detect is not None:
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
                        mean_x, min_x, max_x = np.mean(det_pcd[:,0]), np.min(det_pcd[:,0]), np.max(det_pcd[:,0])
                        mean_y, min_y, max_y = np.mean(det_pcd[:,1]), np.min(det_pcd[:,1]), np.max(det_pcd[:,1])
                        mean_z, max_z, min_z = np.mean(det_pcd[:,2]), np.max(det_pcd[:,2]), np.min(det_pcd[:,2])    
                        width = max_x-min_x 
                        length = max_y-min_y
                        height = max_z-min_z
                        volume = width * length * height
                        box_info.append([mean_x,mean_y,max_z,length, width, height,0.0,6])

def main(args=None):
    rclpy.init(args=None)

    lidar = detect_cam()
    rclpy.spin(lidar)

    lidar.destroy_node()
    rclpy.shutdown()


if __name__=='__main__':
    main()
                       