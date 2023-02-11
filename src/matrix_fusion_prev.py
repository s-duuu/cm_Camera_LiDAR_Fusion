#!/usr/bin/env python

import rospy
import os
import numpy as np
import numpy.linalg as lin
import kalman_filter
import math
import cv2
import pandas as pd


from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from camera_lidar_fusion.msg import BoundingBoxes
from camera_lidar_fusion.msg import BoundingBox
from camera_lidar_fusion.msg import LidarObject
from camera_lidar_fusion.msg import LidarObjectList
from filterpy.kalman import KalmanFilter
import time

VELOCITY_MAX_CNT = 10

class fusion():
    def __init__(self):

        self.bridge = CvBridge()
        self.fusion_index_list = []
        self.fusion_distance_list = [[], []]
        self.fusion_velocity_list = [[], []]
        self.lidar_object_list = []
        self.bounding_box_list = []
        self.distance_thresh = 6
        self.angle_thresh = 30
        self.my_speed = 30
        self.bbox_num = 0
        self.min_lidar_x = []
        self.cur_time = 0
        self.time_diff = 0
        self.velocity_cnt = 1
        self.dis_vel_list = []

        rospy.init_node('fusion_node', anonymous=False)
        rospy.Subscriber('tracked_boxes', BoundingBoxes, self.camera_object_callback)
        rospy.Subscriber('lidar_objects', LidarObjectList, self.lidar_object_callback)
        rospy.Subscriber('tracked_image', Image, self.visualize)

    # 카메라 Bounding Box Callback 함수
    def camera_object_callback(self, data):

        self.object_id_point = []
        self.bounding_box_list = data.bounding_boxes
        self.bbox_num = len(self.bounding_box_list)

        for i in range(self.bbox_num):
            self.object_id_point.append([])
            self.dis_vel_list.append([])
            self.min_lidar_x.append(math.inf)
            self.angle_thresh
        
        # rospy.Subscriber('lidar_objects', LidarObjectList, self.lidar_object_callback)

    # 레이더 XYZV Callback 함수
    def lidar_object_callback(self, data):

        self.lidar_object_list = data.LidarObjectList
    
    # 레이더 2D point가 Bounding Box 내에 위치하는지 return (True or False)
    def is_in_bbox(self, bbox, lidar_2d):

        if lidar_2d[0] > bbox.xmin and lidar_2d[0] < bbox.xmax and lidar_2d[1] > bbox.ymin and lidar_2d[1] < bbox.ymax:
            return True
        
        else:
            return False
    
    # 3D point -> 2D point Projection
    def transform(self, lidar_object):
        intrinsic_matrix = np.array([[640/math.tan(0.5*math.radians(50)), 0, 640], [0, 640/math.tan(0.5*math.radians(50)), 480], [0, 0, 1]])
        projection_matrix = np.array([[-0.1736, -0.9848, 0, -1.08], [0, 0, -1, -0.5], [0.9848, -0.1736, 0, 0.368]]) 

        world_point = np.array([[lidar_object.x], [lidar_object.y], [lidar_object.z], [1]])

        transformed_matrix = intrinsic_matrix @ projection_matrix @ world_point

        scaling = transformed_matrix[2][0]

        transformed_matrix /= scaling

        x = round(transformed_matrix[0][0])
        y = round(transformed_matrix[1][0])

        # cv2.line(self.image, (x,y), (x,y), (0, 255, 0), 3)

        return (x,y)
            

    # # Bounding Box 밑변 중점 Z=-0.5 가정하고 2D point -> 3D point Projection
    # def transformation_demo(self):
    #     Rt = np.array([[0.1736, -0.9848, 0], [0, 0, -1], [0.9848, 0.1736, 0]]).T

    #     # YOLO detecting 될 때
    #     if len(self.bounding_box_list) != 0:
    #         for bbox in self.bounding_box_list:

    #             x = (bbox.xmin + bbox.xmax) / 2
    #             y = bbox.ymax

    #             fx = 640/math.tan(math.radians(25))
    #             fy = fx

    #             u = (x - 640) / fx
    #             v = (y - 480) / fy

    #             Pc = np.array([[u], [v], [1]])

    #             t = np.array([[1.3842], [0.5], [2.0914]])

    #             pw = Rt @ (Pc-t)
    #             cw = Rt @ (-t)

    #             k = (cw[2][0] + 0.5) / (cw[2][0] - pw[2][0])

    #             world_point = cw + k*(pw-cw)
                
    #             x_c = world_point[0][0]
    #             y_c = world_point[1][0]

    #             camera_object = (x_c, y_c)

    #             self.matching(bbox, camera_object)

    #     # YOLO detecting 끊겼을 때 (radar_risk_calculate 함수 호출)
    #     else:
    #         min_x = math.inf
    #         for radar_object in self.radar_object_list:
    #             if radar_object.x < min_x:
    #                 min_x = radar_object.x
            
    #         self.radar_risk_calculate(min_x)
            
    # 동일 객체 판단 및 최종 거리, 속도 데이터 산출
    def matching(self):
        # print("# bbox num : ", self.bbox_num)

        for lidar_point in self.lidar_object_list:
            # print(lidar_point.x)
            for bbox in self.bounding_box_list:
                if self.is_in_bbox(bbox, self.transform(lidar_point)) == True:
                    if lidar_point.x < self.min_lidar_x[bbox.id-1]:
                        self.min_lidar_x[bbox.id-1] = lidar_point.x
        
        # for min_point in self.min_lidar_x:
            # print("Min point x : ", min_point)
        
        fusion_distance_list.append(self.min_lidar_x[0])

        for bbox in self.bounding_box_list:
            
            end_time = time.time()

            if len(self.fusion_distance_list[bbox.id-1]) == 0:
                self.fusion_distance_list[bbox.id-1].append(self.min_lidar_x[bbox.id-1])
                self.dist_risk_calculate(bbox, self.min_lidar_x[bbox.id-1])
                self.cur_time = end_time
            
            else:
                velocity = -3.6*(self.min_lidar_x[bbox.id-1] - self.fusion_distance_list[bbox.id-1][-1]) / (self.time_diff)
                self.dis_vel_list[bbox.id - 1] = [self.min_lidar_x[bbox.id - 1], velocity]
                if bbox.id == 1:
                    velocity_list.append(velocity)
                # print("Distance difference : ", self.min_lidar_x[bbox.id-1] - self.fusion_distance_list[bbox.id-1][-1])
                # print("Cycle Time difference : ", end_time - self.start_time)
                # print("Loop Time difference : ", end_time - self.cur_time)
                # print("self.time_diff : ", self.time_diff)
                print("Velocity : ", velocity)
                self.fusion_distance_list[bbox.id-1].append(self.min_lidar_x[bbox.id-1])
                # velocity_list.append(velocity)
                self.cur_time = end_time
            
            print("-------------------------")

        print("==============================")

    def risk_calculate(self, bbox, distance, velocity):
        
        if distance < 6:
            cv2.rectangle(self.image, (0, 0), (1280, 960), (0,0,255), 50, 1)
        
        else:
            car_velocity = self.my_speed + velocity

            print("car velocity : ", car_velocity)

            crash_time = distance * 3600 / (1000 * (car_velocity - math.sin(math.radians(85))*self.my_speed))

            print("Crash time : ", crash_time)

            crash_list.append(crash_time)
            
            lane_change_time = 3.5 * 3600 / (1000*self.my_speed * math.cos(math.radians(85)))
            
            # Ok to change lane
            if crash_time - lane_change_time >= 3.5 or self.my_speed > car_velocity:
                pass
            
            # Warning
            elif crash_time - lane_change_time < 3.5 and crash_time - lane_change_time >= 2.5:
                cv2.rectangle(self.image, (0, 0), (1280, 960), (0,130,255), 50, 1)
                cv2.line(self.image, (1000, 800), (int((bbox.xmin + bbox.xmax)/2), bbox.ymax), (0, 130, 255), 5, 1)
                cv2.line(self.image, (850, 800), (1150, 800), (0, 130, 255), 5, 1)
                cv2.line(self.image, (bbox.xmin, bbox.ymax), (bbox.xmax, bbox.ymax), (0, 130, 255), 5, 1)

            # Dangerous
            else:
                cv2.rectangle(self.image, (0, 0), (1280, 960), (0,0,255), 50, 1)
                cv2.line(self.image, (1000, 800), (int((bbox.xmin + bbox.xmax)/2), bbox.ymax), (0, 0, 255), 5, 1)
                cv2.line(self.image, (850, 800), (1150, 800), (0, 0, 255), 5, 1)
                cv2.line(self.image, (bbox.xmin, bbox.ymax), (bbox.xmax, bbox.ymax), (0, 0, 255), 5, 1)
    
    def dist_risk_calculate(self, bbox, distance):
        
        # Ok to change lane
        if distance > 20:
            pass
        
        # Warning
        elif 10 < distance <= 20:
            cv2.rectangle(self.image, (0, 0), (1280, 960), (0,130,255), 50, 1)
            cv2.line(self.image, (1000, 800), (int((bbox.xmin + bbox.xmax)/2), bbox.ymax), (0, 130, 255), 5, 1)
            cv2.line(self.image, (850, 800), (1150, 800), (0, 130, 255), 5, 1)
            cv2.line(self.image, (bbox.xmin, bbox.ymax), (bbox.xmax, bbox.ymax), (0, 130, 255), 5, 1)

        # Dangerous
        else:
            cv2.rectangle(self.image, (0, 0), (1280, 960), (0,0,255), 50, 1)
            cv2.line(self.image, (1000, 800), (int((bbox.xmin + bbox.xmax)/2), bbox.ymax), (0, 0, 255), 5, 1)
            cv2.line(self.image, (850, 800), (1150, 800), (0, 0, 255), 5, 1)
            cv2.line(self.image, (bbox.xmin, bbox.ymax), (bbox.xmax, bbox.ymax), (0, 0, 255), 5, 1)


    # # 카메라 detecting 안될 때 레이더 데이터로만 경고
    # def radar_risk_calculate(self, distance):

    #     if distance < 7:
    #         cv2.rectangle(self.image, (0, 0), (1280, 960), (0,0,255), 50, 1)
        
    #     elif distance < 12:
    #         cv2.rectangle(self.image, (0, 0), (1280, 960), (0,130,255), 50, 1)
        
    #     else:
    #         pass
    
    # Image 송출 함수
    def visualize(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        # self.start_time = time.time()
        if self.velocity_cnt == VELOCITY_MAX_CNT:
            cur_time = time.time()
            self.time_diff = cur_time - self.cur_time
            self.cur_time = cur_time
            self.matching()
            self.velocity_cnt = 1
        else:
            self.velocity_cnt += 1

        for bbox in self.bounding_box_list:
            if bbox.id == 1:
                self.risk_calculate(bbox, self.dis_vel_list[bbox.id - 1][0], self.dis_vel_list[bbox.id - 1][1])
        # self.image = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgr8")
    
        cv2.imshow("Display", self.image)
        cv2.waitKey(1)

if __name__ == '__main__':
    
    kf = KalmanFilter(dim_x=2, dim_z=1)

    time_list = []
    only_camera_distance_list = []
    only_radar_distance_list = []
    # fusion_radar_list = []
    fusion_distance_list = []
    velocity_list = []
    crash_list = []
    up_list = []
    down_list = []

    if not rospy.is_shutdown():
        fusion()
        rospy.spin()
    
    
    # # 결과 CSV 파일로 저장
    os.chdir('/home/heven/CoDeep_ws/src/cm_Camera_LiDAR_Fusion/src/csv/test')

    # df = pd.DataFrame({'Fusion': fusion_distance_list})        
    # df.to_csv("distance_fusion_test.csv", index=False)

    df2 = pd.DataFrame({'Velocity' : velocity_list})
    df2.to_csv("velocity_fusion_test.csv", index=False)

    # df3 = pd.DataFrame({'dist_diff' : up_list, 'time_diff' : down_list})
    # df3.to_csv("Difference_test.csv", index=False)

    df3 = pd.DataFrame({'Crash time' : crash_list})
    df3.to_csv("crash_fusion_result.csv", index=False)

    # # df4 = pd.DataFrame({'Radar': only_radar_distance_list})
    # # df4.to_csv("only_radar_distance.csv", index=False)
