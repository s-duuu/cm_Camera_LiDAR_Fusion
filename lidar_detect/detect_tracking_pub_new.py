import sys
import os

from sklearn.neighbors import NearestNeighbors

lib_path = os.path.abspath(os.path.join(__file__, '..'))
sys.path.insert(0, lib_path)

from lidar_detect.tracker import Tracker
from lidar_detect.detect_utils import *

from math import *
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import numpy as np
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
# from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA, Int16, Float32MultiArray, Float64MultiArray, Header
from custom_msgs.msg import Object, ObjectInfos

class tracking_pub(Node):
    def __init__(self):
        super().__init__('tracking_pub')

        cur_path = os.getcwd()+'/src/lidar_detect/lidar_detect'
        self.cross_4pts = np.loadtxt(cur_path+'/crosswalk/crosswalk_with_id.txt',dtype=np.float32,delimiter=',')
        self.cross_4idx = self.cross_4pts[:,0]
        self.cross_4pts = self.cross_4pts[:,1:3]
        self.nn_crss_obj = NearestNeighbors(n_neighbors=1, radius=10)
        self.nn_crss_obj.fit(self.cross_4pts)
        # self.near_crss_id = None
        self.crss_box = None

        self.deep_detects = None
        self.rule_detects = None
        self.ego_info = None

        self.deep_tracker = Tracker(2.0, 10, 10)
        self.rule_tracker = Tracker(0.5, 10, 4)

        self.box_tracking = self.create_publisher(Marker, '/lidar/tracking', 1)
        self.marker_all = self.create_publisher(Marker, '/lidar/raw_detect', 1)
        self.info_pub = self.create_publisher(ObjectInfos, '/lidar/detect_infos', 1)

        sub_deep = self.create_subscription(
            ObjectInfos(),
            '/lidar/detect_deep',
            self.receive_deep,
            1
        )
        sub_deep

        sub_rule = self.create_subscription(
            ObjectInfos(),
            '/lidar/detect_rule',
            self.receive_rule,
            1
        )
        sub_rule

        sub_ego = self.create_subscription(
            Float64MultiArray(),
            "/localization/ego_info",
            self.receive_ego,
            1
        )
        sub_ego

        timer_period = 0.05
        self.timer = self.create_timer(timer_period, self.main)

    def receive_deep(self, deep_info):
        deep = deep_info.objects
        det_np = np.array([])
        for i in range(len(deep)):  #x, y, z, w, l, h, yaw, cls
            cur_ = np.array([deep[i].point.x,deep[i].point.y,deep[i].point.z,
                            deep[i].w,deep[i].l,deep[i].h,deep[i].yaw,deep[i].label])
            if i == 0:
                det_np = cur_
            else:
                det_np = np.vstack((det_np,cur_))
        self.deep_detects = det_np.reshape(-1,8)

    def receive_rule(self, rule_info):
        rule = rule_info.objects
        # if len(rule) > 0:
        #     self.near_crss_id = rule[0].nearest_crss_id
        det_np = np.array([])
        for i in range(len(rule)):  #x, y, z, w, l, h, yaw, cls
            cur_ = np.array([rule[i].point.x,rule[i].point.y,rule[i].point.z,
                            rule[i].w,rule[i].l,rule[i].h,rule[i].yaw,rule[i].label])
            if i == 0:
                det_np = cur_
            else:
                det_np = np.vstack((det_np,cur_))
        self.rule_detects = det_np.reshape(-1,8)

    def receive_ego(self, msg):
        ego_info_d = msg.data
        self.ego_info= np.frombuffer(ego_info_d, dtype=np.float64).reshape(-1,7)[:,:3]  #x, y, heading, 등등
        self.ego_info = self.ego_info.reshape(3,)

    def get_crss_box(self, obj, ego): 
        head = ego[2]
        x = obj[0]
        y = obj[1]
        # current_cross = self.cross_4pts[self.cross_4idx == self.near_crss_id]
        R = np.array([[cos(head), -sin(head), 0],[sin(head), cos(head), 0], [0, 0, 1]])
        obj_global_loc = (np.matmul(R, np.array([x,y,0]).T).T)[:2]+ego[:2]
        print(obj_global_loc)
        curr_id = self.cross_4idx[self.nn_crss_obj.kneighbors(np.array([obj_global_loc]), return_distance=False)]
        print(curr_id)
        current_cross = self.cross_4pts[np.where(self.cross_4idx == curr_id[0])]
        cross_ego_vector = current_cross - ego[:2]
        cross_ego_vector_mag = np.linalg.norm(cross_ego_vector,axis=1)
        max_idx = np.argmax(cross_ego_vector_mag)
        min_idx = np.argmin(cross_ego_vector_mag)

        diff_min_vector = cross_ego_vector-cross_ego_vector[min_idx]
        diff_min_idx = np.linalg.norm(diff_min_vector, axis=1).argsort()
        diff_min_vector = diff_min_vector[diff_min_idx]
        crss_loc = np.zeros((current_cross.shape[0], 3))
        crss_loc[:,:2] = current_cross[diff_min_idx] - ego[:2]
        R = np.array([[cos(-head), -sin(-head), 0],[sin(-head), cos(-head), 0], [0, 0, 1]])
        crss_loc[:,:2] = (np.matmul(R, crss_loc.T).T)[:,:2]  #local 좌표 4개,z=0 (4x3)

        # # # print(obj)
        # x0 = (crss_loc[1,0]-crss_loc[0,0])/(crss_loc[1,1]-crss_loc[0,1])*(obj[1]-crss_loc[0,1])+crss_loc[0,0]
        # x1 = (crss_loc[2,0]-crss_loc[0,0])/(crss_loc[2,1]-crss_loc[0,1])*(obj[1]-crss_loc[0,1])+crss_loc[0,0]
        # x2 = (crss_loc[1,0]-crss_loc[3,0])/(crss_loc[1,1]-crss_loc[3,1])*(obj[1]-crss_loc[3,1])+crss_loc[3,0]
        # x3 = (crss_loc[2,0]-crss_loc[3,0])/(crss_loc[2,1]-crss_loc[3,1])*(obj[1]-crss_loc[3,1])+crss_loc[3,0]

        # y0 = (crss_loc[1,1]-crss_loc[0,1])/(crss_loc[1,0]-crss_loc[0,0])*(obj[0]-crss_loc[0,0])+crss_loc[0,1]
        # y1 = (crss_loc[2,1]-crss_loc[0,1])/(crss_loc[2,0]-crss_loc[0,0])*(obj[0]-crss_loc[0,0])+crss_loc[0,1]
        # y2 = (crss_loc[1,1]-crss_loc[3,1])/(crss_loc[1,0]-crss_loc[3,0])*(obj[0]-crss_loc[3,0])+crss_loc[3,1]
        # y3 = (crss_loc[2,1]-crss_loc[3,1])/(crss_loc[2,0]-crss_loc[3,0])*(obj[0]-crss_loc[3,0])+crss_loc[3,1]

        # shapely
        poly = Polygon([crss_loc[0],crss_loc[1],crss_loc[3],crss_loc[2]])
        print('in',poly.contains(Point(obj[:2])))
        if poly.contains(Point(obj[:2])):
            heading_vector = diff_min_vector[1]
            width = np.linalg.norm(diff_min_vector[2])
            length = np.linalg.norm(diff_min_vector[1])

            local_ctr = (crss_loc[0]+crss_loc[3])/2
            box_heading = np.arctan2(heading_vector[0],heading_vector[1]) - head
            return local_ctr[0], local_ctr[1], 0.6, length, width, 2, box_heading
        else:
            return None

        # under_x=max(x0,x1)
        # upper_x=min(x2,x3)
        # x_bound = np.array([x0,x1,x2,x3])
        # x_bound.sort()
        # y_bound = np.array([y0,y1,y2,y3])
        # y_bound.sort()
        # # print(x_bound, y_bound, obj)
        # if x_bound[1] < obj[0] < x_bound[2] and y_bound[1] < obj[1] < y_bound[2]:  #object 횡단보도에 있음
        #     heading_vector = diff_min_vector[1]
        #     width = np.linalg.norm(diff_min_vector[2])
        #     length = np.linalg.norm(diff_min_vector[1])

        #     local_ctr = (crss_loc[0]+crss_loc[3])/2
        #     box_heading = np.arctan2(heading_vector[0],heading_vector[1]) - head
        #     return local_ctr[0], local_ctr[1], 0.6, length, width, 2, box_heading
        # else:
        #     return None

    def main(self):
        start = time.time()
        all_infos = Marker()
        all_infos.header.frame_id = 'Pandar40P'
        all_infos.type = 5  #line list
        all_infos.scale.x = 0.1

        pub_infos = ObjectInfos()
        cur_object_deep = np.array([])
        if self.deep_detects is not None:
            # print('deep',self.deep_detects.shape)
            self.deep_tracker.update(self.deep_detects, v_x=0, v_y=0, dt=0.05)

            for det in self.deep_detects:  #deep learning raw 결과 : 노랑색
                point_list = draw_box(det)
                cur_color = ColorRGBA()
                cur_color.a = 0.8
                cur_color.r, cur_color.g, cur_color.b = rainbow_colormap[2]
                color_list = [cur_color]*len(point_list)
                all_infos.colors = all_infos.colors + color_list
                all_infos.points = all_infos.points + point_list

            if len(self.deep_tracker.tracks) != 0:
                for i in range(len(self.deep_tracker.tracks)):
                    if (len(self.deep_tracker.tracks[i].trace) >= 1):
                        if self.deep_tracker.tracks[i].start_frames == 3:
                            x = round(self.deep_tracker.tracks[i].trace[-1][0,0],4)
                            y = round(self.deep_tracker.tracks[i].trace[-1][0,1],4)
                            z = round(self.deep_tracker.tracks[i].trace[-1][0,2],4)
                            w = round(self.deep_tracker.tracks[i].trace[-1][0,3],4)
                            l = round(self.deep_tracker.tracks[i].trace[-1][0,4],4)
                            h = round(self.deep_tracker.tracks[i].trace[-1][0,5],4)
                            yaw = round(self.deep_tracker.tracks[i].trace[-1][0,6],4)
                            cls = self.deep_tracker.tracks[i].trace[-1][0,7]
                            
                            rel_v_x = round(self.deep_tracker.tracks[i].trace[-1][0,-2],4)
                            rel_v_y = round(self.deep_tracker.tracks[i].trace[-1][0,-1],4)
                            # rel_w = round(self.deep_tracker.tracks[i].trace[-1][0,-1],4)  #상대 angular velosity

                            Id = self.deep_tracker.tracks[i].trackId
                            if cur_object_deep.shape[0] == 0:
                                cur_object_deep = np.array([Id, x, y, z, w, l, h, yaw, rel_v_x, rel_v_y, cls])

                            else:
                                cur_object_deep = np.vstack((cur_object_deep, np.array([Id, x, y, z, w, l, h, yaw, rel_v_x, rel_v_y, cls])))
        
        cur_object_rule = np.array([])
        if self.rule_detects is not None:
            # print('rule',self.rule_detects.shape)
            self.rule_tracker.update(self.rule_detects, v_x=0, v_y=0, dt=0.05)

            for det in self.rule_detects:  #rule learning raw 결과 : 흰색
                point_list = draw_box(det)
                cur_color = ColorRGBA()
                cur_color.a = 0.8
                cur_color.r, cur_color.g, cur_color.b = rainbow_colormap[18]
                color_list = [cur_color]*len(point_list)
                all_infos.colors = all_infos.colors + color_list
                all_infos.points = all_infos.points + point_list
                # print('raw',det)

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
                            # print('tracking',cur_object_rule)
        
        if cur_object_rule.shape[0] > 0 and cur_object_deep.shape[0] == 0:
            cur_object = cur_object_rule
        elif cur_object_rule.shape[0] == 0 and cur_object_deep.shape[0] > 0:
            cur_object = cur_object_deep
        elif cur_object_rule.shape[0] > 0 and cur_object_deep.shape[0] > 0:
            cur_object = np.vstack((cur_object_deep,cur_object_rule))
        else:
            cur_object = np.array([])
        # print(cur_object)
        tracking_boxes = Marker()
        tracking_boxes.header.frame_id = 'Pandar40P'
        tracking_boxes.type = 5  #line list
        tracking_boxes.scale.x = 0.07
        tracking_boxes.color.a = 0.6  #투명도
        if cur_object.shape[0] != 0:
            if len(cur_object.shape) == 1:
                cur_object = cur_object.reshape(1,11)
            for i, cur_ in enumerate(cur_object):
                self.crss_box = None
                # if cur_[-1] == 4:
                self.crss_box = self.get_crss_box([cur_[1],cur_[2]],self.ego_info)
                # print(cur_, self.crss_box)
                if cur_[-1] == 2:  #person
                    head = self.ego_info[2]
                    x, y = cur_[1],cur_[2]
                    R = np.array([[cos(head), -sin(head), 0],[sin(head), cos(head), 0], [0, 0, 1]])
                    obj_global_loc = (np.matmul(R, np.array([x,y,0]).T).T)[:2]+self.ego_info[:2]
                    obj_global_loc_0 = (np.matmul(R, np.array([x-0.05,y,0]).T).T)[:2]+self.ego_info[:2]
                    obj_global_loc_1 = (np.matmul(R, np.array([x+0.05,y,0]).T).T)[:2]+self.ego_info[:2]
                    obj_global_loc_2 = (np.matmul(R, np.array([x+0.05,y,0]).T).T)[:2]+self.ego_info[:2]
                    obj_global_loc_3 = (np.matmul(R, np.array([x-0.05,y,0]).T).T)[:2]+self.ego_info[:2]
                    print('deep : ', obj_global_loc)
                    print('cclk : ',obj_global_loc_0,obj_global_loc_1,obj_global_loc_2,obj_global_loc_3)
                if self.crss_box is not None: # crsswalk
                    # cur_[1:8] = self.crss_box
                    # if cur_[-1] == 4:
                    #     cur_[-1] = 5  #crsswalk
                    # else:
                    # tracking_boxes.points = tracking_boxes.points + draw_box(cur_[1:8])
                    # tracking_boxes.color.r, tracking_boxes.color.g, tracking_boxes.color.b = rainbow_colormap[6]
                    tracking_boxes.points = tracking_boxes.points + draw_box(self.crss_box)
                    tracking_boxes.color.r, tracking_boxes.color.g, tracking_boxes.color.b = rainbow_colormap[6]
                    info = Object()
                    info.point.x, info.point.y, info.point.z = self.crss_box[0:3]
                    info.point.x += 1.0  #lidar 후륜중심 offset
                    info.l, info.w, info.h = cur_[3:6]
                    info.yaw = cur_[6]
                    # info.rel_v_x, info.rel_v_y = cur_[8:10]
                    info.label = 5
                    pub_infos.objects.append(info)
        
                # else:
                tracking_boxes.points = tracking_boxes.points + draw_box(cur_[1:8])
                tracking_boxes.color.r, tracking_boxes.color.g, tracking_boxes.color.b = rainbow_colormap[6]

                info = Object()
                info.point.x, info.point.y, info.point.z = cur_[1:4]
                info.point.x += 1.0  #lidar 후륜중심 offset
                info.l, info.w, info.h = cur_[4:7]
                info.yaw = cur_[7]
                info.rel_v_x, info.rel_v_y = cur_[8:10]
                info.label = int(cur_[-1])
                pub_infos.objects.append(info)

                
        self.marker_all.publish(all_infos)
        self.box_tracking.publish(tracking_boxes)
        self.info_pub.publish(pub_infos)
        # print('tracking & pub : ', (time.time()-start)*1000, 'ms')

def main(args=None):
    rclpy.init(args=None)

    lidar = tracking_pub()
    rclpy.spin(lidar)

    lidar.destroy_node()
    rclpy.shutdown()


if __name__=='__main__':
    main()
