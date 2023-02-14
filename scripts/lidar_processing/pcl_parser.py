#!/usr/bin/env python

import rospy
import pcl
import math
import pcl_helper

import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from camera_lidar_fusion.msg import LidarObject
from camera_lidar_fusion.msg import LidarObjectList

class pcl_data_calc():
    def __init__(self):
        rospy.init_node('PclParser', anonymous=False)
        rospy.Subscriber('/pointcloud/os1_pc2', PointCloud2, self.pcl_callback)

        self.pub = rospy.Publisher('/pointcloud/filtered', PointCloud2, queue_size=1)
        self.pub_ran = rospy.Publisher('/pointcloud/ransac', PointCloud2, queue_size=1)
        self.lidar_object_pub = rospy.Publisher('lidar_objects', LidarObjectList, queue_size=1)
    
    def pcl_callback(self, data):
        
        # Parameters for ROI setting and Removing noise
        self.xmin = 3.0
        self.xmax = 4.0
        self.mean_k = 1
        self.thresh = 0.0005
        # Convert sensor_msgs/PointCloud2 -> pcl
        cloud = pcl_helper.ros_to_pcl(data)
        if cloud.size > 0:

            # Removing ground
            _, _, cloud = self.do_ransac_plane_normal_segmentation(cloud, 0.15)

            new_data_2 = pcl_helper.pcl_to_ros(cloud)
            cloud = pcl_helper.XYZRGB_to_XYZ(cloud)

            # Clustering
            cloud, indices = self.do_euclidean_clustering(cloud)
            # self.do_euclidean_clustering(cloud)

            cloud = pcl_helper.XYZ_to_XYZRGB(cloud, (255,255,255))

            Objects = LidarObjectList()

            for filtered_data in cloud:
                x = filtered_data[0]
                y = filtered_data[1]
                z = filtered_data[2]

                object_variable = LidarObject()
                object_variable.x = x
                object_variable.y = y
                object_variable.z = z

                Objects.LidarObjectList.append(object_variable)

            
            self.lidar_object_pub.publish(Objects)
            
            # Convert pcl -> sensor_msgs/PointCloud2
            new_data = pcl_helper.pcl_to_ros(cloud)

            # Publish filtered LiDAR pointcloud data (sensor_msgs/pointcloud2)
            self.pub.publish(new_data)
            self.pub_ran.publish(new_data_2)

            rospy.loginfo("Filtered Point Published")
        
        else:
            pass
    
    
    def do_ransac_plane_normal_segmentation(self, pcl_data, input_max_distance):

        segmenter = pcl_data.make_segmenter_normals(ksearch=50)
        segmenter.set_optimize_coefficients(True)
        segmenter.set_model_type(pcl.SACMODEL_NORMAL_PLANE)  #pcl_sac_model_plane
        segmenter.set_normal_distance_weight(0.1)
        segmenter.set_method_type(pcl.SAC_RANSAC) #pcl_sac_ransac
        segmenter.set_max_iterations(1000)
        segmenter.set_distance_threshold(input_max_distance) #0.03)  #max_distance
        indices, coefficients = segmenter.segment()

        inliers = pcl_data.extract(indices, negative=False)
        outliers = pcl_data.extract(indices, negative=True)

        return indices, inliers, outliers

    
    def do_euclidean_clustering(self, pcl_data):

        tree = pcl_data.make_kdtree()

        ec = pcl_data.make_EuclideanClusterExtraction()
        ec.set_ClusterTolerance(0.9) #0.02
        ec.set_MinClusterSize(10)
        ec.set_MaxClusterSize(5000)
        ec.set_SearchMethod(tree)
        cluster_indices = ec.Extract()

        color_cluster_point_list = []

        for j, indices in enumerate(cluster_indices):
            for i, indice in enumerate(indices):
                color_cluster_point_list.append([pcl_data[indice][0],
                                                pcl_data[indice][1],
                                                pcl_data[indice][2]
                                                ])

        cluster_cloud = pcl.PointCloud()
        cluster_cloud.from_list(color_cluster_point_list)

        return cluster_cloud,cluster_indices

if __name__ == '__main__':
    try:
        pcl_data_calc()
        rospy.spin()
    
    except rospy.ROSInterruptException:
        pass