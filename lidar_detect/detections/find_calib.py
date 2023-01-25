import os
import numpy as np
import cv2
from math import *

file_num = len(os.listdir(os.getcwd()+'/cam'))
pcd2pix = np.ones(())

wld_xyz = open(os.getcwd()+'/wld_xyz.txt','w')
pix_xyz = open(os.getcwd()+'/pix_xyz.txt','w')
for i in range(file_num):
    cam = np.load(os.getcwd()+'/cam/'+str(i)+'.npy')
    pcd = np.load(os.getcwd()+'/lidar/'+str(i)+'.npy')

    # print(cam.shape, pcd.shape)
    pix_xy = [cam[2]-cam[0],cam[3]-cam[1]]
    pcd_xyz = [pcd[0]-pcd[3]/2/cos(pcd[6]),pcd[1],pcd[2]]

    pix_xyz.write(str(pix_xy[0])+' '+str(pix_xy[1])+'\n')
    wld_xyz.write(str(pcd_xyz[0])+' '+str(pcd_xyz[1])+' '+str(pcd_xyz[2])+'\n')

    # print(pix_xy,pcd_xyz)
wld_xyz.close()
pix_xyz.close()