import numpy as np
from math import *
from geometry_msgs.msg import Point

################ Visualization utils ################
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

def draw_box(ref_boxes):
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

    return wireframe(FL, FR, RL, RR, height)


def wireframe( FL, FR, RL, RR, z):
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

def rotationMatrixToEulerAngles(R) :
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