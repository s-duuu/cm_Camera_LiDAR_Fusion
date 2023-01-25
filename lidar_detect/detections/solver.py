from sympy import Symbol, solve, Eq
import numpy as np
from itertools import combinations
import scipy.linalg as lin

pix_xy = np.genfromtxt('pix_xyz.txt',delimiter=' ')
pcd_xyz = np.genfromtxt('wld_xyz.txt', delimiter=' ')
print(pix_xy.shape, pcd_xyz.shape)
cam2pix = np.array([[1852.666, 0, 982.862],
              [0, 1866.610, 612.790],
              [0, 0, 1]])


r11=Symbol('r11')
r12=Symbol('r12')
r13=Symbol('r13')
r21=Symbol('r21')
r22=Symbol('r22')
r23=Symbol('r23')
r31=Symbol('r31')
r32=Symbol('r32')
r33=Symbol('r33')
t1=Symbol('t1')
t2=Symbol('t2')
t3=Symbol('t3')
# s=Symbol('s')

# idx_combi = list(combinations(np.arange(pix_xy.shape[0]),4))
# print(len(idx_combi))
iter = pix_xy.shape[0]//6
solve_rt = []
for i in range(iter):
    cur_pix = pix_xy[i*6:(i+1)*6]
    cur_pcd = pcd_xyz[i*6:(i+1)*6]

    pix_1 = np.ones((cur_pix.shape[0],3))
    pix_1[:,:2] = cur_pix
    cur_cam = np.matmul(lin.inv(cam2pix),pix_1.T).T  #nx3

    eqs = []
    for j in range(6):
        s = r31*cur_pcd[j,2]+r32*cur_pcd[j,1]+r33*cur_pcd[j,2]+t3
        eqs.append(Eq(s*(cur_cam[j,0]),r11*cur_pcd[j,0]+r12*cur_pcd[j,1]+r13*cur_pcd[j,2]+t1))
        print(eqs[-1])
        eqs.append(Eq(s*(cur_cam[j,1]),r21*cur_pcd[j,0]+r22*cur_pcd[j,1]+r23*cur_pcd[j,2]+t2))
        print(eqs[-1])
    result = solve(eqs, [r11,r12,r13,r21,r22,r23,r31,r32,r33,t1,t2,t3],dict=True)
    print(result)
