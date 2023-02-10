import os
import csv

os.chdir('/home/heven/CoDeep_ws/src/cm_Camera_LiDAR_Fusion/src/csv/practice')

f = open('practice1.csv', 'w')

dict = {}

dict[0] = []
dict[1] = []

dict[0].append([1,1])
dict[0].append([2,3])

writer = csv.writer(f)
writer.writerows(dict[0])

f.close()