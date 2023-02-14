import matplotlib.pyplot as plt
import numpy as np
import os
import csv

t = []
angle = []
heading_x = []
heading_y = []
reference_x = []
reference = []
line_1 = []
line_2 = []
line_3 = []
before = []
after = []

os.chdir('/home/heven/CoDeep_ws/src/cm_Camera_LiDAR_Fusion/src/csv/test')

with open('heading_fusion_result.csv', 'r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    cnt = 1
    for row in lines:
        t.append(float(row[0]))
        angle.append(float(row[1]))
        heading_x.append(float(row[2]))
        heading_y.append(-float(row[3]))
        line_1.append(1.75)
        line_2.append(5.25)
        line_3.append(-1.75)
        if cnt % 5 == 0:
            reference_x.append(float(row[2]))
            reference.append(0)
        # c.append(float(row[2]))
        # r.append(float(row[3]))
        # f.append(float(row[0]))
        # fr.append(float(row[5]))
        # vel.append(float(row[1])*-3.6)
        # filtered_vel.append(float(row[2])*-3.6)
        cnt += 1

csvfile.close()

# with open('1d_kalman_velocity_test.csv', 'r') as csv2:
#     line = csv.reader(csv2, delimiter=',')
#     for r in line:
#         after.append(float(r[0]))

# print(after)
unit_vec = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])

# print(len(c))
# print(len(r))
# print(len(f))
time = np.array(t)
fig = plt.figure(figsize=(30, 13))
plt.quiver(heading_x, heading_y, unit_vec[0], unit_vec[1], scale=30, width=0.003, color="red")
plt.quiver(reference_x, reference, 1, 0, scale=10, width=0.005, color="blue")
plt.plot(heading_x, line_1, linestyle="--", color = "black", linewidth = 3)
plt.plot(heading_x, line_2, linestyle="--", color = "black", linewidth = 3)
plt.plot(heading_x, line_3, linestyle="--", color = "black", linewidth = 3)
# plt.plot(time, r, linestyle="--", color = "orange", linewidth = 3)
# plt.plot(time, f, linestyle="-", color = "green", linewidth = 5)
# plt.plot(f, linestyle="-", color = "green", linewidth = 5)
# plt.plot(time, reference, linestyle="-", color = "red", linewidth = 3)
# plt.plot(time, fr, linestyle="--", color = "black", linewidth = 3)
# # plt.plot(x, crash, linestyle = "-", color = "green", linewidth = 5)
# plt.plot(time, vel, linestyle="-", color = "black", linewidth = 2)
# plt.plot(time, filtered_vel, linestyle="-", color = "blue", linewidth = 4)
# plt.plot(time, reference, linestyle="--", color = "red", linewidth = 2)
# plt.plot(x, before, linestyle="-", color = "black", linewidth = 3)
# plt.plot(x, after, linestyle="-", color = "green", linewidth = 5)
plt.xlabel("X [m]", fontsize=28)
plt.ylabel("Y [m]", fontsize=28)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# plt.xlim([0, 30])
plt.ylim([-2, 6])
# plt.hlines(float(-5/3.6), 0, 747, colors="red", linewidth=3)
# plt.legend(['Fusion'], fontsize= 28)
plt.legend(["Heading", "My Car", "Lane"], fontsize=28)
# plt.legend(['Crash Time'], fontsize= 28)
# plt.legend(["Before Kalman Filter", "After Kalman Filter"], fontsize=28)
# plt.legend(["Crash Time"], fontsize=28)

# ax1 = fig.add_subplot(1, 1, 1)
# ax1.plot(x, c, color = 'red', label='Camera')
# ax1.tick_params(axis='y', labelcolor="red")

# ax2 = fig.add_subplot(1, 1, 1)
# ax2.plot(x, r, color = "green", label='Radar')
# ax2.tick_params(axis='y', labelcolor="green")
# # ax2.legend(loc="upper right")

# ax3 = fig.add_subplot(1, 1, 1)
# ax3.plot(x, f, color = "blue", label='Fusion')
# ax3.tick_params(axis='y', labelcolor='blue')
# # ax3.legend(loc="upper right")

plt.show()