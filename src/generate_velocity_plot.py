import matplotlib.pyplot as plt
import numpy as np
import os
import csv

t = []
ref_t = []
c = []
r = []
f = []
fr = []
crash = []
vel = []
ref_v = []
filtered_vel = []
reference = []
before = []
after = []

os.chdir('/home/heven/CoDeep_ws/src/cm_Camera_LiDAR_Fusion/src/csv/test')

with open('velocity_fusion_test.csv', 'r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:
        t.append(float(row[0]))
        # c.append(float(row[2]))
        # r.append(float(row[3]))
        # f.append(float(row[0]))
        # fr.append(float(row[5]))
        vel.append(float(row[1])*-3.6)
        filtered_vel.append(float(row[2])*-3.6)
        reference.append(5)

csvfile.close()

with open('velocity_reference.csv', 'r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:
        ref_t.append(float(row[0]))
        ref_v.append(float(row[1]) * 3.6)

csvfile.close()


# with open('1d_kalman_velocity_test.csv', 'r') as csv2:
#     line = csv.reader(csv2, delimiter=',')
#     for r in line:
#         after.append(float(r[0]))

# print(after)

# print(c)

# print(len(c))
# print(len(r))
# print(len(f))
time = np.array(t)
fig = plt.figure(figsize=(30, 13))
# plt.plot(time, c, linestyle="--", color = "blue", linewidth = 3)
# plt.plot(time, r, linestyle="--", color = "orange", linewidth = 3)
# plt.plot(time, f, linestyle="-", color = "green", linewidth = 5)
# plt.plot(f, linestyle="-", color = "green", linewidth = 5)
# plt.plot(time, reference, linestyle="-", color = "red", linewidth = 3)
# plt.plot(time, fr, linestyle="--", color = "black", linewidth = 3)
# plt.plot(x, crash, linestyle = "-", color = "green", linewidth = 5)
plt.plot(time, vel, linestyle="-", color = "black", linewidth = 2)
plt.plot(ref_t, ref_v, linestyle = '--', color = "red", linewidth = 5)
plt.plot(time, filtered_vel, linestyle="-", color = "blue", linewidth = 4)
# plt.plot(time, reference, linestyle="--", color = "red", linewidth = 2)
# plt.plot(x, before, linestyle="-", color = "black", linewidth = 3)
# plt.plot(x, after, linestyle="-", color = "green", linewidth = 5)
plt.xlabel("Time [s]", fontsize=28)
plt.ylabel("Velocity [km/h]", fontsize=28)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.xlim([0, 30])
plt.ylim([2, 14])
# plt.hlines(float(-5/3.6), 0, 747, colors="red", linewidth=3)
# plt.legend(['Fusion'], fontsize= 28)
plt.legend(["Velocity", "Filtered", "Reference"], fontsize=28)
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