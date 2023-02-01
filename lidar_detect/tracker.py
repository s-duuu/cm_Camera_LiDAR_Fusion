import numpy as np 
from kalmanFilter import KalmanFilter
# from KalmanFilter_v2 import KalmanFilter
from scipy.optimize import linear_sum_assignment
from collections import deque
from scipy.spatial import distance


class Tracks(object):
    """docstring for Tracks"""
    def __init__(self, detection, trackId, v_x, v_y, dt):
        super(Tracks, self).__init__()
        
        #### detections : x, y, z, w, l, h, yaw, cls

        self.KF = KalmanFilter(detection.reshape(8,1), v_x, v_y, dt)
        self.KF.predict(dt)
        self.KF.correct(detection.reshape(8,1),1,dt)

        self.trace = deque(maxlen=10)
        self.prediction = detection.reshape(1,8)

        self.trackId = trackId
        self.skipped_frames = 0
        self.start_frames = 0

class Tracker(object):
    """docstring for Tracker"""
    def __init__(self, dist_threshold, max_frame_skipped, max_trace_length):
        super(Tracker, self).__init__()
        self.dist_threshold = dist_threshold
        self.max_frame_skipped = max_frame_skipped
        self.max_trace_length = max_trace_length
        self.trackId = 0
        self.tracks = []

    def update(self, detections,v_x, v_y, dt):

        ### detections : x, y, z, w, l, h, yaw, cls
        if len(self.tracks) == 0:
            for i in range(detections.shape[0]):
                # print(detections[i])
                track = Tracks(detections[i], self.trackId, v_x, v_y, dt)
                self.trackId +=1
                self.tracks.append(track)

        N = len(self.tracks)
        M = len(detections)
        assignment = [-1]*N

        if N != 0 and M ==0:
            assignment = [-1]*N

        else:
            cost = []
            for i in range(N):
                if np.size(detections) != 0:
                    for j in range(M):
                        # diff = np.linalg.norm(self.tracks[i].prediction[0,:2] - detections[:,0:2], axis=1)
                        diff = distance.minkowski(self.tracks[i].prediction[0,:2],detections[j,0:2],2)
                        cost.append(diff)


            if len(cost) !=0:
                cost = np.array(cost)
                cost = np.reshape(cost,(N,M))
                cost = cost * 0.5
                row, col = linear_sum_assignment(cost)
                assignment = [-1]*N
                for i in range(len(row)):
                    assignment[row[i]] = col[i]

        un_assigned_tracks = []
        for i in range(len(assignment)):
            if assignment[i] != -1:
                if (cost[i][assignment[i]] > self.dist_threshold):
                    assignment[i] = -1
                    un_assigned_tracks.append(i)
            else:
                self.tracks[i].skipped_frames +=1
                self.tracks[i].start_frames -= 1

        un_assigned_detects = []
        for i in range(len(detections)):
            if i not in assignment:
                un_assigned_detects.append(i)
            

        del_tracks = []
        for i in range(len(self.tracks)):
            if self.tracks[i].skipped_frames > self.max_frame_skipped :
                del_tracks.append(i)

        if len(del_tracks) > 0:
            for i in reversed(del_tracks):
                if i < len(self.tracks):
                    del self.tracks[i]
                    del assignment[i]

        if(len(un_assigned_detects) != 0):
            for i in range(len(un_assigned_detects)):
                track = Tracks(detections[un_assigned_detects[i]], self.trackId,v_x,v_y,dt)
                self.trackId +=1
                self.tracks.append(track)

        for i in range(len(assignment)+len(un_assigned_detects)):                
            if self.tracks[i].start_frames <3:
                self.tracks[i].start_frames += 1
            self.tracks[i].prediction = self.tracks[i].KF.predict(dt)
            self.tracks[i].prediction = np.reshape(self.tracks[i].prediction,(1,-1))

        for i in range(len(assignment)):
            if(assignment[i] != -1):
                tmp = self.tracks[i].prediction
                self.tracks[i].skipped_frames = 0
                self.tracks[i].prediction = self.tracks[i].KF.correct(detections[assignment[i]],1,dt)

                
                if abs(detections[assignment[i],6] - self.tracks[i].prediction[0,6]) > 0.5 * np.pi :
                    self.tracks[i].prediction[0,6] = tmp[0,6]

                # print(self.tracks[i].prediction[:])
            else:
                self.tracks[i].prediction = self.tracks[i].KF.correct(self.tracks[i].prediction,0,dt)


        for i in range(len(assignment)):
            if(len(self.tracks[i].trace) > self.max_trace_length):
                for j in range(len(self.tracks[i].trace) -self.max_trace_length):
                    del self.tracks[i].trace[j]

        for i in range(len(assignment)+len(un_assigned_detects)):
            self.tracks[i].trace.append(self.tracks[i].prediction)
            self.tracks[i].KF.lastResult = self.tracks[i].prediction

        un_assigned_tracks = []
        for i in range(len(assignment)):
            if assignment[i] != -1:
                if (cost[i][assignment[i]] > self.dist_threshold):
                    assignment[i] = -1
                    un_assigned_tracks.append(i)
            else:
                self.tracks[i].skipped_frames +=1
                self.tracks[i].start_frames -= 1

        un_assigned_detects = []
        for i in range(len(detections)):