import numpy as np	
from numpy.linalg import inv


class KalmanFilter(object):

    def __init__(self,init_pose,v_x,v_y,dt,stateVariance=1,measurementVariance=1):
        super(KalmanFilter, self).__init__()
        self.stateVariance = stateVariance
        self.measurementVariance = measurementVariance
        self.dt = dt

        #init_pose : x, y, z, w, l, h, yaw, cls,v_x, v_y(10개)
        self.init_pose = init_pose  
        self.init_pose = np.append(self.init_pose,v_x)
        self.init_pose = np.append(self.init_pose,v_y)
        # self.init_pose = np.append(self.init_pose,w)
        
        self.lastResult = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0], [0]])
        self.initModel()

    def initModel(self): 
        self.A = np.eye(10)
        self.A[0,-2] = self.dt  #x + v_x * dt
        self.A[1,-1] = self.dt  #y + v_y * dt
        # self.A[3,-1] = self.dt  #yaw + w * dt

        # self.A = np.array([[1,0,0,0,0,0,0,0,self.dt,0],
        #                     [0,1,0,0,0,0,0,0,0,self.dt],
        #                     [0,0,1,0,0,0,0,0,0,0],
        #                     [0,0,0,1,0,0,0,0,0,0],
        #                     [0,0,0,0,1,0,0,0,0,0],
        #                     [0,0,0,0,0,1,0,0,0,0],
        #                     [0,0,0,0,0,0,1,0,0,0],
        #                     [0,0,0,0,0,0,0,1,0,0],
        #                     [0,0,0,0,0,0,0,0,1,0],
        #                     [0,0,0,0,0,0,0,0,0,1]])

        self.P = self.stateVariance * np.eye(10)
        # self.P = np.matrix(self.stateVariance*np.identity(self.A.shape[0]))
        # self.P = self.stateVariance * np.array([[1,0,0,0,0,0,0,0,0,0],
        #                                         [0,1,0,0,0,0,0,0,0,0],
        #                                         [0,0,1,0,0,0,0,0,0,0],
        #                                         [0,0,0,1,0,0,0,0,0,0],
        #                                         [0,0,0,0,1,0,0,0,0,0],
        #                                         [0,0,0,0,0,1,0,0,0,0],
        #                                         [0,0,0,0,0,0,1,0,0,0],
        #                                         [0,0,0,0,0,0,0,1,0,0],
        #                                         [0,0,0,0,0,0,0,0,1,0],
        #                                         [0,0,0,0,0,0,0,0,0,1]])

        # self.H = np.matrix([[1,0,0,0,0,0,0,0,0,0],
        #                     [0,1,0,0,0,0,0,0,0,0],
        #                     [0,0,1,0,0,0,0,0,0,0],
        #                     [0,0,0,1,0,0,0,0,0,0],
        #                     [0,0,0,0,1,0,0,0,0,0],
        #                     [0,0,0,0,0,1,0,0,0,0],
        #                     [0,0,0,0,0,0,0,0,0,0],
        #                     [0,0,0,0,0,0,0,0,0,0]])

        self.H = np.matrix([[1,0,0,0,0,0,0,0,0,0],  #x
                            [0,1,0,0,0,0,0,0,0,0],  #y
                            [0,0,1,0,0,0,0,0,0,0],  #z
                            [0,0,0,0,0,0,0,0,0,0],  #width
                            [0,0,0,0,0,0,0,0,0,0],  #length
                            [0,0,0,0,0,0,0,0,0,0],  #height
                            [0,0,0,0,0,0,1,0,0,0],  #yaw
                            [0,0,0,0,0,0,0,0,0,0]   #cls
                            ])

        # self.R = np.matrix(self.measurementVariance*np.identity(self.H.shape[0]))
        self.R = self.measurementVariance * np.array([[1,0,0,0,0,0,0,0],  #높을수록 관측 value invalid
                                                        [0,1,0,0,0,0,0,0],
                                                        [0,0,1,0,0,0,0,0],
                                                        [0,0,0,1,0,0,0,0],
                                                        [0,0,0,0,1,0,0,0],
                                                        [0,0,0,0,0,1,0,0],
                                                        [0,0,0,0,0,0,0.100,0],
                                                        [0,0,0,0,0,0,0,1]])

        self.Q = np.array([[1,0,0,0,0,0,0,0,0,0],    #x
                            [0,1,0,0,0,0,0,0,0,0],   #y
                            [0,0,1,0,0,0,0,0,0,0],   #z
                            [0,0,0,0,0,0,0,0,0,0],   #w
                            [0,0,0,0,0,0,0,0,0,0],   #l
                            [0,0,0,0,0,0,0,0,0,0],   #h
                            [0,0,0,0,0,0,0.01,0,0,0],  #yaw  
                            [0,0,0,0,0,0,0,0,0,0],  #cls
                            [0,0,0,0,0,0,0,0,8,0],   #d_x  높을수록 변화폭이 커짐
                            [0,0,0,0,0,0,0,0,0,8]])   #d_y
                            # [0,0,0,0,0,0,0,0,1]])  #d_yaw


        self.erroCov = self.P
        
        self.state = self.init_pose.reshape(10,1)
        
        
    def predict(self,dt):
        self.dt = dt
        # print(self.state)
        self.predictstate = np.matmul(self.A,self.state)

        self.predictedErrorCov = np.matmul(np.matmul(self.A,self.erroCov),self.A.T) + self.Q

        temp = np.asarray(self.predictstate)
        self.lastResult = self.predictstate

        return temp[0],temp[1],temp[2],temp[3],temp[4],temp[5], self.limit_degree(temp[6]), temp[7], temp[8], temp[9]


    def correct(self, currentMeasurement, flag, dt):
        self.dt = dt
        if not flag:
            # currentMeasurement = self.lastResult
            currentMeasurement = self.lastResult[0:8]
        else:
            currentMeasurement = currentMeasurement[0:8]

        currentMeasurement = np.reshape(currentMeasurement,(8,-1))

        self.kalmanGain = np.matmul(np.matmul(self.predictedErrorCov , self.H.T),np.linalg.pinv(
                                np.matmul(np.matmul(self.H,self.predictedErrorCov),self.H.T)+self.R))
        self.state = self.predictstate + np.matmul(self.kalmanGain,(currentMeasurement - (np.matmul(self.H,self.predictstate))))
        self.erroCov = np.matmul((np.identity(self.P.shape[0]) -np.matmul(self.kalmanGain,self.H)),self.predictedErrorCov)
        self.state[6] = self.limit_degree(self.state[6])
        # print(self.state[6])
        return np.reshape(self.state,(1,10))

    def limit_degree(self, ang):
        if -np.pi <= ang <= np.pi:
            new_ang = ang

        else:
            new_ang = ang % (2*np.pi)

            if new_ang < np.pi:
                pass
            else:

                new_ang -= 2*np.pi

        return new_ang  