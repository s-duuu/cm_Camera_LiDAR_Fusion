import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from numpy.linalg import inv

def call_2dkalman(kf, dt, distance, x_i, v_i):
    kf.x = np.array([[x_i], [v_i]])

    kf.F = np.array([[1, dt],
                     [0, 1]])
    kf.H = np.array([[1, 0]])
    # kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=10)
    kf.Q = np.array([[0.6, 0.], [0., 1.75]])
    kf.R = 1.8

    # z = np.array([distance, velocity])
    z = np.array([[distance]])

    kf.predict()
    kf.update(z)
    # kf.predict(u=None, B=None, F=kf.F, Q=kf.Q)+
    # kf.update(z, kf.R, kf.H)
    return kf.x[1][0]
