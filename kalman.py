import numpy as np
import matplotlib.py as plt

class KalmanPhaseTracker:

  def __init__(self,dt, process_noise_phase=1e-6, process_noise_freq= 1e-3, measurement_noise= 0.1, adaptive=True):
    self.dt= dt
    self.x= np.array([0,0]) #initial state
    self.P= np.eye(2)*1      #uncertainity
    self.Q= np.array([[process_noise,0],[0,process_noise]])    #process nosie covariance 
    self.R= np.array([[measurement_noise]])  #measurement noise covariance
    self.F= np.array([[1,dt],[0,1]])    #state transition matrix( phase = phase+ freq*dt)
    self.H= np.array([[1,0]])           #meaurement matrix (only phase)
    self.adaptive= adaptive
    self.history=[]

  def predict(self):
    self.x= self.F @ self.x
    self.P= self.F @ self.P @ self.F.T + self.Q
    return self.x[0]

  def update(self, measurement):
    y= measurement -  self.H @ self.x    # measurement- prediction
    self.history.append(y[0])
    #innovation covariance
    S= self.H @ self.P @ self.H.T + self.R
    #kalman gain
    K= self.P @ self.H @ np.linalg.inv(S)

    #updated state
    self.x= self.x + K @ y
    self.P= (np.eye(2) - K @ slef.H) @ self.P
    
    
