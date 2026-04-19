import numpy as np
import matplotlib.py as plt

class KalmanPhaseTracker:

  def __init__(self,dt, process_noise_phase=1e-6, process_noise_freq= 1e-3, measurement_noise= 0.1, adaptive=True):
    self.dt= dt
    self.x= np.array([0,0]) #initial state
    self.P= np.eye(2)*1      #uncertainity
    self.Q= np.array([[process_noise_phase,0],[0,process_noise_phase]])    #process nosie covariance 
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
    self.P= (np.eye(2) - K @ self.H) @ self.P

def smooth(self, measurement):
  #Rauch_tung Smooting for Kalman tracker
  n=len(measurement)
  x_filtered=[]
  P_filtered=[]
  x_predicted=[]
  P_predicted=[]

  x_predicted.append(self.x.copy())
  P_predicted.append(self,P.copy())

  print("Forward pass")
  for i,z in enumerate(measurement):
    #predict step
    pred_phase = self.predict()  #to update x and P
    x_predicted.append(self.x.copy())
    P_predicted.append(self.P.copy())

    self.update(np.array([z]))
    x_filtered.append(self.x.copy())
    P_filtered.append(self.P.copy())

  #backward smoothing pass
  print("Backward pass")
  x_smooth=np.zeros((len(x_filtered),2))
  P_smooth=np.zeros((len(x_filtered),2,2))

  #last smooth=last filtered
  x_smooth[-1] = x_filtered[-1]
  P_smooth[-1] = P_filtered[-1]

  #backward recursion
  for k in range(len(x_filtered)-2,-1,-1):
    P_pred_inv=np.linalg.inv(P_predicted[k+1])
    C_k= P_filtered[k] @ self.F.T @ P_pred_inv
    x_smooth[k]= x_filtered + c_k @ (x_smooth[k+1]-x_predicted[k+1]) @ C_k.T
    return x_smooth, P_smooth


    
