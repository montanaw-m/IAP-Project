import numpy as np 
import pandas as pd 
import csv
import scipy as sp 
from scipy import signal

Markers = pd.read_csv("testCalibrationMarkers.csv") # reads csv file containing markers data
Devices = pd.read_csv("testCalibrationDevices.csv") # reads csv file containing devices data

L1 = len(Markers) # length of markers data
Lfilt = L1-4 # length of data minus 5 rows of headings
L2 = len(Devices) # length of devices data

df = pd.DataFrame(Devices) # creates dataframe for devices data
dfarray = np.array(df) # puts dataframe in array
LCV = dfarray[4:L2, 31] # load cell voltage column of data
LCVfilt = sp.signal.resample(LCV, Lfilt) # downsamples to markers data size
LCV2d = (np.reshape(LCVfilt, (Lfilt, 1))) # reshapes array

FPC = dfarray[4:L2, 48:51].astype(float) # force plate centre data (Cx, Cy, Cz)
FPCfilt = sp.signal.resample(FPC, Lfilt) # downsamples data
FPC2d = (np.reshape(FPCfilt, (Lfilt, 3)))/1000 # reshapes and converts from mm to m

df2 = pd.DataFrame(Markers) # creates datafram for markers data
df2array = np.array(df2) # puts dataframe in array
T2 = (df2array[4:L1, 2:5].astype(float))/1000 # array with x, y, z positions for each marker in m
T3 = (df2array[4:L1, 5:8].astype(float))/1000
M = (df2array[4:L1, 8:11].astype(float))/1000
B2 = (df2array[4:L1, 11:14].astype(float))/1000
B3 = (df2array[4:L1, 14:17].astype(float))/1000

Fr = np.zeros((Lfilt,3)) # creates matrix of zeros
COP = np.zeros((Lfilt,4))
Mr = np.zeros((Lfilt,3))

T = (T2+T3)/2 # centre at top markers
B = (B3+B2)/2 # centre at bottom markers

Origin = B # origin at centre of bottom markers
AuxV = np.subtract(T, Origin)
AuxLat = np.subtract(B2, Origin)
x = np.cross(AuxV, AuxLat)
x_hat = x/(np.linalg.norm(x)) # x vector
z = np.cross(x_hat, AuxV)
z_hat = z/(np.linalg.norm(z)) # z vector
y_hat = np.cross(z_hat, x_hat) # y vector

for i in range(len(x_hat)):
	xrow = x_hat[i,:] # grabs row of data
	yrow = y_hat[i,:]
	zrow = z_hat[i,:]
	orow = Origin[i,:]
	orowT = np.transpose(orow) # transposes 

	T1 = np.vstack((xrow, yrow, zrow)) # stacks x, y, z values
	T2 = np.column_stack((T1, orowT)) # adds origin column
	Tpad = np.pad(T2, ((0,1), (0,0)),  'constant') # pads matrix 

	d = np.array([0, 0.26, 0, 1]) # offset matrix
	dt = np.transpose(d) # transpose offset matrix 
	COP[i,:] = np.matmul(Tpad,dt) # calculates COP
#print(COP)

Fp = (-225.91*LCV2d)+4.5629 # equation from V to N

UV = y_hat*(-1)
#print(UV) # unit vector of pole's direction

for i in range(len(Fp)):
	Fr[i,:] = Fp[i,:] * UV[i,:] # computes Fx, Fy and Fz forces
	Mr[i,:] = np.cross((COP[i,0:3]-FPC2d[i,:]), Fr[i,:]) 
	# computes Mx, My and Mz moments

R = np.column_stack((Fr, Mr)) # stacks reference forces and moments
RT = np.transpose(R) # transpose to 6x4595
#print(R)

M = (dfarray[4:L2, 45:48].astype(float))/1000 # force plate moment data from Nmm to Nm
F = dfarray[4:L2, 42:45].astype(float) # force plate force data
FPD = np.column_stack((F, M)) # stacks force and moment data
Ffilt = sp.signal.resample(FPD, Lfilt) # downsamples data
S = np.array(Ffilt) # 4595x6 signal matrix from force plate data
#print(S)
Sinv = np.linalg.pinv(S) # pseudoinverse 6x4595 of signal matrix
#print(Sinv)
SinvT = np.transpose(Sinv) # transpose of pseudoinverse of S

C = np.matmul(RT, SinvT) # calculates 6x6 calibration matrix
np.set_printoptions(suppress=True, precision=4)
print(C)

