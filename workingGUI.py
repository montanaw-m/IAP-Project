import os
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from tkinter import ttk

import numpy as np 
import pandas as pd 
import csv
import scipy as sp 
from scipy import signal 

# select csv file containing markers data
def browse_markers():
    global markers_path, Markers
    markersdir = askopenfilename(multiple=False)
    markers_path.set(markersdir)
    print(markersdir)
    Markers = pd.read_csv(markersdir)
    return Markers

# select csv file containing devices data
def browse_devices():
    global devices_path, Devices
    devicedir = askopenfilename(multiple=False)
    devices_path.set(devicedir)
    print(devicedir)
    Devices = pd.read_csv(devicedir)
    return Devices
  
# creates window  
window = Tk()
window.title("Force Plate Calibration")
window.geometry('950x400')
separator = ttk.Separator(window, orient='vertical')
separator.place(relx = 0.25, rely = 0, relheight = 1)
separator2 = ttk.Separator(window, orient='vertical')
separator2.place(relx = 0.39, rely = 0, relheight = 1)

# creates heading labels
label1 = Label(window, text="Data Files:")
label1.place(relx = 0.1, rely = 0.1)
label2 = Label(window, text="6x6 Calibration Matrix:")
label2.place(relx = 0.6, rely = 0.1)
lbl3 = Label(window, text="Force Plate:")
lbl3.place(relx = 0.28, rely = 0.1)

markers_path = StringVar()
lbl1 = Label(window,textvariable=markers_path, wraplength=225)
lbl1.place(relx = 0.01, rely = 0.3)
button2 = Button(text="Browse Markers CSV File", command=browse_markers)
button2.place(relx=0.025, rely=0.2)
devices_path = StringVar()
lbl2 = Label(window,textvariable=devices_path, wraplength=225)
lbl2.place(relx = 0.01, rely = 0.6)
button3 = Button(text="Browse Devices CSV File", command=browse_devices)
button3.place(relx=0.025, rely=0.5)

# calculation of calibration matrix
def FP4():  
	label2['text'] = "6x6 Calibration Matrix: Force Plate 4"
	# define lengths of data
	L1 = len(Markers) 
	Lfilt = L1-4 # account for 5 rows of headings
	L2 = len(Devices)

	# creates dataframes for devices data
	df = pd.DataFrame(Devices) 
	dfarray = np.array(df)
	LCV = dfarray[4:L2, 31] 
	LCVfilt = sp.signal.resample(LCV, Lfilt) 
	LCV2d = (np.reshape(LCVfilt, (Lfilt, 1))) 
	FPC = dfarray[4:L2, 39:42].astype(float) 
	FPCfilt = sp.signal.resample(FPC, Lfilt) 
	FPC2d = (np.reshape(FPCfilt, (Lfilt, 3)))/1000 

	# creates dataframes for markers data
	df2 = pd.DataFrame(Markers) 
	df2array = np.array(df2) 
	T2 = (df2array[4:L1, 2:5].astype(float))/1000 
	T3 = (df2array[4:L1, 5:8].astype(float))/1000
	M = (df2array[4:L1, 8:11].astype(float))/1000
	B2 = (df2array[4:L1, 11:14].astype(float))/1000
	B3 = (df2array[4:L1, 14:17].astype(float))/1000

	# Matrix of zeros for reference forces, moments and COP
	Fr = np.zeros((Lfilt,3)) 
	COP = np.zeros((Lfilt,4))
	Mr = np.zeros((Lfilt,3))

	# calculates x, y, z vectors
	T = (T2+T3)/2 
	B = (B2+B3)/2 
	Origin = B 
	AuxV = np.subtract(T, Origin)
	AuxLat = np.subtract(B2, Origin)
	x = np.cross(AuxV, AuxLat)
	x_hat = x/(np.linalg.norm(x)) 
	y = np.cross(x_hat, AuxV)
	y_hat = z/(np.linalg.norm(y)) 
	z_hat = np.cross(y_hat, x_hat) 

	# grabs row of data 
	for i in range(len(x_hat)):
		xrow = x_hat[i,:] 
		xrowT = np.transpose(xrow)
		yrow = y_hat[i,:]
		yrowT = np.transpose(yrow)
		zrow = z_hat[i,:]
		zrowT = np.transpose(zrow)
		orow = Origin[i,:]
		orowT = np.transpose(orow) 

		# calculates COP for each row of data
		T = np.column_stack((xrowT, yrowT, zrowT, orowT))
		pad = [0, 0, 0, 1]
		Tpad = np.vstack((T, pad)) 
		d = np.array([0, 0, 0.26, 1]) 
		dt = np.transpose(d) 
		COP[i,:] = np.matmul(Tpad, dt)
	# COP[:,2] = 0
	# print(COP)

	# computes reference forces and moments
	Fp = ((-225.91*LCV2d)+4.5629)
	UV = z_hat*(-1) 
	for i in range(Lfilt):
		Fr[i,:] = Fp[i,:] * UV[i,:] 
		Mr[i,:] = np.cross((COP[i,0:3]-FPC2d[i,:]), Fr[i,:]) 
	R = np.column_stack((Fr, Mr)) 
	RT = np.transpose(R)

	# creates signal matrix
	M = (dfarray[4:L2, 36:39].astype(float))/1000 
	F = dfarray[4:L2, 33:36].astype(float) 
	FPD = np.column_stack((F, M)) 
	Ffilt = sp.signal.resample(FPD, Lfilt) 
	S = np.array(Ffilt) 
	Sinv = np.linalg.pinv(S) # pseudoinverse 6x4595 of signal matrix
	SinvT = np.transpose(Sinv) 

	# calculates 6x6 calibration matrix
	C = np.matmul(RT, SinvT) 
	np.set_printoptions(suppress=True, precision=4)
	print(C)

	label3['text'] = "%s" % C

def FP5():  
	label2['text'] = "6x6 Calibration Matrix: Force Plate 5"
	# define lengths of data
	L1 = len(Markers) 
	Lfilt = L1-4 # account for 5 rows of headings
	L2 = len(Devices)

	# creates dataframes for devices data
	df = pd.DataFrame(Devices) 
	dfarray = np.array(df)
	LCV = dfarray[4:L2, 31] 
	LCVfilt = sp.signal.resample(LCV, Lfilt) 
	LCV2d = (np.reshape(LCVfilt, (Lfilt, 1))) 
	FPC = dfarray[4:L2, 48:51].astype(float) 
	FPCfilt = sp.signal.resample(FPC, Lfilt) 
	FPC2d = (np.reshape(FPCfilt, (Lfilt, 3)))/1000 

	# creates dataframes for markers data
	df2 = pd.DataFrame(Markers) 
	df2array = np.array(df2) 
	T2 = (df2array[4:L1, 2:5].astype(float))/1000 
	T3 = (df2array[4:L1, 5:8].astype(float))/1000
	M = (df2array[4:L1, 8:11].astype(float))/1000
	B2 = (df2array[4:L1, 11:14].astype(float))/1000
	B3 = (df2array[4:L1, 14:17].astype(float))/1000

	# Matrix of zeros for reference forces, moments and COP
	Fr = np.zeros((Lfilt,3)) 
	COP = np.zeros((Lfilt,4))
	Mr = np.zeros((Lfilt,3))

	# calculates x, y, z vectors
	T = (T2+T3)/2 
	B = (B2+B3)/2 
	Origin = B 
	AuxV = np.subtract(T, Origin)
	AuxLat = np.subtract(B2, Origin)
	x = np.cross(AuxV, AuxLat)
	x_hat = x/(np.linalg.norm(x)) 
	y = np.cross(x_hat, AuxV)
	y_hat = y/(np.linalg.norm(y)) 
	z_hat = np.cross(y_hat, x_hat) 

	# grabs row of data 
	for i in range(len(x_hat)):
		xrow = x_hat[i,:] 
		xrowT = np.transpose(xrow)
		yrow = y_hat[i,:]
		yrowT = np.transpose(yrow)
		zrow = z_hat[i,:]
		zrowT = np.transpose(zrow)
		orow = Origin[i,:]
		orowT = np.transpose(orow) 

		# calculates COP for each row of data
		T = np.column_stack((xrowT, yrowT, zrowT, orowT))
		pad = [0, 0, 0, 1]
		Tpad = np.vstack((T, pad)) 
		d = np.array([0, 0, 0.26, 1]) 
		dt = np.transpose(d) 
		COP[i,:] = np.matmul(Tpad, dt)
	# COP[:,2] = 0
	# print(COP)

	# computes reference forces and moments
	Fp = ((-225.91*LCV2d)+4.5629)
	UV = z_hat*(-1) 
	for i in range(Lfilt):
		Fr[i,:] = Fp[i,:] * UV[i,:] 
		Mr[i,:] = np.cross((COP[i,0:3]-FPC2d[i,:]), Fr[i,:]) 
	R = np.column_stack((Fr, Mr)) 
	RT = np.transpose(R)

	# creates signal matrix
	M = (dfarray[4:L2, 45:48].astype(float))/1000 
	F = dfarray[4:L2, 42:45].astype(float) 
	FPD = np.column_stack((F, M)) 
	Ffilt = sp.signal.resample(FPD, Lfilt) 
	S = np.array(Ffilt) 
	Sinv = np.linalg.pinv(S) # pseudoinverse 6x4595 of signal matrix
	SinvT = np.transpose(Sinv) 

	# calculates 6x6 calibration matrix
	C = np.matmul(RT, SinvT) 
	np.set_printoptions(suppress=True, precision=4)
	print(C)

	label3['text'] = "%s" % C

def FP6():  
	label2['text'] = "6x6 Calibration Matrix: Force Plate 6"
	# define lengths of data
	L1 = len(Markers) 
	Lfilt = L1-4 # account for 5 rows of headings
	L2 = len(Devices)

	# creates dataframes for devices data
	df = pd.DataFrame(Devices) 
	dfarray = np.array(df)
	LCV = dfarray[4:L2, 31] 
	LCVfilt = sp.signal.resample(LCV, Lfilt) 
	LCV2d = (np.reshape(LCVfilt, (Lfilt, 1))) 
	FPC = dfarray[4:L2, 57:60].astype(float) 
	FPCfilt = sp.signal.resample(FPC, Lfilt) 
	FPC2d = (np.reshape(FPCfilt, (Lfilt, 3)))/1000 

	# creates dataframes for markers data
	df2 = pd.DataFrame(Markers) 
	df2array = np.array(df2) 
	T2 = (df2array[4:L1, 2:5].astype(float))/1000 
	T3 = (df2array[4:L1, 5:8].astype(float))/1000
	M = (df2array[4:L1, 8:11].astype(float))/1000
	B2 = (df2array[4:L1, 11:14].astype(float))/1000
	B3 = (df2array[4:L1, 14:17].astype(float))/1000

	# Matrix of zeros for reference forces, moments and COP
	Fr = np.zeros((Lfilt,3)) 
	COP = np.zeros((Lfilt,4))
	Mr = np.zeros((Lfilt,3))

	# calculates x, y, z vectors
	T = (T2+T3)/2 
	B = (B2+B3)/2 
	Origin = B 
	AuxV = np.subtract(T, Origin)
	AuxLat = np.subtract(B2, Origin)
	x = np.cross(AuxV, AuxLat)
	x_hat = x/(np.linalg.norm(x)) 
	y = np.cross(x_hat, AuxV)
	y_hat = y/(np.linalg.norm(y)) 
	z_hat = np.cross(y_hat, x_hat) 

	# grabs row of data 
	for i in range(len(x_hat)):
		xrow = x_hat[i,:] 
		xrowT = np.transpose(xrow)
		yrow = y_hat[i,:]
		yrowT = np.transpose(yrow)
		zrow = z_hat[i,:]
		zrowT = np.transpose(zrow)
		orow = Origin[i,:]
		orowT = np.transpose(orow) 

		# calculates COP for each row of data
		T = np.column_stack((xrowT, yrowT, zrowT, orowT))
		pad = [0, 0, 0, 1]
		Tpad = np.vstack((T, pad)) 
		d = np.array([0, 0, 0.26, 1]) 
		dt = np.transpose(d) 
		COP[i,:] = np.matmul(Tpad, dt)
	# COP[:,2] = 0
	# print(COP)

	# computes reference forces and moments
	Fp = ((-225.91*LCV2d)+4.5629)
	UV = z_hat*(-1) 
	for i in range(Lfilt):
		Fr[i,:] = Fp[i,:] * UV[i,:] 
		Mr[i,:] = np.cross((COP[i,0:3]-FPC2d[i,:]), Fr[i,:]) 
	R = np.column_stack((Fr, Mr)) 
	RT = np.transpose(R)

	# creates signal matrix
	M = (dfarray[4:L2, 54:57].astype(float))/1000 
	F = dfarray[4:L2, 51:54].astype(float) 
	FPD = np.column_stack((F, M)) 
	Ffilt = sp.signal.resample(FPD, Lfilt) 
	S = np.array(Ffilt) 
	Sinv = np.linalg.pinv(S) # pseudoinverse 6x4595 of signal matrix
	SinvT = np.transpose(Sinv) 

	# calculates 6x6 calibration matrix
	C = np.matmul(RT, SinvT) 
	np.set_printoptions(suppress=True, precision=4)
	print(C)

	label3['text'] = "%s" % C

def FP7():  
	label2['text'] = "6x6 Calibration Matrix: Force Plate 7"
	# define lengths of data
	L1 = len(Markers) 
	Lfilt = L1-4 # account for 5 rows of headings
	L2 = len(Devices)

	# creates dataframes for devices data
	df = pd.DataFrame(Devices) 
	dfarray = np.array(df)
	LCV = dfarray[4:L2, 31] 
	LCVfilt = sp.signal.resample(LCV, Lfilt) 
	LCV2d = (np.reshape(LCVfilt, (Lfilt, 1))) 
	FPC = dfarray[4:L2, 66:69].astype(float) 
	FPCfilt = sp.signal.resample(FPC, Lfilt) 
	FPC2d = (np.reshape(FPCfilt, (Lfilt, 3)))/1000 

	# creates dataframes for markers data
	df2 = pd.DataFrame(Markers) 
	df2array = np.array(df2) 
	T2 = (df2array[4:L1, 2:5].astype(float))/1000 
	T3 = (df2array[4:L1, 5:8].astype(float))/1000
	M = (df2array[4:L1, 8:11].astype(float))/1000
	B2 = (df2array[4:L1, 11:14].astype(float))/1000
	B3 = (df2array[4:L1, 14:17].astype(float))/1000

	# Matrix of zeros for reference forces, moments and COP
	Fr = np.zeros((Lfilt,3)) 
	COP = np.zeros((Lfilt,4))
	Mr = np.zeros((Lfilt,3))

	# calculates x, y, z vectors
	T = (T2+T3)/2 
	B = (B2+B3)/2 
	Origin = B 
	AuxV = np.subtract(T, Origin)
	AuxLat = np.subtract(B2, Origin)
	x = np.cross(AuxV, AuxLat)
	x_hat = x/(np.linalg.norm(x)) 
	y = np.cross(x_hat, AuxV)
	y_hat = y/(np.linalg.norm(y)) 
	z_hat = np.cross(y_hat, x_hat) 

	# grabs row of data 
	for i in range(len(x_hat)):
		xrow = x_hat[i,:] 
		xrowT = np.transpose(xrow)
		yrow = y_hat[i,:]
		yrowT = np.transpose(yrow)
		zrow = z_hat[i,:]
		zrowT = np.transpose(zrow)
		orow = Origin[i,:]
		orowT = np.transpose(orow) 

		# calculates COP for each row of data
		T = np.column_stack((xrowT, yrowT, zrowT, orowT))
		pad = [0, 0, 0, 1]
		Tpad = np.vstack((T, pad)) 
		d = np.array([0, 0, 0.26, 1]) 
		dt = np.transpose(d) 
		COP[i,:] = np.matmul(Tpad, dt)
	# COP[:,2] = 0
	# print(COP)

	# computes reference forces and moments
	Fp = ((-225.91*LCV2d)+4.5629)
	UV = z_hat*(-1) 
	for i in range(Lfilt):
		Fr[i,:] = Fp[i,:] * UV[i,:] 
		Mr[i,:] = np.cross((COP[i,0:3]-FPC2d[i,:]), Fr[i,:]) 
	R = np.column_stack((Fr, Mr)) 
	RT = np.transpose(R)

	# creates signal matrix
	M = (dfarray[4:L2, 63:66].astype(float))/1000 
	F = dfarray[4:L2, 60:63].astype(float) 
	FPD = np.column_stack((F, M)) 
	Ffilt = sp.signal.resample(FPD, Lfilt) 
	S = np.array(Ffilt) 
	Sinv = np.linalg.pinv(S) # pseudoinverse 6x4595 of signal matrix
	SinvT = np.transpose(Sinv) 

	# calculates 6x6 calibration matrix
	C = np.matmul(RT, SinvT) 
	np.set_printoptions(suppress=True, precision=4)
	print(C)

	label3['text'] = "%s" % C

# prints out the calibration matrix
label3 = Label(window, text=" ", font=("Helvetica", 20))
label3.place(relx = 0.45, rely = 0.3)

# creates buttons for force plates
btn1 = Button(window, text="Force Plate 4", command=FP4)
btn1.place(relx = 0.26, rely = 0.2)
btn2 = Button(window, text="Force Plate 5", command=FP5)
btn2.place(relx = 0.26, rely = 0.4)
btn3 = Button(window, text="Force Plate 6", command=FP6)
btn3.place(relx = 0.26, rely = 0.6)
btn4 = Button(window, text="Force Plate 7", command=FP7)
btn4.place(relx = 0.26, rely = 0.8)

window.mainloop()



