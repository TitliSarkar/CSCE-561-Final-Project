# -*- coding: utf-8 -*-
'''
This code calculates protein-protein similarity among 172 proteins with 108091 keys,in parallel batch mode.
'''
from __future__ import print_function
from __future__ import absolute_import
from pycuda.compiler import SourceModule

import pycuda.driver as drv
import pycuda.autoinit  
import numpy as np
import time
import pandas as pd
import glob, os
import os, os.path
import csv
import operator
import re


mod = SourceModule("""
	__global__ void MinMax(double a1[5689], double a2[5689], double mini_gpu[172], double maxi_gpu[5689], int y)
	{
		int idx = threadIdx.x;
		double min = 0.0, max = 0.0;
		if(idx < 1)
		{
			for(int i=0; i<5689; i++)
			{
				if( a1[i] <= a2[i])
				{
					min = min + a1[i];
					max = max + a2[i];
				}
				else
				{
					min = min + a2[i];
					max = max + a1[i];
				}
			}
			mini_gpu[y] = mini_gpu[y] + min;
			maxi_gpu[y] = maxi_gpu[y] + max;
		}
	}
	__global__ void result(double mini_gpu[172], double maxi_gpu[172], double res_gpu[172][172], int x, int y)
	{
		int idx = threadIdx.x;
		if(idx>=y && idx<172)
		{
			res_gpu[x][idx] = (double)mini_gpu[idx]/(double)maxi_gpu[idx];	
			res_gpu[idx][x] = (double)mini_gpu[idx]/(double)maxi_gpu[idx];
			
		}
	}
	""")
print("*********GPU version of Protein-Protein Similarity Calculation**************\n")
#Getting all Protein files and saving them in a list 'Protein[]'
Protein=[]

#Set the ProteinKet data set folder path
os.chdir("F:\\Studies\\Ph.D\\Ph.D Work\\ProteinDataSet")

for file in glob.glob("*.keys"):
	Protein.append(file)

no_of_proteins = len(Protein)
print("#Proteins  = ", no_of_proteins)

#Sorting all Protein files ------------->
for p in Protein:
	with open(p, "r") as p_file:	#open p position file of Protein[] in Read format
		filename = ""+p_file.name
		#print(" File --->", filename)
		f = open(filename, "r")
		lines = f.readlines()
		#print(lines)
		lines.sort(key=lambda a_line: a_line.split()[0])
		f.close()

#Getting all unique keys in a sorted list 'Keys[]' ------------>
keys = []
for pt in Protein:
	with open(pt, "r") as p_file:
		filename = ""+p_file.name
		#print(filename)
		f = open(filename, "r")
		for line in f:
			cntnt = line.split()	
			res = list(map(int, cntnt))
			item = res[0]
			keys.append(item)	#add only 1st number of each line of the files as keys
		f.close()

keys = np.unique(keys)
no_unq_keys = keys.shape[0]
np.save("keys", keys)	#save the keys into Keys file
print("#Unique Keys = ", no_unq_keys)

#Forming Protein-Key Matrix ------------------------>
print("Forming Protein-Key matrix ...")
start_time = time.clock()
PKmat_gpu = np.zeros(shape=(no_of_proteins,no_unq_keys))
for p in Protein:
	with open(p, "r") as pt_file:
		fname = ""+pt_file.name
		fl = open(fname, "r")
		for line in fl:
			content = line.split()
			results = list(map(int, content))
			r = Protein.index(p)
			var = np.where(keys==results[0])
			c = var[0][0]
			PKmat_gpu[r][c] = results[1]
		f.close()
print("Protein-Key matrix fromulation Done!")
print ("Protein-Key matrix fromulation took :",time.clock() - start_time, "seconds.")
np.savetxt("PKmat_gpu.txt", PKmat_gpu, delimiter=' ')
df = pd.DataFrame(PKmat_gpu, columns=keys)
df.to_csv('pkmat_gpu.csv')
print("*************Row Wise Calculation*************")

r = np.shape(PKmat_gpu)[0]		#No. of Proteins
row = np.int32(r)			
c = np.shape(PKmat_gpu)[1]		#No. of Keys
col = np.int32(c)

#This is the Device Kernel Section that calculate the Similarity value
#This Machine GPU having following details : 
#					Device Name : GeForce GTX 960
#					Total Global Memory : 979 MB
#					maximum Shared memory in a block : 48 KB
#					Maximum Threads per Block : 1024

#This Kernel Section calculate the Similarity matrix in Row wise

mini = np.zeros(row)		#Create Mini Column martix on Host Memory
mini_gpu = drv.mem_alloc(mini.nbytes)	#Allocate Device memory for Mini Column matrix
maxi = np.zeros(row)		#Create Maxi Column matrix on Host Memory
maxi_gpu = drv.mem_alloc(maxi.nbytes)	#Allocate Device memory for Maxi Column matrix
res = np.zeros(shape=(row,row))		#Create Result matrix on Host Memory to store the Similarirty value
res_gpu = drv.mem_alloc(res.nbytes)	#Allocate Device memory for Result matrix

drv.memcpy_htod(res_gpu, res)		#Set Device Result matrix with 0's

#Create two Temp row matrix for GPU calculation and allocate there corrosponding memory into Device
a1 = np.zeros(5689)			
a2 = np.zeros(5689)
a1_gpu = drv.mem_alloc(a1.nbytes)
a2_gpu = drv.mem_alloc(a2.nbytes)


print("Entering in the similarity calculation kernel section ...")
start_time = time.clock()		#Start the timer
i = 0
val = np.int32(i)
for k in range(0,row):		#For each row
	x = np.int32(k)
	drv.memcpy_htod(mini_gpu, mini)	#For each row Set device (mini_gpu) matrix with 0's 
	drv.memcpy_htod(maxi_gpu, maxi)	#For each row Set device (maxi_gpu) matrix with 0's

	for l in range(x,row):		#For calculate only Upper Triangular Matrix
		y = np.int32(l)
		pos = 0					#For calculate the Column number
		for i in range(0,19):	#Break rows into 19 Parts for kernel call
			for j in range(0,5689):	#For storing each part with 5689 columns into a1 and a2 Temp array
				if(pos<108091):
					a1[j] = PKmat_gpu[x][pos]
					a2[j] = PKmat_gpu[l][pos]
				else:
					a1[j] = 0
					a2[j] = 0
				pos = pos + 1
		
			drv.memcpy_htod(a1_gpu, a1)		#Copying data of (a1) temp host matrix to (a1_gpu) device memory
			drv.memcpy_htod(a2_gpu, a2)		#Copying data of (a2) temp host matrix to (a2_gpu) device memory

			MinMax = mod.get_function("MinMax")		#Create MinMax function in host that call the MinMax Kernel on GPU or Device
			MinMax(a1_gpu, a2_gpu, mini_gpu, maxi_gpu, y, block = (1,1,1), grid = (1,1))	#Send MinMax Kernel Arguments and Kernel Structure(as block, grid)
		#print(y)

		result = mod.get_function("result")			#Create result function in host that call the result kernel on GPU or Device
		result(mini_gpu, maxi_gpu, res_gpu, x, y, block = (172,1,1), grid = (1,1))			#Send result Kernel Arguments and Kernel Structure(as block, grid)
	#print(x)
print ("Similarity calculation on GPU Done!")
print ("Similarity calculation by GPU took :",time.clock() - start_time, "seconds.")		#Stop timer and Calculate total time

drv.memcpy_dtoh(res, res_gpu)				#Copy data of (res_gpu) device memory to (res) host memory  
np.savetxt("similarity_gpu-1.txt", res, delimiter=' ')		#Save results in similarity_gpu.txt file 

np.set_printoptions(precision=3)
print(res)					#Print Result Matrix

df1 = pd.DataFrame(res)		#Save Result in .csv file
df1.to_csv('similarity_gpu-1.csv')	
