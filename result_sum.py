import sys
sys.path.append('c:\\program files\\anaconda3\\lib\\site-packages')

import glob, os
import os, os.path
import csv
import operator 
import numpy as np
import pandas as pd

print("***********Showing CPU and GPU results are same****************")
os.chdir("F:\\Studies\\Ph.D\\Ph.D Work\\ProteinDataSet\\")
#For Module 1------->
cpu_result_sum = np.loadtxt("similarity_cpu.txt")
print("Similarity CPU Result read Done!")
p = np.shape(cpu_result_sum)[0]
q = np.shape(cpu_result_sum)[1]
cpu_row = np.int32(p)
cpu_col = np.int32(q)

np.set_printoptions(precision=3)
print(cpu_result_sum)

cpu_sum = 0.0
gpu_sum1 = 0.0
gpu_sum2 = 0.0

for i in range(0,cpu_row):
	for j in range(0,cpu_col):
		cpu_sum = cpu_sum + cpu_result_sum[i][j]

print("")
print("CPU Result Sum = ",cpu_sum)

#For Module 2 ------------>
#For Row wise calculation
gpu_result_sum1 = np.loadtxt("similarity_gpu-1.txt")
print("\nSimilarity Row wise Result read Done!")
r = np.shape(gpu_result_sum1)[0]
s = np.shape(gpu_result_sum1)[1]
gpu_row = np.int32(r)
gpu_col = np.int32(s)

np.set_printoptions(precision=3)
print(gpu_result_sum1)


for i in range(0, gpu_row):
	for j in range(0, gpu_col):
		gpu_sum1 = gpu_sum1 + gpu_result_sum1[i][j]

print("\nGPU Result Row wise Sum = ",gpu_sum1)

