# -*- coding: utf-8 -*-
'''
This code calculates protein-protein similarity among 172 proteins with 108091 keys,in batch mode.

Formula used for similarity calculation: Jaccard's Similarity Coefficient.

The input is available as a folder/directory of '.keys' files each of which contains a set of key-value pairs.

First, a Protein-Key matrix is formed from these files with Protein files as rows and list of unique keys as columns.
Then similarity among all Protein(i,j) pairs are calculated using Jaccard's similarity coefficient and the outputs are 
stored in a  Protein-Protein similarity matrix.
'''
import sys
sys.path.append('c:\\program files\\anaconda3\\lib\\site-packages')

import glob, os
import os, os.path
import csv
import operator
import numpy as np
import pandas as pd
import re
import time

print("\n*********CPU version of Protein-Protein Similarity Calculation**************\n")
#Getting all Protein files and saving them in a list 'Protein[]'
Protein=[]

#Set the ProteinKet data set folder path
os.chdir("F:\\Studies\\Ph.D\\Ph.D Work\\ProteinDataSet\\")

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
#print("FileSorting Done!")
#print(Protein)

#Getting all unique keys in a sorted list 'keys[]' ------------>
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
#print(Keys)
#print("Getting Unique keys Done!")

#Forming Protein-Key Matrix ------------------------>
print("Forming Protein-Key matrix ...")
start_time = time.clock()
PKmat = np.zeros(shape=(no_of_proteins,no_unq_keys))
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
			PKmat[r][c] = results[1]
		f.close()
np.savetxt("PKmat.txt", PKmat, delimiter=' ')
df = pd.DataFrame(PKmat, columns=keys)
df.to_csv('pkmat.csv')
print("Protein-Key matrix fromulation Done!")
print ("Protein-Key matrix fromulation took :",time.clock() - start_time, "seconds.")


r = np.shape(PKmat)[0]		#No. of Proteins
row = np.int32(r)			
c = np.shape(PKmat)[1]		#No. of Keys
col = np.int32(c)

similariy = np.zeros(shape=(no_of_proteins,no_of_proteins), dtype=np.float64)

print("Entering in the similarity calculation section ...")
start_time = time.clock()
for i in range(0,row):
	x = np.int32(i)
	for j in range(x,row):
		minsum = 0.0
		maxsum = 0.0
		for k in range(0,col):
			if(PKmat[i][k] <= PKmat[j][k]):
				minsum = minsum + PKmat[i][k]
				maxsum = maxsum + PKmat[j][k]
			else:
				minsum = minsum + PKmat[j][k]
				maxsum = maxsum + PKmat[i][k]
		res = minsum/maxsum
		similariy[i][j] =  res
		similariy[j][i] =  res
	#print(x)
print ("Similarity calculation on CPU Done!")
print ("Similarity calculation by CPU took :",time.clock() - start_time, "seconds.")

np.savetxt("similarity_cpu.txt", similariy, delimiter=' ')

np.set_printoptions(precision=3)	#for print Similarity upto 3 decimal
print (similariy)

df2 = pd.DataFrame(similariy)		#Save Result in .csv file
df2.to_csv('similarity_cpu.csv')	