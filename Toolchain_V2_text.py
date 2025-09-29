"""
SALib sampling
"""
#Sampling and GSA
from SALib.sample import saltelli
from SALib.analyze import sobol
import sobol_seq
import ghalton
from pyDOE import *
#Rbf
from scipy.interpolate import Rbf
#Plot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#Deal with array and matrix
import numpy as np 
#Read csv
import csv

text_file = open('text.txt','r')
input_information = text_file.read()
text_file.close()
exec(input_information)

#sampling
sample_values={}

if method == 'sobol':
#Sobol
	vec = sobol_seq.i4_sobol_generate(num,num_samples_simulation)
	for i in range(num):
		sample_values[variablenames[i]]=list(vec[:,i])
#Halton
elif method == 'halton':		
	sequencer = ghalton.GeneralizedHalton(num)
	points = sequencer.get(num_samples_simulation)
	for i in range(num):
		sample_values[variablenames[i]]=np.array(points)[:,i]
#Latin hypercube
elif method == 'latin':
	latin = lhs(num, samples=num_samples_simulation)
	for i in range(num):
		sample_values[variablenames[i]]=list(latin[:,i])
#Saltelli
elif method == 'saltelli':
#Define problem
	problem={
			'num_vars':num,'names':variablenames,'bounds':bounds
			} 
#Generate samples
# When calc_second_order=False,number of samples= N *( D+2).Otherwise,N *( 2D+2)
	param_values_simulation = saltelli.sample(problem,int(int(num_samples_simulation)/(num+2)),calc_second_order=False)  
	for i in range(num):
		sample_values[variablenames[i]]=list(param_values_simulation[:,i])
else:
	print('Please type the correct method name !(sobol/halton/latin)')
#get result seperately in csv file
with open('input_result.csv','w',newline='') as f:
	writer = csv.writer(f)
	writer.writerow(['method',method,'num variables',num,'num samples',num_samples_simulation])
	writer.writerow(sample_values.keys())
	writer.writerows(zip(*sample_values.values()))


















