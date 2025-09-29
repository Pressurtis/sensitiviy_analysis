#Sampling and GSA
from SALib.sample import saltelli
from SALib.analyze import sobol
import sobol_seq
import ghalton
from pyDOE import *
#Plot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#Deal with array and matrix
import numpy as np 
#Read csv
import csv
import pandas as pd
import xlsxwriter
#remove file
import os

#Defination
#Simulation
num = 3		#Number of variables. e.g. 3
num_samples_simulation = 100	 #Number of samples for simulation e.g. 100
method = 'halton'		#Method for input simulation sampling

skip = '' 		#For sobol method, default='', e.g. 3
seed = ''		#For halton method, default='', e.g. 3
latin_criterion = 'default'		#For latin_hypercube method, default='default'. Alternatives: 'center','maximin','centermaximin','correlation'

#Define variables
variablenames = ['rotor_core_stiffness','bearing_stiffness','foundation_stiffness']		
bounds = [[1e8,3e8],[4e8,8e8],[0.5e9,1.5e9]] 	 

#File name for fit surface result
Filename_simulation_sampling = 'input.csv'

#Names of output
names = ['P5','P6','P7','P8']

k_fold = 10 	#For k-fold cross-validation
#File name for fit surface result
Filename_fit_surface = 'output_GSA.csv'

#saltelli
saltelli_second_order = False 
#Number of saltelli samples for GSA e.g. 500. When second_order is false,num_samples_saltelli/(num+2) should be integer, else num_samples_saltelli/(2*num+2) should be integer
num_samples_saltelli = 500		
#Do you save saltelli samples in separte speadsheet
Save_saltelli_samples= True

File_result = 'result.xlsx'

print('You have finnished the basic definition \nYou still need to continue with defining surrogate model and cross validation model later')

def write_sampling_result(*args,Filename_simulation_sampling=Filename_simulation_sampling):	#args=method,num,num_samples_simulation+ alternative argument(skip,seed etc.)
	with open(Filename_simulation_sampling,'w',newline='') as f:
		writer = csv.writer(f)
		writer.writerow(*args)
		writer.writerow(sample_values.keys())
		writer.writerows(zip(*sample_values.values()))

sample_values={}
#sampling
if method == 'sobol':
#Sobol
	if skip != '' :
		skip = int(skip)
		vec = sobol_seq.i4_sobol_generate(num,num_samples_simulation,skip)
		for i in range(num):
			sample_values[variablenames[i]]=list(vec[:,i])
	else:
		skip = 0
		vec = sobol_seq.i4_sobol_generate(num,num_samples_simulation)
		for i in range(num):
			sample_values[variablenames[i]]=list(vec[:,i])
	write_sampling_result(['method',method,'num variables',num,'num samples',num_samples_simulation,'skip property',skip])

#Halton
elif method == 'halton':
	if seed != '' :	
		seed = int(seed)
		sequencer = ghalton.GeneralizedHalton(num,seed)
		points = sequencer.get(num_samples_simulation)
		for i in range(num):
			sample_values[variablenames[i]]=np.array(points)[:,i]
	else:
		sequencer = ghalton.GeneralizedHalton(num)
		points = sequencer.get(num_samples_simulation)
		for i in range(num):
			sample_values[variablenames[i]]=np.array(points)[:,i]
	write_sampling_result(['method',method,'num variables',num,'num samples',num_samples_simulation,'seed',seed])

#Latin hypercube
elif method == 'latin':
	if latin_criterion == 'default':
		latin = lhs(num, samples=num_samples_simulation)
		for i in range(num):
			sample_values[variablenames[i]]=list(latin[:,i])
	else:
		latin = lhs(num, samples=num_samples_simulation,criterion=latin_criterion)
		for i in range(num):
			sample_values[variablenames[i]]=list(latin[:,i])
	write_sampling_result(['method',method,'num variables',num,'num samples',num_samples_simulation,'criterion',latin_criterion])

else:
	print('Please type the correct method name !(sobol/halton/latin)')		

print('You have written simulation samples to '+str(Filename_simulation_sampling))


#Formula for run simulation
sample_values['P5'] = sample_values[variablenames[0]] +sample_values[variablenames[1]] +sample_values[variablenames[2]] #simulation function
sample_values['P6'] = sample_values[variablenames[0]]**2 +sample_values[variablenames[1]] +sample_values[variablenames[2]]
sample_values['P7'] = sample_values[variablenames[0]]**3 +sample_values[variablenames[1]] +sample_values[variablenames[2]]
sample_values['P8'] = sample_values[variablenames[0]]**3 +sample_values[variablenames[1]] +sample_values[variablenames[2]]


if method == 'sobol':
	write_sampling_result(['method',method,'num variables',num,'num samples',num_samples_simulation,'skip property',skip])
elif method == 'halton':
	write_sampling_result(['method',method,'num variables',num,'num samples',num_samples_simulation,'seed',seed])
elif method == 'latin':
	write_sampling_result(['method',method,'num variables',num,'num samples',num_samples_simulation,'criterion',latin_criterion])
        
print('You have written simulation samples and run simulation reslut to '+str(Filename_simulation_sampling))


#Generate Saltelli samples
problem={
'num_vars':num,'names':variablenames,'bounds':bounds
} 
if saltelli_second_order == False:
	param_values_GSA = saltelli.sample(problem,int(int(num_samples_saltelli)/(num+2)),calc_second_order=False)  
elif saltelli_second_order == True:
	param_values_GSA = saltelli.sample(problem,int(int(num_samples_saltelli)/(2*num+2)),calc_second_order=True) 
samples_GSA = {}
input_interpolation_data = pd.read_csv(Filename_simulation_sampling,header=1)
param_values=[[]for i in range(len(variablenames))]
input_interpolation_variables = [[]for i in range(len(variablenames)+1)]
for i in range(len(variablenames)):
	param_values[i]=param_values_GSA[:,i]
	input_interpolation_variables[i] = input_interpolation_data[variablenames[i]].iloc[:num_samples_simulation]

if Save_saltelli_samples == True:
	saltelli_samples={}
	for i in range(num):
		saltelli_samples[variablenames[i]]=list(param_values_GSA[:,i])
# get result seperately in csv file
	with open('Saltelli_samples.csv','w',newline='') as f:
		writer = csv.writer(f)
		writer.writerow(saltelli_samples.keys())
		writer.writerows(zip(*saltelli_samples.values()))
    print('You have generated GSA samples and save it to Saltelli_samples.csv')
else:
    print('You have generated GSA samples')












