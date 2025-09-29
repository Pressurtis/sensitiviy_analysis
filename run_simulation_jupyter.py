import pandas as pd 
import numpy as np
import csv
#input
variablenames = ['rotor_core_stiffness','bearing_stiffness','foundation_stiffness']
num = 3
names = ['P5','P6','P7']
num_samples_simulation = 100

#function
def simulation_function(num,sampling_x):
	simulation_result = 0
	for i in range(num):
		simulation_result = simulation_result + sampling_x[i]
	return simulation_result

#Run_simulation

#read data
sampling_data = pd.read_csv('input.csv',header=1)
#Get variable values seperately
print(sampling_data)
sampling_x = [[]for i in range(num)]
for i in range(num):
	sampling_x[i] = sampling_data[variablenames[i]].iloc[:num_samples_simulation]
print(type(sampling_x))

#Calculate by using funtion

sampling_output = {}
for i in range(len(names)):
	sampling_output[names[i]] = simulation_function(num,sampling_x)


with open('output_test.csv','w',newline='') as f:
	writer = csv.writer(f)
	writer.writerow(['# Design Points of Design of Experiments'])
	writer.writerow(['custom'])
	writer.writerow(x for x in names)
	writer.writerow(sampling_output.keys())
	writer.writerows(zip(*sampling_output.values()))


