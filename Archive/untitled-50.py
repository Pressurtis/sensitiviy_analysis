from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np 
import csv

variablenames = ['rotor_core_stiffness','bearing_stiffness','foundation_stiffness']
bounds=[[] for i in range(len(variablenames))]
#input bounds
for row in range(0,3):
	bounds[row].append(input('Variable %s left bound is:' %variablenames[row]))
	bounds[row].append(input('Variable %s right bound is:' %variablenames[row]))

problem={
'num_vars':3,'names':['rotor_core_stiffness','bearing_stiffness','foundation_stiffness'],'bounds':[[1e8,3e8],[4e8,8e8],[0.5e9,1.5e9]]
}     #define variables
param_values=saltelli.sample(problem,1000,calc_second_order=False)  # When calc_second_order=True,number of samples= N *( 2*D+2).Otherwise,N *( D+2)
# get sampling data,56 should be used there
with open('sample_5000.csv', 'w', newline='') as f:
    writer = csv.writer(f,delimiter=' ')
    for row in param_values:
        writer.writerow(param_values)
rotor_core_stiffness = param_values[:,0]	#get values of every variable
bearing_stiffness = param_values[:,1]
foundation_stiffness = param_values[:,2]
input_result =[rotor_core_stiffness,bearing_stiffness,foundation_stiffness]
# get result seperately in csv file
with open('input_result_5000.csv','w',newline='') as f:
	variablenames = ['rotor_core_stiffness','bearing_stiffness','foundation_stiffness']
	writer = csv.DictWriter(f,fieldnames=variablenames)
	writer.writeheader()
	for row in range(len(rotor_core_stiffness)):
		writer.writerow({'rotor_core_stiffness':rotor_core_stiffness[row],'bearing_stiffness':bearing_stiffness[row],'foundation_stiffness':foundation_stiffness[row]})
