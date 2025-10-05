"""
SALib sampling
"""
#Sampling and GSA
from SALib.sample import saltelli
from SALib.analyze import sobol
from scipy.interpolate import Rbf
#Plot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#Deal with array and matrix
import numpy as np 
#Read csv
import csv
from itertools import islice


'''
Task2:(1)Have choices to choose what kind of sampling function 
	  (2)Generate sampling module for each of methods
Task3: Write final result( both graphs and values) in xls file
'''

'''
Variables:
num_samples_simulation --- numbers of simulation samples
num_samples_saltelli --- number of saltelli samples
num --- number of variables
variablenames --- put variables names in a list[]
bounds --- put bounds of different variables in a list of list [[]]
method --- method for simulation sampling
'''
'''
GUI for input information
In this script, the GUI for user to type the input information is designed
'''


#Definition script


#Define numbers of variables
num = int(input('Enter number of variables:'))
#define number of samples
num_samples_simulation = int(int(input('Enter numbers of simulation samples: '))/(num+2))

#Input variablenames
variablenames = []
for i in range(0,num):
	variablenames.append(input('Variable %s name is: ' %(i+1)))
bounds=[[] for i in range(num)]
#Input bounds
for j in range(0,num):
	bounds[j].append(int(input('Variable %s left bound is:' %variablenames[j])))
	bounds[j].append(int(input('Variable %s right bound is:' %variablenames[j])))
#define problem
problem={
'num_vars':num,'names':variablenames,'bounds':bounds
}     

#Generate samples
# When calc_second_order=False,number of samples= N *( D+2).Otherwise,N *( 2D+2)
param_values_simulation = saltelli.sample(problem,num_sampless_simulation,calc_second_order=False)  

sample_values={}
for i in range(num):
	sample_values[variablenames[i]]=list(param_values_simulation[:,i])
# get result seperately in csv file
with open('input_result.csv','w',newline='') as f:
	writer = csv.writer(f)
	writer.writerow(sample_values.keys())
	writer.writerows(zip(*sample_values.values()))


#Rbf

#Setup data(get samples)
p5_raw=[]
with open('output_values_500.csv', 'r', newline='') as f:
    reader = csv.reader(f,delimiter=' ')
    next(reader)
    next(reader)
    next(reader)
    next(reader)
    a=[]
    b=[]
    for row in islice(f,100):
        a=''.join(row)
        b=a.split(',')
        p5_raw.append(float(b[5]))

x_simulation = param_values_simulation[:,0]
y_simulation = param_values_simulation[:,1]
z_simulation = param_values_simulation[:,2]

#Get Rbf function
p5_rbf = Rbf(x_simulation,y_simulation,z_simulation,p5_raw)#,function='gaussian'

#Generate new samples
param_values_ROM=saltelli.sample(problem,num_samples_saltelli,calc_second_order=False)

x_ROM = param_values_ROM[:,0]
y_ROM = param_values_ROM[:,1]
z_ROM = param_values_ROM[:,2]

p5_ROM= p5_rbf(x_ROM,y_ROM,z_ROM)
print(p5_500)


#Sensitivity analysis

p5=[]
p6=[]
p7=[]
p8=[]
p9=[]
#Read output values from the file
with open('output_values_500.csv', 'r', newline='') as f:
    reader = csv.reader(f,delimiter=' ')
    next(reader)
    next(reader)
    next(reader)
    next(reader)
    a=[]
    b=[]
    for row in f:
    	a=''.join(row)
    	b=a.split(',')
    	p5.append(float(b[5]))
    	p6.append(float(b[6]))
    	p7.append(float(b[7]))
    	p8.append(float(b[8]))
    	p9.append(float(b[9]))

#get the input of sensitivity analysis, which should be the array

Y_p5=np.array(p5).reshape((500)).astype(np.float)
Y_p6=np.array(p6).reshape((500)).astype(np.float)
Y_p7=np.array(p7).reshape((500)).astype(np.float)
Y_p8=np.array(p8).reshape((500)).astype(np.float)
Y_p9=np.array(p9).reshape((500)).astype(np.float)

#sensitivity analysis

Si_p5= sobol.analyze(problem,Y_p5,calc_second_order=False)	#confidence interval level(default 0.95)
Si_p6= sobol.analyze(problem,Y_p6,calc_second_order=False)
Si_p7= sobol.analyze(problem,Y_p7,calc_second_order=False)
Si_p8= sobol.analyze(problem,Y_p8,calc_second_order=False)
Si_p9= sobol.analyze(problem,Y_p9,calc_second_order=False)

#Write sensitivity analysis result in csv files

#P5
with open('output_results_P5.csv', 'w', newline='') as f:
	Sensitivity=['names','S1','conf_S1','S1_conf','ST','conf_ST','ST_conf'] #Create header names
	writer = csv.DictWriter(f,fieldnames=Sensitivity)
	writer.writeheader()
	writer.writerow({'names':'P5'})
	for row in range(0,3):
		S1=Si_p5['S1'][row]
		conf_S1=Si_p5['S1_conf'][row]
		S1_conf=[Si_p5['S1'][row]-Si_p5['S1_conf'][row],Si_p5['S1'][row]+Si_p5['S1_conf'][row]]
		ST=Si_p5['ST'][row]
		conf_ST=Si_p5['ST_conf'][row]
		ST_conf=[Si_p5['ST'][row]-Si_p5['ST_conf'][row],Si_p5['ST'][row]+Si_p5['ST_conf'][row]]
		writer.writerow({'names':variablenames[row],'S1':S1,'conf_S1':conf_S1,'S1_conf':S1_conf,'ST':ST,'conf_ST':conf_ST,'ST_conf':ST_conf})

numbers_variables=np.arange(len(variablenames))
width=0.2

conf_S1_p5=[]
for i in range(0,3):
	conf_S1_p5.append(Si_p5['S1_conf'][i])
conf_ST_p5=[]
for i in range(0,3):
	conf_ST_p5.append(Si_p5['ST_conf'][i])
#generate graph for P5
sensitivity_S1_p5=[Si_p5['S1'][0],Si_p5['S1'][1],Si_p5['S1'][2]]
sensitivity_ST_p5=[Si_p5['ST'][0],Si_p5['ST'][1],Si_p5['ST'][2]]

plt.figure('P5')
plt.bar(numbers_variables,sensitivity_S1_p5,width,yerr=conf_S1_p5,capsize=7,align='center',alpha=0.5,color='red')
plt.bar(numbers_variables+0.3,sensitivity_ST_p5,width,yerr=conf_ST_p5,capsize=7,align='center',alpha=0.5,color='blue')
plt.xticks(numbers_variables,variablenames)
plt.xlabel('Variables')
plt.ylabel('Sensitivity_values')
plt.title('Sensitivity_analysis_P5')
S1_patch = mpatches.Patch(color='red',label='S1') #create legend
ST_patch = mpatches.Patch(color='blue',label='ST')
plt.legend(handles=[S1_patch,ST_patch],loc=1,fontsize = 'x-small')
plt.tight_layout()
plt.savefig('P5_Graph')
plt.show()

#P6
with open('output_results_P6.csv', 'w', newline='') as f:
	Sensitivity=['names','S1','conf_S1','S1_conf','ST','conf_ST','ST_conf']
	writer = csv.DictWriter(f,fieldnames=Sensitivity)
	writer.writeheader()
	writer.writerow({'names':'P6'})
	for row in range(0,3):
		S1=Si_p6['S1'][row]
		conf_S1=Si_p6['S1_conf'][row]
		S1_conf=[Si_p6['S1'][row]-Si_p6['S1_conf'][row],Si_p6['S1'][row]+Si_p6['S1_conf'][row]]
		ST=Si_p6['ST'][row]
		conf_ST=Si_p6['ST_conf'][row]
		ST_conf=[Si_p6['ST'][row]-Si_p6['ST_conf'][row],Si_p6['ST'][row]+Si_p6['ST_conf'][row]]
		writer.writerow({'names':variablenames[row],'S1':S1,'conf_S1':conf_S1,'S1_conf':S1_conf,'ST':ST,'conf_ST':conf_ST,'ST_conf':ST_conf})

conf_S1_p6=[]
for i in range(0,3):
	conf_S1_p6.append(Si_p6['S1_conf'][i])
conf_ST_p6=[]
for i in range(0,3):
	conf_ST_p6.append(Si_p6['ST_conf'][i])
#generate graph for P6
sensitivity_S1_p6=[Si_p6['S1'][0],Si_p6['S1'][1],Si_p6['S1'][2]]
sensitivity_ST_p6=[Si_p6['ST'][0],Si_p6['ST'][1],Si_p6['ST'][2]]

plt.figure('P6')
plt.bar(numbers_variables,sensitivity_S1_p6,width,yerr=conf_S1_p6,capsize=7,align='center',alpha=0.5,color='r')
plt.bar(numbers_variables+0.3,sensitivity_ST_p6,width,yerr=conf_ST_p6,capsize=7,align='center',alpha=0.5,color='b')
plt.xticks(numbers_variables,variablenames)
plt.xlabel('Variables')
plt.ylabel('Sensitivity_values')
plt.title('Sensitivity_analysis_P6')
plt.savefig('P6_Graph')



#P7
with open('output_results_P7.csv', 'w', newline='') as f:
	Sensitivity=['names','S1','conf_S1','S1_conf','ST','conf_ST','ST_conf']
	writer = csv.DictWriter(f,fieldnames=Sensitivity)
	writer.writeheader()
	writer.writerow({'names':'P7'})
	for row in range(0,3):
		S1=Si_p7['S1'][row]
		conf_S1=Si_p7['S1_conf'][row]
		S1_conf=[Si_p7['S1'][row]-Si_p7['S1_conf'][row],Si_p7['S1'][row]+Si_p7['S1_conf'][row]]
		ST=Si_p7['ST'][row]
		conf_ST=Si_p7['ST_conf'][row]
		ST_conf=[Si_p7['ST'][row]-Si_p7['ST_conf'][row],Si_p7['ST'][row]+Si_p7['ST_conf'][row]]
		writer.writerow({'names':variablenames[row],'S1':S1,'conf_S1':conf_S1,'S1_conf':S1_conf,'ST':ST,'conf_ST':conf_ST,'ST_conf':ST_conf})

#P8
with open('output_results_P8.csv', 'w', newline='') as f:
	Sensitivity=['names','S1','conf_S1','S1_conf','ST','conf_ST','ST_conf']
	writer = csv.DictWriter(f,fieldnames=Sensitivity)
	writer.writeheader()
	writer.writerow({'names':'P8'})
	for row in range(0,3):
		S1=Si_p8['S1'][row]
		conf_S1=Si_p8['S1_conf'][row]
		S1_conf=[Si_p8['S1'][row]-Si_p8['S1_conf'][row],Si_p8['S1'][row]+Si_p8['S1_conf'][row]]
		ST=Si_p8['ST'][row]
		conf_ST=Si_p8['ST_conf'][row]
		ST_conf=[Si_p8['ST'][row]-Si_p8['ST_conf'][row],Si_p8['ST'][row]+Si_p8['ST_conf'][row]]
		writer.writerow({'names':variablenames[row],'S1':S1,'conf_S1':conf_S1,'S1_conf':S1_conf,'ST':ST,'conf_ST':conf_ST,'ST_conf':ST_conf})

#P9
with open('output_results_P9.csv', 'w', newline='') as f:
	Sensitivity=['names','S1','conf_S1','S1_conf','ST','conf_ST','ST_conf']
	writer = csv.DictWriter(f,fieldnames=Sensitivity)
	writer.writeheader()
	writer.writerow({'names':'P9'})
	for row in range(0,3):
		S1=Si_p9['S1'][row]
		conf_S1=Si_p9['S1_conf'][row]
		S1_conf=[Si_p9['S1'][row]-Si_p9['S1_conf'][row],Si_p9['S1'][row]+Si_p9['S1_conf'][row]]
		ST=Si_p9['ST'][row]
		conf_ST=Si_p9['ST_conf'][row]
		ST_conf=[Si_p9['ST'][row]-Si_p9['ST_conf'][row],Si_p9['ST'][row]+Si_p9['ST_conf'][row]]
		writer.writerow({'names':variablenames[row],'S1':S1,'conf_S1':conf_S1,'S1_conf':S1_conf,'ST':ST,'conf_ST':conf_ST,'ST_conf':ST_conf})







