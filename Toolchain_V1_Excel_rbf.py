'''
Task1: combine csv and pandas
Task2: find a way to warn there are wrong values
'''
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
from itertools import islice
import pandas as pd
import xlsxwriter
#rbf
from scipy.interpolate import Rbf
#remove file
import os

# #Read input information
# with open('input information', 'r', newline='') as f:
#     reader = csv.reader(f,delimiter=' ')
#     next(reader)

input_information = pd.read_csv('input_information.csv')
num = int(input_information.iloc[0]['Num_variables'])
variablenames = []
bounds = [[]for i in range(num)]
for i in range(0,num):
	variablenames.append(input_information.iloc[i]['Variables_names'])
	bounds[i].append(int(input_information.iloc[i]["Left_bound"]))
	bounds[i].append(int(input_information.iloc[i]["Right_bound"]))
method = input_information.iloc[0]['Simulation_method']
num_samples_simulation = int(input_information.iloc[0]['Num_simulation_samples'])
num_samples_saltelli = int(input_information.iloc[0]['Num_GSA_samples'])
#sampling
sample_values={}

if method == 'sobol':
#Sobol
	vec = sobol_seq.i4_sobol_generate(num,num_samples_simulation)
	for i in range(num):
		sample_values[variablenames[i]]=list(vec[:,i])
#Halton
elif method == 'halton':		
	seed = int(input_information.iloc[0]['Seed_halton'])
	sequencer = ghalton.GeneralizedHalton(num,seed)
	points = sequencer.get(num_samples_simulation)
	for i in range(num):
		sample_values[variablenames[i]]=np.array(points)[:,i]
#Latin hypercube
elif method == 'latin':
	latin_criterion = input_information.iloc[0]['Criterion_latin']
	if latin_criterion == 'none':
		latin = lhs(num, samples=num_samples_simulation)
	else:
		latin = lhs(num, samples=num_samples_simulation,criterion = latin_criterion)
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
	if input_information.iloc[0]['Second_order_Saltelli'] == 'yes' :
		param_values_simulation = saltelli.sample(problem,int(num_samples_simulation/(2*num+2)),calc_second_order=False)  
	else :
		param_values_simulation = saltelli.sample(problem,int(num_samples_simulation/(num+2)),calc_second_order=False)  
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

#rbf
p5_100=[]
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
        p5_100.append(float(b[5]))


x_100 = param_values_simulation[:,0]
y_100 = param_values_simulation[:,1]
z_100 = param_values_simulation[:,2]

#Get Rbf function
p5_rbf = Rbf(x_100,y_100,z_100,p5_100)#,function='gaussian'

#Generate new samples
param_values_500=saltelli.sample(problem,int(num_samples_saltelli/(num+2)),calc_second_order=False)

x_500 = param_values_500[:,0]
y_500 = param_values_500[:,1]
z_500 = param_values_500[:,2]

p5_500= p5_rbf(x_500,y_500,z_500)

#sensitivity analysis
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

Y_p5=np.array(p5_500).reshape((500)).astype(np.float)
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
plt.close()
#xlsx
workbook = xlsxwriter.Workbook('result.xlsx')
worksheet_P5 = workbook.add_worksheet('P5')
worksheet_P6 = workbook.add_worksheet('P6')
worksheet_P7 = workbook.add_worksheet('P7')
worksheet_P8 = workbook.add_worksheet('P8')
worksheet_P9 = workbook.add_worksheet('P9')

#Write P5 value and graph
worksheet_P5.write('A1','P5')
worksheet_P5.write('A2','Variables')
worksheet_P5.write('B2','S1')
worksheet_P5.write('C2','Conf_S1')
worksheet_P5.write('D2','S1_Conf')
worksheet_P5.write('E2','ST')
worksheet_P5.write('F2','Conf_ST')
worksheet_P5.write('G2','ST_Conf')
worksheet_P5.set_column(0,0,25)
worksheet_P5.set_column(3,3,15)
worksheet_P5.set_column(6,6,15)
for row in range(0,3) :
	worksheet_P5.write('A'+str(row+3),variablenames[row])
	worksheet_P5.write('B'+str(row+3),Si_p5['S1'][row])
	worksheet_P5.write('C'+str(row+3),Si_p5['S1_conf'][row])
	worksheet_P5.write_string('D'+str(row+3),str([round(Si_p5['S1'][row]-Si_p5['S1_conf'][row],3),round(Si_p5['S1'][row]+Si_p5['S1_conf'][row],3)]))
	worksheet_P5.write('E'+str(row+3),Si_p5['ST'][row])
	worksheet_P5.write('F'+str(row+3),Si_p5['ST_conf'][row])
	worksheet_P5.write_string('G'+str(row+3),str([round(Si_p5['ST'][row]-Si_p5['ST_conf'][row],3),round(Si_p5['ST'][row]+Si_p5['ST_conf'][row],3)]))

worksheet_P5.insert_image('B8','P5_Graph.png',{'x_scale':0.7,'y_scale':0.7})
workbook.close()
os.remove('P5_Graph.png')






