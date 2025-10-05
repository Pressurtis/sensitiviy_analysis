"""
SALib sampling
"""
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np 
import csv

variablenames = ['c1','a1','l1']
bounds=[[] for i in range(len(variablenames))]
#input bounds
for row in range(0,3):
	bounds[row].append(input('Variable %s left bound is:' %variablenames[row]))
	bounds[row].append(input('Variable %s right bound is:' %variablenames[row]))

problem={
'num_vars':3,'names':['c1','a1','l1'],'bounds':[[-1,1],[-1,1],[-1,1]]
}     #define variables
param_values=saltelli.sample(problem,1,calc_second_order=False)  # When calc_second_order=False,number of samples= N *( D+2).Otherwise,N *( 2D+2)
# get sampling data,56 should be used there
with open('sample.csv', 'w', newline='') as f:
    writer = csv.writer(f,delimiter=' ')
    for row in param_values:
        writer.writerow(param_values)
c1 = param_values[:,0]	#get values of every variable
a1 = param_values[:,1]
l1 = param_values[:,2]
input_result =[c1,a1,l1]
# get result seperately in csv file
with open('input_result.csv','w',newline='') as f:
	variablenames = ['c1','a1','l1']
	writer = csv.DictWriter(f,fieldnames=variablenames)
	writer.writeheader()
	for row in range(len(c1)):
		writer.writerow({'c1':c1[row],'a1':a1[row],'l1':l1[row]})
data_result=-1*c1+a1+l1
#definition of the formula
def Toster(X):
	return(-1*c1+a1+l1)
#method2:for i in range(len(c1)):
		#data_result=map(lambda c1,a1,l1,sh,H,l2,a2,c2:-1*c1+a1+l1+sh+H+l2+a2-c2,c1,a1,l1,sh,H,l2,a2,c2)
#Using formula to calculate the result
with open('output.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for row in list(data_result):
        writer.writerow(data_result)
#sensitivity analysis

Si= sobol.analyze(problem,data_result)	#confidence interval level(default 0.95)
print(Si['S1_conf'])
print(Si['ST_conf'])
print('c1-c2:',Si['S2'][0,2])
conf_interval=[]
for x in range(0,3):
	conf_interval.append(Si['S1_conf'][x])
print(conf_interval)
#Generate graphs
import matplotlib.pyplot as plt
group_variables=('c1','a1','l1')
numbers_variables=np.arange(len(group_variables))
sensitivity_S1=[Si['S1'][0],Si['S1'][1],Si['S1'][2]]
plt.bar(numbers_variables,sensitivity_S1,yerr=conf_interval,capsize=7,align='center',alpha=0.5)
plt.xticks(numbers_variables,group_variables)
plt.xlabel('Variables')
plt.ylabel('Sensitivity_values_S1')
plt.title('Sensitivity_analysis')
plt.show()

#Sensitivity analysis result
with open('Sensitivity_analysis.csv','w',newline='') as f:
	Sensitivity=['names','S1','ST']
	writer = csv.DictWriter(f,fieldnames=Sensitivity)
	writer.writeheader()
	for row in range(0,3):
		S1=Si['S1'][row]
		conf_S1=Si['S1_conf'][row]
		S1_conf=[Si['S1'][row]-Si['S1_conf'][row],Si['S1'][row]+Si['S1_conf'][row]]
		ST=Si['ST'][row]
		conf_ST=Si['ST_conf'][row]
		ST_conf=[Si['ST'][row]-Si['ST_conf'][row],Si['ST'][row]+Si['ST_conf'][row]]
		writer.writerow({'names':variablenames[row],'S1':S1_conf,'ST':ST_conf})


