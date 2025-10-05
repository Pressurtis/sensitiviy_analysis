from SALib.sample import saltelli
from SALib.analyze import sobol
from scipy.interpolate import Rbf
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import xlsxwriter
import pandas as pd
import csv
#remove file
import os
variablenames = ['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10']
num_variables = np.arange(len(variablenames))
bounds=[[5e8,3.5e9],[6.5e8,9.5e8],[-5e7,-2e7],[6.5e8,9.5e8],[1.05e11,1.2e11],[1.5e8,4.5e8],[3.5e8,5.5e8],[6.5e8,9.5e8],[6.5e8,9.5e8],[6.2e3,8.2e3]]
problem={ 'num_vars':len(variablenames),'names':variablenames,'bounds':bounds}
names = ['Y']
saltelli_second_order = False
num_samples_saltelli=240

#rbf
input_rbf_data = pd.read_csv('output.csv',header=3)
y = input_rbf_data[names].iloc[:100]
input_rbf_variables = [[]for i in range(len(variablenames)+1)]
for i in range(len(variablenames)):
	input_rbf_variables[i] = input_rbf_data[variablenames[i]].iloc[:100]
input_rbf_variables[len(variablenames)] = y
P5_rbf = Rbf(*input_rbf_variables)
# generate 100 saltelli samples
param_values_GSA = saltelli.sample(problem,20,calc_second_order=False)
param_values=[[]for i in range(len(variablenames))]
for i in range(len(variablenames)):
	param_values[i]=param_values_GSA[:,i]
P5_rbf = P5_rbf(*param_values)

P5_samples = {}
for i in range(len(variablenames)):
	P5_samples[variablenames[i]] = param_values[i]
P5_samples['Y'] = P5_rbf

with open ('output_GSA.csv','w',newline='') as f:
	writer = csv.writer(f)
	writer.writerow(P5_samples.keys())
	writer.writerows(zip(*P5_samples.values()))




#Read output values from the file
def GSA(names=names,saltelli_samples=num_samples_saltelli ):
	#Total data for all names
	#Read data
	input_GSA_input_data = pd.read_csv('output_GSA.csv',header=0)
	
	#Store names data in seperate list
	input_GSA_names = [[] for i in range(len(names))]
	for i in range(len(names)):
		input_GSA_names[i] = input_GSA_input_data[names[i]].iloc[:num_samples_saltelli]
		input_GSA_names[i] = np.array(input_GSA_names[i],dtype='float')
# # #Sensitivity analysis

	GSA_conf_S1 = [[]for i in range(len(names))]
	GSA_conf_ST = [[]for i in range(len(names))]
	GSA_result_S1 = [[]for i in range(len(names))]
	GSA_result_ST = [[]for i in range(len(names))]

	sobol_analyze_result = [[]for i in range(len(names))]
# 	#sobol_anlyze
	for i in range(len(names)):
		sobol_analyze_result[i] = sobol.analyze(problem,input_GSA_names[i],calc_second_order=False)
		#Get S1,ST,Conf result in sobol_analyze_result
		S1=sobol_analyze_result[i]['S1'][i]
		conf_S1=sobol_analyze_result[i]['S1_conf'][i]
		S1_conf=[sobol_analyze_result[i]['S1'][i]-sobol_analyze_result[i]['S1_conf'][i],sobol_analyze_result[i]['S1'][i]+sobol_analyze_result[i]['S1_conf'][i]]
		ST=sobol_analyze_result[i]['ST'][i]
		conf_ST=sobol_analyze_result[i]['ST_conf'][i]
		ST_conf=[sobol_analyze_result[i]['ST'][i]-sobol_analyze_result[i]['ST_conf'][i],sobol_analyze_result[i]['ST'][i]+sobol_analyze_result[i]['ST_conf'][i]]

		

		#Get sensitivity analysis result,S1,ST,conf_S1,conf_ST
		GSA_result_S1[i] = sobol_analyze_result[i]['S1']
		GSA_result_ST[i] = sobol_analyze_result[i]['ST']
		GSA_conf_S1[i] = sobol_analyze_result[i]['S1_conf']
		GSA_conf_ST[i] = sobol_analyze_result[i]['ST_conf']


#Plot

	for i in range(len(names)):
		width=0.2
		plt.bar(num_variables,GSA_result_S1[i],width,yerr=GSA_conf_S1[i],capsize=7,align='center',alpha=0.5,color='red')
		plt.bar(num_variables+0.3,GSA_result_ST[i],width,yerr=GSA_conf_ST[i],capsize=7,align='center',alpha=0.5,color='blue')
		plt.xticks(num_variables,variablenames)
		plt.xlabel('Variables')
		plt.ylabel('Sensitivity_values')
		plt.title('Sensitivity_analysis_'+names[i])
		S1_patch = mpatches.Patch(color='red',label='S1') #create legend
		ST_patch = mpatches.Patch(color='blue',label='ST')
		plt.legend(handles=[S1_patch,ST_patch],loc=1,fontsize = 'x-small')
		plt.tight_layout()
		# graph name = names[i]
		plt.savefig(names[i])
		plt.close()

#xlsx
#Write worksheet
	workbook = xlsxwriter.Workbook('result.xlsx')
	for i in range(len(names)):	
		worksheet_title = workbook.add_worksheet(names[i])
		worksheet_title.write('A1',names[i])
		worksheet_title.write('A2','Variables')
		worksheet_title.write('B2','S1')
		worksheet_title.write('C2','Conf_S1')
		worksheet_title.write('D2','S1_Conf')
		worksheet_title.write('E2','ST')
		worksheet_title.write('F2','Conf_ST')
		worksheet_title.write('G2','ST_Conf')
		worksheet_title.set_column(0,0,25)
		worksheet_title.set_column(3,3,15)
		worksheet_title.set_column(6,6,15)
		#Write result(values) in the grid 
		for row in range(len(variablenames)) :
			worksheet_title.write('A'+str(row+3),variablenames[row])
			worksheet_title.write('B'+str(row+3),sobol_analyze_result[i]['S1'][row])
			worksheet_title.write('C'+str(row+3),sobol_analyze_result[i]['S1_conf'][row])
			worksheet_title.write_string('D'+str(row+3),str([round(sobol_analyze_result[i]['S1'][row]-sobol_analyze_result[i]['S1_conf'][row],3),round(sobol_analyze_result[i]['S1'][row]+sobol_analyze_result[i]['S1_conf'][row],3)]))
			worksheet_title.write('E'+str(row+3),sobol_analyze_result[i]['ST'][row])
			worksheet_title.write('F'+str(row+3),sobol_analyze_result[i]['ST_conf'][row])
			worksheet_title.write_string('G'+str(row+3),str([round(sobol_analyze_result[i]['ST'][row]-sobol_analyze_result[i]['ST_conf'][row],3),round(sobol_analyze_result[i]['ST'][row]+sobol_analyze_result[i]['ST_conf'][row],3)]))
			#insert image
			worksheet_title.insert_image('B8',names[i]+'.png',{'x_scale':0.7,'y_scale':0.7})
			#close the workbook
	workbook.close()
			#remove the graph from folder
	for i in range(len(names)):
		os.remove(names[i]+'.png')

GSA = GSA()








