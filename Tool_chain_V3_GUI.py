'''
Task2: different buttons(不同的窗口实现不同的功能)
Task3: let user define file names
Task4: beautiful interface
Task6: Deal with none performing value
Task9: test fmu
'''

'''
Task1: 完成GUI完整版本
Task2: 完成jupyter notebook分版本
'''

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
import pandas as pd
import xlsxwriter
#rbf
from scipy.interpolate import Rbf
#remove file
import os
#GUI
import sys 
from PyQt5.QtWidgets import (QWidget,QToolTip,QPushButton,QComboBox,
	QApplication,QLabel,QLineEdit,QTextEdit,QGridLayout)
from PyQt5.QtGui import QFont


class Window(QWidget):
	def __init__(self):
		super().__init__()
		self.initUI()



	def initUI(self):
		QToolTip.setFont(QFont('SansSerif',12))
		self.setToolTip('This is a <b>QWidget</b>')
		#Quit button
		qbtn = QPushButton('Quit',self)
		qbtn.clicked.connect(QApplication.instance().quit) #funtion of the button ,QApplication.instance() run,self.closeEvent
		qbtn.setToolTip('This is a <b>Quit<b/> button')
		qbtn.resize(qbtn.sizeHint())
		qbtn.move(450,450)
		#Run button
		rbtn = QPushButton('Run',self)
		rbtn.clicked.connect(self.runMethod)
		rbtn.setToolTip('This is a <b>Run<b/> button')
		rbtn.resize(rbtn.sizeHint())
		rbtn.move(250,450)
		
		#Creat lables and text line for input
# num
		lbl_num = QLabel('Number of variables',self)
		lbl_num.move(15,10)
		lbl_num.setFont(QFont('New roman times',11,QFont.Bold))
		self.lbl_num_Edit = QLineEdit(self)
		self.lbl_num_Edit.resize(150,20)
		self.lbl_num_Edit.move(150,10)
# num_samples_simulation
		lbl_num_samples_simulation = QLabel('Number of simmulation samples :',self)
		lbl_num_samples_simulation.move(350,10)
		lbl_num_samples_simulation.setFont(QFont('New roman times',11,QFont.Bold))
		self.lbl2_num_samples_simulation_Edit = QLineEdit(self)
		self.lbl2_num_samples_simulation_Edit.resize(150,20)
		self.lbl2_num_samples_simulation_Edit.move(550,10)
# num_samples_saltelli
		lbl_num_samples_saltelli = QLabel('Number of saltelli samples :',self)
		lbl_num_samples_saltelli.move(15,40)
		lbl_num_samples_saltelli.setFont(QFont('New roman times',11,QFont.Bold))
		self.lbl_num_samples_saltelli_Edit = QLineEdit(self)
		self.lbl_num_samples_saltelli_Edit.resize(100,20)
		self.lbl_num_samples_saltelli_Edit.move(200,40)
#method
		lbl_method = QLabel('Sampling method:',self)
		lbl_method.move(350,40)
		self.lbl_method_Edits = ['sobol',
						'halton',
						'latin',
								]
		self.lbl_method_Edit = QComboBox(self)
		self.lbl_method_Edit.addItems(self.lbl_method_Edits)
		self.lbl_method_Edit.setMinimumWidth(80)
		self.lbl_method_Edit.move(460,40)

		#Sobol
		lbl_sobol = QLabel('Sobol Sampling',self)
		lbl_sobol.move(15,70)
		#skip	
		lbl_sobol_skip = QLabel('Skip',self)
		lbl_sobol_skip.move(200,70)
		self.lbl_sobol_skip_Edit = QLineEdit(self)
		self.lbl_sobol_skip_Edit.resize(200,20)
		self.lbl_sobol_skip_Edit.move(260,70)
		self.lbl_sobol_skip_Edit.setPlaceholderText('e.g. 2(default:0)')

		#Halton
		lbl_halton = QLabel('Halton Sampling',self)
		lbl_halton.move(15,100)
		#seed	
		lbl_halton_seed = QLabel('Seed',self)
		lbl_halton_seed.move(200,100)
		self.lbl_halton_seed_Edit = QLineEdit(self)
		self.lbl_halton_seed_Edit.resize(200,20)
		self.lbl_halton_seed_Edit.move(260,100)
		self.lbl_halton_seed_Edit.setPlaceholderText('e.g. 2')

		#Latin Hypercube
		lbl_latin = QLabel('Latin-Hypercube Sampling',self)
		lbl_latin.move(15,130)
		#criterion
		lbl_latin_criterion = QLabel('Criterion',self)
		lbl_latin_criterion.move(200,130)
		self.lbl_latin_criterion_Edits = ['default',
						'center',
						'maximin',
						'centermaximin',
						'correlation']
		self.lbl_latin_criterion_Edit = QComboBox(self)
		self.lbl_latin_criterion_Edit.addItems(self.lbl_latin_criterion_Edits)
		self.lbl_latin_criterion_Edit.setMinimumWidth(200)
		self.lbl_latin_criterion_Edit.move(260,130)

		#Saltelli
		lbl_saltelli_second = QLabel('Calc_second_order',self)
		lbl_saltelli_second.move(200,160)

		#second order
		lbl_saltelli = QLabel('Saltelli',self)
		lbl_saltelli.move(15,160)
		self.lbl_saltelli_Edits = ['False',
					'True']
		self.lbl_saltelli_Edit = QComboBox(self)
		self.lbl_saltelli_Edit.addItems(self.lbl_saltelli_Edits)
		self.lbl_saltelli_Edit.setMinimumWidth(140)
		self.lbl_saltelli_Edit.move(320,160)
		
#Rbf
		lbl_Rbf = QLabel('Rbf',self)
		lbl_Rbf.move(15,190)
		#k-fold method
		lbl_Rbf_k_fold = QLabel('Number of folds',self)
		lbl_Rbf_k_fold.move(200,190)
		self.lbl_Rbf_k_fold_Edit = QLineEdit(self)
		self.lbl_Rbf_k_fold_Edit.resize(140,20)
		self.lbl_Rbf_k_fold_Edit.move(320,190)
		self.lbl_Rbf_k_fold_Edit.setPlaceholderText('e.g. 10')

		#Rbf function
		lbl_rbf_function = QLabel('Function',self)
		lbl_rbf_function.move(480,190)
		self.lbl_rbf_function_Edits = ['multiquadric',
										'inverse',
										'gaussian',
										'linear',
										'cubic',
										'quintic',
										'thin_plate']
		self.lbl_rbf_function_Edit = QComboBox(self)
		self.lbl_rbf_function_Edit.addItems(self.lbl_rbf_function_Edits)
		self.lbl_rbf_function_Edit.setMinimumWidth(140)
		self.lbl_rbf_function_Edit.move(550,190)

#Sensitivity analysis
		lbl_GSA = QLabel('names',self)
		lbl_GSA.move(15,220)
		#names
		lbl_GSA_names = QLabel('names',self)
		lbl_GSA_names.move(200,220)
		self.lbl_GSA_names_Edit = QLineEdit(self)
		self.lbl_GSA_names_Edit.resize(140,29)
		self.lbl_GSA_names_Edit.move(320,220)
		self.lbl_GSA_names_Edit.setPlaceholderText('e.g. P5,P6,P7')


#Variable names
		lbl_variable_names = QLabel('Variables name :',self)
		lbl_variable_names.move(15,320)
		lbl_variable_names.setFont(QFont('New roman times',10))
		self.lbl_variable_names_Edit = QLineEdit(self)
		self.lbl_variable_names_Edit.resize(500,20)
		self.lbl_variable_names_Edit.move(120,320)
		self.lbl_variable_names_Edit.setPlaceholderText('e.g. a,b,c')

#Bounds
		lbl_bounds = QLabel('Bounds :',self)
		lbl_bounds.move(15,350)
		lbl_bounds.setFont(QFont('New roman times',10))
		self.lbl_bounds_Edit = QLineEdit(self)
		self.lbl_bounds_Edit.resize(500,20)
		self.lbl_bounds_Edit.move(120,350)
		self.lbl_bounds_Edit.setPlaceholderText('e.g. 1,2|3,4|5,6')



		self.setGeometry(300,100,800,500)	#(Location:x,y)(Size;width.Height)
		self.setWindowTitle('Tool chain for GSA')
		self.show()

	def center(self):
		qr = self.frameGeometry()
		cp = QDesktopWidget().availableGeometry().center()
		qr.moveCenter(cp)
		self.move(qr.topLeft())

	def runMethod(self):
		#Get input information
		num = int(self.lbl_num_Edit.text()) #type in number of variables
		num_samples_simulation = self.lbl2_num_samples_simulation_Edit.text()
		if num_samples_simulation != '':
			num_samples_simulation = int(num_samples_simulation)
		method = self.lbl_method_Edit.currentText() #sampling method:'sobol','halton','latin'
		variablenames = self.lbl_variable_names_Edit.text() #['a','b','c']
		variablenames = variablenames.split(',')
		#rbf
		k_fold = int(self.lbl_Rbf_k_fold_Edit.text())
		names = [y for y in self.lbl_GSA_names_Edit.text().split(',')] #['P5','P6']
		rbf_function = self.lbl_rbf_function_Edit.currentText()

		sample_values={}
		skip = ''#skip is for sobol sample argument. '' means you don't want to use it. e.g. 2. 
		seed =

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
			with open('input.csv','w',newline='') as f:
				writer = csv.writer(f)
				writer.writerow(['method',method,'num variables',num,'num samples',num_samples_simulation,'skip property',skip])
				writer.writerow(sample_values.keys())
				writer.writerows(zip(*sample_values.values()))
		
		#Halton
		elif method == 'halton':
			seed = self.lbl_halton_seed_Edit.text()
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
			with open('input.csv','w',newline='') as f:
				writer = csv.writer(f)
				writer.writerow(['method',method,'num variables',num,'num samples',num_samples_simulation,'seed',seed])
				writer.writerow(sample_values.keys())
				writer.writerows(zip(*sample_values.values()))

		#Latin hypercube
		elif method == 'latin':
			latin_criterion = self.lbl_latin_criterion_Edit.currentText()
			if latin_criterion == 'default':
				latin = lhs(num, samples=num_samples_simulation)
				for i in range(num):
					sample_values[variablenames[i]]=list(latin[:,i])
			else:
				latin = lhs(num, samples=num_samples_simulation,criterion=latin_criterion)
				for i in range(num):
					sample_values[variablenames[i]]=list(latin[:,i])
			with open('input.csv','w',newline='') as f:
				writer = csv.writer(f)
				writer.writerow(['method',method,'num variables',num,'num samples',num_samples_simulation,'criterion',latin_criterion])
				writer.writerow(sample_values.keys())
				writer.writerows(zip(*sample_values.values()))
		else:
			print('Please type the correct method name !(sobol/halton/latin)')
		
		#Generate Saltelli samples for GSA
		saltelli_second_order = self.lbl_saltelli_Edit.currentText()
		num_samples_saltelli = self.lbl_num_samples_saltelli_Edit.text()
		if num_samples_saltelli != '':
			num_samples_saltelli = int(num_samples_saltelli)
		bounds = self.lbl_bounds_Edit.text()
		bounds = [[y for y in x.split(',')] for x in bounds.split('|')]
		bounds = [[float(column) for column in row] for row in bounds]
		



		#Define problem
		problem={
			'num_vars':num,'names':variablenames,'bounds':bounds
		} 
		#Generate samples for test
		# When calc_second_order=False,number of samples= N *( D+2).Otherwise,N *( 2D+2)
		if saltelli_second_order == 'False':
			param_values_simulation = saltelli.sample(problem,int(int(num_samples_simulation)/(num+2)),calc_second_order=False)  
		elif saltelli_second_order == 'True':
			param_values_simulation = saltelli.sample(problem,int(int(num_samples_simulation)/(2*num+2)),calc_second_order=True)  				


		#Rbf
		def k_fold_generator(X, y, k_fold,num):
		    rbf_result=[]
		    subset_size = int(len(X) / k_fold)
		    #Convert nd.array to list (Once the array is created, th size can't be changed)
		    X = X.tolist()
		    for k in range(k_fold):
		        rbf_train = [[] for i in range(num+1)]
		        #Get X train data
		        X_train = X[:k * subset_size] + X[(k + 1) * subset_size:]
		        X_train = np.array(X_train,dtype='float')
		        
		        for i in range(num):
		        	rbf_train[i] = X_train[:,i].tolist()
		        #Get y train data
		        y = np.array(y,dtype='float').tolist()
		        y_train = y[:k * subset_size] + y[(k + 1) * subset_size:]     
		        rbf_train[num] = y_train
		        #Get train Rbf model
		        train_rbf = Rbf(*rbf_train,function=rbf_function)

		        #Get valid data
		        rbf_valid = [[] for i in range(num)]
		        X_valid = X[k * subset_size:][:subset_size]
		        X_valid = np.array(X_valid,dtype='float')
		        for i in range(num):
		        	rbf_valid[i] = X_valid[:,i].tolist()  
		        #Get valid result
		        rbf_valid_result = train_rbf(*rbf_valid).tolist()
		        rbf_result = rbf_result+ rbf_valid_result
		    
		    rbf_result = np.array(rbf_result,dtype='float')
			
			#Calculate RMSE
		    rmse = round(np.sqrt(((rbf_result-y)**2).mean()),3)
		    #Calculate r square
		    y_bar = np.sum(y)/len(y)  		#Calculate the mean of y
		    ssres = np.sum((y-rbf_result)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
		    sstot = np.sum((y-y_bar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
		    r_square = round(ssreg / sstot,3)
			
		    #Plot scatter graph
		    plt.figure('k-fold method')
		    plt.scatter(y,rbf_result,c='r',cmap=plt.cm.coolwarm,zorder=2,s=6)
		    plt.xlabel('Samples values')
		    plt.ylabel('Rbf values')
		    plt.title('K-folds method for Rbf')
		    r_square_patch = mpatches.Patch(color='red',label='r_square: %s'%r_square)
		    rmse_patch = mpatches.Patch(color='blue',label='RMSE: %s'%rmse )
		    plt.legend(handles=[r_square_patch,rmse_patch],loc=0,fontsize = 'x-small')
		    #Plot y=x 
		    lims = [np.min([plt.gca().get_xlim(),plt.gca().get_ylim()]),np.max([plt.gca().get_xlim(),plt.gca().get_ylim()])] #plt.gca().get_ylim() get lim in y axis
		    plt.plot(lims,lims,alpha=0.75)

		    #Create text for r_square and RMSE
		    plt.text(np.max(plt.gca().get_xlim()),np.min(plt.gca().get_ylim())+0.1,'r_square: %s'%r_square,verticalalignment='bottom', horizontalalignment='right',fontsize= 10)
		    plt.text(np.max(plt.gca().get_xlim()),np.min(plt.gca().get_ylim())+0.5,'RMSE: %s'%rmse,verticalalignment='bottom', horizontalalignment='right',fontsize= 10) 
		    plt.tight_layout()
		    plt.show()
		    plt.close()

		input_rbf_data = pd.read_csv('rbf_data.csv',header=3)
		y = [[]for i in range(len(names))]
		for i in range(len(names)):
			y[i] = input_rbf_data[names[i]].iloc[:num_samples_simulation]
			k_fold_generator(param_values_simulation,y[i],k_fold,num)
		
		#description: How to calculate output values(P5,P6,P7) with rbf
		# names=['P5','P6','P7']
		# x=[[]for i in range(len(names))]
		# for i in range(len(names)):	
		# 	x[i]=train_rbf[i](1,1,7)
		# x=[[4],[5],[6]]

#Run Rbf to interpolate 
#Calculate output values for saltelli samples


#Sensitivity analysis

		def GSA(names=names,saltelli_samples=num_samples_saltelli ):
			#Total data for all names
			#Read data
			input_GSA_input_data = pd.read_csv('output.csv',header=3)
			
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
				if saltelli_second_order == 'False':
					sobol_analyze_result[i] = sobol.analyze(problem,input_GSA_names[i],calc_second_order=False)
				elif saltelli_second_order == 'True':
					sobol_analyze_result[i] = sobol.analyze(problem,input_GSA_names[i],calc_second_order=True)
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
				plt.bar(np.arange(num),GSA_result_S1[i],width,yerr=GSA_conf_S1[i],capsize=7,align='center',alpha=0.5,color='red')
				plt.bar(np.arange(num)+0.3,GSA_result_ST[i],width,yerr=GSA_conf_ST[i],capsize=7,align='center',alpha=0.5,color='blue')
				plt.xticks(np.arange(num),variablenames)
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






if __name__ == '__main__':
	app = QApplication(sys.argv)
	w=Window()
	sys.exit(app.exec_())







