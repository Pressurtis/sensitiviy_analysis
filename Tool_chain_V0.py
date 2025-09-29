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
		qbtn.move(450,380)
		#Run button
		rbtn = QPushButton('Run',self)
		rbtn.clicked.connect(self.runMethod)
		rbtn.setToolTip('This is a <b>Run<b/> button')
		rbtn.resize(rbtn.sizeHint())
		rbtn.move(250,380)
		
		#Creat lables and text line for input
# num
		lbl1 = QLabel('Number of variables',self)
		lbl1.move(15,10)
		lbl1.setFont(QFont('New roman times',11,QFont.Bold))
		self.lbl1Edit = QLineEdit(self)
		self.lbl1Edit.resize(150,20)
		self.lbl1Edit.move(150,10)
# num_samples_simulation
		lbl2 = QLabel('Number of simmulation samples :',self)
		lbl2.move(350,10)
		lbl2.setFont(QFont('New roman times',11,QFont.Bold))
		self.lbl2Edit = QLineEdit(self)
		self.lbl2Edit.resize(150,20)
		self.lbl2Edit.move(550,10)
# num_samples_saltelli
		lbl3 = QLabel('Number of saltelli samples :',self)
		lbl3.move(15,40)
		lbl3.setFont(QFont('New roman times',11,QFont.Bold))
		self.lbl3Edit = QLineEdit(self)
		self.lbl3Edit.resize(100,20)
		self.lbl3Edit.move(200,40)
#method
		lbl4 = QLabel('Sampling method:',self)
		lbl4.move(350,40)
		self.lbl4Edits = ['sobol',
						'halton',
						'latin',
						'saltelli']
		self.lbl4Edit = QComboBox(self)
		self.lbl4Edit.addItems(self.lbl4Edits)
		self.lbl4Edit.setMinimumWidth(80)
		self.lbl4Edit.move(460,40)

#variable 1
		lbl5 = QLabel('Variables 1 name :',self)
		lbl5.move(15,70)
		lbl5.setFont(QFont('New roman times',10))
		self.lbl5Edit = QLineEdit(self)
		self.lbl5Edit.resize(100,20)
		self.lbl5Edit.move(120,70)

		lbl6 = QLabel('Left bound :',self)
		lbl6.move(260,70)
		lbl6.setFont(QFont('New roman times',10))
		self.lbl6Edit = QLineEdit(self)
		self.lbl6Edit.resize(100,20)
		self.lbl6Edit.move(350,70)

		lbl7 = QLabel('Right bound :',self)
		lbl7.move(510,70)
		lbl7.setFont(QFont('New roman times',10))
		self.lbl7Edit = QLineEdit(self)
		self.lbl7Edit.resize(100,20)
		self.lbl7Edit.move(600,70)
#variable 2
		lbl8 = QLabel('Variables 2 name :',self)
		lbl8.move(15,100)
		lbl8.setFont(QFont('New roman times',10))
		self.lbl8Edit = QLineEdit(self)
		self.lbl8Edit.resize(100,20)
		self.lbl8Edit.move(120,100)

		lbl9 = QLabel('Left bound :',self)
		lbl9.move(260,100)
		lbl9.setFont(QFont('New roman times',10))
		self.lbl9Edit = QLineEdit(self)
		self.lbl9Edit.resize(100,20)
		self.lbl9Edit.move(350,100)

		lbl10 = QLabel('Right bound :',self)
		lbl10.move(510,100)
		lbl10.setFont(QFont('New roman times',10))
		self.lbl10Edit = QLineEdit(self)
		self.lbl10Edit.resize(100,20)
		self.lbl10Edit.move(600,100)
#variable 3
		lbl11 = QLabel('Variables 3 name :',self)
		lbl11.move(15,130)
		lbl11.setFont(QFont('New roman times',10))
		self.lbl11Edit = QLineEdit(self)
		self.lbl11Edit.resize(100,20)
		self.lbl11Edit.move(120,130)

		lbl12 = QLabel('Left bound :',self)
		lbl12.move(260,130)
		lbl12.setFont(QFont('New roman times',10))
		self.lbl12Edit = QLineEdit(self)
		self.lbl12Edit.resize(100,20)
		self.lbl12Edit.move(350,130)

		lbl13 = QLabel('Right bound :',self)
		lbl13.move(510,130)
		lbl13.setFont(QFont('New roman times',10))
		self.lbl13Edit = QLineEdit(self)
		self.lbl13Edit.resize(100,20)
		self.lbl13Edit.move(600,130)
#variable 4
		lbl14 = QLabel('Variables 4 name :',self)
		lbl14.move(15,160)
		lbl14.setFont(QFont('New roman times',10))
		self.lbl14Edit = QLineEdit(self)
		self.lbl14Edit.resize(100,20)
		self.lbl14Edit.move(120,160)

		lbl15 = QLabel('Left bound :',self)
		lbl15.move(260,160)
		lbl15.setFont(QFont('New roman times',10))
		self.lbl15Edit = QLineEdit(self)
		self.lbl15Edit.resize(100,20)
		self.lbl15Edit.move(350,160)

		lbl16 = QLabel('Right bound :',self)
		lbl16.move(510,160)
		lbl16.setFont(QFont('New roman times',10))
		self.lbl16Edit = QLineEdit(self)
		self.lbl16Edit.resize(100,20)
		self.lbl16Edit.move(600,160)

#variable 5
		lbl17 = QLabel('Variables 5 name :',self)
		lbl17.move(15,190)
		lbl17.setFont(QFont('New roman times',10))
		self.lbl17Edit = QLineEdit(self)
		self.lbl17Edit.resize(100,20)
		self.lbl17Edit.move(120,190)

		lbl18 = QLabel('Left bound :',self)
		lbl18.move(260,190)
		lbl18.setFont(QFont('New roman times',10))
		self.lbl18Edit = QLineEdit(self)
		self.lbl18Edit.resize(100,20)
		self.lbl18Edit.move(350,190)

		lbl19 = QLabel('Right bound :',self)
		lbl19.move(510,190)
		lbl19.setFont(QFont('New roman times',10))
		self.lbl19Edit = QLineEdit(self)
		self.lbl19Edit.resize(100,20)
		self.lbl19Edit.move(600,190)

#variable 6
		lbl20 = QLabel('Variables 6 name :',self)
		lbl20.move(15,220)
		lbl20.setFont(QFont('New roman times',10))
		self.lbl20Edit = QLineEdit(self)
		self.lbl20Edit.resize(100,20)
		self.lbl20Edit.move(120,220)

		lbl21 = QLabel('Left bound :',self)
		lbl21.move(260,220)
		lbl21.setFont(QFont('New roman times',10))
		self.lbl21Edit = QLineEdit(self)
		self.lbl21Edit.resize(100,20)
		self.lbl21Edit.move(350,220)

		lbl22 = QLabel('Right bound :',self)
		lbl22.move(510,220)
		lbl22.setFont(QFont('New roman times',10))
		self.lbl22Edit = QLineEdit(self)
		self.lbl22Edit.resize(100,20)
		self.lbl22Edit.move(600,220)		
#variable 7
		lbl23 = QLabel('Variables 7 name :',self)
		lbl23.move(15,250)
		lbl23.setFont(QFont('New roman times',10))
		self.lbl23Edit = QLineEdit(self)
		self.lbl23Edit.resize(100,20)
		self.lbl23Edit.move(120,250)

		lbl24 = QLabel('Left bound :',self)
		lbl24.move(260,250)
		lbl24.setFont(QFont('New roman times',10))
		self.lbl24Edit = QLineEdit(self)
		self.lbl24Edit.resize(100,20)
		self.lbl24Edit.move(350,250)

		lbl25 = QLabel('Right bound :',self)
		lbl25.move(510,250)
		lbl25.setFont(QFont('New roman times',10))
		self.lbl25Edit = QLineEdit(self)
		self.lbl25Edit.resize(100,20)
		self.lbl25Edit.move(600,250)

#variable 8
		lbl26 = QLabel('Variables 8 name :',self)
		lbl26.move(15,280)
		lbl26.setFont(QFont('New roman times',10))
		self.lbl26Edit = QLineEdit(self)
		self.lbl26Edit.resize(100,20)
		self.lbl26Edit.move(120,280)

		lbl27 = QLabel('Left bound :',self)
		lbl27.move(260,280)
		lbl27.setFont(QFont('New roman times',10))
		self.lbl27Edit = QLineEdit(self)
		self.lbl27Edit.resize(100,20)
		self.lbl27Edit.move(350,280)

		lbl28 = QLabel('Right bound :',self)
		lbl28.move(510,280)
		lbl28.setFont(QFont('New roman times',10))
		self.lbl28Edit = QLineEdit(self)
		self.lbl28Edit.resize(100,20)
		self.lbl28Edit.move(600,280)

#variable 9
		lbl29 = QLabel('Variables 9 name :',self)
		lbl29.move(15,310)
		lbl29.setFont(QFont('New roman times',10))
		self.lbl29Edit = QLineEdit(self)
		self.lbl29Edit.resize(100,20)
		self.lbl29Edit.move(120,310)

		lbl30 = QLabel('Left bound :',self)
		lbl30.move(260,310)
		lbl30.setFont(QFont('New roman times',10))
		self.lbl30Edit = QLineEdit(self)
		self.lbl30Edit.resize(100,20)
		self.lbl30Edit.move(350,310)

		lbl31 = QLabel('Right bound :',self)
		lbl31.move(510,310)
		lbl31.setFont(QFont('New roman times',10))
		self.lbl31Edit = QLineEdit(self)
		self.lbl31Edit.resize(100,20)
		self.lbl31Edit.move(600,310)

#variable 10
		lbl32 = QLabel('Variables 10 name :',self)
		lbl32.move(15,340)
		lbl32.setFont(QFont('New roman times',10))
		self.lbl32Edit = QLineEdit(self)
		self.lbl32Edit.resize(100,20)
		self.lbl32Edit.move(120,340)

		lbl33 = QLabel('Left bound :',self)
		lbl33.move(260,340)
		lbl33.setFont(QFont('New roman times',10))
		self.lbl33Edit = QLineEdit(self)
		self.lbl33Edit.resize(100,20)
		self.lbl33Edit.move(350,340)

		lbl34 = QLabel('Right bound :',self)
		lbl34.move(510,340)
		lbl34.setFont(QFont('New roman times',10))
		self.lbl34Edit = QLineEdit(self)
		self.lbl34Edit.resize(100,20)
		self.lbl34Edit.move(600,340)

		lbl35 = QLabel('Seed :',self)
		lbl35.move(580,40)
		lbl35.setFont(QFont('New roman times',10))
		lbl35.setFont(QFont('New roman times',11,QFont.Bold))
		self.lbl35Edit = QLineEdit(self)
		self.lbl35Edit.resize(80,20)
		self.lbl35Edit.move(620,40)

		self.setGeometry(300,100,800,450)	#(Location:x,y)(Size;width.Height)
		self.setWindowTitle('Tool chain for GSA')
		self.show()

	# def closeEvent(self,event):
	# 	reply = QMessageBox.question(self,'Message','Are you sure to quit?',QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
	# 	if reply == QMessageBox.Yes:
	# 		event.accept()
	# 	else:
	# 		event.ignore()

	def center(self):
		qr = self.frameGeometry()
		cp = QDesktopWidget().availableGeometry().center()
		qr.moveCenter(cp)
		self.move(qr.topLeft())

	def runMethod(self):
		#Get input information
		num = int(self.lbl1Edit.text())
		num_samples_simulation = self.lbl2Edit.text()
		if num_samples_simulation != '':
			num_samples_simulation = int(num_samples_simulation)
		num_samples_saltelli = self.lbl3Edit.text()
		if num_samples_saltelli != '':
			num_samples_saltelli = int(num_samples_saltelli)
		method = self.lbl4Edit.currentText()
		variablenames = [self.lbl5Edit.text(),self.lbl8Edit.text(),self.lbl11Edit.text(),self.lbl14Edit.text(),self.lbl17Edit.text(),self.lbl20Edit.text(),self.lbl23Edit.text(),self.lbl26Edit.text(),self.lbl29Edit.text(),self.lbl32Edit.text(),]
		variablenames = [x for x in variablenames if x is not '']
		bounds = [[self.lbl6Edit.text(),self.lbl7Edit.text()],[self.lbl9Edit.text(),self.lbl10Edit.text()],[self.lbl12Edit.text(),self.lbl13Edit.text()],[self.lbl15Edit.text(),self.lbl16Edit.text()],[self.lbl18Edit.text(),self.lbl19Edit.text()],[self.lbl21Edit.text(),self.lbl22Edit.text()],[self.lbl24Edit.text(),self.lbl25Edit.text()],[self.lbl27Edit.text(),self.lbl28Edit.text()],[self.lbl30Edit.text(),self.lbl31Edit.text()],[self.lbl33Edit.text(),self.lbl34Edit.text()]]		#bounds = [[int(self.lbl6Edit.text()),int(self.lbl7Edit.text())],[int(self.lbl9Edit.text()),int(self.lbl10Edit.text())],[int(self.lbl12Edit.text()),int(self.lbl13Edit.text())],[int(self.lbl15Edit.text()),int(self.lbl16Edit.text())],[int(self.lbl18Edit.text()),int(self.lbl19Edit.text())],[int(self.lbl21Edit.text()),int(self.lbl22Edit.text())],[int(self.lbl24Edit.text()),int(self.lbl25Edit.text())],[int(self.lbl27Edit.text()),int(self.lbl28Edit.text())],[int(self.lbl30Edit.text()),int(self.lbl31Edit.text())],[int(self.lbl33Edit.text()),int(self.lbl34Edit.text())]]
		bounds = [x for x in bounds if x != ['','']]
		bounds = [[float(column) for column in row] for row in bounds]
		seed = self.lbl35Edit.text()
		if seed != '' :
			seed = int(seed)

		sample_values={}

		#sampling
		if method == 'sobol':
		#Sobol
			vec = sobol_seq.i4_sobol_generate(num,num_samples_simulation)
			for i in range(num):
				sample_values[variablenames[i]]=list(vec[:,i])
		#Halton
		elif method == 'halton':
			
			sequencer = ghalton.GeneralizedHalton(num,seed)
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


if __name__ == '__main__':
	app = QApplication(sys.argv)
	w=Window()
	sys.exit(app.exec_())
'''
Task4: Creat 'help' button for user
Task5: Let user define the file name
Task6: Error tips
Task7: Let user choose calc_second_order
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


# #Rbf

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


# #Sensitivity analysis

# p5=[]
# p6=[]
# p7=[]
# p8=[]
# p9=[]
# #Read output values from the file
# with open('output_values_500.csv', 'r', newline='') as f:
#     reader = csv.reader(f,delimiter=' ')
#     next(reader)
#     next(reader)
#     next(reader)
#     next(reader)
#     a=[]
#     b=[]
#     for row in f:
#     	a=''.join(row)
#     	b=a.split(',')
#     	p5.append(float(b[5]))
#     	p6.append(float(b[6]))
#     	p7.append(float(b[7]))
#     	p8.append(float(b[8]))
#     	p9.append(float(b[9]))

# #get the input of sensitivity analysis, which should be the array

# Y_p5=np.array(p5).reshape((500)).astype(np.float)
# Y_p6=np.array(p6).reshape((500)).astype(np.float)
# Y_p7=np.array(p7).reshape((500)).astype(np.float)
# Y_p8=np.array(p8).reshape((500)).astype(np.float)
# Y_p9=np.array(p9).reshape((500)).astype(np.float)

# #sensitivity analysis

# Si_p5= sobol.analyze(problem,Y_p5,calc_second_order=False)	#confidence interval level(default 0.95)
# Si_p6= sobol.analyze(problem,Y_p6,calc_second_order=False)
# Si_p7= sobol.analyze(problem,Y_p7,calc_second_order=False)
# Si_p8= sobol.analyze(problem,Y_p8,calc_second_order=False)
# Si_p9= sobol.analyze(problem,Y_p9,calc_second_order=False)

# #Write sensitivity analysis result in csv files

# #P5
# with open('output_results_P5.csv', 'w', newline='') as f:
# 	Sensitivity=['names','S1','conf_S1','S1_conf','ST','conf_ST','ST_conf'] #Create header names
# 	writer = csv.DictWriter(f,fieldnames=Sensitivity)
# 	writer.writeheader()
# 	writer.writerow({'names':'P5'})
# 	for row in range(0,3):
# 		S1=Si_p5['S1'][row]
# 		conf_S1=Si_p5['S1_conf'][row]
# 		S1_conf=[Si_p5['S1'][row]-Si_p5['S1_conf'][row],Si_p5['S1'][row]+Si_p5['S1_conf'][row]]
# 		ST=Si_p5['ST'][row]
# 		conf_ST=Si_p5['ST_conf'][row]
# 		ST_conf=[Si_p5['ST'][row]-Si_p5['ST_conf'][row],Si_p5['ST'][row]+Si_p5['ST_conf'][row]]
# 		writer.writerow({'names':variablenames[row],'S1':S1,'conf_S1':conf_S1,'S1_conf':S1_conf,'ST':ST,'conf_ST':conf_ST,'ST_conf':ST_conf})

# numbers_variables=np.arange(len(variablenames))
# width=0.2

# conf_S1_p5=[]
# for i in range(0,3):
# 	conf_S1_p5.append(Si_p5['S1_conf'][i])
# conf_ST_p5=[]
# for i in range(0,3):
# 	conf_ST_p5.append(Si_p5['ST_conf'][i])
# #generate graph for P5
# sensitivity_S1_p5=[Si_p5['S1'][0],Si_p5['S1'][1],Si_p5['S1'][2]]
# sensitivity_ST_p5=[Si_p5['ST'][0],Si_p5['ST'][1],Si_p5['ST'][2]]

# plt.figure('P5')
# plt.bar(numbers_variables,sensitivity_S1_p5,width,yerr=conf_S1_p5,capsize=7,align='center',alpha=0.5,color='red')
# plt.bar(numbers_variables+0.3,sensitivity_ST_p5,width,yerr=conf_ST_p5,capsize=7,align='center',alpha=0.5,color='blue')
# plt.xticks(numbers_variables,variablenames)
# plt.xlabel('Variables')
# plt.ylabel('Sensitivity_values')
# plt.title('Sensitivity_analysis_P5')
# S1_patch = mpatches.Patch(color='red',label='S1') #create legend
# ST_patch = mpatches.Patch(color='blue',label='ST')
# plt.legend(handles=[S1_patch,ST_patch],loc=1,fontsize = 'x-small')
# plt.tight_layout()
# plt.savefig('P5_Graph')
# plt.show()

# #P6
# with open('output_results_P6.csv', 'w', newline='') as f:
# 	Sensitivity=['names','S1','conf_S1','S1_conf','ST','conf_ST','ST_conf']
# 	writer = csv.DictWriter(f,fieldnames=Sensitivity)
# 	writer.writeheader()
# 	writer.writerow({'names':'P6'})
# 	for row in range(0,3):
# 		S1=Si_p6['S1'][row]
# 		conf_S1=Si_p6['S1_conf'][row]
# 		S1_conf=[Si_p6['S1'][row]-Si_p6['S1_conf'][row],Si_p6['S1'][row]+Si_p6['S1_conf'][row]]
# 		ST=Si_p6['ST'][row]
# 		conf_ST=Si_p6['ST_conf'][row]
# 		ST_conf=[Si_p6['ST'][row]-Si_p6['ST_conf'][row],Si_p6['ST'][row]+Si_p6['ST_conf'][row]]
# 		writer.writerow({'names':variablenames[row],'S1':S1,'conf_S1':conf_S1,'S1_conf':S1_conf,'ST':ST,'conf_ST':conf_ST,'ST_conf':ST_conf})

# conf_S1_p6=[]
# for i in range(0,3):
# 	conf_S1_p6.append(Si_p6['S1_conf'][i])
# conf_ST_p6=[]
# for i in range(0,3):
# 	conf_ST_p6.append(Si_p6['ST_conf'][i])
# #generate graph for P6
# sensitivity_S1_p6=[Si_p6['S1'][0],Si_p6['S1'][1],Si_p6['S1'][2]]
# sensitivity_ST_p6=[Si_p6['ST'][0],Si_p6['ST'][1],Si_p6['ST'][2]]

# plt.figure('P6')
# plt.bar(numbers_variables,sensitivity_S1_p6,width,yerr=conf_S1_p6,capsize=7,align='center',alpha=0.5,color='r')
# plt.bar(numbers_variables+0.3,sensitivity_ST_p6,width,yerr=conf_ST_p6,capsize=7,align='center',alpha=0.5,color='b')
# plt.xticks(numbers_variables,variablenames)
# plt.xlabel('Variables')
# plt.ylabel('Sensitivity_values')
# plt.title('Sensitivity_analysis_P6')
# plt.savefig('P6_Graph')



# #P7
# with open('output_results_P7.csv', 'w', newline='') as f:
# 	Sensitivity=['names','S1','conf_S1','S1_conf','ST','conf_ST','ST_conf']
# 	writer = csv.DictWriter(f,fieldnames=Sensitivity)
# 	writer.writeheader()
# 	writer.writerow({'names':'P7'})
# 	for row in range(0,3):
# 		S1=Si_p7['S1'][row]
# 		conf_S1=Si_p7['S1_conf'][row]
# 		S1_conf=[Si_p7['S1'][row]-Si_p7['S1_conf'][row],Si_p7['S1'][row]+Si_p7['S1_conf'][row]]
# 		ST=Si_p7['ST'][row]
# 		conf_ST=Si_p7['ST_conf'][row]
# 		ST_conf=[Si_p7['ST'][row]-Si_p7['ST_conf'][row],Si_p7['ST'][row]+Si_p7['ST_conf'][row]]
# 		writer.writerow({'names':variablenames[row],'S1':S1,'conf_S1':conf_S1,'S1_conf':S1_conf,'ST':ST,'conf_ST':conf_ST,'ST_conf':ST_conf})

# #P8
# with open('output_results_P8.csv', 'w', newline='') as f:
# 	Sensitivity=['names','S1','conf_S1','S1_conf','ST','conf_ST','ST_conf']
# 	writer = csv.DictWriter(f,fieldnames=Sensitivity)
# 	writer.writeheader()
# 	writer.writerow({'names':'P8'})
# 	for row in range(0,3):
# 		S1=Si_p8['S1'][row]
# 		conf_S1=Si_p8['S1_conf'][row]
# 		S1_conf=[Si_p8['S1'][row]-Si_p8['S1_conf'][row],Si_p8['S1'][row]+Si_p8['S1_conf'][row]]
# 		ST=Si_p8['ST'][row]
# 		conf_ST=Si_p8['ST_conf'][row]
# 		ST_conf=[Si_p8['ST'][row]-Si_p8['ST_conf'][row],Si_p8['ST'][row]+Si_p8['ST_conf'][row]]
# 		writer.writerow({'names':variablenames[row],'S1':S1,'conf_S1':conf_S1,'S1_conf':S1_conf,'ST':ST,'conf_ST':conf_ST,'ST_conf':ST_conf})

# #P9
# with open('output_results_P9.csv', 'w', newline='') as f:
# 	Sensitivity=['names','S1','conf_S1','S1_conf','ST','conf_ST','ST_conf']
# 	writer = csv.DictWriter(f,fieldnames=Sensitivity)
# 	writer.writeheader()
# 	writer.writerow({'names':'P9'})
# 	for row in range(0,3):
# 		S1=Si_p9['S1'][row]
# 		conf_S1=Si_p9['S1_conf'][row]
# 		S1_conf=[Si_p9['S1'][row]-Si_p9['S1_conf'][row],Si_p9['S1'][row]+Si_p9['S1_conf'][row]]
# 		ST=Si_p9['ST'][row]
# 		conf_ST=Si_p9['ST_conf'][row]
# 		ST_conf=[Si_p9['ST'][row]-Si_p9['ST_conf'][row],Si_p9['ST'][row]+Si_p9['ST_conf'][row]]
# 		writer.writerow({'names':variablenames[row],'S1':S1,'conf_S1':conf_S1,'S1_conf':S1_conf,'ST':ST,'conf_ST':conf_ST,'ST_conf':ST_conf})







