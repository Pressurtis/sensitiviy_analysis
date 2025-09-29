"""
SALib sampling
"""
#Sampling and GSA
from SALib.sample import saltelli
from SALib.analyze import sobol
import sobol_seq
import ghalton
from pyDOE import *
#Deal with array and matrix
import numpy as np 
#Read csv
import csv
#GUI
import sys 
from PyQt5.QtWidgets import (QWidget,QToolTip,QPushButton,
	QApplication,QLabel,QLineEdit,QTextEdit,QGridLayout,QComboBox)
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
		lbl1 = QLabel('Number of variables',self)
		lbl1.move(15,10)
		lbl1.setFont(QFont('New roman times',11,QFont.Bold))
		self.lbl1Edit = QLineEdit(self)
		self.lbl1Edit.resize(150,20)
		self.lbl1Edit.move(150,10)


		lbl2 = QLabel('Number of simmulation samples :',self)
		lbl2.move(350,10)
		lbl2.setFont(QFont('New roman times',11,QFont.Bold))
		self.lbl2Edit = QLineEdit(self)
		self.lbl2Edit.resize(150,20)
		self.lbl2Edit.move(550,10)

		lbl3 = QLabel('Number of saltelli samples :',self)
		lbl3.move(15,40)
		lbl3.setFont(QFont('New roman times',11,QFont.Bold))
		self.lbl3Edit = QLineEdit(self)
		self.lbl3Edit.resize(100,20)
		self.lbl3Edit.move(200,40)

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
		# lbl4.setFont(QFont('New roman times',11,QFont.Bold))
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

		lbl9 = QLabel('lLeft bound :',self)
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
			print('Please choose the correct method name !(sobol/halton/latin/saltelli)')
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
