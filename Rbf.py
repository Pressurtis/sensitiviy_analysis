from SALib.sample import saltelli
from SALib.analyze import sobol
from scipy.interpolate import Rbf
from itertools import islice
import matplotlib.pyplot as plt
import numpy as np 
import csv

#Rbf

#Setup data(get samples)
input_rbf_data = pd.read_csv('output.csv',header=3)
p5_ansys = input_rbf_data['P5'].iloc[:100]
#Get Rbf function
p5_rbf = Rbf(*fuc)#,function='gaussian'

#Generate new samples
param_values_500=saltelli.sample(problem,100,calc_second_order=False)

x_500 = param_values_500[:,0]
y_500 = param_values_500[:,1]
z_500 = param_values_500[:,2]

p5_500= p5_rbf(x_500,y_500,z_500)
print(p5_500)
# #k-fold cross validation
# num_folds = 10
# subset_size = len(p5_500)/num_folds
# for i in range(numfolds):
#     testing_rbf = 


# #simlation,ROM


