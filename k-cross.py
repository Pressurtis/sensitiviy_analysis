from SALib.sample import saltelli
from SALib.analyze import sobol
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np 
import pandas as pd

def k_fold_generator(X, y, k_fold,num,rbf_function):
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
    ssreg = np.sum((y-y_bar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((rbf_result-y_bar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    r_square = round(ssreg / sstot,3)
	
    #Plot scatter graph
    plt.figure('k-fold method')
    plt.scatter(y,rbf_result,c='r',cmap=plt.cm.coolwarm,zorder=2,s=6)
    plt.xlabel('Samples values')
    plt.ylabel('Rbf values')
    plt.title('K-folds method for Rbf')
    #Create legend
    r_square_patch = mpatches.Patch(color='red',label='r_square: %s'%r_square)
    rmse_patch = mpatches.Patch(color='blue',label='RMSE: %s'%rmse )
    plt.legend(handles=[r_square_patch,rmse_patch],loc=0,fontsize = 'x-small')
    #Plot y=x 
    lims = [np.min([plt.gca().get_xlim(),plt.gca().get_ylim()]),np.max([plt.gca().get_xlim(),plt.gca().get_ylim()])] #plt.gca().get_ylim() get lim in y axis
    plt.plot(lims,lims,alpha=0.75)

    #Create text for r_square and RMSE

    plt.tight_layout()
    plt.show()
    plt.close()



#Read Data
num_samples_simulation=100
names = ['P5','P6','P7']
input_rbf_data = pd.read_csv('output.csv',header=3)
y = [[]for i in range(len(names))]
for i in range(len(names)):
	y[i] = input_rbf_data[names[i]].iloc[:num_samples_simulation]

#Define input
num = 3
k_fold = 10

problem={
'num_vars':num,'names':['rotor_core_stiffness','bearing_stiffness','foundation_stiffness'],'bounds':[[1e8,3e8],[4e8,8e8],[0.5e9,1.5e9]]
}
param_values=saltelli.sample(problem,20,calc_second_order=False)
rbf_function = 'multiquadric' #function: multiquadric(default),inverse,gaussian,linear,cubic,quintic,thin_plate
for i in range(len(names)):
	k_fold_generator(param_values,y[i],k_fold,num,rbf_function)




