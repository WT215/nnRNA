
import pyreadr
from os import walk
import gzip
import numpy as np
import pandas as pd
import scipy

import matplotlib.pyplot as plt
import sys



sys.path.insert(1, 'D:/RNAseqProject/Julia_allele/nnRNA')
import nnRNA
import nnRNA_wrapper
import sim
import applytodata as app


def nnRNA_wrapper(scRNA,Gname, file_name,store_path,allele_double=False,repeats=1,threshold_gc = [5000,2,100],prior="Fano", acceptance_level=[0,0.0],convert_data=True,meanBeta=0.06,beta_fixed=False,beta_loc=2.74,beta_scale=0.39,epochs=10000,neurons=100,layers=3,lr=1e-4,dropout_rate=0.2,contour2d=False,inputbeta_vec=None,test=True):

    cell_number =  int(np.shape(scRNA)[1])
    num_genes=int(np.shape(scRNA)[0])
    
    if (inputbeta_vec is None):
        inputbeta = sim.empirBETA(scRNA,meanBeta=meanBeta)
    else:
        inputbeta=inputbeta_vec

    
    sim.sum_stat(num_genes, cell_number, "_"+str(cell_number)+"_"+file_name, scRNA, np.zeros(shape=(3,num_genes)), inputbeta, repeats=1)
    
    
    
    train_genes = int(max(10000,100*5000/cell_number))

     # Use the NN to predict properties for the dataset.
    name = sim.naming(train_genes, cell_number, beta_fixed=beta_fixed,meanBeta=meanBeta,beta_loc=beta_loc,beta_scale=beta_scale,repeats=repeats,prior=prior,allele_double=allele_double)
    grid_name = "SumStat"+name

    nnRNA.execute(train_genes,cell_number,show_plot=False,train=True,test=test,meanBeta=meanBeta,contour2d=contour2d,beta_fixed=beta_fixed,beta_loc=beta_loc,beta_scale=beta_scale,neurons=neurons, layers=layers,lr=lr, dropout_rate=dropout_rate,epochs=epochs,acceptance_level=acceptance_level,repeats=repeats,prior=prior,allele_double=allele_double,inputbeta_vec=inputbeta_vec)


    ext = "_lbeta_"+str(beta_loc)+"_sbeta_"+str(beta_scale)
    outname = "NN_"+file_name+".csv"

    y_pred = app.predictions(cell_number,grid_name,file_name)
    
    if (Gname is not None):
        output = np.column_stack((Gname,y_pred))
    else:
        output=y_pred


    
    #output[:,0:5].shape
    
    
    
    dfnn = pd.DataFrame(output)
    

    
    dfnn.to_csv(store_path+outname, index = False)
    return dfnn




"""
cell_number =  int(np.shape(scRNA)[1])
num_genes=int(np.shape(scRNA)[0])

if (inputbeta_vec is None):
    inputbeta = sim.empirBETA(scRNA,meanBeta=meanBeta)
else:
    inputbeta=inputbeta_vec


sim.sum_stat(num_genes, cell_number, "_"+str(cell_number)+"_"+file_name, scRNA, np.zeros(shape=(3,num_genes)), inputbeta, repeats=1)



train_genes = int(max(10000,100*5000/cell_number))

 # Use the NN to predict properties for the dataset.
name = sim.naming(train_genes, cell_number, beta_fixed=beta_fixed,meanBeta=meanBeta,beta_loc=beta_loc,beta_scale=beta_scale,repeats=repeats,prior=prior,allele_double=allele_double)
grid_name = "SumStat"+name

nnRNA.execute(train_genes,cell_number,show_plot=False,train=True,test=True,meanBeta=meanBeta,contour2d=contour2d,beta_fixed=beta_fixed,beta_loc=beta_loc,beta_scale=beta_scale,neurons=neurons, layers=layers,lr=lr, dropout_rate=dropout_rate,epochs=epochs,acceptance_level=acceptance_level,repeats=repeats,prior=prior,allele_double=allele_double,inputbeta_vec=inputbeta_vec)


ext = "_lbeta_"+str(beta_loc)+"_sbeta_"+str(beta_scale)
outname = "NN_"+file_name+".csv"

y_pred = app.predictions(cell_number,grid_name,file_name)
output = np.column_stack((Gname,y_pred))
dfnn = pd.DataFrame(output)
dfnn.to_csv(store_path+outname, index = False)
"""