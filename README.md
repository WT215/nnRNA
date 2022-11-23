# How to run nnRNA

```python
#load library###
import pyreadr
from os import walk
import gzip
import numpy as np
import pandas as pd
import scipy

import matplotlib.pyplot as plt
import sys
import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

#folder of nnRNA
sys.path.insert(1, '~/nnRNA')
import nnRNA
import nnRNA_wrapper
import sim
import applytodata as app


#output path
store_path="~/nnout/"

#load toy example 
df = pd.read_csv("~/sim500run2"+"/"+"Dat_down_2.csv")



DData = df.to_numpy()


#prepare count matrix (genes in rows and cells in columns)
scRNA_input=DData.astype(float)

#estimate cell-specific capture efficiencies
inputbeta = sim.empirBETA(scRNA_input,meanBeta=0.06)
#no gene name provided, so just set to be None
Gname_input=None

#create directory for storing output fron nnRNA
os.chdir(store_path)
os.getcwd()
file_name='sim500run2'

#Run nnRNA
#set `allele_double=True` for UMI data, otherwise `allele_double=False` for allele specific scRNA-seq data.
dfnn_c=nnRNA_wrapper.nnRNA_wrapper(scRNA=scRNA_input,Gname=Gname_input, file_name=file_name,store_path=store_path,allele_double=False,repeats=1,threshold_gc=[5000,2,100],prior="Fano",inputbeta_vec=inputbeta)

```
Output will be in your output directory with name `NN_${file_name}.csv`.
