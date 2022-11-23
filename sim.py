import numpy as np
import pandas as pd
import scipy.stats as scs
import pickle
from sklearn.neighbors import KernelDensity
from joblib import Parallel, delayed
#====================================================================
# Function for computing ground truth of 3d parameter space.

#set number of cores for parallele computing
n_jobs=4

def Groundtruth_lars(num_samples):

    ML_concat = pd.read_pickle('./SS3_cast_UMIs_concat_ML.pkl')
    ML_concat2 = pd.read_pickle('./SS3_c57_UMIs_concat_ML.pkl')
    ML_concat = ML_concat[ML_concat[1]][0]
    ML_concat2 = ML_concat2[ML_concat2[1]][0]

    kon = list(np.log10(pd.Series([gene[0] for gene in ML_concat], index=ML_concat.index)))
    kon.extend(np.log10(pd.Series([gene[0] for gene in ML_concat2], index=ML_concat2.index)))
    koff = list(np.log10(pd.Series([gene[1] for gene in ML_concat], index=ML_concat.index)))
    koff.extend(np.log10(pd.Series([gene[1] for gene in ML_concat2], index=ML_concat2.index)))
    ksyn = list(np.log10(pd.Series([gene[2] for gene in ML_concat], index=ML_concat.index)))
    ksyn.extend(np.log10(pd.Series([gene[2] for gene in ML_concat2], index=ML_concat2.index)))

    data = np.column_stack((kon,koff,ksyn))

#    mean = np.mean(data, axis=0)
#    cov = np.cov(data, rowvar=0)

#    samples = np.random.multivariate_normal(mean, cov, size=num_samples, check_valid='warn', tol=1e-8)

    kde = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(data)

    samples = kde.sample([num_samples])

    sim_kon = 10**samples[:,0]
    sim_koff = 10**samples[:,1]
    sim_ksyn = 10**samples[:,2]
    
    new_param_space = [sim_kon, sim_koff, sim_ksyn]
    return new_param_space

#====================================================================
# Function for computing ground truth of 3d parameter space.

def Groundtruth_uniform(num_samples):
    #simulate kon from uniform distribution.
    sim_kon = 10**np.random.uniform(low=-2.0, high=2.0, size=num_samples)
    #simulate ratio of kon and koff from normal distribution
#    sim_koff = 10**np.random.uniform(low=-2.0, high=np.log10(sim_kon), size=num_samples)
#    sim_koff = 10**np.random.uniform(low=-2.0, high=4.0, size=num_samples)
    #simulate ratio of kon and koff from normal distribution
    sim_inter = abs(np.random.normal(loc=0.05, scale=0.5, size=num_samples))
    #compute koff
    sim_koff=sim_kon / sim_inter
    #compute ksyn
    sim_ksyn = 10**np.random.uniform(low=-2.0, high=5.0, size=num_samples)
    #finally, obtain the ground truth:
    new_param_space = [sim_kon, sim_koff, sim_ksyn]
    return new_param_space

#====================================================================
# Function for computing ground truth of 3d parameter space.

def Groundtruth(num_samples):
    #simulate kon from uniform distribution.
    sim_kon = 10**np.random.uniform(low=-2.0, high=2.0, size=num_samples)
    #simulate ratio of kon and koff from normal distribution
    sim_inter = abs(np.random.normal(loc=0.05, scale=0.5, size=num_samples))
    #compute koff
    sim_koff=sim_kon / sim_inter
    #simulate fano factor
    sim_fano = 10**np.random.uniform(low=np.log10(1.001), high=np.log10(30), size=num_samples)
    #compute ksyn
    sim_ksyn=(sim_fano - 1.0) * (sim_kon + sim_koff) * (sim_kon + sim_koff + 1.0) / sim_koff
    #finally, obtain the ground truth:
    new_param_space = [sim_kon, sim_koff, sim_ksyn]
    return new_param_space

#=================================================================
# Compute beta

def BETA(cell_number,beta_fixed=False,meanBeta=0.06,beta_loc=2.74,beta_scale=0.39):
    # The mean capture efficiency is around 6% to 10% for droplet based RNA-seq
    # The capture rate will vary between cells but there is the option to fix it to a constant value.
    if beta_fixed:
        inputbeta = meanBeta
    else:
        exp_lib_sizes = np.random.lognormal(mean=beta_loc, sigma=beta_scale, size=cell_number)
        inputbeta = exp_lib_sizes / np.mean(exp_lib_sizes) * meanBeta
        
        #WT added 
        inputbeta[inputbeta>=1]=np.max(inputbeta[inputbeta<1])
        inputbeta[inputbeta<=0]=np.min(inputbeta[inputbeta>0])
    return inputbeta 

#=================================================================
# Simulation from BETA-POISSON
def rbp_self(cell_number,kon, koff, ksyn):
    simout=np.random.poisson(np.random.beta(kon, koff, cell_number)*ksyn, cell_number)
    return simout

#=================================================================
##4: simulation from BETA-POISSON+ Binomial downsampling. A toy example:
#numm: number of cells
#tempvec=rbp_self(numm,input_par[0], input_par[1], input_par[2])
#binomial downsampling
#tempvec_down=np.random.binomial(tempvec,inputbeta,numm)

#=================================================================
##5: estimate BETA from scRNA-seq data:
def empirBETA(DData,meanBeta=0.06):
    #compute total counts for each cell
    temp_beta=np.sum(DData,axis=0)
    #normalized total counts to mean capture efficiency
    temp_beta2=temp_beta/np.mean(temp_beta)*meanBeta
    #clip extreme values, because beta represent probability, should range between 0 to 1.
    temp_beta2[temp_beta2<=0]=np.min(temp_beta2[temp_beta2>0])
    temp_beta2[temp_beta2>=1]=np.max(temp_beta2[temp_beta2<1])
    return temp_beta2

#=================================================================
# Main, executing the above functions

def create_sim(num_genes, cell_number, name,allele_double=False, beta_fixed=False,meanBeta=0.06,beta_loc=2.74,beta_scale=0.39,repeats=1,prior="Fano",inputbeta_vec=None):

    print("Running simulations for %i genes and %i cells (repeating %i times)..." %(num_genes, cell_number,repeats))

    if prior == "Fano":
        param_space = Groundtruth(num_genes)
    if prior == "Uniform":
        param_space = Groundtruth_uniform(num_genes)
    if prior == "larsson":
        param_space = Groundtruth_lars(num_genes)
        
    if (inputbeta_vec is None):
        inputbeta = BETA(cell_number,beta_fixed=beta_fixed,meanBeta=meanBeta,beta_loc=beta_loc,beta_scale=beta_scale)
    else:
        inputbeta=inputbeta_vec
        

    
    scRNA = []
    for i in range(num_genes):
        for j in range(repeats):
            tempvec=rbp_self(cell_number,param_space[0][i], param_space[1][i], param_space[2][i])
            if allele_double:
                tempvec2=rbp_self(cell_number,param_space[0][i], param_space[1][i], param_space[2][i])
                tempvec = tempvec+tempvec2
            tempvec_down=np.random.binomial(tempvec,inputbeta,cell_number)
            if i == 0 and j == 0:
                scRNA = tempvec_down
            else:
                scRNA = np.vstack((scRNA,tempvec_down))

    # Save the output to a pandas array and then to a csv-file.
    # The array as number of genes rows and the number of cells columns. Every gene has a different value of
    # kon, koff, and ksyn, which is what we want to infer. So, the summary statistics will have to be over all
    # cells.
    
    
    print("begin storing CSV")
    df = pd.DataFrame(scRNA)
    df.to_csv ("./DataFrame"+name, index = False)

    print("CSV file stored.")

    return scRNA, param_space, inputbeta

#=================================================================

def sum_stat(num_genes, cell_number, name, scRNA, param_space, inputbeta,repeats=1,prefix="./"):

    print("Computing summary statistics for %i genes and %i cells..." %(num_genes, cell_number))

    sum_stat = []
    for i in range(num_genes):
        kon = param_space[0][i]
        koff = param_space[1][i]
        ksyn = param_space[2][i]

        for j in range(repeats):
            # Data for each gene adjusted for capture rate
            gene = scRNA[i*repeats+j,:]
            
            #nonan = ~np.isnan(gene)
            #gene = gene[nonan]
            #inputbeta = inputbeta[nonan]

            raw_m1 = np.mean(gene)
            raw_count = np.sum(gene)

            gene = gene/inputbeta
            counts = np.sum(gene)
            gene = gene/counts

            zcount = len(np.where(gene == 0)[0])/cell_number
            ir = max(gene)-min(gene)
            maxgene = max(gene)
            m1 = np.mean(gene) 
            p10 = np.percentile(gene,5)
            p25 = np.percentile(gene,25)
            p50 = np.median(gene)
            p75 = np.percentile(gene,75)
            p95 = np.percentile(gene,95)
            m2 = np.var(gene)
            m3 = scs.skew(gene) # Third moment, skewness
            m4 = scs.kurtosis(gene) # Fourth moment, kurtosis
            cov = scs.variation(gene)

            sum_gene = [kon,koff,ksyn,np.log10(ir),np.log10(maxgene),p10,p25,p50,p75,p95,m1,np.log10(m2),m3,m4,np.log10(counts),zcount,np.log10(cov),np.log10(raw_m1),np.log10(raw_count)]

            if i == 0 and j == 0:
                sum_stat = sum_gene
            else:
                sum_stat = np.vstack((sum_stat,sum_gene))

    df = pd.DataFrame(sum_stat)
    df.to_csv(prefix+"SumStat"+name, index = False)

    print("CSV file stored.")

    return sum_stat

#=================================================================

def naming(num_genes, cell_number, allele_double=False, beta_fixed=False,meanBeta=0.06,beta_loc=2.74,beta_scale=0.39,repeats=1,prior="Fano"):
    if not beta_fixed:
        ext = "_sbeta_"+str(beta_scale)+"_lbeta_"+str(beta_loc)
    else:
        ext = "_Fixed"
    if not prior == "Fano":
        ext += "_"+prior
    if allele_double:
        ext += "_double_allele"

    #name = "_scRNA_Genes_"+str(num_genes)+"_Cells_"+str(cell_number)+"_repeats_"+str(repeats)+"_mbeta_"+str(meanBeta)+ext+".csv"
    name = "_scRNA_Genes_"+str(num_genes)+"_Cells_"+str(cell_number)+".csv"
    return name 

#=================================================================

def run(num_genes, cell_number, allele_double=False, beta_fixed=False,meanBeta=0.06,beta_loc=2.74,beta_scale=0.39,repeats=1,prefix="./",prior="Fano",inputbeta_vec=None):
    name = naming(num_genes, cell_number, allele_double=allele_double, beta_fixed=beta_fixed,meanBeta=meanBeta,beta_loc=beta_loc,beta_scale=beta_scale,repeats=repeats,prior=prior)
    #scRNA, param_space, inputbeta = create_sim(num_genes, cell_number, name, allele_double=allele_double, beta_fixed=beta_fixed,meanBeta=meanBeta,beta_loc=beta_loc,beta_scale=beta_scale,repeats=repeats,prior=prior,inputbeta_vec=inputbeta_vec)
    #summary = sum_stat(num_genes, cell_number, name, scRNA, param_space, inputbeta,repeats=repeats,prefix=prefix)
    
    
    #modified by wt
    scRNA, param_space, inputbeta = create_sim_wt(num_genes, cell_number, name, allele_double=allele_double, beta_fixed=beta_fixed,meanBeta=meanBeta,beta_loc=beta_loc,beta_scale=beta_scale,repeats=repeats,prior=prior,inputbeta_vec=inputbeta_vec)

    summary = sum_stat_wt(num_genes, cell_number, name, scRNA, param_space, inputbeta,repeats=repeats,prefix=prefix)

#=================================================================

# Example:
#num_genes = 10000
#cell_number = 5000
#run(num_genes, cell_number, beta_fixed=False,meanBeta=0.06,beta_loc=2.74,beta_scale=0.39)
#




def create_sim_wt(num_genes, cell_number, name,allele_double=False, beta_fixed=False,meanBeta=0.06,beta_loc=2.74,beta_scale=0.39,repeats=1,prior="Fano",inputbeta_vec=None,n_jobs=n_jobs):

    print("Running simulations for %i genes and %i cells (repeating %i times)..." %(num_genes, cell_number,repeats))

    if prior == "Fano":
        param_space = Groundtruth(num_genes)
    if prior == "Uniform":
        param_space = Groundtruth_uniform(num_genes)
    if prior == "larsson":
        param_space = Groundtruth_lars(num_genes)
        
    if (inputbeta_vec is None):
        inputbeta = BETA(cell_number,beta_fixed=beta_fixed,meanBeta=meanBeta,beta_loc=beta_loc,beta_scale=beta_scale)
    else:
        inputbeta=inputbeta_vec
        
    param_space_repeat=[np.repeat(param_space[0], repeats, axis=0),np.repeat(param_space[1], repeats, axis=0),np.repeat(param_space[2], repeats, axis=0)]
    
    if allele_double:
        scRNA1=Parallel(n_jobs=n_jobs,verbose=0, prefer="threads")(delayed(rbp_self)(cell_number,param_space_repeat[0][ii],param_space_repeat[1][ii],param_space_repeat[2][ii]) for ii in range(num_genes))
        scRNA1=np.asarray(scRNA1)
        scRNA2=Parallel(n_jobs=n_jobs,verbose=0, prefer="threads")(delayed(rbp_self)(cell_number,param_space_repeat[0][ii],param_space_repeat[1][ii],param_space_repeat[2][ii]) for ii in range(num_genes))
        scRNA2=np.asarray(scRNA2)
        scRNA=scRNA1+scRNA2
        
    else:
        scRNA=Parallel(n_jobs=n_jobs,verbose=0, prefer="threads")(delayed(rbp_self)(cell_number,param_space_repeat[0][ii],param_space_repeat[1][ii],param_space_repeat[2][ii]) for ii in range(num_genes))
        scRNA=np.asarray(scRNA)
    scRNA=np.random.binomial(scRNA,inputbeta,scRNA.shape)

    
    print("begin storing CSV")
    df = pd.DataFrame(scRNA)
    df.to_csv ("./DataFrame"+name, index = False)

    print("CSV file stored.")

    return scRNA, param_space_repeat, inputbeta

def sum_sub_wt(i,scRNA,cell_number,repeats,param_space,inputbeta):
    kon = param_space[0][i]
    koff = param_space[1][i]
    ksyn = param_space[2][i]
    #gene = scRNA[i*repeats+j,:]
    gene = scRNA[i,:]
    
    #nonan = ~np.isnan(gene)
    #gene = gene[nonan]
    #inputbeta = inputbeta[nonan]

    raw_m1 = np.mean(gene)
    raw_count = np.sum(gene)

    gene = gene/inputbeta
    counts = np.sum(gene)
    gene = gene/counts

    zcount = len(np.where(gene == 0)[0])/cell_number
    ir = max(gene)-min(gene)
    maxgene = max(gene)
    m1 = np.mean(gene) 
    p10 = np.percentile(gene,5)
    p25 = np.percentile(gene,25)
    p50 = np.median(gene)
    p75 = np.percentile(gene,75)
    p95 = np.percentile(gene,95)
    m2 = np.var(gene)
    m3 = scs.skew(gene) # Third moment, skewness
    m4 = scs.kurtosis(gene) # Fourth moment, kurtosis
    cov = scs.variation(gene)

    sum_gene = [kon,koff,ksyn,np.log10(ir),np.log10(maxgene),p10,p25,p50,p75,p95,m1,np.log10(m2),m3,m4,np.log10(counts),zcount,np.log10(cov),np.log10(raw_m1),np.log10(raw_count)]
    return sum_gene





def sum_stat_wt(num_genes, cell_number, name, scRNA, param_space, inputbeta,repeats=1,prefix="./",n_jobs=n_jobs):

    print("Computing summary statistics for %i genes and %i cells..." %(num_genes, cell_number))
    sum_stat=Parallel(n_jobs=n_jobs,verbose=0, prefer="threads")(delayed(sum_sub_wt)(ii,scRNA,cell_number=cell_number,repeats=1,param_space=param_space,inputbeta=inputbeta) for ii in range(num_genes))
    sum_stat=np.asarray(sum_stat)

    df = pd.DataFrame(sum_stat)
    df.to_csv(prefix+"SumStat"+name, index = False)

    print("CSV file stored.")

    return sum_stat


#Parallel(n_jobs=2, prefer="threads")(delayed(np.sqrt)(i ** 2) for i in range(10))
