import pandas as pd
import numpy as np
import sim
import nnRNA
import seaborn as sns
import tensorflow
import pickle
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import os
import pygtc

#======================================================================

def predictions(cell_number,grid_name,file_name):

    model = tensorflow.keras.models.load_model("infe_model_"+grid_name+".h5", compile=False)

    with open("infe_scy_"+grid_name+".pkl", 'rb') as run:
        scy = pickle.load(run)
    with open("infe_sc_"+grid_name+".pkl", 'rb') as run:
        sc = pickle.load(run)

    df = pd.read_csv("SumStat_"+str(cell_number)+"_"+file_name)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df.to_numpy()

    dimy = 3

    X_data = df[:,[3,4,5,6,7,8,9,11,12,13,14,15,16,17,18]]
    cc = X_data[:,-2]
    print(X_data)
    X_data = sc.transform(X_data)

    genes = len(X_data)
    samples = 1000

    y_pred = []
    p16 = []
    p84 = []
    ratio_pred = []
    r16 = []
    r84 = []
    rstd=[]
    mstd=[]

    for i in range(genes):
        print("Running %f per cent" % ((100*float(i)+1.)/genes), end="\r")
        x = X_data[i,:]
        y = []
        for j in range(samples-1):
            x = np.vstack((x,X_data[i,:]))
        y = scy.inverse_transform(model.predict(x)[:,:dimy])

#        yr = 10**y[:,2]/10**y[:,1]
        yr = y[:,2]-y[:,1]
        yr2=np.reshape(yr,[samples,1])
        y_combine = np.hstack((y,yr2))




        

        r1 = np.mean(yr)
        r2 = np.percentile(yr,16)
        r3 = np.percentile(yr,84)
        r4 = np.percentile(yr,2.5)
        r5 = np.percentile(yr,97.5)

        m1 = np.mean(y,axis=0)
        m2 = np.percentile(y,16,axis=0)
        m3 = np.percentile(y,84,axis=0)
        m4 = np.percentile(y,2.5,axis=0)
        m5 = np.percentile(y,97.5,axis=0)
        std_m=np.std(y,axis=0)
        std_r=np.std(yr,axis=0)

        
        if i == 0:
            y_pred = m1
            p16 = m2
            p84 = m3
            p02 = m4
            p97 = m5
            ratio_pred = r1
            r16 = r2
            r84 = r3
            r02 = r4
            r97 = r5
            rstd=std_r
            mstd=std_m
            
        else:
            y_pred = np.vstack((y_pred,m1))
            p16 = np.vstack((p16,m2))
            p84 = np.vstack((p84,m3))
            ratio_pred = np.vstack((ratio_pred,r1))
            r16 = np.vstack((r16,r2))
            r84 = np.vstack((r84,r3))
            r02 = np.vstack((r02,r4))
            r97 = np.vstack((r97,r5))
            p02 = np.vstack((p02,m4))
            p97 = np.vstack((p97,m5))      
            rstd=np.vstack((rstd,std_r))
            mstd=np.vstack((mstd,std_m))

    numc = cell_number*np.ones(genes)

    return np.column_stack((y_pred,ratio_pred,rstd,mstd,p16,p84,r16,r84,numc,cc,r02,r97,p02,p97))

#======================================================================

def plotting_data(raw_cell_number,y_data,grid_name,file_name,colour=True):

#    plt.figure()

    y_pred = y_data[:,:3]
    ratio_pred = y_data[:,9]
    p16 = y_data[:,3:6]
    p84 = y_data[:,6:9]
    r16 = y_data[:,10]
    r84 = y_data[:,11]
    numc = y_data[:,12]
    cc = y_data[:,13]

    cmap = "plasma"
    #convert time to a color tuple using the colormap used for scatter
    norm = matplotlib.colors.Normalize(vmin=min(cc), vmax=max(cc), clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    y_color = np.array([(mapper.to_rgba(v)) for v in cc])

    for x, y, xel, xeu, yel, yeu, color in zip(ratio_pred, y_pred[:,0], r16,r84, p16[:,0], p84[:,0], y_color):
        yerr = np.vstack(([y-yel],[yeu-y]))
        xerr = np.vstack(([x-xel],[xeu-x]))
        if not colour:
            color="grey"
        plt.errorbar(x, y, yerr = yerr, xerr=xerr,capsize=3,alpha=0.3,color=color)
    if not colour:
        plt.scatter(ratio_pred,y_pred[:,0],s=numc/raw_cell_number*75,c="grey",zorder=2,alpha=0.3)
    else:   
        plt.scatter(ratio_pred,y_pred[:,0],s=numc/raw_cell_number*75,c=cc,zorder=2,alpha=0.3,cmap=cmap)

    plt.xlabel(r"$\mathrm{Burst\,size}\,[\log_{10}(k_\mathrm{syn}/k_\mathrm{off})]$",fontsize=13)
    plt.ylabel(r"$\mathrm{Burst\,frequency}\,[\log_{10}(k_\mathrm{on})]$",fontsize=13)
    plt.xticks(size=13)
    plt.yticks(size=13)

    if colour:
        cbar = plt.colorbar()
        for t in cbar.ax.get_yticklabels():
            t.set_fontsize(13)
        cbar.ax.get_yaxis().labelpad = 10
        t = r"$\mathrm{Mean\,\,expression \,\,}[\log_{10}(N_\mathrm{mol})]$"
        cbar.ax.set_ylabel(t, rotation=90,fontsize=13)

#    plt.savefig(file_name+grid_name+"_Burstfreqvssize.png",bbox_inches='tight')

#======================================================================

def read_data(direc, file_name,repeats=1,threshold_gc = [5000,2,100],prior="Fano", acceptance_level=[0,0.0],convert_data=True,meanBeta=0.06,beta_fixed=False,beta_loc=2.74,beta_scale=0.39,epochs=10000,neurons=100,layers=3,lr=1e-4,dropout_rate=0.2,contour2d=False,inputbeta_vec=None):

    # Load data
    df0 = pd.read_csv(direc+"/"+file_name)
    mask = list(set(df0.isnull().sum(axis=1).to_numpy()))
    y_data = []
    discarded = 0
    if (inputbeta_vec is None):
        inputbeta_1 = None
    else:
        inputbeta_1=inputbeta_vec


    for i in mask:
        # Find number width of mask
        df = df0[df0.isnull().sum(axis=1).to_numpy() == i]
        DData = df.to_numpy()

        for j in range(len(DData)):
            gene = DData[j,:]
            if ((inputbeta_vec is not None)):
                inputbeta_used=inputbeta_1[~np.isnan(gene)]
            gene = gene[~np.isnan(gene)]         
            if j == 0:
                scRNA = gene
            else:
                scRNA = np.vstack((scRNA,gene))

        # Compute summary statistics
        num_genes = int(np.shape(scRNA)[0])
        raw_cell_number = int(np.shape(DData)[1])
        if j == 0:
            cell_number = int(np.shape(scRNA)[0])
            num_genes = 1
        else:
            num_genes = int(np.shape(scRNA)[0])
            cell_number =  int(np.shape(scRNA)[1])  # Only count non-NaN entries

        if num_genes >= threshold_gc[1] and num_genes < threshold_gc[0] and cell_number >= threshold_gc[2]:
            if convert_data:
                # Compute beta and summary stat.
                if (inputbeta_vec is None):
                    inputbeta_2 = sim.empirBETA(scRNA,meanBeta=meanBeta)
                else:
                    inputbeta_2=inputbeta_used
                    
                sim.sum_stat_wt(num_genes, cell_number, "_"+str(cell_number)+"_"+file_name, scRNA, np.zeros(shape=(3,num_genes)), inputbeta_2, repeats=1)

            X_data = pd.read_csv("SumStat_"+str(cell_number)+"_"+file_name)
            X_data = X_data.to_numpy()
            counts = min(X_data[:,-1])
            zcount = 1-max(X_data[:,-4])
            if 10**counts >= acceptance_level[0] and zcount*cell_number/raw_cell_number >= acceptance_level[1]:

            # Compute grid with same number of cells but more genes for training. 
            # The number of genes in the training set is chosen such that it is equivalent to the number of measurements you get with 5000 cells and 10000 genes.

                train_genes = int(max(10000,100*5000/cell_number))

                # Use the NN to predict properties for the dataset.

                name = sim.naming(train_genes, cell_number, beta_fixed=beta_fixed,meanBeta=meanBeta,beta_loc=beta_loc,beta_scale=beta_scale,repeats=repeats,prior=prior)
                grid_name = "SumStat"+name

                if not os.path.isfile("infe_model_"+grid_name+".h5"):
                    if (inputbeta_vec is None):
                        inputbeta_inter = None
                    else:
                        inputbeta_inter=inputbeta_2
                        
                    nnRNA.execute(train_genes,cell_number,show_plot=False,train=True,test=False,meanBeta=meanBeta,contour2d=contour2d,beta_fixed=beta_fixed,beta_loc=beta_loc,beta_scale=beta_scale,neurons=neurons, layers=layers,lr=lr, dropout_rate=dropout_rate,epochs=epochs,acceptance_level=acceptance_level,repeats=repeats,prior=prior,inputbeta_vec=inputbeta_inter)
                else:
                    print("Trained model found")

                y_pred = predictions(cell_number,grid_name,file_name)

                if len(y_data) == 0:
                    y_data = y_pred
                else:
                    y_data = np.vstack((y_data,y_pred))
            else:
                discarded += 1

    print("%i data points discarded due to too low number of detections" %(discarded))

    df = pd.DataFrame(y_data)
    if beta_fixed:
        ext = "_Fixed"
    else:
        ext = "_lbeta_"+str(beta_loc)+"_sbeta_"+str(beta_scale)

    df.to_csv("ATD_"+file_name+"_"+prior+"_AL_"+str(acceptance_level[0])+"_"+".csv", index = False)

    print("CSV file stored.")

    #plotting_data(raw_cell_number,y_data,grid_name,file_name)

#======================================================================

def load_ATDFILE(file_name,repeats=1,prior="Fano",meanBeta=0.06,beta_fixed=False,beta_loc=2.74,beta_scale=0.39,acceptance_level=[0,0.0]):

    if beta_fixed:
        ext = "_Fixed"
    else:
        ext = "_lbeta_"+str(beta_loc)+"_sbeta_"+str(beta_scale)

    #name = "ATD_"+file_name+"_"+prior+"_AL_"+str(acceptance_level[0])+"_"+str(acceptance_level[1])+"_beta_"+str(meanBeta)+ext+".csv"
    #df = pd.read_csv(name)
    df=pd.read_csv(file_name)
    y_data = df.to_numpy()

    print("Genes", len(y_data))

    return np.column_stack((y_data[:,2]-y_data[:,1],y_data[:,0])), file_name, y_data

#======================================================================

def just_plot(file_names,raw_cell_number,repeats=1,meanBeta=0.06,beta_fixed=False,beta_loc=2.74,beta_scale=0.39,acceptance_level=[0,0.0],all_priors=True,followup=False,colour=True):

    plt.figure(1,figsize=(20,10))
    tit = [r"$\mathrm{c57, \, NN \, (Fano \, prior)}$",r"$\mathrm{c57, \, NN \, (Uniform \, prior)}$",r"$\mathrm{cast, \, NN \, (Fano\,prior)}$",r"$\mathrm{cast, \, NN \, (Uniform \, prior)}$" ]
    tit2 = [r"$\mathrm{c57}$",r"$\mathrm{cast}$"]

    for i, file_name in enumerate(file_names):

        if "c57" in file_name:
            ML_concat = pd.read_pickle('./SS3_c57_UMIs_concat_ML.pkl')
        elif "cast" in file_name:
            ML_concat = pd.read_pickle('./SS3_cast_UMIs_concat_ML.pkl')
        ML_concat = ML_concat[ML_concat[1]][0]
 
        BFL = list(np.log10(pd.Series([gene[0] for gene in ML_concat], index=ML_concat.index)))
        BSL = list(np.log10(pd.Series([gene[2]/gene[1] for gene in ML_concat], index=ML_concat.index)))

        BF = BFL
        BS = BSL
        priors = [r"$\mathrm{Larsson\, et\, al.\, (2019)}$"]*len(BF)

        samples1, grid_name1, y_data1 = load_ATDFILE(file_name,prior="Fano",meanBeta=meanBeta,beta_fixed=beta_fixed)
        samples2, grid_name2, y_data2 = load_ATDFILE(file_name,prior="Uniform",meanBeta=meanBeta,beta_fixed=beta_fixed)
        priors.extend([r"$\mathrm{NN, \, Fano}$"]*len(samples1))
        priors.extend([r"$\mathrm{NN, \, Uniform}$"]*len(samples2))

        BF.extend(list(samples1[:,1]))
        BF.extend(list(samples2[:,1]))
        BS.extend(list(samples1[:,0]))
        BS.extend(list(samples2[:,0]))

        df = {r"$\mathrm{Burst\,size}\,[\log_{10}(k_\mathrm{syn}/k_\mathrm{off})]$": BS,
              r"$\mathrm{Burst\,frequency}\,[\log_{10}(k_\mathrm{on})]$":  BF,
              r"$\mathrm{Method}$": priors,
              }
        df = pd.DataFrame(df)

        sns.jointplot(data=df, x=r"$\mathrm{Burst\,size}\,[\log_{10}(k_\mathrm{syn}/k_\mathrm{off})]$", y=r"$\mathrm{Burst\,frequency}\,[\log_{10}(k_\mathrm{on})]$", hue = r"$\mathrm{Method}$",  kind="kde")
        
        plt.figure(1)
        if all_priors:
            plt.subplot(2,2,i*2+1)
            plt.title(tit[2*i])
            plotting_data(raw_cell_number,y_data1,grid_name1,file_name,colour=colour)
            plt.subplot(2,2,i*2+2)
            plt.title(tit[2*i+1])
            plotting_data(raw_cell_number,y_data2,grid_name2,file_name,colour=colour)
        else:
            plt.subplot(1,2,i+1)
            plt.title(tit2[i])
            plotting_data(raw_cell_number,y_data1,grid_name1,file_name,colour=colour)

    if not followup:
        plt.figure(1)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3)
        plt.savefig("Appl_"+grid_name1+".png",bbox_inches='tight')

#======================================================================

def final_plot():

    meanBeta = 1.00
    beta_fixed = True

    just_plot(["df_c57_used.csv","df_cast_used.csv"],224,beta_fixed=beta_fixed,meanBeta=meanBeta,all_priors=True,followup=True,colour=False)
    cmap = matplotlib.cm.get_cmap('Spectral')
    rgba = cmap(0.01)
    for j in range(4):
        plt.subplot(2,2,j+1)
        plt.text(-0.5,-1,r"$\beta = 1.00$",color="grey",fontsize=13)
        plt.text(2.9,1,r"$\beta = 0.06$",color=rgba,fontsize=13)
        x1 = np.linspace(-1.5,5.5,100)
        plt.plot(x1,x1,"k:")
        plt.ylim(-3.1,2.1)
        plt.xlim(-1.5,5.5)

    meanBeta = 0.06
    beta_fixed = False

    just_plot(["df_c57_used.csv","df_cast_used.csv"],224,beta_fixed=beta_fixed,meanBeta=meanBeta,all_priors=True,followup=False,colour=True)

def whoiswho(direc,file_name,nogenenameNNcsvpath,threshold_gc=[5000,2,100],meanBeta=0.06,beta_fixed=False,beta_loc=2.74,beta_scale=0.39,acceptance_level=[0,0.0],prior="larsson"):

    df0 = pd.read_csv(direc+"/"+file_name)
    NoNA = df0.isnull().sum(axis=1).to_numpy()
    index = df0.index.to_numpy()

    mask = list(set(df0.isnull().sum(axis=1).to_numpy()))

    INDEX = []

    for i in mask:
        df = df0[df0.isnull().sum(axis=1).to_numpy() == i]
        DData = df.to_numpy()

        for j in range(len(DData)):
            gene = DData[j,:]
            gene = gene[~np.isnan(gene)]
            if j == 0:
                scRNA = gene
            else:
                scRNA = np.vstack((scRNA,gene))

        # Compute summary statistics
        num_genes = int(np.shape(scRNA)[0])
        raw_cell_number = int(np.shape(DData)[1])
        if j == 0:
            cell_number = int(np.shape(scRNA)[0])
            num_genes = 1
        else:
            num_genes = int(np.shape(scRNA)[0])
            cell_number =  int(np.shape(scRNA)[1])  # Only count non-NaN entries

        if num_genes >= threshold_gc[1] and num_genes < threshold_gc[0] and cell_number >= threshold_gc[2]:
            for j, nn in enumerate(NoNA):
                if nn==i:
                    INDEX.extend([index[j]]) 

    samples, grid_name, y_data = load_ATDFILE(nogenenameNNcsvpath,prior=prior,meanBeta=meanBeta,beta_fixed=beta_fixed)

    y_pred = y_data[:,:3]
    ratio_pred = y_data[:,9]
    p16 = y_data[:,3:6]
    p84 = y_data[:,6:9]
    r16 = y_data[:,10]
    r84 = y_data[:,11]
    numc = y_data[:,12]
    cc = y_data[:,13]
    r02 = y_data[:,14]
    r97 = y_data[:,15]
    p02 = y_data[:,16:19]
    p97 = y_data[:,19:22]

    summary = np.column_stack((INDEX,y_pred,ratio_pred,p02,p97,r02,r97,p16,p84,r16,r84))

    df = pd.DataFrame(summary)

    df.to_csv(direc+"NN_"+file_name+".csv", index = False)

    print(summary)

#whoiswho("./Larsson_data","df_c57_used.csv")
#whoiswho("./Larsson_data","df_cast_used.csv")

#final_plot()

#prior = "Fano"
#ul = 500
#ll = 2

#meanBeta = 1.00
#beta_fixed = True

#read_data("./Larsson_data","df_c57_used.csv",lr=1e-4,acceptance_level =[0,0.0], convert_data=True,threshold_gc=[ul,ll,100],prior=prior,meanBeta=meanBeta,beta_fixed=beta_fixed)
#read_data("./Larsson_data","df_cast_used.csv",lr=1e-4,acceptance_level =[0,0.0], convert_data=True,threshold_gc=[ul,ll,100],prior=prior,meanBeta=meanBeta,beta_fixed=beta_fixed)
