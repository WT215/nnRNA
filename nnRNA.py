import pandas as pd
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import metrics
import numpy as np
from sklearn.metrics import r2_score
import tensorflow.keras
import elfi
import pygtc
import pickle
from tensorflow.keras.layers import Dropout, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf
import matplotlib
import matplotlib.cm as cm
import seaborn as sns
import pygtc
import sim
import os.path
from tensorflow.keras.callbacks import EarlyStopping

#========================================================================

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

#========================================================================

#grid_name="D:/RNAseqProject/Julia_allele/Davie_2018/Data/for_nnRNA/Results/test_SumStat_scRNA_Genes_55865_Cells_179.csv"
'''
def load_data(grid_name,acceptance_level=[0,0],beta_fixed=False):
    df = pd.read_csv(grid_name)
    df = df.dropna()
    data = df.to_numpy()
    y = np.log10(data[:,0:3])

    X = data[:,[3,4,5,6,7,8,9,11,12,13,14,15,16,17,18]]
#    X = data[:,[3,11,14,15,16,17,18]]

    cc = 10**data[:,-1]
    dl = data[:,-4]
    print("Cleaning data")
    genes = len(X)
    empty = 0
    X_clean = []
    y_clean = []

    for i in range(genes):
        if cc[i] <= acceptance_level[0] or dl[i] >= 1- acceptance_level[1]: 
            empty +=1
        else:
            if len(X_clean) == 0:
                X_clean = X[i,:]
                y_clean = y[i,:]
            else:
                X_clean = np.vstack((X_clean,X[i,:]))
                y_clean = np.vstack((y_clean,y[i,:]))

    print("%i out of %i data points were discarded as they provided no information (no genes were registered)" %(empty, genes))

    return X_clean, y_clean
'''


def load_data(grid_name,acceptance_level=[0,0],beta_fixed=False):
    df = pd.read_csv(grid_name)
    df = df.dropna()
    data = df.to_numpy()
    y = np.log10(data[:,0:3])

    X = data[:,[3,4,5,6,7,8,9,11,12,13,14,15,16,17,18]]
#    X = data[:,[3,11,14,15,16,17,18]]

    cc = 10**data[:,-1]
    dl = data[:,-4]
    print("Cleaning data")
    genes = len(X)
    empty = 0
    X_clean = []
    y_clean = []
    
    
    used_ind=np.where((cc > acceptance_level[0]) & (dl < 1- acceptance_level[1]))[0]
    X_clean = X[used_ind,:]
    y_clean = y[used_ind,:]


    return X_clean, y_clean

#========================================================================

def explore(grid_name,acceptance_level=[0,0],beta_fixed=False):
    X,y = load_data(grid_name,acceptance_level=acceptance_level,beta_fixed=beta_fixed)
    dimx = int(np.shape(X)[1])
    for i in range(dimx):
        plt.figure()
        plt.plot(y[:,1],X[:,i],'.',alpha=0.4,label=str(i))
        plt.legend()
    plt.show()

#========================================================================

def standardize(grid_name,num_genes,acceptance_level=[0,0],beta_fixed=False,repeats=1):

    X_train, y_train = load_data(grid_name,acceptance_level=acceptance_level,beta_fixed=beta_fixed)
    small_grids = grid_name.replace("_repeats_"+str(repeats),"_repeats_"+str(1))
    small_grids = small_grids.replace("Genes_"+str(num_genes),"Genes_"+str(int(num_genes*0.2*repeats)))
    X_val, y_val = load_data("val_"+small_grids,acceptance_level=acceptance_level,beta_fixed=beta_fixed)
    X_test, y_test = load_data("test_"+small_grids,acceptance_level=acceptance_level,beta_fixed=beta_fixed)

#    X, y = load_data(grid_name,acceptance_level=acceptance_level,beta_fixed=beta_fixed)

#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X_val = sc.transform(X_val)

#    scy = MinMaxScaler() 
    scy = StandardScaler()
    y_train = scy.fit_transform(y_train)
    y_test = scy.transform(y_test)
    y_val = scy.transform(y_val)

    nrt = np.zeros(len(y_train))
    nrv = np.zeros(len(y_val))
    dimy = np.shape(y_train)[1]
    for i in range(dimy):
        y_train = np.column_stack((y_train,nrt))
        y_val = np.column_stack((y_val,nrv))

    save_object(sc, "infe_sc_"+grid_name+".pkl")
    save_object(scy, "infe_scy_"+grid_name+".pkl")

    return X_train, X_val, X_test, y_train, y_val, y_test, sc, scy

#========================================================================

# aleatoric loss function
# "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning" by Y. Gal and Z. Ghahramani.
# "Evaluating Scalable Uncertainty Estimation Methods for DNN-Based Molecular Property Prediction" by G. Scalia et al.
# "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" Kendall & Gal
# Implementation is furthermore based on the code example by Michel Kana's blog "Uncertainty in Deep learning. How to measure?" to be found on 
# https://towardsdatascience.com/my-deep-learning-model-says-sorry-i-dont-know-the-answer-that-s-absolutely-ok-50ffa562cb0b

def aleatoric_loss(y_true, y_pred):
    dimy = int(np.shape(y_true)[1]/2)
    se = K.pow((y_true[:,:dimy]-y_pred[:,:dimy]),2)
    inv_std = K.exp(-y_pred[:,dimy:])
    mse = K.mean(K.batch_dot(inv_std,se))
    reg = K.mean(y_pred[:,dimy:])
    return 0.5*(mse + reg)

#========================================================================

def neunet(grid_name,num_genes,neurons=100,layers=3,lr=1e-4, dropout_rate=0.2,epochs=5000,training=True,acceptance_level=[0,0],beta_fixed=False,show_plot=True,repeats=1,min_delta=0.01,patience=250):

    X_train, X_val, X_test, y_train, y_val, y_test, sc, scy = standardize(grid_name,num_genes,acceptance_level=acceptance_level,beta_fixed=beta_fixed,repeats=repeats)

    dimx = int(np.shape(X_train)[1])
    dimy = int(np.shape(y_train)[1]/2)

    inputs = Input(shape=(dimx,))
    hl = Dense(100, kernel_initializer='uniform', activation='relu')(inputs)
    
    
    #add WT
    #strategy = tf.distribute.MirroredStrategy()
    #with strategy.scope():
    
    
    for i in range(layers):
        hl = Dense(neurons, kernel_initializer='uniform', activation='relu')(hl)
        hl = Dropout(rate = dropout_rate)(hl, training=training)
    outputs = Dense(2*dimy, kernel_initializer='uniform')(hl)
    model = Model(inputs, outputs)

    opt = tensorflow.keras.optimizers.Adam(learning_rate=lr,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-09,
            )
    model.compile(loss=aleatoric_loss, optimizer=opt, metrics=['accuracy', metrics.MeanSquaredError()])

    print(model.summary())

    # Early stopping callback
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience, min_delta=min_delta)

    history = model.fit(X_train, y_train, batch_size=min([250,int(len(X_train)/3)]), epochs = epochs, callbacks=[es], shuffle=True, validation_data=(X_val, y_val), use_multiprocessing=True)

    print("The neural network has converged")

    # loss: accuracy: mean_squared_error: val_loss: val_accuracy: val_mean_squared_error

    plt.figure()
    plt.title('Loss')
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(grid_name[:-4]+"_valloss.png",bbox_inches='tight')

    plt.figure()
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.figure()
    plt.title('Mean Squared Error')
    plt.plot(history.history['mean_squared_error'], label='mean_squared_error')
    plt.plot(history.history['val_mean_squared_error'], label='val_mean_squared_error')
    plt.ylabel("MSE")
    plt.xlabel("Epoch")
    plt.legend()
    if show_plot:
        plt.show()

    y_pred = scy.inverse_transform(model.predict(X_val)[:,:dimy])

    y_val = scy.inverse_transform(y_val[:,:dimy])

    for i in range(dimy):
        plt.figure()
        plt.plot(10**y_val[:,i],10**y_val[:,i],'r.')
        plt.plot(10**y_val[:,i],10**y_pred[:,i],'ko',alpha=0.4)
        plt.yscale("log")
        plt.xscale("log")
        plt.figure(2*dimy+10)
        y = abs(y_pred[:,i] - y_val[:,i])/np.max(abs(y_pred[:,i] - y_val[:,i]))
        plt.plot(y_val[:,i]/np.max(y_val[:,i]),y,'o')

    plt.savefig(grid_name[:-4]+"_explvar.png",bbox_inches='tight')

    print("R2-score:", r2_score(y_val, y_pred[:,:dimy]))
    print("Acceptance levels", acceptance_level)

    model.save("infe_model_"+grid_name+".h5")

    if show_plot:
        plt.show()
    else:
        plt.close('all')

    save_object(X_test, "infe_X_test_"+grid_name+".pkl")
    save_object(y_test, "infe_y_test_"+grid_name+".pkl")

    return model, sc.inverse_transform(X_test), y_test, y_pred

#========================================================================

def make_2d_contour(samples,truths):

    truths = list(truths)
    truths.extend([truths[2]-truths[1]])

    names = ['$k_\mathrm{on}$',
             '$k_\mathrm{off}$',
             '$k_\mathrm{syn}$',
             '$k_\mathrm{syn}/k_\mathrm{off}$',
            ]

    plotName = "./CONTOURS/Test_kon_"+str(truths[0])+"_koff_"+str(truths[1])+"_ksyn_"+str(truths[2])+".png"

    GTC = pygtc.plotGTC(chains=[samples],
    figureSize='MNRAS_page',
    truths = truths,
    paramNames=names,
    plotName=plotName,
    )

    plt.close()

#========================================================================

def testing(grid_name,num_cells,contour2d=False,beta_fixed=False,meanBeta=0.06,med=False):

    print("Testing")

    model = tensorflow.keras.models.load_model("infe_model_"+grid_name+".h5", compile=False)

    with open("infe_scy_"+grid_name+".pkl", 'rb') as run:
        scy = pickle.load(run)
    with open("infe_sc_"+grid_name+".pkl", 'rb') as run:
        sc = pickle.load(run)
    with open("infe_X_test_"+grid_name+".pkl", 'rb') as run:
        X_test = pickle.load(run)
    with open("infe_y_test_"+grid_name+".pkl", 'rb') as run:
        y_test = pickle.load(run)

    dimy = int(np.shape(y_test)[1])

    y_test = scy.inverse_transform(y_test[:,:dimy])

    genes = len(X_test)
    samples = 1000

    y_pred = []
    p16 = []
    p84 = []
    ratio_pred = []
    r16 = []
    r84 = []

    for i in range(genes):
        print("Running %f per cent" % ((100*float(i)+1.)/genes), end="\r")
        x = X_test[i,:]
        y = []
        for j in range(samples-1):
            x = np.vstack((x,X_test[i,:]))
        y = scy.inverse_transform(model.predict(x)[:,:dimy])

        yr = y[:,2]-y[:,1]
        
        if contour2d:
            make_2d_contour(np.column_stack((y,yr)),y_test[i,:])

        if med:
            r1 = np.median(yr)
        else:
            r1 = np.mean(yr)
        r2 = np.percentile(yr,16)
        r3 = np.percentile(yr,84)

        if med:
            m1 = np.median(y,axis=0)
        else:
            m1 = np.mean(y,axis=0)
        m2 = np.percentile(y,16,axis=0)
        m3 = np.percentile(y,84,axis=0)
        if i == 0:
            y_pred = m1
            p16 = m2
            p84 = m3
            ratio_pred = r1
            r16 = r2
            r84 = r3
        else:
            y_pred = np.vstack((y_pred,m1))
            p16 = np.vstack((p16,m2))
            p84 = np.vstack((p84,m3))
            ratio_pred = np.vstack((ratio_pred,r1))
            r16 = np.vstack((r16,r2))
            r84 = np.vstack((r84,r3))

    # Starting plots
    fig = plt.figure(figsize=(18,10))

    # Preparing input for box plots.
    ratio_true = y_test[:genes,2] - y_test[:genes,1]
    truth = np.asarray(list(y_test[:genes,0])+list(y_test[:genes,1])+list(y_test[:genes,2])+list(ratio_true))
    pred = np.asarray(list(y_pred[:,0])+list(y_pred[:,1])+list(y_pred[:,2])+list(np.concatenate(ratio_pred)))
    MMD = pred-truth
    sigmas = (np.asarray(list(p84[:,0])+list(p84[:,1])+list(p84[:,2])+list(np.concatenate(r84))) - np.asarray(list(p16[:,0])+list(p16[:,1])+list(p16[:,2])+list(np.concatenate(r16))))/2.
    RMMD = MMD/sigmas
    param = [r"$k_\mathrm{on}$"]*genes
    param.extend([r"$k_\mathrm{off}$"]*genes)
    param.extend([r"$k_\mathrm{syn}$"]*genes)
    param.extend([r"$k_\mathrm{syn}/k_\mathrm{off}$"]*genes)
    method = [""]*(dimy+1)*genes

    df = {"Parameter": param,
            "MMD" : MMD,
          "method": method,
          "RMMD"  : RMMD,
        }

    df = pd.DataFrame(df)

    # Box plot showing logarithmic error
    plt.subplot(2, dimy, dimy+2)
    plt.axhline(y=0,linestyle="--",color="grey")
    ax = sns.boxplot(x="method", y="MMD", hue="Parameter",
                    data=df, palette="Set3")
    plt.axhline(y=1,linestyle=":",color="grey")
    plt.axhline(y=-1,linestyle=":",color="grey")
    plt.legend(loc='upper right')
    plt.xlabel("")
    plt.ylabel(r"$\log_{10}(\theta_\mathrm{pred})-\log_{10}(\theta_\mathrm{o})$",fontsize=13)
    plt.xticks(size=13)
    plt.yticks(size=13)

    # Box plot showing deviation in sigma
    plt.subplot(2, dimy, dimy+3)
    plt.axhline(y=0,linestyle="--",color="grey")
    plt.axhline(y=1,linestyle=":",color="grey")
    plt.axhline(y=-1,linestyle=":",color="grey")
    ax = sns.boxplot(x="method", y="RMMD", hue="Parameter",
                    data=df, palette="Set3")
    plt.ylim(-5,5)
    plt.legend(loc='upper right')
    plt.xlabel("")
    plt.ylabel(r"$\log_{10}(\theta_\mathrm{pred})-\log_{10}(\theta_\mathrm{o}) \,\, \mathrm{[\sigma]}$",fontsize=13)
    plt.xticks(size=13)
    plt.yticks(size=13)

    # Scatter plots for kon, koff, and ksyn
    x_label = [r"$\mathrm{True} \, \log_{10}(k_\mathrm{on})$",r"$\mathrm{True} \, \log_{10}(k_\mathrm{off})$",r"$\mathrm{True} \, \log_{10}(k_\mathrm{syn})$"]
    y_label = [r"$\mathrm{Pred.} \, \log_{10}(k_\mathrm{on})$",r"$\mathrm{Pred.} \, \log_{10}(k_\mathrm{off})$",r"$\mathrm{Pred.} \, \log_{10}(k_\mathrm{syn})$"]

    X_test = sc.inverse_transform(X_test)

    if beta_fixed:
        ext = "\, fixed"
    else:
        ext = ""

    Title = r"$\mathrm{Neural \, Network \,("+str(int(num_cells))+"\, cells, \, beta = "+str(meanBeta)+ext+")}$"

    alpha = 0.5

    for i in range(dimy):
        err = np.vstack((p16[:,i],p84[:,i]))
        plt.subplot(2, dimy, i+1)
        x1 = sorted(y_test[:genes,i])
        cc = X_test[:genes,-4]
        if i == 0:
            fig.suptitle(Title, fontsize=16)

        #convert time to a color tuple using the colormap used for scatter
        norm = matplotlib.colors.Normalize(vmin=min(cc), vmax=max(cc), clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='viridis')
        y_color = np.array([(mapper.to_rgba(v)) for v in cc])

        #loop over each data point to plot
        for x, y, el, eu, color in zip(y_test[:genes,i], y_pred[:genes,i], p16[:genes,i],p84[:genes,i], y_color):
            plt.errorbar(x, y, yerr = np.vstack(([y-el],[eu-y])), lw=1, capsize=3, color=color, alpha = alpha)
        plt.plot(x1,x1,'r--',zorder=0)
        plt.scatter(y_test[:genes,i],y_pred[:,i],c=cc,zorder=2,alpha=alpha,cmap='viridis')
#        plt.yscale("log")
#        plt.xscale("log")
        plt.xlabel(x_label[i],fontsize=13)
        plt.ylabel(y_label[i],fontsize=13)
        plt.xticks(size=13)
        plt.yticks(size=13)

    # Plot for ratio of ksyn to koff
    plt.subplot(2, dimy, dimy+1)
    x1 = sorted(ratio_true)
    plt.plot(x1,x1,'r--',zorder=0)
    for x, y, el, eu, color in zip(ratio_true, ratio_pred, r16,r84, y_color):
            plt.errorbar(x, y, yerr = np.vstack(([y-el],[eu-y])), lw=1, capsize=3, color=color, alpha = alpha)
    plt.scatter(ratio_true,ratio_pred,c=cc,zorder=2,alpha=alpha,cmap='viridis')
#    plt.yscale("log")
#    plt.xscale("log")
    plt.xlabel(r"$\mathrm{True}\,k_\mathrm{syn}/k_\mathrm{off}$",fontsize=13)
    plt.ylabel(r"$\mathrm{Pred.}\,k_\mathrm{syn}/k_\mathrm{off}$",fontsize=13)
    plt.xticks(size=13)
    plt.yticks(size=13)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
    cbar = plt.colorbar()
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(13)
    cbar.ax.get_yaxis().labelpad = 10
    t = r"$\mathrm{Dropout\,\, rate}$"
    cbar.ax.set_ylabel(t, rotation=90,fontsize=13)

    print("R2-score:", r2_score(y_test[:genes,:], y_pred))

    plt.savefig(grid_name[:-4]+"_test.png",bbox_inches='tight')

#========================================================================

def execute(num_genes,num_cells,train=True,exploration=False,test=True,allele_double=False,meanBeta=0.06,show_plot=False,med=False,contour2d=False,beta_fixed=False,beta_loc=2.74,beta_scale=0.39,neurons=100,layers=3,lr=1e-4, dropout_rate=0.2,epochs=5000,training=True,acceptance_level=[0,0],repeats=1, min_delta=0.01,patience=250,prior="Fano",inputbeta_vec=None):

    name = sim.naming(num_genes, num_cells, allele_double=allele_double, beta_fixed=beta_fixed,meanBeta=meanBeta,beta_loc=beta_loc,beta_scale=beta_scale,repeats=repeats,prior=prior)
    grid_name = "SumStat"+name

    print(grid_name)

    if os.path.isfile(grid_name):
        print("Simulations for training found")
    else:
        print("Simulations for training not found. Computing series...")
        sim.run(num_genes, num_cells,allele_double=allele_double,beta_fixed=beta_fixed,meanBeta=meanBeta,beta_loc=beta_loc,beta_scale=beta_scale,repeats=repeats,prior=prior,inputbeta_vec=inputbeta_vec)

    small_grids = grid_name.replace("_repeats_"+str(repeats),"_repeats_"+str(1))
    small_grids = small_grids.replace("Genes_"+str(num_genes),"Genes_"+str(int(num_genes*0.2*repeats)))

    if os.path.isfile("val_"+small_grids):
        print("Simulations for validation found")
    else:
        print("Simulations for validation not found. Computing series...")
        sim.run(int(num_genes*0.2*repeats), num_cells, allele_double=allele_double, beta_fixed=beta_fixed,meanBeta=meanBeta,beta_loc=beta_loc,beta_scale=beta_scale,repeats=1,prefix="val_",prior=prior,inputbeta_vec=inputbeta_vec)

    if os.path.isfile("test_"+small_grids):
        print("Simulations for testing found")
    else:
        print("Simulations for testing not found. Computing series...")
        sim.run(int(num_genes*0.2*repeats), num_cells, allele_double=allele_double, beta_fixed=beta_fixed,meanBeta=meanBeta,beta_loc=beta_loc,beta_scale=beta_scale,repeats=1,prefix="test_",prior=prior,inputbeta_vec=inputbeta_vec)

    if exploration:
        explore(grid_name,acceptance_level=acceptance_level,beta_fixed=beta_fixed)
    if train:
        neunet(grid_name,num_genes,neurons=neurons,layers=layers,lr=lr, show_plot=show_plot, patience=patience,min_delta=min_delta, dropout_rate=dropout_rate,epochs=epochs,training=training,acceptance_level=acceptance_level,beta_fixed=beta_fixed,repeats=repeats)
    if test:
        testing(grid_name,num_cells,contour2d=contour2d,beta_fixed=beta_fixed,meanBeta=meanBeta,med=med)

#======================================================================

def convert(direc, file_name, train_genes=10000, new_NN=True,allele_double=False,repeats=1,prior="Fano", acceptance_level=[0,0.0],meanBeta=0.06,beta_fixed=False,beta_loc=2.74,beta_scale=0.39):
    # Load data
    df = pd.read_csv(direc+"/"+file_name)
    scRNA = df.to_numpy()

    num_genes = int(np.shape(scRNA)[0])
    cell_number = int(np.shape(scRNA)[1])

    inputbeta = sim.empirBETA(scRNA,meanBeta=meanBeta)
    sim.sum_stat(num_genes, cell_number, "_"+str(cell_number)+"_"+file_name, scRNA, np.zeros(shape=(3,num_genes)), inputbeta, repeats=1)

    name = sim.naming(train_genes, cell_number, allele_double, beta_fixed=beta_fixed,meanBeta=meanBeta,beta_loc=beta_loc,beta_scale=beta_scale,repeats=repeats,prior=prior)
    grid_name = "SumStat"+name

    if not os.path.isfile("infe_model_"+grid_name+".h5"):
        nnRNA.execute(train_genes,cell_number,meanBeta=meanBeta,beta_fixed=beta_fixed,beta_loc=beta_loc,beta_scale=beta_scale,acceptance_level=acceptance_level,repeats=repeats,prior=prior)
    else:
        print("Trained model found")

    return cell_number, grid_name

#======================================================================

def predicting(cell_number,grid_name,file_name,acceptance_level=[0,0]):

    model = tensorflow.keras.models.load_model("infe_model_"+grid_name+".h5", compile=False)

    with open("infe_scy_"+grid_name+".pkl", 'rb') as run:
        scy = pickle.load(run)
    with open("infe_sc_"+grid_name+".pkl", 'rb') as run:
        sc = pickle.load(run)

    df = pd.read_csv("SumStat_"+str(cell_number)+"_"+file_name)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.loc[df.isnull().any(axis=1), :] = np.nan
    index = df.index.to_numpy()
    df = df.to_numpy()

    dimy = 3

    X_data = df[:,[3,4,5,6,7,8,9,11,12,13,14,15,16,17,18]]
    counts = X_data[:,-1]
    zcounts = 1-X_data[:,-4]
    mask = 10**counts <= acceptance_level[0]
    mask = np.logical_or(mask, zcounts<=acceptance_level[1])
    X_data[mask,0] = np.nan
    index = index[~np.isnan(X_data[:,0])]
    X_data = X_data[~np.isnan(X_data[:,0]),:]

    X_data = sc.transform(X_data)

    genes = len(X_data)
    samples = 1000

    y_pred = []
    p16 = []
    p84 = []
    ratio_pred = []
    r16 = []
    r84 = []

    for i in range(genes):
        print("Running %f per cent" % ((100*float(i)+1.)/genes), end="\r")
        x = X_data[i,:]
        y = []
        for j in range(samples-1):
            x = np.vstack((x,X_data[i,:]))
        y = scy.inverse_transform(model.predict(x)[:,:dimy])

        yr = y[:,2]-y[:,1]

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

    return np.column_stack((index,y_pred,p16,p84,p02,p97,ratio_pred,r16,r84,r02,r97))

# Example
#meanBeta=0.06
#num_genes = 10000
#num_cells = 5000
#train = True
#explore = False
#test = True
#contour2d = False
#beta_fixed = False
#beta_loc = 2.74
#beta_scale = 0.39
#acceptance_level = [50,0.01]

#execute(num_genes,num_cells,train=train,exploration=exploration,test=test,meanBeta=meanBeta,contour2d=contour2d,beta_fixed=beta_fixed,beta_loc=beta_loc,beta_scale=beta_scale,neurons=100,layers=3,lr=1e-4, dropout_rate=0.2,epochs=5000,acceptance_level=acceptance_level)

