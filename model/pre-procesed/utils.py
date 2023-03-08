''' Funciones to:
- Filtros: Para seleccionar los features con mayor entropia
- Para gráficar matrcies de confusión.
'''


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_selection import SelectKBest  #Se usa con test statical y select # of the better features


from sklearn.model_selection import train_test_split,GridSearchCV, RandomizedSearchCV
from sklearn.metrics import plot_roc_curve,classification_report,plot_confusion_matrix,plot_roc_curve,accuracy_score,f1_score,precision_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import sklearn

from termcolor import colored
from pprint import pprint
from time import time
from astropy.table import Table
import os
import fnmatch


def Filtros(filtro, numero_features, data_scala, target, columnas):
    '''Determina los Features más informativos y tiene como salida el modelo ajustado y 
    una dataframe con los score.
    '''
    
    # Define feature selection
    fs = SelectKBest(score_func=filtro, k=numero_features)
    
    # Ajusto el modelo
    #df_data_scala = pd.DataFrame(data_scala).copy()
    fs_fit = fs.fit(data_scala, target)

    #Observo los puntajes que tuvieron los features
    df_scores = pd.DataFrame(fs_fit.scores_) 

    #las columnas del dataframe
    df_columns = pd.DataFrame(columnas) 

    #concat two dataframes for better visualization 
    df_Final = pd.concat([df_columns, df_scores], axis=1)

    #naming the dataframe columns
    df_Final.columns = ['columnas','Score'] 
    
    #organizo los primeros
    df_top =  df_Final.sort_values('Score', ascending=False).iloc[0:numero_features, :].copy()
    
    return fs, df_top
    



########## Plots


mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_loss(history, label, n):
    # Use a log scale on y-axis to show the wide range of values.
    plt.semilogy(history.epoch, history.history['loss'],
                 color=colors[n], label='Train ' + label)
    plt.semilogy(history.epoch, history.history['val_loss'],
                 color=colors[n], label='Val ' + label,
                 linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

def plot_metrics(history):
    metrics = ['loss', 'prc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])

        plt.legend()

def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    print('No-Galaxies Detected (True Negatives): ', cm[0][0])
    print('No-Galaxies Incorrectly Detected (False Positives): ', cm[0][1])
    print('Galaxies Missed (False Negatives): ', cm[1][0])
    print('Galaxies Detected (True Positives): ', cm[1][1])
    print('Total Galaxies: ', np.sum(cm[1]))

def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)
    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5,60])
    plt.ylim([0,100.0])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')

def plot_prc(name, labels, predictions, **kwargs):
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions)

    plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


def plot_confusion_matrix(y_true,y_pred,oldFashioned=False):
    np.set_printoptions(precision=2)
    
    if oldFashioned:
        TP = (y_true == y_pred) & (y_true == 0)
        FP = (y_true != y_pred) & (y_true == 0)
        TN = (y_true == y_pred) & (y_true == 1)
        FN = (y_true != y_pred) & (y_true == 1)
        
        totalP = sum(y_true == 0)
        totalN = sum(y_true == 1)
        
        confusion = np.array([[sum(TP),sum(FP)],[sum(FN),sum(TN)]])
        norm_confusion = np.array([[sum(TP)/totalP,sum(FP)/totalP],[sum(FN)/totalN,sum(TN)/totalN]])
    else:
        y_real = y_true.values
        
        TP = (y_real == y_pred) & (y_real == 1)
        TN = (y_real == y_pred) & (y_real == 0)
        FN = (y_real != y_pred) & (y_real == 1)
        FP = (y_real != y_pred) & (y_real == 0)
        FullP = sum(y_real == 1)
        FullN = sum(y_real == 0)
    
        confusion = np.array([[sum(TN),sum(FP)],[sum(FN),sum(TP)]])
        norm_confusion = np.array([[sum(TN)/FullN,sum(FP)/FullN],[sum(FN)/FullP,sum(TP)/FullP]])
    
    
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.matshow(norm_confusion,cmap='Purples',vmin=0.0,vmax=1.0)
    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(['no-galaxy','galaxy'],size=16)
    ax.set_yticklabels(['no-galaxy','galaxy'],size=16,rotation=90)
    ax.set_ylabel('Actual',size=16)
    ax.set_xlabel('Prediction',size=16)
    ax.xaxis.set_label_position('top') 
    
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, "%d\n%.2f" % (confusion[i, j],norm_confusion[i,j]),
                           ha="center", va="center", color="k",backgroundcolor=[1,1,1,0.2],size=14)
    
    fig.tight_layout()
    plt.colorbar(im,ax=ax,shrink=0.8)

    
def plot_confusion_matrix_3(y_true, y_pred):
    np.set_printoptions(precision=2)
    '''- detached -> 0
       - contact -> 1
       - semi -> 2
        La matriz de confusión con entrada fila i y columna j indica el número de 
        muestras cuya etiqueta verdadera es de clase i y la etiqueta predicha es de clase j.
        La misma está normalizada sobre las condiciones verdaderas, es decir, las filas de la matriz.'''
        
    TD = 0
    FC_D = 0
    FSD_D = 0 #(y_ture == 0) & (y_pred == 2)
    
    FD_C = 0 #(y_ture == 1) & (y_pred == 0)
    TC = 0
    FSD_C = 0
    
    FD_SD = 0
    FC_SD = 0 #(y_ture == 2) & (y_pred == 1)
    TSD = 0
    
    for i in range(len(y_true)):
        
        # Fila Detached
        if (y_true[i] == 0) & (y_pred[i] == 0):
            TD = TD + 1
        elif (y_true[i] == 0) & (y_pred[i] == 1):
            FC_D = FC_D + 1
        elif (y_true[i] == 0) & (y_pred[i] == 2):
            FSD_D = FSD_D + 1

        # Fila Contac 
        elif (y_true[i] == 1) & (y_pred[i] == 0):
            FD_C = FD_C + 1
        elif (y_true[i] == 1) & (y_pred[i] == 1):
            TC = TC + 1
        elif (y_true[i] == 1) & (y_pred[i] == 2):
            FSD_C = FSD_C + 1

        # Fila SemiDetached
        elif (y_true[i] == 2) & (y_pred[i] == 0):
            FD_SD = FD_SD + 1
        elif (y_true[i] == 2) & (y_pred[i] == 1):
            FC_SD = FC_SD + 1
        else: # (y_true == 2) & (y_pred == 2)
            TSD = TSD + 1

    # suma por fila 
    total_D = TD + FC_D + FSD_D
    total_C = FD_C + TC + FSD_C
    total_SD = FD_SD + FC_SD + TSD
    
    confusion = np.array([[TD, FC_D, FSD_D], [FD_C, TC, FSD_C], [FD_SD, FC_SD, TSD]])
    norm_confusion = np.array([[TD/total_D, FC_D/total_D, FSD_D/total_D], 
                               [FD_C/total_C, TC/total_C, FSD_C/total_C], 
                               [FD_SD/total_SD, FC_SD/total_SD, TSD/total_SD]])
    
    
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.matshow(norm_confusion, cmap='Oranges', vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(['D', 'C', 'SD'], size=14)
    ax.set_yticklabels(['D', 'C', 'SD'], size=14, rotation=90)
    ax.set_ylabel('Real', size=16)
    ax.set_xlabel('Prediction',size=16)
    ax.xaxis.set_label_position('top') 
    
    for i in range(3):
        for j in range(3):
            text = ax.text(j, i, "%d\n%.2f" % (confusion[i, j], norm_confusion[i,j]),
                           ha="center", va="center", color="k", backgroundcolor=[1,1,1,0.2], size=14)
    
    fig.tight_layout()
    plt.colorbar(im, ax=ax, shrink=0.8)
    
def report(results, n_top=3):
    """
    Utility function to report
    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        candidate = candidates[0]
        print("Model with rank: {0}".format(i))
        print("Mean validation score: {0:.3f} (std: {1:.3f})"
              .format(results['mean_test_score'][candidate],
                      results['std_test_score'][candidate]))
        print("Parameters: {0}".format(results['params'][candidate]))
        print("")
    
def search_params(model,param_dist,X_train,y_train,X_test,y_test,scores = ['precision', 'recall', 'f1'],n_iter=100):
    for score in scores:
        print(colored("# Tuning hyper-parameters for %s" % score, 'green', attrs=['bold']))
        print()
        
        clf = RandomizedSearchCV(model, param_distributions=param_dist,n_iter=n_iter,cv=5, scoring='%s_micro' % score, n_jobs=-1,random_state=42,verbose=2)
        
        start = time()
        clf.fit(X_train, y_train)
        print("RandomizedSearchCV took %.2f seconds for %d candidates parameter settings." % ((time() - start), n_iter))
        
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))

    return(clf)

def expand_flux_radius_column(df):
    df[['FLUX_RADIUS_0','FLUX_RADIUS_1','FLUX_RADIUS_2']] = df['FLUX_RADIUS'].apply(lambda x: x.replace('(','').replace(')','')).str.split(',',expand=True).astype(float)
    
def compute_colors(df):
    df['colorJH']  = df['MAG_APER_J_CORR'] - df['MAG_APER_H_CORR']
    df['colorHKs'] = df['MAG_APER_H_CORR'] - df['MAG_APER_KS_CORR']
    df['colorJKs'] = df['MAG_APER_J_CORR'] - df['MAG_APER_KS_CORR']
    return(True)
    
def load_galaxy_table(filename):
    data = pd.read_csv(filename)
    data.set_index('ID',inplace=True)
    return(data)
    
def load_galaxy_table_all(folder="../../../Data/VVV/Galaxias_por_tile/",pattern="d???_galaxies.csv"):
    listOfFiles = os.listdir(folder)
    data = pd.DataFrame()
    for entry in listOfFiles:
        if fnmatch.fnmatch(entry,pattern):
            data = data.append(load_galaxy_table(folder+entry))
    data = data[~data.index.duplicated()]
    return(data)

def load_extragalactic_table(filename):
    df = Table.read(filename, format='fits')
    names = [name for name in df.colnames if len(df[name].shape) <= 1]
    data = df[names].to_pandas()
    data[['FLUX_RADIUS_0','FLUX_RADIUS_1','FLUX_RADIUS_2']] = pd.DataFrame(df['FLUX_RADIUS'].tolist())
    data['ID'] = data['ID'].str.decode(encoding = 'ASCII')
    data.set_index('ID',inplace=True)
    return(data)

def load_extragalactic_table_all(folder="../../../Data/VVV/Extragalactic/",pattern = "d???_extragalactic.fits"):
    listOfFiles = os.listdir(folder)
    data = pd.DataFrame()
    for entry in listOfFiles:
        if fnmatch.fnmatch(entry, pattern):
            data = data.append(load_extragalactic_table(folder+entry))
    data = data[~data.index.duplicated()]
    return(data)
    
def load_data_old():
    filename = "data_115.csv"
    data_115 = pd.read_csv(filename,delim_whitespace=False)
    filename = "data_077.csv"
    data_077 = pd.read_csv(filename,delim_whitespace=False)
    filename = "data_078.csv"
    data_078 = pd.read_csv(filename,delim_whitespace=False)
    filename = "data_116.csv"
    data_116 = pd.read_csv(filename,delim_whitespace=False)
    data = pd.concat([data_115,data_116,data_077,data_078])
    return(data[columns])


def load_extragalactic_table_VVVx(filename):
    data = Table.read(filename, format='fits')

    one_value_columns = [name for name in data.colnames if len(data[name].shape) <= 1]
    multi_values_coulumns = [name for name in data.colnames if len(data[name].shape) > 1]

    _data = data[one_value_columns].to_pandas()

    _data[['MAG_APER_KS_0','MAG_APER_KS_1','MAG_APER_KS_2','MAG_APER_KS_3']] = pd.DataFrame(data['MAG_APER'].tolist())
    _data[['MAGERR_APER_KS_0','MAGERR_APER_KS_1','MAGERR_APER_KS_2','MAGERR_APER_KS_3']] = pd.DataFrame(data['MAGERR_APER'].tolist())
    _data[['FLUX_RADIUS_0','FLUX_RADIUS_1','FLUX_RADIUS_2']] = pd.DataFrame(data['FLUX_RADIUS'].tolist())

    _data.rename(columns = {'MAG_APER_KS_0': 'MAG_APER_KS','MAG_APER_KS_0_CORR': 'MAG_APER_KS_CORR', 'MAGERR_APER_KS_0': 'MAGERR_APER_KS'}, inplace = True)
    _data.rename(columns = {'MAG_APER_KS': 'MAG_APER'}, inplace = True)
    _data.rename(columns = {'MAG_APER_J_0': 'MAG_APER_J', 'MAG_APER_J_0_CORR': 'MAG_APER_J_CORR', 'MAGERR_APER_J_0': 'MAGERR_APER_J'}, inplace = True)
    _data.rename(columns = {'MAG_APER_H_0': 'MAG_APER_H', 'MAG_APER_H_0_CORR': 'MAG_APER_H_CORR', 'MAGERR_APER_H_0': 'MAGERR_APER_H'}, inplace = True)

    _data['ID'] = _data['ID'].str.decode(encoding = 'ASCII')
    _data.set_index('ID',inplace=True)

    #create_ID(_data)
    return(_data)


def create_ID(df,alpha_col='ALPHA_J2000',delta_col='DELTA_J2000'):
    df['ID_match'] = ""
    for j,s in df.iterrows():
        delta = s[delta_col]
        alpha = "%010.6f" % s[alpha_col]
        if delta > 0.0:
            delta = "+%010.6f" % delta
        else:
            delta = "%010.6f" % delta
        ID = alpha + delta
        df.loc[j,'ID_match'] = ID

def load_galaxy_table_VVVx(filename):
    data = pd.read_csv(filename,delim_whitespace=True,usecols=['ALPHA_J2000','DELTA_J2000','Galaxy'])
    create_ID(data)
    return(data)
