'''MODELOS M1 - M2 - M3
'''

# Librery
import pandas as pd
from collections import Counter

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline

import numpy as np

import joblib


# function

def voto_count(x):
    '''This function is used to count the classifitions
    of the M# sub models.
    '''
    
    p1 = x.value_counts()
    
    return p1

######################################################## Modelo M1 #####################
def M1(X_IM):

    #### M1
    # Ruta
    rutaM1 = 'Modelos/M1/'

    # load
    KN1 = joblib.load(rutaM1 + 'KN1.joblib')
    DTC1 = joblib.load(rutaM1 + 'DTC1.joblib')
    LSVC1 = joblib.load(rutaM1 + 'LSVC1.joblib')
    RF1 = joblib.load(rutaM1 + 'RF1.joblib')

    KN1_os = joblib.load(rutaM1 + 'KN1_os.joblib')
    DTC1_os = joblib.load(rutaM1 + 'DTC1_os.joblib')
    LSVC1_os = joblib.load(rutaM1 + 'LSVC1_os.joblib')
    RF1_os = joblib.load(rutaM1 + 'RF1_os.joblib')

    KN1_smt = joblib.load(rutaM1 + 'KN1_smt.joblib')
    DTC1_smt = joblib.load(rutaM1 + 'DTC1_smt.joblib')
    LSVC1_smt = joblib.load(rutaM1 + 'LSVC1_smt.joblib')
    RF1_smt = joblib.load(rutaM1 + 'RF1_smt.joblib')


    X_KN1 = KN1.predict(X_IM) 
    X_DTC1 = DTC1.predict(X_IM) 
    X_LSVC1 = LSVC1.predict(X_IM)
    X_RF1 = RF1.predict(X_IM)

    X_KN1_os = KN1_os.predict(X_IM) 
    X_DTC1_os = DTC1_os.predict(X_IM) 
    X_LSVC1_os = LSVC1_os.predict(X_IM) 
    X_RF1_os = RF1_os.predict(X_IM) 

    X_KN1_smt = KN1_smt.predict(X_IM) 
    X_DTC1_smt = DTC1_smt.predict(X_IM) 
    X_LSVC1_smt = LSVC1_smt.predict(X_IM) 
    X_RF1_smt = RF1_smt.predict(X_IM)

    ### Cuento los votos y resulta un vetor con la clasifcación por M1
    M1 = np.stack((X_KN1, X_DTC1, X_LSVC1, X_RF1, 
                   X_KN1_os, X_DTC1_os, X_LSVC1_os, X_RF1_os, 
                   X_KN1_smt, X_DTC1_smt, X_LSVC1_smt, X_RF1_smt), axis=1)

    M1 = pd.DataFrame(M1, columns=['KN1', 'DTC1', 'LSVC1', 'RF1', 
                   'KN1_os', 'DTC1_os', 'LSVC1_os', 'RF1_os', 
                   'KN1_smt', 'DTC1_smt', 'LSVC1_smt', 'RF1_smt'])

    P1 = M1.apply(voto_count, axis=1)
    P1 = P1.fillna(0)
    X1 = P1.idxmax(axis=1)
    
    return X1

######################################################## Modelo M2 #####################
def M2(x1):
    '''Input: an element to be classified between D and SD.
    '''

    #### M2
    # Ruta
    rutaM2 = 'Modelos/M2/'

    # load
    KN2 = joblib.load(rutaM2 + 'KN2.joblib')
    DTC2 = joblib.load(rutaM2 + 'DTC2.joblib')
    LSVC2 = joblib.load(rutaM2 + 'LSVC2.joblib')
    RF2 = joblib.load(rutaM2 + 'RF2.joblib')

    KN2_os = joblib.load(rutaM2 + 'KN2_os.joblib')
    DTC2_os = joblib.load(rutaM2 + 'DTC2_os.joblib')
    LSVC2_os = joblib.load(rutaM2 + 'LSVC2_os.joblib')
    RF2_os = joblib.load(rutaM2 + 'RF2_os.joblib')

    KN2_smt = joblib.load(rutaM2 + 'KN2_smt.joblib')
    DTC2_smt = joblib.load(rutaM2 + 'DTC2_smt.joblib')
    LSVC2_smt = joblib.load(rutaM2 + 'LSVC2_smt.joblib')
    RF2_smt = joblib.load(rutaM2 + 'RF2_smt.joblib')


    X_KN2 = KN2.predict(x1) 
    X_DTC2 = DTC2.predict(x1) 
    X_LSVC2 = LSVC2.predict(x1)
    X_RF2 = RF2.predict(x1)

    X_KN2_os = KN2_os.predict(x1) 
    X_DTC2_os = DTC2_os.predict(x1) 
    X_LSVC2_os = LSVC2_os.predict(x1) 
    X_RF2_os = RF2_os.predict(x1) 

    X_KN2_smt = KN2_smt.predict(x1) 
    X_DTC2_smt = DTC2_smt.predict(x1) 
    X_LSVC2_smt = LSVC2_smt.predict(x1) 
    X_RF2_smt = RF2_smt.predict(x1)

    ### Cuento los votos y resulta un vetor con la clasifcación por M2
    M2 = np.stack((X_KN2, X_DTC2, X_LSVC2, X_RF2, 
                   X_KN2_os, X_DTC2_os, X_LSVC2_os, X_RF2_os, 
                   X_KN2_smt, X_DTC2_smt, X_LSVC2_smt, X_RF2_smt), axis=1)

    M2 = pd.DataFrame(M2, columns=['KN2', 'DTC2', 'LSVC2', 'RF2', 
                   'KN2_os', 'DTC2_os', 'LSVC2_os', 'RF2_os', 
                   'KN2_smt', 'DTC2_smt', 'LSVC2_smt', 'RF2_smt'])

    P2 = M2.apply(voto_count, axis=1)
    P2 = P2.fillna(0)
    x2 = P2.idxmax(axis=1)
    
    return M2, x2

######################################################## Modelo M3 #####################
def M3(x1):
    '''Input: an element to be classified between D and SD.
    '''

    #### M3
    # Ruta
    rutaM3 = 'Modelos/M3/'

    # load
    KN3 = joblib.load(rutaM3 + 'KN3.joblib')
    DTC3 = joblib.load(rutaM3 + 'DTC3.joblib')
    LSVC3 = joblib.load(rutaM3 + 'LSVC3.joblib')
    RF3 = joblib.load(rutaM3 + 'RF3.joblib')

    KN3_os = joblib.load(rutaM3 + 'KN3_os.joblib')
    DTC3_os = joblib.load(rutaM3 + 'DTC3_os.joblib')
    LSVC3_os = joblib.load(rutaM3 + 'LSVC3_os.joblib')
    RF3_os = joblib.load(rutaM3 + 'RF3_os.joblib')

    KN3_smt = joblib.load(rutaM3 + 'KN3_smt.joblib')
    DTC3_smt = joblib.load(rutaM3 + 'DTC3_smt.joblib')
    LSVC3_smt = joblib.load(rutaM3 + 'LSVC3_smt.joblib')
    RF3_smt = joblib.load(rutaM3 + 'RF3_smt.joblib')


    X_KN3 = KN3.predict(x1) 
    X_DTC3 = DTC3.predict(x1) 
    X_LSVC3 = LSVC3.predict(x1)
    X_RF3 = RF3.predict(x1)

    X_KN3_os = KN3_os.predict(x1) 
    X_DTC3_os = DTC3_os.predict(x1) 
    X_LSVC3_os = LSVC3_os.predict(x1) 
    X_RF3_os = RF3_os.predict(x1) 

    X_KN3_smt = KN3_smt.predict(x1) 
    X_DTC3_smt = DTC3_smt.predict(x1) 
    X_LSVC3_smt = LSVC3_smt.predict(x1) 
    X_RF3_smt = RF3_smt.predict(x1)

    ### Cuento los votos y resulta un vetor con la clasifcación por M3
    M3 = np.stack((X_KN3, X_DTC3, X_LSVC3, X_RF3, 
                   X_KN3_os, X_DTC3_os, X_LSVC3_os, X_RF3_os, 
                   X_KN3_smt, X_DTC3_smt, X_LSVC3_smt, X_RF3_smt), axis=1)

    M3 = pd.DataFrame(M3, columns=['KN3', 'DTC3', 'LSVC3', 'RF3', 
                   'KN3_os', 'DTC3_os', 'LSVC3_os', 'RF3_os', 
                   'KN3_smt', 'DTC3_smt', 'LSVC3_smt', 'RF3_smt'])

    P3 = M3.apply(voto_count, axis=1)
    P3 = P3.fillna(0)
    x3 = P3.idxmax(axis=1)
    
    return M3, x3