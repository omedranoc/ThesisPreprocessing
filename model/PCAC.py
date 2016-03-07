import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
import pandas as pd
import copy


    
        
        
def getPca(training , test):
    
    
    n=1
    pca = PCA(n_components=n)
    pca.fit(training)
    energy=pca.explained_variance_ratio_.sum() 
    
    while (energy<0.85):
        n=n+1
        pca = PCA(n_components=n)
        pca.fit(training)
        
        energy=pca.explained_variance_ratio_.sum() 
    newTest=pca.transform(test)  
    newTraining=pca.transform(training)
    
    return newTraining, newTest
def pcaf(trainingi,testi):
     
    
    training= copy.deepcopy(trainingi)
    test= copy.deepcopy(testi)
    #trainingm=training.replace([True,False], [1,0])
    print training
    print test.dtypes
    cols=['L']
    try:
        for x in cols:
            del training[x]
            del test[x]
    except:
        pass
    
    trainingm=training.replace(['True' ,'False'], [1,0])
    testr=test.replace(['True','False'], [1,0])
    
    trainm=trainingm.as_matrix(columns=None)
    print testr
    testm=testr.as_matrix(columns=None)  
    arraypcatr,arraypcatest=getPca(trainm, testm)
    M,N= arraypcatr.shape
    size=range(N)
    trainingd = pd.DataFrame(arraypcatr, columns=size)
    testd = pd.DataFrame(arraypcatest, columns=size)
    return trainingd , testd
def pcaf1(X):
    
    m= training= copy.deepcopy(X)
    
    
    arraypca=getPca1(m)
    
    o,p = arraypca.shape
    size=range(100,100+p)
    arraypca = pd.DataFrame(arraypca, columns=size)
    return arraypca

def getPca1(training ):
    
    
    n=1
    pca = PCA(n_components=n)
    pca.fit(training)
    energy=pca.explained_variance_ratio_.sum() 
    
    while (energy<0.85):
        n=n+1
        pca = PCA(n_components=n)
        pca.fit(training)
        
        energy=pca.explained_variance_ratio_.sum() 
     
    newTraining=pca.transform(training)
    
    return newTraining