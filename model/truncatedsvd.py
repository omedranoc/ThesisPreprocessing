from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
import numpy as np
from sklearn.utils.extmath import randomized_svd
import pandas as pd
import copy
#test array

class svd():
	"""docstring for svd"""
	def __init__(self,x):

		self.x=x
	
	def getsvd(self,X):		
		
		n=0
		# U, Sigma, VT = randomized_svd(X, n_components=2,random_state=42)
		print 'svd'
		svd = TruncatedSVD(n_components=n, random_state=42)
		svd.fit(X) 
		energy=svd.explained_variance_ratio_.sum()
		print energy
		while (energy<0.8):
			n=n+1
			
			svd = TruncatedSVD(n_components=n+1, random_state=42)	
			svd.fit(X) 
			energy=svd.explained_variance_ratio_.sum()
			
		 
		new=svd.transform(X)
		return new
def SVDf(X):
   
    X=X.replace(['True','False'], [1,0])
    X=X.replace([True,False], [1,0])
    
    m=X.as_matrix(columns=None)
    SVD=svd(m)
    array=SVD.getsvd(m)
    M,N= array.shape
    LA=range(N)
    array = pd.DataFrame(array, columns=LA)
    return array
def SVDf1(X):

    Target= copy.deepcopy(X)
  
    cols=['L']
    try:
        for x in cols:
            del Target[x]
           
    except:
        pass
    SVD=svd(X)
    array=SVD.getsvd(X)
    M,N= array.shape
    LA=range(N)
    print 'k columns:\n', LA
    array = pd.DataFrame(array, columns=LA)
    return array