from join import Join as join
from joindocuments import joindocuments 
import pandas as pd
from pandas import Series, DataFrame
from oddratio import OddRatio as ratio
from laplace import Laplacian_matrix
from scipy.linalg import eigh as eigenvectors
import nltk
import numpy as np
import numpy.matlib
import arff
from PCAC import pcaf1
from truncatedsvd import SVDf1
from pruebalgorithm import randomforest
from sklearn.metrics.cluster import normalized_mutual_info_score
from topwords import topwords
from ngrams import ngrams
from mutual_info import mutualinfo
def addbigrams(dft,dfte,df1,selector=0,n=50):
	
	
	
	top = topwords(df1,'Clean tweet',n)
	bigrams=ngrams(df1,'Clean tweet')
	
	
	bigramsw=bigrams.bigrams
	main_domain = join(dft,'Clean tweet')
	main_domain1 = join(dfte,'Clean tweet')
	main_domain.joinall(bigramsw,2)
	main_domain1.joinall(bigramsw,2)
	
	return main_domain.df, main_domain1.df

class espectralfeature():
	""" espectralfeature: algorithm from the paper cross domain sentiment classification"""
	def __init__(self,df1,df2 ):
	    
	    self.df1=pd.read_csv(df1+'.csv'  )
	    self.df2=pd.read_csv(df2+'.csv'  )
	def domain_specificbyeigenvector(self,dfjoined):
		m,n=self.laplacian_matrix.shape
		eigenvaluesv=0
		l=0
		
		# while eigenvaluesv>0:
			
		# 	l=l+1
			
		# 	w, matrix_eigenvectors = eigenvectors(self.laplacian_matrix, eigvals=(n-l,n-1))
		# 	eigenvaluesv=int(w[0])
			
		# 	if eigenvaluesv<=0:
		# 		l=l-1
		# 		w, matrix_eigenvectors = eigenvectors(self.laplacian_matrix, eigvals=(n-l,n-1))
		matrix_eigenvectors=pcaf1(self.laplacian_matrix)

		
		print 'matrix', matrix_eigenvectors
		matrix_eigenvectors=np.array(matrix_eigenvectors)
		lenght= len(self.domain_specific)

		eigenvectordomain=matrix_eigenvectors[:lenght,:]
		
		df = dfjoined
		
		s= self.matrixtodott[self.domain_specific]
		dataframeceros= s.replace(['False', 'True'],  [0,1]) 
		matrix=dataframeceros.as_matrix(columns=None)
		l1=np.dot(matrix,eigenvectordomain ) # multiplies the matrix of Xi by the matrix of the domain especific eigenvectors
		
		m,n =l1.shape
		classes = range(1,n+1)
		data=DataFrame(l1,columns=classes)
		headers_eigen=list(data.columns.values)
		unidost = df.join(data)
		return unidost,headers_eigen

	    
 	def spectralcluster(self,A=1, varydocument=0,joineig=0,undersamplingv=False):
 	  #1 join the documents  and get the test sample 
 	  	
 	  
		def joind(df1,df2,size=0.1,undersamplingv=False, varydocument=0):
			df1.L=df1.L.replace(['y','Y','n','n '], ['True','True','False','False'])  
			df2.L=df2.L.replace(['y','Y','n','n '], ['True','True','False','False']) 
		    
			joindf=joindocuments(df1,df2)
			if varydocument==0:
				df1,otro=joindf.gettrainingandtestp(df1,size)
			if varydocument==1:
			    df2,otro=joindf.gettrainingandtestp(df2,size)
			joinc=joindocuments(df1,df2)
			if undersamplingv==True:
				
				df2=joinc.undersampling(df2)
				df1=joinc.undersampling(df1)
				
			undersampleddf1=joinc.join(df1,df2)
			
			return undersampleddf1
		undersampleddf1=joind(self.df1,self.df2,A, undersamplingv,varydocument)
		
		  #undersampleddf1, undertest=joinc.joinsourcetarget(A,varydocument)

		undertest=pd.read_csv('documents\csv\drunk\drunkTEXT400U'+'.csv'  )
		undertest.L=undertest.L.replace(['y','Y','n','n '], [True,True,False,False]) 
		  #join the domain specific features to the training and sample
		
		laplacian= Laplacian_matrix(self.df1,self.df2,'Clean tweet')
		la,ds,di=laplacian.LAPLACE_NORMALIZED()
	
		self.domain_specific=ds #+['sentiment_polarity','sentiment_subjectivity','absPolarity']
		self.laplacian_matrix=la
		joiner=join(undersampleddf1,'Clean tweet')
		def getmostcommon(df,df1,n=10):
				
				main_domain = join(df,'Clean tweet')
				main_domain1 = join(df1,'Clean tweet')
				top = topwords(self.df2,'Clean tweet',n)
				bigrams=ngrams(self.df2,'Clean tweet',n)
				
				topw=top.top
				bigramsw=bigrams.bigrams
				
				main_domain.joinall(topw,1)
				main_domain.joinall(bigramsw,2)
				main_domain1.joinall(topw,1)
				main_domain1.joinall(bigramsw,2)
				return main_domain.df,main_domain1.df,
		tainingt, testt=getmostcommon(undersampleddf1,undertest,10)
		
		###################################
		#tainingt, testt=addbigrams(tainingt,testt,self.df2)
		
	
		self.matrixtodott,self.matrixtodotest=joiner.jointwodocuments(undersampleddf1,undertest,ds,1)
		
		if joineig==0:
		  trainingset,headerst=self.domain_specificbyeigenvector(tainingt)
		  testset,headerstest=self.domain_specificbyeigenvector(testt)
		elif (joineig==1) or (joineig==2):
			trainingset=tainingt
			testset=testt
			if joineig==2:
				trainingset,headerst=self.domain_specificbyeigenvector(tainingt)
				testset,headerstest=self.domain_specificbyeigenvector(testt)
				headerst=headerst+['L']
				trainingset=trainingset[headerst]
				testset=testset[headerst]
				
		
		
		

		return  trainingset , testset
def todo(document1,document2,target,target1,A=1,varydocument=0,joineig=0,undersamplingv=0): #varydocument= 0 it varies the source and Joineig=0 it adds the spectral features
	print 'size: ', 'A= ',A,'eigenvectors=', joineig, 'with or without eigenvectors 1=without 0=with 2=withoutdi'
	spectral=espectralfeature(document1,document2)	
	df, test=spectral.spectralcluster(A,varydocument,joineig,undersamplingv)
	print "PASO 1 COMPLETED"
	headers_names=list(df.columns.values)
	
	cols=['Clean tweet','tweet','url']
	for x in cols:
	 try:	
		del df[x]
		del test[x]
	 except:
	 	pass
	try:
		df=df.replace(['True','False'], [True,False])   
	except:
		pass
	try:
	 test=test.replace(['True','False'], [True,False])    
	except:
		pass
	print headers_names
	headers_names=list(df.columns.values)
	headers_names.remove('L')
	headers_names.append('L')
	print headers_names
	
	
	
	print type(headers_names)
	test = test[headers_names]
	df= df[headers_names]

	A=str(A)
	joineig=str(joineig)
	varydocument=str(varydocument)
	undersamplingv=str(undersamplingv)
	
	df.to_csv(target+'\Training'+A+'.csv',index=False)
	test.to_csv(target+'\Test'+A+'.csv',index=False)
	print "COMPLETED 0", df.dtypes

	TRAINING=df.as_matrix(columns=None)
	print "COMPLETED 0.1"
	arff.dump(target1+r'\training'+A+varydocument+ joineig+undersamplingv+'.arff',TRAINING, relation="whatever", names=headers_names)
	TEST=test.as_matrix(columns=None)	 
	arff.dump(target1+ r'\test'+A+varydocument+joineig+undersamplingv+'.arff',TEST, relation="whatever", names=headers_names)
	print "COMPLETED"
	
if __name__ == '__main__':	
	
	A=[[0.2,1,0,1],[0.5,1,0,1],[0.8,1,0,1],[1.0,1,0,1],[0.2,1,0,0],[0.5,1,0,0],[0.8,1,0,0],[1.0,1,0,0],[0.2,1,1,1],[0.5,1,1,1],[0.8,0,1,1],[1.0,1,1,1],[0.2,1,2,1],[0.5,1,2,1],[0.8,1,2,1],[1.0,1,2,1]]    #,(1,0),(1,1),(1,2),(0.5,0),(0.5,1),(0.5,2)
	print 'vamos'
	
	for s,i,j,k in A:
		print s,i,j,k 
		todo('documents\csv\pregnancy\GOOD LABELING 170620151','documents\csv\drunk\drunk labeling 1300',r'documents\csv\pregnancy',r'documents\Arff\transferlearning\ctarget',s,i,j,k)
	
		