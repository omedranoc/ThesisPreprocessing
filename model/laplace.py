import re
from join import Join as join
from oddratio import OddRatio as ratio
from topwords import topwords
import pandas as pd
from pandas import DataFrame
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import numpy.matlib
from ngrams import ngrams
from mutual_info import mutualinfo
from joindocuments import joindocuments 
class  Laplacian_matrix:   
	def __init__ (self,df1,df2,column='Clean tweet'):

       	# most frecuent words domain specific
	    self.df1 = df1
	    self.df2 = df2
	    self.column=column
	def intersectionoddfrecuent(self,oddwords ): #intersection of the odd words from the source with the target domain words

		topds = topwords(self.df2,self.column,50)
		inter=set(oddwords).intersection(topds.top)
		intersectionall= list(inter)
		return intersectionall, topds.top
	def getlaplacian(self):

		
		def getintersection(df,selector=0,n=50):
				
				main_domain = join(df,'Clean tweet')
				
				top = topwords(df,'Clean tweet',n)
				bigrams=ngrams(df,'Clean tweet')
				
				topw=top.top
				bigramsw=bigrams.bigrams
				
				main_domain.joinall(topw,1)
				mutualwordsu= mutualinfo(main_domain.df)
				main_domain.joinall(bigramsw,2)
				mutualwordsb= mutualinfo(main_domain.df)
				mutualwordsb=[e for e in mutualwordsb if e not in mutualwordsu]
				ratiov=ratio(main_domain.df,'L')
				ratios=ratiov.getoddratios(top.top)
				dratios=list(ratios.keys())
				return topw, bigramsw, dratios,mutualwordsu,mutualwordsb



													#top = topwords(self.df1,self.column,50)
													
													#main_domain = join(self.df1,self.column)
													#main_domain.joinall(top.top,1)
													
													#ratiov=ratio(main_domain.df,'L')
													#ratios=ratiov.getoddratios(top.top)
													#print 'odd ratio words: \n', ratios		
													#ds=list(ratios.keys())
													#ds=top.top # domain specific feature without oddratio
		def intersectiontwobigrams(bigrams1, bigrams2):
			equal=[]
			for xb, yb in bigrams1:
				for cb in bigrams2:
					if xb in cb and yb in cb:
						equal.append(cb)




		unigrams, bigrams,ratiolist, mutuallistu,mutuallistb= getintersection(self.df1)
		
		unigrams1, bigrams1,ratiolist1, mutuallistu1, mutuallistb1= getintersection(self.df2)

		ds1=mutuallistu+mutuallistu1
		dsb=mutuallistb+mutuallistb1
		bigrams=bigrams+bigrams1
		ds=ds1+dsb

		
		
		intersectionwords,ds1=self.intersectionoddfrecuent(unigrams)
		
		ds=ds
		ds = list(set(ds))
		ds=[e for e in ds if e not in intersectionwords]
		
		di=intersectionwords
		
		#print 'ds ',ds
		#print 'di ',di
		matrixm=self.matrix(ds,di,bigrams)
		
		matrixall=self.simetry_matrix(matrixm)
		
		return matrixall,ds,di
		#matrixm.to_csv('documents\csv\prueba.csv',index=False)

	def matrix(self,domain,independent,domainb=[('without value','withoutvalue')]): # creates a  matrix M from the paper cross domain sentiment classification
 		  
 		  stemmer=SnowballStemmer("english")
 		  
 		  	
          ####################################################
		  domaincheck=domain
		  domainl=domain
		  
		  domain1,domain2=map(list, zip(*domainb))
		  domain1= list(map(stemmer.stem, domain1))
		  domain2= list(map(stemmer.stem, domain2))
		 
		  matrixM=DataFrame(0,index=domainl, columns=independent)
		  joinf=joindocuments(df1,df2)
		  undersampleddf=joinf.join(self.df1,self.df2)
		  for x in undersampleddf[self.column].values:

		    
			tokens = x.split()
			tokens=[x.lower() for x in tokens]
			
			stemm_words=[]
			tokens_clean=[]
			for j in tokens:
		      
				sa=re.sub('[^A-Za-z]+', '', j)
				tokens_clean.append(sa)
		    
			for s in tokens_clean:
				try:
				  stem= stemmer.stem(s)
				  if s!='':
				   stemm_words.append(str(stem)) 
				except:
				  pass

			
			inter=set(domain).intersection(stemm_words) #find the intersection two lists
			intersection1= list(inter)
			inter1=set(independent).intersection(stemm_words) #find the intersection two lists
			intersection2= list(inter1)
			inter3=set(domain1).intersection(stemm_words) #find the intersection two lists
			intersection3= list(inter3)
			inter4=set(domain2).intersection(stemm_words) #find the intersection two lists
			intersection4= list(inter4)

 
			if intersection1:
			    if intersection2:
			      
			      for  x in intersection1:
			        
			        for y in intersection2:
			        
			          	matrixM.xs(x)[y]=matrixM.xs(x)[y]+1
			if intersection3:
			    if intersection4:
			      if intersection2:
				      for  x1 in intersection3:
				        
				        for y1 in intersection4:
				        	for z1 in intersection2:
				        		label=x1+y1
				        		
				        		if label in domain:
				          			matrixM.xs(label)[z1]=matrixM.xs(label)[z1]+1          
		  
		  return matrixM
	def simetry_matrix(self,dataframe):
	    matrix=dataframe.as_matrix(columns=None)# transform a dataframe to a matrix
	    n,m=matrix.shape
	    
	    trans= matrix.transpose() # Transpose matrix M
	    
	    matrixzero1= np.matlib.zeros((n, n))
	    matrixzero2= np.matlib.zeros((m, m))
	    right =  np.concatenate((matrixzero1,matrix), axis=1)
	    right2 = np.concatenate((trans,matrixzero2), axis=1)
	    allM = np.concatenate((right,right2), axis=0) # Matrix A
	    return allM
	def  LAPLACE_NORMALIZED(self): #algorithm Spectral domain specific feature...
		simetricmatrix,ds,di=self.getlaplacian()
		sumD=np.sum(simetricmatrix, axis=1)
		sumDT= sumD.transpose()
		arraysumDT= sumDT.tolist()
		D= np.diag(arraysumDT[0]) # D diagonal Matrix
		Dexp= np.sqrt(D) # square of a matrix
		transDsqr= Dexp.transpose()# d^-1/2
		#print transDsqr # D^(-1/2)
		l1=np.dot(transDsqr, simetricmatrix)
		l =np.dot(l1,transDsqr)
		ldimension=l.shape
	  	return l,ds,di

	  	
df1=pd.read_csv('documents\csv\pregnancy\GOOD LABELING 170620151'+'.csv'  )
df2=pd.read_csv('documents\csv\drunk\drunk labeling 1300'+'.csv'  )
test=Laplacian_matrix(df1,df2)
test.getlaplacian()