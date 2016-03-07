from sklearn.metrics.pairwise import cosine_similarity
from join import Join as join
from joindocuments import joindocuments 
import pandas as pd
from oddratio import OddRatio as ratio
from topwords import topwords
from ngrams import ngrams
from truncatedsvd import SVDf1
from PCAC import pcaf
import arff
import numpy as np
from laplace import Laplacian_matrix
import arff
from sklearn.cross_validation import train_test_split
#svd transfer1

class svdtransfer():
 	
	def __init__(self, df):
	 			
		self.df1=df
	
	def getdataframe(self,selector=0,n=50):
		
		main_domain = join(self.df1,'Clean tweet')
		
		top = topwords(self.df1,'Clean tweet',n)
		bigrams=ngrams(self.df1,'Clean tweet')
		print 'bigrams'

		print bigrams.bigrams
		topw=top.top
		bigramsw=bigrams.bigrams
		

		main_domain.joinall(bigramsw,2)
		main_domain.joinall(topw,1)
		if selector==1:
			cols=[]
		else:	
			cols=['tweet','Clean tweet','L']
		
		for x in cols:
			try:
			 print x ,"fsd"
			 del main_domain.df[x]
			except:
			 pass
		main_domain.df=main_domain.df.replace(['True','False',True,False], [1,0,1,0])
		main_domain.df.to_csv('prueba1.csv',index=False)
		return main_domain.df, list(main_domain.df.columns.values),topw, bigramsw

	def joinunigramandbigrams(self,df,unigrams, bigrams):
		main_domain = join(df,'Clean tweet')
		main_domain.joinall(bigrams,2)
		main_domain.joinall(unigrams,1)
		main_domain.df=main_domain.df.replace(['True','False',True,False], [1,0,1,0])
		main_domain.df.to_csv('prueba3.csv',index=False)

		return main_domain.df
		



	def intersectionf(self,a,b):
		inter=set(a).intersection(b)
		intersectionall= list(inter)
		return intersectionall
	

	def getfcomplement(self,domain):
		
		
		term_frecuency=domain.as_matrix(columns=None)
		term_frecuency_transpose=term_frecuency.transpose()
		print 'terms vs long document: \n',term_frecuency_transpose.shape, '\n',term_frecuency_transpose
		return term_frecuency_transpose

	def getutxk(self,domain):

		array=self.getfcomplement(domain)
		U=SVDf1(array)
		
  		return U

	def Cosenosimilarity(self,u,lenght,totallenght):

	 	
	 	
	 	similarity=[]
		X=cosine_similarity(u[0:lenght],u[lenght:totallenght])
		columnindex=0
		d,f=X.shape
		
		for l in range(1,d+1):
			X[l-1][l-1]=0
			print columnindex
			a=X[l-1:l]
			
			vectorequal=np.amax(X[l-1:l])
			i,j=np.where(X[l-1:l] == vectorequal)
			
			if vectorequal>=0.99:
				n,m=i[0],j[0]
				print 'similarity coeficient', a[n][m], columnindex,' ',m, ' ', X[columnindex][m],m+lenght
				similarity.append((columnindex,lenght+m))
			columnindex=columnindex+1
		return similarity
	def mapsimilarity(self,clasifier,columninter, columnt,whithoutmaping=0):
		print 'intersection and target', columninter,columnt
		columnintersection = clasifier[columninter].values
		column =clasifier[columnt].values

		
		
		
		for i,wordi in enumerate(columnintersection):

			if whithoutmaping==0:
			
				if wordi==1 or wordi==True or wordi=='True':
				  clasifier.set_value(i, columnt, 1)
			
			  
			  
			#print columninter,columnt,i, 'columnt', column[i],'columnintersection',columnintersection[i]
			
		
		
		return clasifier

	def joind(self,df1,df2, size=0, sourceOrTarget=0,undersample=0):
            if sourceOrTarget==0:

              df1, testdf = train_test_split(df1, test_size =1-size)
            else:
              df2, testdf = train_test_split(df2, test_size = 1-size)
            df1.L=df1.L.replace([True,False], ['True','False'])
            df2.L=df2.L.replace([True,False], ['True','False'])
            joinc=joindocuments(df1,df2)
            undersampleddf1=joinc.join(df1,df2)
            under=joinc.undersampling(undersampleddf1)
            under.L=under.L.replace(['True','False'], [True,False])
            return undersampleddf1

def arfffunc(training,test, target,target1,size=1,source=0,undersample=0):
	cols=['Clean tweet'] 
	for x in cols:
			 del training[x]
			 del test[x]
	A=str(size)
	joineig=str(source)
	undersample=str(undersample) #needs to be completed
	
	training.L=training.L.to_frame(name=None)
	test.L=test.L.to_frame(name=None)
	
	training.L=training.L.replace(['True','False'], [True,False]) 
	test.L=test.L.replace(['True','False'], [True,False])
	
	training.to_csv(target+'\Training'+A+'.csv',index=False)
	test.to_csv(target+'\Test'+A+'.csv',index=False)
	headers_names=list(training.columns.values)
	headers_names.remove('L')
	headers_names.append('L')
	test = test[headers_names]
	training=training[headers_names]
	TRAINING=training.as_matrix(columns=None)

	arff.dump(target1+r'\training'+A+joineig+'.arff',TRAINING, relation="whatever", names=headers_names)
	TEST=test.as_matrix(columns=None)	 
	arff.dump(target1+ r'\test'+A+joineig+'.arff',TEST, relation="whatever", names=headers_names)
def all(dfsource,dftarget,textclassifier,size=0.7,sourceOrTarget=0,undersample=0):
	
	
	
	 
	source=svdtransfer(dfsource)
	target=svdtransfer(dftarget)
	targetdf, wordst,unigramst,bigramst =target.getdataframe()
	sourcedf,words,unigramss, bigramss=source.getdataframe()
	bigramsall=bigramst+bigramss
	intersection =target.intersectionf(words,wordst)

	onlytarget= list(set(wordst) - set(intersection))

	featuresall=onlytarget+intersection

	alldf=targetdf[featuresall]






	lengthtarget=len(onlytarget)
	lenghtall=len(featuresall)
	A=target.getutxk(targetdf)
	similar=target.Cosenosimilarity(A,lengthtarget,lenghtall)
 
	dfclasifiertraining=target.joind(dfsource,dftarget,size,sourceOrTarget,undersample)
	print "similar",similar
	def mapping(dfclasifier,unigramst,bigramsall,onlytarget, featuresall):
		alldata=svdtransfer(dfclasifier)	
		dataframeall=alldata.joinunigramandbigrams(dfclasifier,unigramst,bigramsall)
		dataframeall.L=dataframeall.L.replace([1,0], ['True','False'])
		
		for i,j in similar:
			
			#print 'intersection',onlytarget[i].encode('ascii','ignore'), 'target',featuresall[j].encode('ascii','ignore')
			clasifier=target.mapsimilarity(dataframeall,featuresall[j],onlytarget[i])
			dataframeall=clasifier
		return dataframeall
	training=mapping(dfclasifiertraining,unigramst,bigramsall,onlytarget, featuresall)
	textset=mapping(textclassifier,unigramst,bigramsall,onlytarget, featuresall)


	arfffunc(training, textset,r'documents\csv\svdtransfer',r'documents\Arff\svdtransfer',size,sourceOrTarget,undersample)

sizesdataframes=[1.0]

mapping=[0,1]
dfsource=pd.read_csv('documents\csv\pregnancy\GOOD LABELING 170620151'+'.csv'  )
dftarget=pd.read_csv('documents\csv\drunk\drunk labeling 1300'+'.csv'  )
textclassifier=pd.read_csv('documents\csv\drunk\drunkTEXT400U'+'.csv'  )
dfsource.L=dfsource.L.replace(['y','Y','n','n '], [True,True,False,False])  
dftarget.L=dftarget.L.replace(['y','Y','n','n '], [True,True,False,False]) 
textclassifier.L=textclassifier.L.replace(['y','Y','n','n '], [True,True,False,False]) 

for sourceOrTarget in mapping:
	for size in sizesdataframes:
		print 'size:',size
		print 'mapping', sourceOrTarget
		all(dfsource,dftarget,textclassifier, size,sourceOrTarget)