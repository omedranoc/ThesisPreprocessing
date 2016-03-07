from join import Join as join
from joindocuments import joindocuments 
import pandas as pd
from oddratio import OddRatio as ratio
from topwords import topwords
from ngrams import ngrams
from truncatedsvd import SVDf
from PCAC import pcaf
import arff
import numpy as np
#joinoddratios
#it gets the data preprocessing
def domain(document, crossvalidationundersampling,ArffL,A=0, undersampler=0,sentiment=0 ):
	test=pd.read_csv('documents\csv\drunk\drunkTEXT400'+'.csv'  )
	test.L=test.L.replace(['y','n'], ['True','False'])
	df1=pd.read_csv(document+'.csv'  )
	df1.L=df1.L.replace(['y','n'], ['True','False'])
	joinc=joindocuments(df1,df1)
	top = topwords(df1,'Clean tweet',100)
	main_domain = join(df1,'Clean tweet')
	
	bigrams=ngrams(df1,'Clean tweet')
	print 'bigrams'
	print bigrams.bigrams
	main_domain.joinall(bigrams.bigrams,2)
	main_domain.joinall(top.top,1)
	
	
	
	main_domain.df.to_csv('prueba.csv',index=False)
	ratiov=ratio(main_domain.df,'L')
	ratios=ratiov.getoddratios(top.top)
	print 'ratios'
	print ratios		
	ds=list(ratios.keys())
	testobject = join(test,'Clean tweet')
	oddradiojoin=join(df1,'Clean tweet')
	oddradiojoin.joinall(ds,1)
	testobject.joinall(ds,1)
	oddradiojoin.joinall(bigrams.bigrams,2)
	testobject.joinall(bigrams.bigrams,2)
	test=testobject.df
	cols=['Clean tweet']
	if sentiment==1:
		cols=['Clean tweet','sentiment_polarity', 'sentiment_subjectivity', 'absPolarity']

	try:
		for x in cols:
			del oddradiojoin.df[x]
			del test[x]
	except:
		pass
	#training, test=joinc.gettrainingandtestp(oddradiojoin.df)
	print 'matrix of elements to reduce'
	print "saul,",oddradiojoin.df.shape
	#########################################################
	if undersampler==1:
	  print "saul,",oddradiojoin.df.shape
	  oddradiojoin.df=joinc.undersampling(oddradiojoin.df)
	  print oddradiojoin.df.shape
	if A==1:
		
		
		
		dftraining, dftest=pcaf(oddradiojoin.df,test)
		oddradiojoin.df =dftraining.join(oddradiojoin.df["L"])
		
		
		test=dftest.join(test["L"])

	
	print oddradiojoin.df.shape
	training=oddradiojoin.df
	
	training=training.replace(['True','False'], [True,False])	
	test=test.replace(['True','False'], [True,False])
	training=training.astype(np.float64)
	test=test.astype(np.float64)
	training['L']=training['L'].astype(bool)
	test['L']=test['L'].astype(bool)
	A=str(A)
	sentiment=str(sentiment)
	oddradiojoin.df.to_csv('crossvalidation.csv',index=False)
	#undersampleddf1.to_csv(str(crossvalidationundersampling) +'\undersampling'+A+'.csv',index=False)
	headers_names=list(training.columns.values)
	headers_names.remove('L')
	headers_names.append('L')
	headers_names1=list(test.columns.values)
	print headers_names,'heathers test',headers_names1
	test = test[headers_names]
	training = training[headers_names]
	print 'training' +str(training.dtypes)
	test.to_csv(str(crossvalidationundersampling) + r'\test1'+A+'.csv',index=False)
	training.to_csv(str(crossvalidationundersampling) +r'\training1'+A+'.csv',index=False)
	TRAINING=training.as_matrix(columns=None)
	TEST=test.as_matrix(columns=None)
	print 'training'
	print training.dtypes
	
	arff.dump(ArffL +r'\trainingwu'+A+str(undersampler)+sentiment+'.arff',TRAINING, relation="whatever", names=headers_names)
	 
	arff.dump(ArffL +r'\testwu'+A+str(undersampler)+sentiment+'.arff',TEST, relation="whatever", names=headers_names)

#domain('documents\csv\divorce\divorce',r'documents\csv\divorce',r'documents\Arff\divorce',1)
#domain('documents\csv\pregnancy\GOOD LABELING 170620151',r'documents\csv\pregnancy',r'documents\Arff\pregnancy',1)#1 indicates the value of A

A=[[0,0,0],[0,1,0],[1,0,0],[1,1,0],[0,0,1],[0,1,1],[1,0,1],[1,1,1]]    #,(1,0),(1,1),(1,2),(0.5,0),(0.5,1),(0.5,2)
	
for s ,i,sentiment in A:
  
 print s, i, sentiment
 domain('documents\csv\drunk\drunk labeling 1300',r'documents\csv\drunk',r'documents\Arff\drunk',s,i,sentiment)
 #domain('documents\csv\pregnancy\GOOD LABELING 170620151',r'documents\csv\pregnancy',r'documents\Arff\pregnancy',s,i, sentiment)#1 indicates the value of A


