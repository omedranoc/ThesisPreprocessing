from topwords import topwords
import pandas as pd
from join import Join as join
import arff
import numpy as np
from PCAC import pcaf	
from join import Join as join

def kclustering(top=100,pca=0):
	training=pd.read_csv('documents\csv\drunk\drunk labeling 1300'+'.csv'  )
	test=pd.read_csv( 'documents\csv\drunk\drunkTEXT400U'+'.csv' )
	main_domain = join(training,'Clean tweet')
	top = topwords(test,'Clean tweet',top)
	main_domain = join(training,'Clean tweet')
	main_domain1 = join(test,'Clean tweet')
	main_domain.joinall(top.top,1)
	main_domain1.joinall(top.top,1)
	training=main_domain.df
	test=main_domain1.df


	cols=['Clean tweet']

	try:
		for x in cols:
			del training[x]
			del test[x]
	except:
		pass


	
	print training['L']
	training.L=training.L.replace(['y','n'], [True,False])
	test.L=test.L.replace(['y','n'], [True,False])
	if pca==1:

		dftraining, dftest=pcaf(training,test)
		training =dftraining.join(training["L"])
		test=dftest.join(test["L"])
	
	try:
		training=training.replace(['True','False'], [True,False])	
		test=test.replace(['True','False'], [True,False])
	except:
		pass
	headers_names=list(training.columns.values)
	training=training.astype(np.float64)
	test=test.astype(np.float64)
	training['L']=training['L'].astype(bool)
	test['L']=test['L'].astype(bool)
	headers_names.remove('L')
	headers_names.append('L')
	
	pca=str(pca)
	test = test[headers_names]
	training = training[headers_names]
	TRAINING=training.as_matrix(columns=None)
	TEST=test.as_matrix(columns=None)
	print training.dtypes
	main_domain.df.to_csv(r'documents\csv\unsupervised\test.csv',index=False)
	main_domain.df.to_csv(r'documents\csv\unsupervised\test.csv',index=False)
	arff.dump(r'documents\Arff\unsupervised'+r'\training'+pca+'.arff',TRAINING, relation="whatever", names=headers_names)
	arff.dump(r'documents\Arff\unsupervised'+r'\test'+pca+'.arff',TEST, relation="whatever", names=headers_names)

#kclustering(100,0)
kclustering(100,1)