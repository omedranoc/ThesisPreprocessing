from sklearn import tree
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import warnings
warnings.filterwarnings('error')

def weka(clf, svmv=0):
	df1=pd.read_csv('undersampling1'+'.csv'  )
	newsT=df1['L']#.replace(['y','Y','Y ','y ' ,'n','n '], [True,True,True,True,False,False]) 


	L= ['L','Clean tweet']
	for x in L:
	 	del df1[x]
	news=df1
	TRAINING=df1.as_matrix(columns=None)
	TEST=newsT.as_matrix(columns=None)
	sss = StratifiedShuffleSplit(TEST, 1, test_size=0.3)
	#sss = StratifiedKFold(TEST, n_folds=10)
	print len(sss)
	      
	recall=[]
	precision=[]
	precisionN=[]
	recallN=[]
	roc=[]
	for train_index, test_index in sss:
		
		X_train, X_test = TRAINING[train_index], TRAINING[test_index]
		y_train, y_test = TEST[train_index], TEST[test_index]
		
		
		print y_train.shape,(y_train == True).sum(),(y_train == False).sum()

		print y_test.shape,(y_test == True).sum(),(y_test == False).sum()
		
		#print y_test
		#X_train=X_train[0:100,:]
		#y_train= y_train[0:100]
		#print X_train.shape, X_test.shape
		print '1'
		try:
			clf.fit(X_train, y_train)
			y_pred = clf.predict(X_test)
			
			if svmv==0:
				y_pred1 = clf.predict_proba(X_test)[:, 1]
			elif svmv==1:
				y_pred1= linear_svm.score(X_test, y_test)
			
			recall.append(recall_score(y_test, y_pred))
			precision.append(precision_score(y_test, y_pred))

			precisionN.append(precision_score(y_test, y_pred,pos_label=0))
			recallN.append(recall_score(y_test, y_pred,pos_label=0))
			 
			roc.append(roc_auc_score(y_test, y_pred1))
		except Warning:
			print 'it does not work'
			
	if not precision:	
		print'empty array'
	else:
		print  'recall: ', np.mean(recall), 'recallN: ', np.mean(recallN),'precision: ', np.mean(precision), 'precisionN: ', np.mean(precisionN),'roc: ', np.mean(roc)

clf = RandomForestClassifier(n_estimators=100)
weka(clf)
clf1 = DecisionTreeClassifier()
weka(clf1)
neigh = KNeighborsClassifier(n_neighbors=5)
weka(neigh)

kernel_svm = svm.SVC(probability=True,kernel= 'linear')
#linear_svm = svm.LinearSVC(probability=True)
weka(kernel_svm)
