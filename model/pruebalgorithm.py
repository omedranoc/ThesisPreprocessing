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


def randomforest(df1,df2):
	
	
	newsT=df1.L
	L= ['L']
	for x in L:
	 	del df1[x]
	news=df1
	TRAINING=df1.as_matrix(columns=None)
	TEST=newsT.as_matrix(columns=None)
	
	newsT=df2['L']
	L= ['L']
	for x in L:
	 	del df2[x]
	X_test=df2.as_matrix(columns=None)
	y_test=newsT.as_matrix(columns=None)

	clf = RandomForestClassifier(n_estimators=200)
	clf.fit(TRAINING, TEST)
	y_pred1 = clf.predict_proba(X_test)[:, 1]
	y_pred = clf.predict(X_test)
	recall_score(y_test, y_pred)
	precision_score(y_test, y_pred)
	precision_score(y_test, y_pred,pos_label=0)
	recall_score(y_test, y_pred,pos_label=0)
	roc_auc_score(y_test, y_pred1)
	print 'roc: ',roc_auc_score(y_test, y_pred1)
	print 'precision: ',precision_score(y_test, y_pred)
	print 'recall:', recall_score(y_test, y_pred)
	print 'precision Negatives: ',precision_score(y_test, y_pred,pos_label=0)
	print 'recall Negatives: ', recall_score(y_test, y_pred,pos_label=0)
	
	return roc_auc_score(y_test, y_pred1),precision_score(y_test, y_pred),recall_score(y_test, y_pred),precision_score(y_test, y_pred,pos_label=0), recall_score(y_test, y_pred,pos_label=0)

df1=pd.read_csv(r'documents\csv\pregnancy\Training0.1.csv' )
df2=pd.read_csv(r'documents\csv\pregnancy\Test0.1.csv' )
#randomforest(df1,df2)