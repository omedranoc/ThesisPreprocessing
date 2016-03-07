from join import Join as join
import pandas as pd
from pandas import Series, DataFrame
from laplace import Laplacian_matrix
from sklearn.cluster import spectral_clustering
from joindocuments import joindocuments 
from oddratio import OddRatio as ratio
import arff
from mutual_info import mutualinfo
class cluster(object):
	"""docstring for ClassName"""
	def __init__(self, laplacian,ncluster,classesnames):
		self.laplacian = laplacian
		self.ncluster = ncluster
		m,n=laplacian.shape
		print 'size Laplacian_matrix: ',m, n
		labels = spectral_clustering(laplacian, n_clusters=ncluster)

		x=range(n+1)
		wordsall=zip(x, classesnames)
		lc= zip(labels,x)
		print "labels", lc
		allwordsclustered=[]
		for m in range(ncluster):
			sort=[item[1] for item in lc if item[0] == m]
			wordsclustered=[]

			for y in sort:

				for item in wordsall:
				 if item[0] == y:
				  wordsclustered.append(item[1])
			if len(wordsclustered) >1:	
				allwordsclustered.append(wordsclustered)

		print'clusteredwords'
		print allwordsclustered
		
		self.cluster=  len(allwordsclustered),allwordsclustered
def alltogether(A,varydocument=0,x=0): 
	df1=pd.read_csv('documents\csv\pregnancy\GOOD LABELING 170620151'+'.csv'  )
	df2=pd.read_csv('documents\csv\drunk\drunk labeling 1300'+'.csv'  )
	laplacian= Laplacian_matrix(df1,df2,'Clean tweet')
	la,ds,di=laplacian.LAPLACE_NORMALIZED()
	classesname=ds+di
	n=len(classesname)
	print 'titles'
	print n
	allclusters=[]
	
	print x
	prueba= cluster(la,x,classesname)
	lencluster, clusterd=prueba.cluster
	allclusters.append(clusterd)
	clusterslong=[]
	for x in allclusters:
		for y in x:
			if len(y)>1 and len(y)<100:
				clusterslong.append(y)
	print clusterslong
	clustersall=[]
	for i in clusterslong:
	  if i not in clustersall:
	    clustersall.append(i)
	print len(clustersall)
	
	print 'pass1'
	joinc=joindocuments(df1,df2)
	print 'pass1.a'
	undersampleddf1, undertest=joinc.joinsourcetarget(A,varydocument)
	joiner=join(undersampleddf1,'Clean tweet')
	print 'pas2'

	tainingt, testt=joiner.jointwodocuments(undersampleddf1,undertest,clustersall,4)

	print 'ta'
	


	ratiov=ratio(tainingt,'L')
	a=[ 'L', 'absPolarity', 'sentiment_polarity', 'sentiment_subjectivity']
	
	cols=['Clean tweet']
	try:
		for x in cols:
			del tainingt[x]
			del testt[x]
	except:
		pass
	headers_names=list(tainingt.columns.values)
	headers_names.remove('L')
	headers_names.append('L')
	tainingt=tainingt[headers_names] 
	testt=testt[headers_names]
	
	tainingt=tainingt.replace(['True','False'], [True,False])   
	testt=testt.replace(['True','False'], [True,False]) 
	TRAINING=tainingt.as_matrix(columns=None)
	A=str(A)	
	arff.dump( r'documents\Arff\cluster\trainning'+A+'.arff',TRAINING, relation="whatever", names=headers_names)
	TEST=testt.as_matrix(columns=None)	 
	arff.dump(r'documents\Arff\cluster\test'+A+'.arff',TEST, relation="whatever", names=headers_names)


##varydocument= 0 it varies the SOURCE
clusterd=[(0.2,0),(0.5,0),(0.8,0),(1.0,0)]
for p,j in clusterd:
	alltogether(p,j,15)
