import math

from sklearn.metrics.cluster import normalized_mutual_info_score
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import pandas as pd



test=pd.read_csv(r'documents\csv\pregnancy\Training0.1.csv' )
def mutualinfo(df):
	dfin=df
	Label=dfin['L']
	VALUES=['sentiment_polarity','sentiment_subjectivity','absPolarity','Clean tweet', 'L']
	
	
	headers_names=list(dfin.columns.values)
	headers_names = [x for x in headers_names if x not in VALUES]
	
	mutualinfowords=[]
	for header in headers_names:
		
		mutualcolumn= dfin[header]
		mutualvalue= normalized_mutual_info_score(mutualcolumn,Label)
		if mutualvalue>0.02:
			#print'mutual info',header, mutualvalue 
			mutualinfowords.append(header)
	return mutualinfowords
#mutualinfo(test)