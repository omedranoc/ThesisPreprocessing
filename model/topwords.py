import re
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from pandas import DataFrame
from nltk.corpus import stopwords
from collections import Counter
from pandas import Series
class topwords:
    def __init__(self,df, column,n ): # gets the most frecuent words in a document
      
        texto = " ".join(str(x) for x in df[column].values)
        tokens = texto.split()
        tokens=[x.lower() for x in tokens]
        #stopset = set(stopwords.words('english')) # dictionary of stop words
        #tokens = [w for w in tokens if not w in stopset]
        stemmer=SnowballStemmer("english")
        stemm_words=[]
        tokens_clean=[]
        for j in tokens:
          
          sa=re.sub('[^A-Za-z]+', '', j)
          tokens_clean.append(sa)
        #print tokens_clean
        for s in tokens_clean:
          try:
            stem= stemmer.stem(s)
            if s!='':
             stemm_words.append(str(stem)) 
          except:
            pass
        cuenta = len(tokens_clean)
        largo =  Counter(stemm_words).most_common(n)
        topdic = dict(largo)
        asortado = Series(topdic)
        asortadol = asortado.columns = ['a', 'b']
        ordenado = asortado.order(ascending=False)
        ordenadolist= topdic.keys() #+stemm_words
        self.top=ordenadolist