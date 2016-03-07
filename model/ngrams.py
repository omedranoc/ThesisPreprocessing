from nltk.corpus import stopwords
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
import re
import nltk
class ngrams:
  def __init__(self, df,column,n=10):

    texto = " ".join(str(x) for x in df[column].values)
    tokens = texto.split()
    tokens=[x.lower() for x in tokens]
    stopset = set(stopwords.words('english')) # dictionary of stop words
    tokens = [w for w in tokens if not w in stopset]
    stemmer=SnowballStemmer("english")
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
    cuenta = len(tokens_clean)
    bigrams = nltk.bigrams(tokens_clean)
    trigrams=nltk.trigrams(tokens_clean)
    fdist = nltk.FreqDist(bigrams)
    fdist1 = nltk.FreqDist(trigrams)
    #for i,j in fdist.items():
    #   print i,j
    frecuentbigrams=fdist.most_common(n)
    frecuenttrigrams=fdist1.most_common(10)
    bigramslist=[]
    trigramslist=[]
    
    for x in frecuentbigrams:
      a,b=x
      l,m=a
      if l !='' and m !='' and l!=m:
        
          bigramslist.append(a)
    
    bigramsduplicates=[]  
    for idx, x in enumerate(bigramslist):
      for idy, y in enumerate(bigramslist):
        if idx!=idy:
          
          if x[0]==y[1]:
            duplicate=(x[1],x[0])
            
            #print bigramsduplicates
            #print x
            if x not in bigramsduplicates:
              bigramslist.pop(idx)
              bigramsduplicates.append(x)
              bigramsduplicates.append(duplicate)
       
    for x in frecuenttrigrams:
      a,b=x
      trigramslist.append(a)
    
    self.bigrams=bigramslist
    self.trigrams=trigramslist
    