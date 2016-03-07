import re
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from pandas import DataFrame
class Join: # it looks for a word inside the tweets and returns a new column indicating if the word  is in each tweet
  def __init__(self , df, column=''):
    
    self.column= column
    self.df= df 
    self.stem()
  def stem(self):
    df=self.df

    word1 = df[self.column].values.tolist()
    new_column = []
        
    for x in word1:
      
      tokens = x.split()
      tokens=[x.lower() for x in tokens]
      words_stem= []
      for y in tokens:

        stringwt = str(y)
        sa=re.sub('[^A-Za-z]+', '', stringwt)
        stemmer=SnowballStemmer("english")
        word= stemmer.stem(sa)
        words_stem.append(word)     
      new_column.append(words_stem)  
    self.stemwords= new_column
  def dataframegenerator(self, new_column,label):
    dfnew_column=DataFrame(new_column, columns=[label])
    headers_names=list(self.df.columns.values)
    #print headers_names
    if label in headers_names:
      print " "

    else:
      self.df = self.df.join(dfnew_column)

    no_wanted_columns=['Unnamed']
    for col in self.df.columns:
      for x in no_wanted_columns:
        if x in col:
          
          #print unidost[col]
          del self.df[col] 
      
    self.label=label

  def unigram(self, term):
    term=re.sub('[^A-Za-z]+', '', term)
    stemmer=SnowballStemmer("english")
    new_column=[]
    
    for words_stem in self.stemwords:
      if term in words_stem:
            new_column.append('True')
      else:
            new_column.append('False')

    df=self.dataframegenerator(new_column,term)
    
    
  def bigram(self,term):
    x,y =term
    stemmer=SnowballStemmer("english")
    x= stemmer.stem(x)
    y= stemmer.stem(y)
    label=x+y
    new_column=[]
    for words_stem in self.stemwords:
      if x in words_stem and y in words_stem:
          new_column.append('True')
      else:
          new_column.append('False')   
    df=self.dataframegenerator(new_column,label)   
    
  def trigram(self,term):
    x,y,z =term
    stemmer=SnowballStemmer("english")
    x= stemmer.stem(x)
    y= stemmer.stem(y)
    z= stemmer.stem(z)
    label=x+y+z 
    new_column=[]
    for words_stem in self.stemwords:       
      if x in words_stem and y in words_stem and z in words_stem:
          new_column.append('True')
      else:
          new_column.append('False')
    self.dataframegenerator(new_column,label) 
    
  def clusters(self, clusterterms=[]):#special case to join the words that were clustered by  the K means
    
    label=''
    terms=[]
    for x in clusterterms:
      label=str(x)+label
      stemmer=SnowballStemmer("english")
      x= stemmer.stem(x)
      terms.append(x)
    
    new_column=[]
    for words_stem in self.stemwords:
        n=0
        for i in terms:
          if n!=1:
            if i in words_stem:
              new_column.append('True')
              n=1
        if n==0:
              new_column.append('False')
    self.dataframegenerator(new_column,label) 
    
  def instancelevel(self, words=[]): #create a dataframe based on the columns
    self.df= self.df.replace(['y', 'n'], ['True','False'])
    for x in words:
      
      df= self.df[self.df['L'] == 'True']
    self.df=df

  def joinall(self,topwords,ngram):#join the columns from a list of the most frecuent words in a document. 

    if ngram==1:
      for x in topwords:
        df=self.unigram(x)
        #headers_names=list(df.columns.values)
        
    if ngram==2:
      for x in topwords:
        df=self.bigram(x)
      
    if ngram==3:
      for x in topwords:
        df=self.trigram(x)     
    if ngram==4:
     
      for x in topwords:
        df=self.clusters(x)

  def jointwodocuments(self,df,dftest,joindata,typejoin=1):
    self.df=df
   
    self.joinall(joindata,typejoin)
    training=self.df
    self.df=dftest
  
    self.joinall(joindata,typejoin)
    testt=self.df
    return  training, testt
#############################################3
#testing area
