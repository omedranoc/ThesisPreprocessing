import pandas as pd
from pandas import Series, DataFrame
import numpy as np
class joindocuments(): # receives the domains and x is the value to divide the document and 0 or 1 to indicate if a test set is needed.
  def __init__(self,df1,df2):
    
    self.df1=df1
    self.df2=df2

  def join(self,df1,df2):
    s = pd.concat([df1, df2], ignore_index=True)
    return s
  def varysize(self,document=2,size=1):

    if document==1:
      n=df[df['L']].count()
      n=n*size
      self.df1=self.df1[0:n]
    if document==2:
      self.df2=self.df2[0:n]

  def gettrainingandtest(self,df,trainingsize=1): # 1 means not test size at all
     
    n,m=df.shape
    
    p=int(n*trainingsize)
    df=df.iloc[np.random.permutation(len(df))]
    

    
    training=df[1:p]
    test=df[p:n]
    return training,test
  def getpositivesnegatives(self,df):
    df=df.replace(['y','Y','Y ','y ' ,'n','n '], ['True','True','True','True','False','False'])   
    
    
    positivesarray=df[df['L'] == 'True']   
    Negativesarray=df[df['L'] == 'False']
    
    return positivesarray,Negativesarray

  
   
  def gettrainingandtestp(self,df,value=0.1):
    
    p,n=self.getpositivesnegatives(df)
    
    pt,ptest=self.gettrainingandtest(p,value)
    nt,ntest=self.gettrainingandtest(n,value)
    training=self.join(pt,nt)
    test=self.join(ptest,ntest)
    

    return training,test

  def undersampling(self,df):
   
    p,n=self.getpositivesnegatives(df)
    positives=p[p['L']=='True'].count()
    negatives=n[n['L']=='False'].count()
    
    t=negatives/float(positives['L'])
    total=t['L']
    
    if total>2 and total>1:
       neg1=n[1:positives['L']]
       neg2=n[positives['L']:2*positives['L']]
    elif total<2 and total>1:
       neg1=n[1:positives['L']]
       neg2=n[positives['L']:negatives['L']]
    elif total<0.5:
       pos1=p[0:negatives['L']]
       pos2=p[negatives['L']:2*negatives['L']]
    elif total>0.5 and total<1:
      pos1=p[0:negatives['L']]
      pos2=p[negatives['L']:positives['L']]
    if total >1:
      undersampling1=p.append(neg1, ignore_index=True)
      undersampling2=p.append(neg2, ignore_index=True)
    if total<= 1:
      undersampling1=n.append(pos1, ignore_index=True)
      undersampling2=n.append(pos2, ignore_index=True)
    
    return undersampling1
      
  def joinsourcetarget(self,A,varydocument=0): #varydocument= 0 it varies the source
    training, test=self.gettrainingandtestp(self.df2)
    if varydocument ==0:
      dfvaried, training2=self.gettrainingandtestp(self.df1,A)
      
      #training, training2=self.gettrainingandtestp(training)
      documentsjoined=self.join(dfvaried,training)
    elif varydocument==1:  
      training, training2=self.gettrainingandtestp(training,A)
      documentsjoined=self.join(self.df1,training)
   
    undersampleddf1=self.undersampling(documentsjoined)
    undertest=self.undersampling(test)
    return undersampleddf1, undertest
   


