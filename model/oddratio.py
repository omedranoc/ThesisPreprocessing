
class  OddRatio:
  def __init__(self, df,label): #find the odd ratio of a  term
    self.df=df
    self.df= df.replace(['y','Y','Y ','y ' ,'n','n '], ['True','True','True','True','False','False'])
    self.label=label
  def getoddratio(self,column):  
    
    T_word = self.df[column].values #  is the term you want to see is in the tweet
    
    T_label =self.df[self.label].values
    
    i=0
    j=0
    k=0
    l=0
      
    
    for word,label  in zip(T_word, T_label):
      
      if word=='True':
        
        if label == 'True':
          i += 1
        else:
          k+= 1 
  
      elif word=='False':
        if label=='True':
          j+= 1
        elif label!='nan':
          l+= 1
                  
    r1 = i*l
    r2 =float(k*j)
    if r2!=0:
      ratio=r1/r2
    else:
      ratio=0
    
    return ratio
  def getoddratios(self,topwords ):# print the odd radios of a list of words
      ratios={}
      for x in topwords:
        
        
          ratio=  self.getoddratio(x)
          if ratio>1.2:
            ratios[x]=ratio
               
          
      
      return ratios

        


