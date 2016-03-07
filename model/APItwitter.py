'''
1. You need register a twitter account and register an app with them to get the token, in order to calling their API. the url below https://dev.twitter.com/. There are some example code the url http://nbviewer.ipython.org/github/ptwobrussell/Mining-the-Social-Web-2nd-Edition/blob/master/ipynb/Chapter%201%20-%20Mining%20Twitter.ipynb

2. You need to install python. I use python 2.7, so you'd better use python 2.x, because latest 3.x has big change, maybe there is some issue for version.  https://www.python.org/download/releases/2.7.5/

3. You need to install some python module, which have been used in the code. You can follow their instruction.
      3.1   glob      https://docs.python.org/2/library/glob.html
      3.2   twitter    twitter API for python https://pypi.python.org/pypi/twitter
      3.3   textblob  text sentiment api  http://textblob.readthedocs.org/en/dev/

4. You need fill the parameters below:
        SearchTerm
        Count
        api token abcd from step1
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding: utf-8
import codecs
import re
import urllib2
import glob
import base64, json, time
import datetime
import os.path
import twitter
from textblob import TextBlob
import csv
import sys
from xml.etree import ElementTree
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import SubElement
import xml.etree.ElementTree as ET
import json
import collections
import nltk
import numpy as np
from collections import Counter
import pandas as pd
from pandas import Series, DataFrame
from itertools import *
import itertools
import datetime
SearchTerm= "pregnant"
Count=30000
g=1
api = twitter.Api(consumer_key='c2NwTYrCTqFVBYtbrWFUCDSwS',
                      consumer_secret='LWAuv9MhjhGba9kBbh2UtDftma8DL1UmgBbQgWHqvJoxxaKCD9',
                      access_token_key='2996364660-7kJtNqsODiNcBwCxypJj2IVJjZCIxQHyRnHdBua',
                      access_token_secret='9vZ2puzzrZDlGX9h6yJNUVEwDy70acq8kKpp5NusujWwl')

def sentiment(message, idtwitter, Urltwitter, reTwitter, Cleanurl):
 
  text = TextBlob(message)
  
  response = [str(EachResult.id),Var_url ,str(round(text.polarity, 2)), str(abs(text.polarity) ) , str(nourls),  str(round(text.subjectivity, 2)), str(EachResult.text.replace(",", " ").replace("\u2018", "'").replace("\u2019", "'").replace("u'", "'").encode('utf-8'))]
  #['id', 'url',  'sentiment_polarity','abs Polarity',  'Clean tweet','sentiment_subjectivity','tweet']
  return response
  UTF8Writer = codecs.getwriter('utf8')
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
until=501886166715531264  #how does it work
n=0

while(n<(Count+1)):
    SearchResult= api.GetSearch(term=SearchTerm, since_id=until, lang='en', result_type='mixed') # data = json.load(SearchResult)
    repetidos = [] 
    for  EachResult in SearchResult:
        until=EachResult.id
        if('RT' not in EachResult.text) and ('porn' not in EachResult.text):
            nourls = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", EachResult.text)
            
            repetidos.append(nourls)
             
                    


         
            result = json.loads(str(EachResult)) # the functions with s have string parameters
            var_retweet= result["retweeted"]
            #print (var_retweet)
            if "urls" in result:
              Var_url = 1  
                
              
            else:
              Var_url = 0 
            
            now = datetime.datetime.today().strftime("%B %d, %Y")
            print now
            sentiment_result = sentiment(EachResult.text, EachResult.id, Var_url, var_retweet, nourls)
            #text1 = TextBlob(EachResult.text)
            #result1 ={ 'id':EachResult.id,'url':Var_url, 'Clean tweet':str(nourls), 'sentiment_polarity' : str(round(text1.polarity, 2)) , 'sentiment_subjectivity' : str(round(text1.subjectivity, 2)), 'tweet': str(EachResult.text.replace(",", " ").replace("\u2018", "'").replace("\u2019", "'").replace("u'", "'").encode('utf-8'))}       
            frame = DataFrame({'id':[], 'url':[],  'sentiment_polarity':[],'abs Polarity':[],  'Clean tweet':[],'sentiment_subjectivity':[],'tweet':[],'now':[]},  columns=['id', 'url',  'sentiment_polarity','abs Polarity',  'Clean tweet','sentiment_subjectivity','tweet','now'])
            text1 = TextBlob(EachResult.text)
            if Var_url == 0 :
              cuenta =1
              cuenta= cuenta +1
              print cuenta
              response1 = {'id':str(EachResult.id),'url':Var_url ,'sentiment_polarity':str(round(text1.polarity, 2)), 'abs Polarity':str(abs(text1.polarity) ) ,'Clean tweet': str(nourls), 'sentiment_subjectivity': str(round(text1.subjectivity, 2)), 'tweet':str(EachResult.text.replace(",", " ").replace("u'", "'").encode('utf-8')), 'now':now}
              #['id', 'url',  'sentiment_polarity','abs Polarity',  'Clean tweet','sentiment_subjectivity','tweet']
              frame1=frame.append(response1, ignore_index=True)
              frame2= frame1.drop_duplicates(['tweet'])
            
            #print (frame) 
            

            print (g)
            if g==1 and Var_url==0:
            
              
              with open(r'documents\csv\pregnancy\pregnant.csv', 'a') as f:
                   frame2.to_csv(f, encoding='utf-8')
                   print (g)     
              g= g+1

            
            if g!=1 and Var_url==0:
              with open(r'documents\csv\pregnancy\pregnant.csv', 'a') as f:
                   frame2.to_csv(f, encoding='utf-8',header=False)


           

          
                            
             
                          
           
            n=n+1
         

    time.sleep(6)


    

            