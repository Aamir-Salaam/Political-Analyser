import preprocessor as p
from textblob import TextBlob
import pymongo
import json
import re
from pymongo import MongoClient
import matplotlib.pyplot as plt
import pandas as pd

tmp_list =[]
clean_list=[]
tweet_senti = []
tweet_dict = {}
ptweet=0
ntweet=0
ntrl=0
flag=0

connection = MongoClient()
db = connection.indian_national_congress
posts = db.bjp

count = posts.find().count()
print (count)

p.set_options(p.OPT.RESERVED, p.OPT.URL, p.OPT.MENTION, p.OPT.HASHTAG, p.OPT.NUMBER, p.OPT.SMILEY, p.OPT.EMOJI)   # remove smiley and emoji if they are to be included in final text

for d in posts.find({},{'text':1,'_id': False}):
    tmp_item = str(d)
    tmp_list.append(p.clean(tmp_item))

for item in tmp_list:
    item=item.strip()
    item=item.replace('RT','')
    item = item.replace('\\n', '')
    item = " ".join(re.findall("[a-zA-Z]+",item))
    tmp_var = re.sub(r'^\S*\s', '', item)
    clean_list.append(tmp_var)


df= pd.DataFrame(clean_list,columns = ["text"])
df.to_csv('inc.csv', index=False)