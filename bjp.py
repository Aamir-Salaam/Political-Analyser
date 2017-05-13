import preprocessor as p
from textblob import TextBlob
import pymongo
import json
import re
from pymongo import MongoClient

tmp_list =[]
clean_list=[]
tweet_senti = []
tweet_dict = {}
ptweet=0
ntweet=0
netweet = 0


connection = MongoClient()
db = connection.bhartiya_janta_party
bjp = db.bjp

count = bjp.find().count()
print (count)

p.set_options(p.OPT.RESERVED, p.OPT.URL, p.OPT.NUMBER, p.OPT.SMILEY, p.OPT.EMOJI)   # remove smiley and emoji if they are to be included in final text

for d in bjp.find({},{'text':1,'_id': False}):
    tmp_item = str(d)
    tmp_list.append(p.clean(tmp_item))

for item in tmp_list:
    # trim
    item=item.strip()
    #Removing RT
    item=item.replace('RT','')
    #Removing new line character
    item = item.replace('\\n', '')

    # Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', item)
    # Convert @username to username
    tweet = re.sub(r'@([^\s]+)', r'\1', item)

    item = " ".join(re.findall("[a-zA-Z]+",item))
    tmp_var = re.sub(r'^\S*\s', '', item)
    clean_list.append(tmp_var)


for item in clean_list:
        #print(item)
        # create TextBlob object of passed tweet text
        analysis = TextBlob(item)
        # set sentiment
        if analysis.sentiment.polarity > 0:
            # saving sentiment of tweet
            tweet_score = 'positive'
            ptweet = ptweet + 1
            tweet_dict[item] = tweet_score
        elif analysis.sentiment.polarity == 0:
            # saving sentiment of tweet
            tweet_score = 'neutral'
            netweet = netweet + 1
            tweet_dict[item] = tweet_score
        else:
            # saving sentiment of tweet
            tweet_score = 'negative'
            ntweet = ntweet + 1
            tweet_dict[item] = tweet_score

for k, v in tweet_dict.items():
    print(k,':',v)

neg = 100 * (ntweet) / ((ntweet) + (ptweet))
pos = 100 * (ptweet) / ((ntweet) + (ptweet))

print("\nNegative tweets percentage: {} %".format(neg))
print("Positive tweets percentage: {} %".format(pos))