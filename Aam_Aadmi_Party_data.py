import pymongo
import time
import tweepy
from tweepy import OAuthHandler
import json

# Connection to Mongo DB
try:
    conn = pymongo.MongoClient()
    print("Connected successfully!!!")
except pymongo.errors.ConnectionFailure as e:
    print("Could not connect to MongoDB:")
finally:
    print(0)

# Twitter API credentials
consumer_key = 'iDDEROs6dT9g6o3AFuPNyvUP5'
consumer_secret = 'y14XOX3L1jLzJiYfvMQiYgwPkczmK90gOA1HpkuTbzUeNnet4G'
access_key = '785123558879993857-jLlJqxvCxnCgN7crwPONUi7LPX2ZTCu'
access_secret = 'h5EhJ2dwDEIyAIpoFrGI3gI2PgbPrsx6oXppxh73vKdvz'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
# refer http://docs.tweepy.org/en/v3.2.0/api.html#API
# tells tweepy.API to automatically wait for rate limits to replenish


# Define my mongoDB database
db = conn.aam_aadmi_party
# Define my collection where I'll insert my search
aap = db.aap

# Put your search term
searchquery = "arvind kejriwal OR aap"

users = tweepy.Cursor(api.search, q=searchquery, geocode="28.7041,77.1025,40km").items()
count = 0
errorCount = 0

while True:
    try:
        user = next(users)
        # use count-break during dev to avoid twitter restrictions
        if(count > 100):
            break
    except tweepy.TweepError:
        # catches TweepError when rate limiting occurs, sleeps, then restarts.
        # nominally 15 minnutes, make a bit longer to avoid attention.
        print("sleeping....")
        time.sleep(60 * 16)
        user = next(users)
    except StopIteration:
        break
    try:
        language = user._json.get("lang")
        if(language == "en"):
            count += 1
            print("Writing to JSON tweet number:" + str(count))
            aap.insert(user._json)
    except UnicodeEncodeError:
        errorCount += 1
        print("UnicodeEncodeError,errorCount =" + str(errorCount))

print("completed, errorCount =" + str(errorCount) + " total tweets=" + str(count))

count = aap.find().count()
print(count)

#for d in aap.find({}, {'text': 1}):
    #print(d)
