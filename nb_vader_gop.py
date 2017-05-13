import preprocessor as p
import pandas as pd
import random
import json
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score 
from unidecode import unidecode
from nltk import word_tokenize
from nltk.classify import NaiveBayesClassifier
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import extract_unigram_feats, mark_negation

data = pd.read_csv("Senti_t1.csv")
# 25000 movie reviews
print data.shape # (25000, 3) 
print data["text"][0]         # Check out the review
print data["sentiment"][0]          # Check out the sentiment (0/1)

sentiment_data = zip(data["text"], data["sentiment"])
random.shuffle(sentiment_data)
# 80% for training
train_X, train_y = zip(*sentiment_data[:8000])
# Keep 20% for testing
test_X, test_y = zip(*sentiment_data[8000:])

#p.set_options(p.OPT.RESERVED, p.OPT.URL, p.OPT.MENTION, p.OPT.HASHTAG, p.OPT.NUMBER, p.OPT.SMILEY, p.OPT.EMOJI)

    
def clean_text(text):

    text = text.replace("<br />", " ")
    text = text.decode("utf-8")
    #text=text.strip()
    text=text.replace('RT','')
    text = text.replace('\\n', '')
    #p.clean(text)
    text = " ".join(re.findall("[a-zA-Z]+",text))
    text = re.sub(r'^\S*\s', '', text)
 
    return text

sentiment = 0.0
tokens_count = 0

text = clean_text(train_X[0])


# mark_negation appends a "_NEG" to words after a negation untill a punctuation mark.
# this means that the same after a negation will be handled differently 
# than the word that's not after a negation by the classifier
print mark_negation("I like the movie .".split())        # ['I', 'like', 'the', 'movie.']
print mark_negation("I don't like the movie .".split())  # ['I', "don't", 'like_NEG', 'the_NEG', 'movie._NEG']
# The nltk classifier won't be able to handle the whole training set
TRAINING_COUNT = 5000
analyzer = SentimentAnalyzer()
vader = SentimentIntensityAnalyzer()
vocabulary = analyzer.all_words([mark_negation(word_tokenize(unidecode(clean_text(instance)))) 
for instance in train_X[:TRAINING_COUNT]])
print "Vocabulary: ", len(vocabulary) # 1356908
print "Computing Unigran Features ..."
unigram_features = analyzer.unigram_word_feats(vocabulary, min_freq=10)
print "Unigram Features: ", len(unigram_features) # 8237
analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_features)


def vader_polarity(text):
    """ Transform the output to a binary 0/1 result """
    score = vader.polarity_scores(text)
    '''
    if score['neu'] > ((score['pos'] + score['neg'])* 6.0):
        point = 'Neutral'
    elif score['pos'] > score['neg']:
        point = 'Positive'
    else:
        point = 'Negative'
        
    return point
    '''
    #return 0 if score['neg'] > score['pos'] else 1
    return 1 if score['pos'] >= score['neg'] else 0


# Build the training set
_train_X = analyzer.apply_features([mark_negation(word_tokenize(unidecode(clean_text(instance)))) 
for instance in train_X[:TRAINING_COUNT]], labeled=False)

# Build the test set
_test_X = analyzer.apply_features([mark_negation(word_tokenize(unidecode(clean_text(instance)))) 
for instance in test_X], labeled=False)

print "Vader Classifier:"

print vader.polarity_scores(train_X[0])
print vader_polarity(train_X[0]), train_y[0] # 0 1
print vader_polarity(train_X[1]), train_y[1] # 0 0
print vader_polarity(train_X[2]), train_y[2] # 1 1
print vader_polarity(train_X[3]), train_y[3] # 0 1
print vader_polarity(train_X[4]), train_y[4] # 0 0


pred_y = [vader_polarity(text) for text in test_X]
print "Vader Accuracy:", accuracy_score(test_y, pred_y) # 0.6892
print "Vader Precision:", precision_score(test_y, pred_y, average = 'binary')
print "Vader Recall:", recall_score(test_y, pred_y, average = 'binary')

trainer = NaiveBayesClassifier.train
classifier = analyzer.train(trainer, zip(_train_X, train_y[:TRAINING_COUNT]))
score = analyzer.evaluate(zip(_test_X, test_y))
print score
print "NB Accuracy: ", score['Accuracy'] # 0.8064 for TRAINING_COUNT=5000
classifyed = NaiveBayesClassifier.classify(_test_X)
print classifyed
