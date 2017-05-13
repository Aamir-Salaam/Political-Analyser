
import preprocessor as p
import pandas as pd
import random
import json
import re
from unidecode import unidecode
from nltk import word_tokenize
from nltk.classify import NaiveBayesClassifier
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import extract_unigram_feats, mark_negation
from textblob import TextBlob


fname = input("Enter file name(aap.csv,bjp.csv, or inc.csv): ")
data = pd.read_csv(fname)




temp = 1
true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

while(temp < 52):
    analysis = TextBlob(data["text"][temp])

    if analysis.sentiment.polarity > 0:
        if(data["sentiment"][temp] == 1):
            true_positive +=1
        if(data["sentiment"][temp] == 0):
            false_positive +=1
    elif analysis.sentiment.polarity < 0:
        if(data["sentiment"][temp] == 1):
            false_negative += 1
        if (data["sentiment"][temp] == 0):
            true_negative += 1

    temp +=1




print('True Positive = ',true_positive)

print('True Negative = ',true_negative)

print('False Positive = ',false_positive)

print('False Negative = ',false_negative)

precision = true_positive/(true_positive+true_negative)
print('Precision = ',precision)

recall = true_positive/(false_negative+true_positive)
print('Recall = ',recall)

accuracy = (true_positive + true_negative)/(true_negative+true_positive+false_negative+false_positive)
print('Accuracy = ',accuracy)

true_negative_rate = true_negative/(true_negative+true_positive)
print('True Negative Rate = ',true_negative_rate)


f_measure = (2*precision*recall)/(precision+recall)
print('F Measure = ',f_measure)

