import preprocessor as p
import pandas as pd
import random
import json
import re
import nltk
import numpy as np
import itertools
import time
# import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk.classify.util
from unidecode import unidecode
from nltk import word_tokenize
from nltk.classify import NaiveBayesClassifier
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import extract_unigram_feats, mark_negation
import matplotlib.pyplot as plt

data = pd.read_csv("bjp.csv")
# 25000 movie reviews
print data.shape  # (25000, 3)
print data["text"][0]  # Check out the review
print data["sentiment"][0]  # Check out the sentiment (0/1)

sentiment_data = zip(data["text"], data["sentiment"])
# random.shuffle(sentiment_data)
# 80% for training
train_X, train_y = zip(*sentiment_data[:50])
# Keep 20% for testing
test_X, test_y = zip(*sentiment_data[100:500])


# p.set_options(p.OPT.RESERVED, p.OPT.URL, p.OPT.MENTION, p.OPT.HASHTAG, p.OPT.NUMBER, p.OPT.SMILEY, p.OPT.EMOJI)


def clean_text(text):
    text = text.replace(',', ' ')
    text = text.replace("<br />", " ")
    text = text.decode("utf-8")
    # text=text.strip()
    text = text.replace('RT', '')
    text = text.replace('\\n', '')
    # p.clean(text)
    text = " ".join(re.findall("[a-zA-Z]+", text))
    text = re.sub(r'^\S*\s', '', text)

    return text


sentiment = 0.0
tokens_count = 0

text = clean_text(train_X[0])

# mark_negation appends a "_NEG" to words after a negation untill a punctuation mark.
# this means that the same after a negation will be handled differently
# than the word that's not after a negation by the classifier
print mark_negation("I like the movie .".split())  # ['I', 'like', 'the', 'movie.']
print mark_negation("I don't like the movie .".split())  # ['I', "don't", 'like_NEG', 'the_NEG', 'movie._NEG']
# The nltk classifier won't be able to handle the whole training set
TRAINING_COUNT = 50
vader = SentimentIntensityAnalyzer()
analyzer = SentimentAnalyzer()
vocabulary = analyzer.all_words([mark_negation(word_tokenize(unidecode(clean_text(instance))))
                                 for instance in train_X[:TRAINING_COUNT]])
print "Vocabulary: ", len(vocabulary)  # 1356908
print "Computing Unigram Features ..."
unigram_features = analyzer.unigram_word_feats(vocabulary, min_freq=10)
print "Unigram Features: ", len(unigram_features)  # 8237
analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_features)


def vader_polarity(text):
    """ Transform the output to a binary 0/1 result """
    score = vader.polarity_scores(text)

    return 1 if score['pos'] >= score['neg'] else 0
    #return 0 if score['neg'] > score['pos'] else 1


# Build the training set
_train_X = analyzer.apply_features([mark_negation(word_tokenize(unidecode(clean_text(instance))))
                                    for instance in train_X[:TRAINING_COUNT]], labeled=False)

# Build the test set
_test_X = analyzer.apply_features([mark_negation(word_tokenize(unidecode(clean_text(instance))))
                                   for instance in test_X], labeled=False)

print "Sample training result scores(obtained / actual):"
print vader_polarity(train_X[0]), train_y[0]
print vader_polarity(train_X[1]), train_y[1]
print vader_polarity(train_X[2]), train_y[2]
print vader_polarity(train_X[3]), train_y[3]
print vader_polarity(train_X[5]), train_y[5]
print vader_polarity(train_X[6]), train_y[6]
print vader_polarity(train_X[7]), train_y[7]
print vader_polarity(train_X[8]), train_y[8]
print vader_polarity(train_X[9]), train_y[9]

pred_x = [vader_polarity(text) for text in train_X]
acc = accuracy_score(train_y, pred_x)
pres = precision_score(train_y, pred_x, average = 'binary')
rec = recall_score(train_y, pred_x, average = 'binary')
f1 = f1_score(train_y, pred_x, average = 'binary')
cnf_matrix = confusion_matrix(train_y, pred_x)
target_names = ['Negative', 'Positive']
class_names = ['Positive','Negative']
class_rep = classification_report(train_y, pred_x, target_names=target_names)
print "Vader Accuracy:", acc # 0.6892
print "Vader Precision:", pres
print "Vader Recall:", rec
print "Vader F1 measure:", f1
print cnf_matrix
print class_rep


print "Tweets sentiment scores are: "
pred_y = [vader_polarity(text) for text in test_X]
print pred_y


ntweet=0
ptweet=0

#code for plotting graph
for item in pred_y:
    if item == 0:
        ntweet=ntweet+1
    elif item == 1:
        ptweet=ptweet+1

print ptweet
print ntweet

x = [1,2]
y = [ptweet,ntweet]
plt.title('BHARTIYA JANTA PARTY')
plt.ylabel("Tweet Count")
plt.xlabel("Sentiment")
plt.xticks([1,2],['Positive','Negative'])
#xlbl=['Positive','Neutral','Negative']
barlist=plt.bar(x, y,width = 0.9, color = 'red', linewidth = 2 ,align = 'center')
barlist[0].set_color('blue')
#barlist[0].set_edgecolor('black')
barlist[0].set_linewidth(2)
#plt.bar(x, y,width = 0.9, color = 'red', edgecolor = 'black', linewidth = 2 ,align = 'center')
plt.show()
#plt.savefig(os.path.join('bjp_results.png'), dpi=300, format='png', bbox_inches='tight') 
#################


##################

'''
p = [1,2,3,4]
y = [acc,pres,rec,f1]
plt.title('Sentiment Metrics')
plt.ylabel("Value")
plt.xlabel("Metrics")
plt.xticks([1,2,3,4],['accuracy','precision','recall','f1_measure'])
#xlbl=['Positive','Neutral','Negative']
plt.bar(x, y,width = 0.9, color = 'blue', edgecolor = 'black', linewidth = 2 ,align = 'edge')
plt.show()
'''
####################
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

#####################


def show_values(pc, fmt="%.2f", **kw):
    
    from itertools import izip
    pc.update_scalarmappable()
    ax = pc.get_axes()
    for p, color, value in izip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)


def cm2inch(*tupl):
   
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=False, cmap='RdBu'):
    '''
    '''

    # Plot it out
    fig, ax = plt.subplots()    
    #c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=cmap)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)      

    # Remove last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()    
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell 
    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()       

    # resize 
    fig = plt.gcf()
    #fig.set_size_inches(cm2inch(40, 20))
    #fig.set_size_inches(cm2inch(40*4, 20*4))
    fig.set_size_inches(cm2inch(figure_width, figure_height))



def plot_classification_report(classification_report, title='Classification report(BJP Dataset) ', cmap='RdBu'):
  
    lines = classification_report.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[2 : (len(lines) - 2)]:
        t = line.strip().split()
        if len(t) < 2: continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        print(v)
        plotMat.append(v)

    print('plotMat: {0}'.format(plotMat))
    print('support: {0}'.format(support))

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    #yticklabels = ['Positive','Negative']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height, correct_orientation, cmap=cmap)

# trainer = NaiveBayesClassifier.train
# classifier = nltk.NaiveBayesClassifier.train(_train_X)
# classifier = analyzer.train(trainer, zip(_train_X, train_y[:TRAINING_COUNT]))
# classifier.classify(zip(_test_X))
# score = analyzer.evaluate(zip(train_y, pred_x))
# print score
# print "Accuracy: ", score['Accuracy'] # 0.8064 for TRAINING_COUNT=5000
plot_classification_report(class_rep)
plt.savefig('test_plot_classif_report_bjp.png', dpi=200, format='png', bbox_inches='tight')
plt.close()

