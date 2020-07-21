## THIS IS A CONTINUATION TO THE SENTIMENT ANALYSIS CODE, WHICH COMPARES TWO
## COMMON METHODS OF SENTIMENT ANALYSIS, WHICH ARE TEXTBLOB AND
## NAIVEBAYESCLASSIFIER. BOTH THE METHODS USE NLTK AS THEIR BASE, BUT TEXTBLOB
## ALSO TELLS WHETHER THE TEXT IS SUBJECTIVE OR OBJECTIVE. NAIVEBAYESCLASSIFIER
## IS FOUND TO BE MORE ACCURATE.

import nltk.classify.util                ##THESE ARE THE LIBRARIES USED IN THIS CODE.
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist
import pickle
from textblob import TextBlob
import tkinter.simpledialog


##THIS CODE IS THE SAME AS THE SENTIMENT ANALYSIS ONE, EXCEPT THAT WE HAVE ADDED THE ANALYSIS OF TEXTBLOB AS WELL FOR THE SAKE OF COMPARISON.

fdist = {}
classifier_f = open('Sentiment Analysis.pickle','rb')
classifier = pickle.load(classifier_f)
classifier_f.close()
neg_reviews = []
pos_reviews = []
def input_to_classifier(words):
    useful_words = []
    for i in words:
        if i not in stopwords.words('english'):
            useful_words.append(i)
    word_dict = {}
    for i in useful_words:
        if len(i) > 1 and i.isalpha():
            word_dict[i.lower()] = True
    return word_dict

#classifier.show_most_informative_features(2000)
with open('Negative_Reviews.txt','r') as f:
    text = f.readlines()
    for i in text:
        neg_reviews.append(''.join(i.split()))
with open('Positive_Reviews.txt','r') as f:
    text = f.readlines()
    for i in text:
        pos_reviews.append(''.join(i.split()))

def positivity(phrase, pos_reviews):
    pos_words = 0
    for i in phrase.keys():
        if i in pos_reviews:
            pos_words = pos_words + 1
    return pos_words

def negativity(phrase, neg_reviews):
    neg_words = 0
    for i in phrase.keys():
        if i in neg_reviews:
            neg_words += 1
    return neg_words
input_text = "Something must be wrong here! This is not a good movie at all. Boring and cringy, an SJW fail. Black Panther was quite cool in Captain America, but don't be fooled. I'm sorry I paid for this, spare your money and don't support this crap or they will make more crap."

##THIS IS THE ADDITIONAL CODE WHICH GETS THE POLARITY OF THE TEXT FROM THE SENTIMENT ANALYSER BUILT IN TEXTBLOB. IT GIVES THE POLARITY AND THE SUBJECTIVITY OF A TEXT.
##THE POLAIRTY IS FROM -1 TO 1, WITH LESS THAN 0 BEING NEGATIVE, 0 BEING NUETRAL AND GREATER THAN ZERO BEING POSITIVE.

analysis = TextBlob(input_text)
pol,subj = analysis.sentiment
if pol < 0:
    print('This text can be classified as NEGATIVE using textblob\n')
elif pol > 0:
    print('This text can be classified as POSITIVE using textblob\n')
else:
    print('This text can be classified as NUETRAL using textblob\n')

input_text = word_tokenize(input_text)
for i in FreqDist(input_text):
    if i in fdist:
        fdist[i] += 1
    else:
        fdist[i] = 1
input_text = input_to_classifier(input_text)

pos = positivity(input_text ,pos_reviews)
neg = negativity(input_text ,neg_reviews)

if pos != 0 or neg != 0:
    pos_percent = (pos/(pos + neg)) * 100
    neg_percent = (neg/(pos + neg)) * 100
else:
    pos_percent = 0
    neg_percent = 0

    
print('This text has ' + str(pos_percent) + '% positivity and ' + str(neg_percent) + '% negativity according to the way the classifier has been trained.\n')
print('Overall this text can be classified as ' + classifier.classify(input_text).upper() + ' using NaiveBayesClassifier')
