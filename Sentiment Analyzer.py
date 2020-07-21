## THIS IS THE CODE WHICH IS CLASSIFYING THE TEXT. WE HAVE ALSO USED A LIST OF
## POSITIVE AND NEGATIVE WORDS TO GIVE THE PERCENTAGE OF POSITIVITY AND
## NEGATIVITY IN A TEXT. WE HAVE ALSO IMPLEMENTED A SIMPLE GUI, WHICH PROMPTS
## THE USER FOR INPUT AND GIVES THE OUTPUT USING ANOTHER PROMPT.
import nltk.classify.util                           ##THESE ARE ALL THE LIBRARIES THAT HAVE BEEN USED AT SOME POINT IN THIS CODE.
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist
import pickle
from textblob import TextBlob
import tkinter.messagebox
import tkinter.simpledialog
from tkinter import *

fdist = {}
classifier_f = open('Sentiment Analysis.pickle','rb')  ##THIS IS THE COMMAND TO OPEN THE PICKLE FILE AND LOAD THE CLASSIFIER THAT WAS MADE IN THIS CLASSIFIER TRAINER
                                                       ##FILE.
classifier = pickle.load(classifier_f)
classifier_f.close()


neg_reviews = []
pos_reviews = []

def input_to_classifier(words):          ##THIS IS THE FUNCTION WHICH FILTERS THE INPUT TEXT IN ORDER TO MAKE IT READY TO BE USED AND ANALYSED BY THE CLASSIFIER.
    useful_words = []
    for i in words:
        if i not in stopwords.words('english'):
            useful_words.append(i)
    word_dict = {}
    for i in useful_words:
        if len(i) > 1 and i.isalpha():
            word_dict[i.lower()] = True
    return word_dict

#classifier.show_most_informative_features(2000)    ##THIS IS THE COMMAND WHICH SHOWS THE PILLARS OF THE CLASSIFIER WITH EVERY WORD BEING SHOWN WITH ITS POSITIVE
                                                    ##TO NEGATIVE RATIO. THIS COMMAND SHOWS ALL THE WORDS IN THE CLASSIFIER.

##IN THE UPCOMING TWO 'WITH' COMMANDS, WE ARE OPENING THE NEGATIVE AND POSITIVE REVIEWS FILES THAT WE CREATED IN THE CLASSIFIER TRAINER CODE AND READING THE WORDS
##THAT ARE IN THE FILE AND APPENDING THEM TO THEIR RESPECTIVE LISTS.

with open('Negative_Reviews.txt','r') as f:
    text = f.readlines()
    for i in text:
       neg_reviews.append(''.join(i.split()))
with open('Positive_Reviews.txt','r') as f:
    text = f.readlines()
    for i in text:
        pos_reviews.append(''.join(i.split()))


##IN THE TWO UPCOMING FUNCTIONS 'POSITIVITY' AND 'NEGATIVITY', WE ARE COUNTING THE NUMBER OF POSITIVE TO NEGATIVE WORDS IN OUR INPUT BY COMPARING IT TO THE
##NEGATIVE AND POSITIVE REVIEWS LIST.

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


##THE NEXT TWO COMMAND HAVE BEEN USED TO IMPLEMENT A BASIC GUI, IN WHICH THE USER WILL BE ASKED FOR INPUT THROUGH A PROMPT AND THE OUTPUT WILL BE GIVEN THROUGH
##ANOTHER PROMPT.


#tkinter.messagebox.showinfo('Welcome','Type your text to be analysed here')
#input_text = tkinter.simpledialog.askstring('Text','Enter text to be analysed')

##INPUT TEXT:

input_text = "Something must be wrong here! This is not a good movie at all. Boring and cringy, an SJW fail. Black Panther was quite cool in Captain America, but don't be fooled. I'm sorry I paid for this, spare your money and don't support this crap or they will make more crap."


input_text = word_tokenize(input_text) ##WE HAVE USED THE TOKENIZE FUNCTION IN NLTK TO BREAK OUR INPUT INTO TOKENS. IT WORKS SIMILAR TO HOW THE SPLIT FUNCTION WORKS
                                       ##IN PYTHON.


for i in FreqDist(input_text):      ##FREQDIST IS ALSO A FUNCTION IN NLTK WHICH IS AN OBJECT WHICH CONTAINS THE FREQUENCY OF EVERY WORD APPEARING IN A TEXT.
                                    ##WE HAVE CALLED FREQDIST AND THEN ITERATED OVER IT TO OUTPUT 'FDIST', WHICH CONTAINS THE FREQUENCY OF EVERY WORD IN OUT INPUT.
    if i in fdist:
        fdist[i] += 1
    else:
        fdist[i] = 1

input_text = input_to_classifier(input_text)    ##AFTER BREAKING THE INPUT INTO TOKENS, WE SEND IT TO THE INPUT TO CLASSIFIER FUNCTION, TO GET IT READY TO BE
                                                ##BE ANALYSED BY THE CLASSIFIER.

#output  = classifier.classify(input_text)

##POS AND NEG STORE THE NUMBER OF POSITIVE WORDS AND NEGATIVE WORDS IN OUR INPUT, THAT WE GET FROM THE POSITIVITY AND NEGATIVITY FUNCTIONS DEFINED ABOVE.

pos = positivity(input_text,pos_reviews)         
neg = negativity(input_text,neg_reviews)



if pos != 0 or neg != 0:                  ##HERE WE ARE CALCULATING THE PERCENTAGE OF POSITIVITY AND NEGATIVITY USING THE NUMBER OF POS AND NEG WORDS
                                          ##IN OUR INPUT HELPED BY A SIMPLE IF CONDITION TO PREVENT RUNTIME ERROR.
    pos_percent = (pos/(pos + neg)) * 100
    neg_percent = (neg/(pos + neg)) * 100
    
else:
    pos_percent = 0
    neg_percent = 0

##THE NEXT TWO PRINT COMMANDS GIVE THE FINAL OUTPUT BY TELLING THE PERCENTAGE OF POSITIVITY AND NEGATIVITY AND THE FINAL OUTPUT OF THE CLASSIFIER, WHICH IS ONE OF
##'NEGATIVE' OR 'POSITIVE'. THE OUTPUT FROM THE CLASSIFIER IS GOT USING THE 'classifier.classify(input_text)' COMMAND IN SECOND PRINT COMMAND.


print('This text has ' + str(pos_percent) + '% positivity and ' + str(neg_percent) + '% negativity according to the way the classifier has been trained.\n')
print('Overall this text can be classified as ' + classifier.classify(input_text).upper())

##THE COMMAND BELOW IS IF WE WISH TO GET THE OUTPUT THROUGH A PROMPT.

#tkinter.messagebox.showinfo('Result','This text has ' + str(pos_percent) + ' % positivity and ' + str(neg_percent) + ' % negativity. Overall, this text can be classified as ' + output)

