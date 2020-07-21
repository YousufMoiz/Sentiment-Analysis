### THIS IS THE CODE FOR THE TRAINING OF THE CLASSIFIER. WE ARE USING NAIVE
### BAYES CLASSIFIER FOR OUR SENTIMENT ANALYSIS. WE HAVE USED THE MOVIE REVIEWS
### CORPUS WHICH HAS 1000 POSITIVE AND 1000 NEGATIVE MOVIE REVIEWS AND USED 1500
### OF THESE REVIEWS TO TRAIN THE CLASSIFIER AND 500 REVIEWS TO TEST IT. THE
### CLASSIFIER, AFTER BEING TRAINED COMES OUT TO BE 89.6 % ACCURATE.
import nltk.classify.util                           ##THESE ARE ALL THE LIBRARIES USED IN THIS CODE.
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
from textblob import TextBlob
def input_to_classifier(words):     ##THIS IS THE FUNCTION WHICH GETS THE INPUT THAT THE CLASSIFIER IS EXPECTING, THAT IS ALL THE STOP WORDS
                                    ##DELETED AND NUMERIC CHARACTERS DISCARDED AND THE INPUT BEING IN THE FORM OF A DICTIONARY WITH KEY BEING THE WORD AND VALUE
                                    ##BEING TRUE.
    useful_words = []
    for i in words:
        if i not in stopwords.words('english'):
            useful_words.append(i)
    word_dict = {}
    for i in useful_words:
        if len(i) > 1 and i.isalpha():
            word_dict[i] = True
    return word_dict

negative_reviews_list = []
positive_reviews_list = []
category = movie_reviews.categories()
for i in category:              ##THIS IS THE LOOP TO ACCESS THE MOVIE REVIEWS CORPUS, WHICH HAS TWO CATEGORIES 'NEG' AND 'POS'.
                                ##THE WORDS IN THE REVIEWS, AFTER BEING FILTERED ARE BEING STORED IN NEG AND POS REVIEW LISTS AS TUPLES
                                ##OF DICTIONARIES CONTAINING THE WORDS AS KEY AND TRUE AS VALUE AND THE CATEGORY FROM WHICH IT WAS TAKEN.
    if i == 'neg':
        for fileid in movie_reviews.fileids(i):
            words = movie_reviews.words(fileid)
            s = input_to_classifier(words)
            negative_reviews_list.append((s,'negative'))
    else:
        for fileid in movie_reviews.fileids(i):
            words = movie_reviews.words(fileid)
            s = input_to_classifier(words)
            positive_reviews_list.append((s,'positive'))

            
file1 = open('Negative_Reviews.txt','w')  ##THREE FILES HAVE BEEN OPENED WHICH STORES NEGATIVE,POSITIVE AND NUETRAL WORDS FROM THE MOVIE REVIEWS CORPUS.
file2 = open('Positive_Reviews.txt','w')
file3 = open('Nuetral_Reviews.txt','w')


for i in negative_reviews_list:
    for j in i[0].keys():                  ##TO FILTER THE WORD AS BEING POS,NEG OR NUETRAL WE ARE USING THE TEXTBLOB LIBRARY, WHICH HAS A FUNCTION
                                           ##WHICH TELLS THE POLARITY OF THE WORD ON A SCALE OF -1 TO 1, WITH 0 TO -1 BEING NEGATIVE, 0 BEING
                                           ##NUETRAL AND 0 TO 1 BEING POSITIVE. THEN WE ARE WRITING THE WORD TO ITS RESPECTIVE FILE ACCORDING TO THE POLARITY.
        pol, subj = TextBlob(j).sentiment
        if pol < 0 :
            file1.writelines(j + '\n')
        elif pol ==0 :
            file3.writelines(j + '\n')
        else:
            file2.writelines(j + '\n')

for i in positive_reviews_list:
    for j in i[0].keys():
        pol, subj = TextBlob(j).sentiment
        if pol < 0 :
            file1.writelines(j + '\n')
        elif pol ==0 :
            file3.writelines(j + '\n')
        else:
            file2.writelines(j + '\n')
file1.close()        
file2.close()
file3.close()

##THE CLASSIFIER IS USUALLY TRAINED AND TESTED USING A RATIO OF 75:25 TRAIN TO TEST DATA RATIO. THOUGH A 50:50 RATIO ALSO WORKS AS WELL.
##THE ADVANTAGE WITH NAIVEBAYESCLASSIFIER IS THAT IT REQUIRES VERY LESS DATA TO FUNCTION PROPERLY, THOUGH ACCURACY INCREASES AS MORE AND MORE DATA WE USE.

train_set = negative_reviews_list[:750] + positive_reviews_list[:750] ##THIS IS THE TRAIN SET, WHICH WILL BE USED TO TRAIN THE NAIVEBAYESCLASSIFIER. IT CONSISTS OF
                                                                      ##750 REVIEWS FROM NEGATIVE REVIEWS LIST AND 750 FROM POSITIVE REVIEWS LIST.

test_set =  negative_reviews_list[250:] + positive_reviews_list[250:] ##THIS IS THE TEST SET THAT WE USED TO TEST THE ACCURACY OF OUR CLASSIFIER, WITH 250 NEGATIVE
                                                                      ##REVIEWS AND 250 POSITIVE REVIEWS.


classifier = NaiveBayesClassifier.train(train_set)     ##THIS IS THE COMMAND TO TRAIN OUR CLASSIFIER.


accuracy = nltk.classify.util.accuracy(classifier,test_set)    ##THIS IS THE COMMAND TO TEST THE ACCURACY OF OUR CLASSIFIER.

print('Accuracy of the above model is:' , str(accuracy*100))


classifier.show_most_informative_features(5)            ##THIS IS THE COMMAND WHICH SHOWS THE PILLARS OF THE CLASSIFIER WITH EVERY WORD BEING SHOWN WITH ITS POSITIVE
                                                        ##TO NEGATIVE RATIO. THIS COMMAND ONLY SHOWS THE FIRST FIVE WORDS.

save_classifier = open('Sentiment Analysis.pickle','wb') ##THIS THE COMMAND WHICH OPENS A PICKLE FILE, WHICH STORES OBJECTS, TO STORE OUR CLASSIFIER
                                                         ##SO THAT WE DO NOT HAVE TO TRAIN IT AGAIN AND AGAIN WHENEVER WE HAVE TO DO ANALYSIS OF ANY TEXT.
pickle.dump(classifier, save_classifier)
save_classifier.close()

