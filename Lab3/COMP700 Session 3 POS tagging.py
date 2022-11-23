##########################################################################################################################
##Lets use some POS taggers and see how they perform.
##Regualar expression tagger

import nltk
from nltk.corpus import brown
brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')
patterns = [
     (r'.*ing$', 'VBG'),               # gerunds
     (r'.*ed$', 'VBD'),                # simple past
     (r'.*es$', 'VBZ'),                # 3rd singular present
     (r'.*ould$', 'MD'),               # modals
     (r'.*\'s$', 'NN$'),               # possessive nouns
     (r'.*s$', 'NNS'),                 # plural nouns
     (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
     (r'.*', 'NN')                     # nouns (default)
 ]
regexp_tagger = nltk.RegexpTagger(patterns)
print(brown_sents[3])
print(regexp_tagger.tag(brown_sents[3]))
print("Accuracy: ",  regexp_tagger.evaluate((brown_tagged_sents)))


##########################################################################################################################
# Separating the training and the test data
import nltk
from nltk.corpus import brown
brown_tagged_sents = brown.tagged_sents(categories='news')
size = int(len(brown_tagged_sents) * 0.9)
print(size)
train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]
unigram_tagger = nltk.UnigramTagger(train_sents)
print(unigram_tagger.evaluate(test_sents))
########################################################################################################################
# Storing a trained model, retrieving it and using it.

#Store it.
# from pickle import dump
# output = open('ugTagger.pkl', 'wb')
# dump(unigram_tagger, output, -1)
# output.close()
#
# #Retrieve it from a file
# from pickle import load
# input = open('ugTagger.pkl', 'rb')
# tagger = load(input)
# input.close()
# #Use it.
# text = "The board's action shows what free enterprise is up against in our complex maze of regulatory laws ."
# tokens = text.split()
# print(tagger.tag(tokens))
# print(unigram_tagger.evaluate(test_sents))
#########################################################################################################################
# Using confusion matrix for evaluation.
# from nltk.metrics import *
# ref  = 'DET NN VB DET JJ NN NN IN DET NN'.split()
# tagged = 'DET VB VB DET NN NN NN IN DET NN'.split()
# cm = ConfusionMatrix(ref, tagged)
# print(cm)
# print("Precision: ",precision(set(ref),set(tagged)))
# print("Recall: ",recall(set(ref),set(tagged)))
# print("F measure: ",f_measure(set(ref),set(tagged)))
# print("Accuracy: ",accuracy(ref,tagged))
# print("-------------------------------------------------------------")
# #another way, library, whats the difference?
# from sklearn import metrics
# print(metrics.classification_report(ref, tagged))
#########################################################################################################################
