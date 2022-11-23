import nltk
import os

path = "D:\\PycharmProject\\TVLab\\Lab3\\news.txt"

with open(path, 'r') as f:
    txt = f.read()
# print(txt)

tokens = nltk. word_tokenize(txt)
postags = nltk.pos_tag(tokens)
print(postags)

from nltk.metrics import *
ref = 'PRP JJ NNP IN NNP VBZ RB NNP WRB PRP RBR VBF VBF IN DT JJ NN IN CD IN DT NN IN NN DT JJ NNP NNP VBZ VBG DT NN IN NNPS VBG JJ NNS NNS WDT VBP DT NNS IN DT NNS RB VBD CC NNP IN NNS IN NNP'.split()
tagged = 'PRP$ JJ NN IN NN VBZ RB NNP WRB PRP RB VBD VBG IN DT JJ NN IN CD IN DT NN IN NN DT JJ NN NN VBZ VBG DT NN IN NNS VBG JJ NNS NNS WDT VBP DT NNS IN DT NNS RB VBD CC NN IN NNS IN NNP'.split()
# cm = ConfusionMatrix(ref, tagged)
# print(cm)
print("Precision: ",precision(set(ref),set(tagged)))
print("Recall: ",recall(set(ref),set(tagged)))
print("F measure: ",f_measure(set(ref),set(tagged)))
print("Accuracy: ",accuracy(ref,tagged))
