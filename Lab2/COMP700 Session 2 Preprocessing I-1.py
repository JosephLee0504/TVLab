import nltk
######################Simple Tagging###############################################################################################
text = nltk.word_tokenize("The city of Auckland is in New Zeland which is in the Pacific")
print(text)
text = nltk.word_tokenize("It doesn't matter what the name is.")
print(text)
text = nltk.word_tokenize("This is an exmaple of pre-processing")
print(text)
#########################################################################################################################
#different tokenizer
from gensim.utils import tokenize
print(list(tokenize("This is an exmaple of pre-processing")))
print(list(tokenize("It doesn't matter what the name is.")))
#########################################################################################################################
""" Extracting n grams from text """
import nltk

text = nltk.word_tokenize("The quick brown fox jumped on the dog")
def find_bigrams(input_list):
  bigram_list = []
  for i in range(len(input_list)-1):
      bigram_list.append((input_list[i], input_list[i+1]))

  return bigram_list
#get individual items from the bigram
bigrams = find_bigrams(text)
print(bigrams)
print(bigrams[0].__getitem__(0))
print(bigrams[0].__getitem__(1))

#Now write a function to generate trigrams.
"""using the nltk ngrams function"""

from nltk import ngrams
sentence = 'The quick brown fox jumped over the dog.'
n = 4
gramsList = ngrams(sentence.split(), n)
ngrams = []
for grams in gramsList:
  ngrams.append(grams)
print(ngrams)

##########################################################################################################################

