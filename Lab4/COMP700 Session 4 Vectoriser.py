"""Convert words to vectores that can be used with classifiers"""

from sklearn.feature_extraction.text import CountVectorizer
# list of text documents
text = ["The quick brown fox jumped over the lazy dog."]
# create the transform
vectorizer = CountVectorizer()
#tokenize and build vocab
vectorizer.fit(text)
# summarize
print(vectorizer.vocabulary_)
# encode document
vector = vectorizer.transform(text)
# summarize encoded vector
print(vector.shape)
print(vector.toarray())
print(vectorizer.vocabulary_)

#Try another sentence
text2 = ["the quick puppy"] # note that puppy is in the vector represenation.
vector = vectorizer.transform(text2)
print(vector.toarray())

"""BOW model is not very effeictive. represents presence or absence of a token in a document.
Lets keep count of tokens in a document

Using TFIDF instead of BOW, TFIDF also takes into account the frequency instead of just the occurance.
calculated as:
Term frequency (normalized) = (Number of Occurrences of a word)/(Total words in the document) : normalizes based on the size of the document.
IDF(word) = Log((Total number of documents)/(Number of documents containing the word)) : reduces the impact  words that are common across documents, eg. the.
TF-IDF is the product of the two."""


from sklearn.feature_extraction.text import TfidfVectorizer
# list of text documents
text = ["The quick brown fox jumped over the lazy dog.",
		"The dog jumped over the fox.",
		"The fox"]
# create the transform
vectorizer = TfidfVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# summarize
print(vectorizer.vocabulary_)
print(vectorizer.idf_)
# encode document
vector1 = vectorizer.transform([text[0]])
vector2 = vectorizer.transform([text[1]])
# summarize encoded vector
print(vector1.shape)
print(vector1.toarray())

###################Distance metrices########################################################################
from nltk.metrics import *
print("Edit Distnance same string: ",edit_distance(text[0],text[0]))
print("Edit Distnance: ",edit_distance(text[0],text[1]))
print("Binary Distnance: ",binary_distance(set(text[0]),set(text[1])))
print("Jaccard Distnance: ",jaccard_distance(set(text[0]),set(text[1])))
print("Masi Distnance: ",masi_distance(set(text[0]),set(text[1])))

from sklearn.metrics.pairwise import euclidean_distances
print("Euclidean Distnance: ",euclidean_distances(vector1,vector2))
#########################################################################################################################
