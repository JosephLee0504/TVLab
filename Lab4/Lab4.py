import os

import numpy

path1 = 'D:\\PycharmProject\\TVLab\\Lab4\\news1.txt'
path2 = 'D:\\PycharmProject\\TVLab\\Lab4\\news2.txt'

txt1 = []
txt2 = []
with open(path1, 'r', encoding='utf-8') as f:
    txt1.append(f.read())
with open(path2, 'r', encoding='utf-8') as f:
    txt2.append(f.read())
# print(txt2)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer1 = CountVectorizer()
vectorizer1.fit(txt1)
# print(vectorizer1.vocabulary_)
vector1 = vectorizer1.transform(txt1)
# print(vector1.shape)
# print(vector1.toarray())
# print(vectorizer1.vocabulary_)
vectorizer2 = CountVectorizer()
vectorizer2.fit(txt2)
# print(vectorizer2.vocabulary_)
vector2 = vectorizer2.transform(txt2)
# print(vector2.shape)
# print(vector2.toarray())
# print(vectorizer2.vocabulary_)

# from sklearn.feature_extraction.text import TfidfVectorizer
# # vectorizer1 = TfidfVectorizer()
# # vectorizer1.fit(txt1)
# # print(vectorizer1.vocabulary_)
# # print(vectorizer1.idf_)
# # vector1 = vectorizer1.transform([txt1[0]])
# # print(vector1.shape)
# # print(vector1.toarray)
# vectorizer2 = TfidfVectorizer()
# vectorizer2.fit(txt2)
# print(vectorizer2.vocabulary_)
# print(vectorizer2.idf_)
# vector2 = vectorizer2.transform([txt2[0]])
# print(vector2.shape)
# print(vector2.toarray)

from sklearn.metrics.pairwise import euclidean_distances
print("Euclidean Distnance: ",euclidean_distances(vector1,vector2))

vec1 = vector1.toarray()
vec2 = vector2.toarray()
import numpy as np
dist = numpy.sqrt(numpy.sum(numpy.square(vec1 - vec2)))
print(dist)



