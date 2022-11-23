import collections
import os

path = "D:\\PycharmProject\\TVLab\\Lab2\\terrorism data"

os.chdir(path)

txts = ''
def read_text_file(file_path):
    with open(file_path, 'r') as f:
        data = f.read()
    return data

for file in os.listdir():
    if file.endswith(".txt"):
        file_path = f"{path}\\{file}"
        # print(file_path)
        txts = txts + read_text_file(file_path)
# print(txts)

import nltk
from nltk.util import ngrams

nltk_tokens = nltk.word_tokenize(txts)
print(list(nltk.bigrams(nltk_tokens)))

nltk_tokens = nltk.word_tokenize(txts)
print(list(nltk.trigrams(nltk_tokens)))

tokenized = txts.split()
trigrams = ngrams(tokenized, 3)
trigramsFreq = collections.Counter(trigrams)
print(trigramsFreq.most_common(1))