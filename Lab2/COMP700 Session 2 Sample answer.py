import os

with os.scandir('data/terrorism2') as entries:
     lines_in_file = ""
     for entry in entries:
         print(entry.name)
         with open(entry, 'r') as file:
             lines_in_file = lines_in_file + file.read().lower()
#print(lines_in_file)

#############################################################################################################
def remove_stop_words(text):
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    stop_words = stopwords.words('english')
    newStopWords = ['it', 'its', 'when',',',':',';'] # add your own stop words to the list here.
    stop_words = stop_words + newStopWords
    word_tokens = word_tokenize(text)

    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    return filtered_sentence
################################################################################################################
def find_ngrams(n, textList):
    from nltk import ngrams
    gramsList = ngrams(textList, n)
    ngrams = []
    for grams in gramsList:
        ngrams.append(grams)
    return ngrams
################################################################################################################
#print(remove_stop_words(lines_in_file))

#print(find_ngrams(3,remove_stop_words(lines_in_file)))

import collections
count_grams = collections.Counter(find_ngrams(3,remove_stop_words((lines_in_file))))
print(count_grams)
most_common_gram = count_grams.most_common(2)

#most common
print ("Most common")
print(most_common_gram[0])
print(most_common_gram[0].__getitem__(0))


print ("Second most common")
print(most_common_gram[1])
print(most_common_gram[1].__getitem__(0).__getitem__(2))
print("\n")
