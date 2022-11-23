import nltk

text = nltk.word_tokenize("The talk was boring")
print(nltk.pos_tag(text))

text = nltk.word_tokenize("You should talk more in class")
print(nltk.pos_tag(text))