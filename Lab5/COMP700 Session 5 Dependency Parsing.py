
# #Collect all nouns and their modifiers
# import spacy
# nlp = spacy.load('en_core_web_sm')
# doc = nlp("Wall Street Journal just published an interesting piece on crypto currencies")
# for chunk in doc.noun_chunks:
#     print(chunk.text, chunk.label_, chunk.root.text)

""" Dep parsing example"""
import spacy
"""
You will need to install the following particular version of spacy.
 pip3 install spacy==2.3.5
You will also need to install en_core_web_sm using the following.
pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz

"""
nlp = spacy.load('en_core_web_sm')
doc = nlp('John ate icecream.')#with knife
#doc = nlp('John ate icecream and Peter ate apple')
# doc = nlp('Wall Street Journal just published an interesting piece on crypto currencies')
#doc = nlp('A man with a knife and a boy hit the dazed shopkeeper on the head yesterday.')
for token in doc:
    print("{0}/{1} <--{2}-- {3}/{4}".format(token.text, token.tag_, token.dep_, token.head.text, token.head.tag_))

################################# Finding a verb with a subject##########################################
# from spacy.symbols import nsubj, VERB
#
# for possible_subject in doc:
#     if possible_subject.dep == nsubj and possible_subject.head.pos == VERB:
#         verbs.add(possible_subject.head)
#         print(possible_subject, "   ", possible_subject.head)
