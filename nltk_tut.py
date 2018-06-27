#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 22:07:40 2018

@author: rushikesh
"""
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
text = 'Hello Mr. Rushikesh, you are really awesome.'

#Tokenizing the document into words
word = word_tokenize(text)

#Tokenizing the document into sentences
print(sent_tokenize(text))


#Removing words of less importance(stopwords)
from nltk.corpus import stopwords

#Locally storing the list of stopwords
stop_words = set(stopwords.words("english"))

filtered_sentence = [w for w in word if w not in stop_words]


#Stemming
from nltk.stem import PorterStemmer 

ps = PorterStemmer()

example_words = ['python','pythoned','pythoner','pythoning','pythonly']
stemmed_example_words = [ps.stem(w) for w in example_words ]

new_text = ' It is very important to be very pythonly when you are pythoning with python.'
words = word_tokenize(new_text)
for w in words:
    print(ps.stem(w))

nltk.download('state_union')    
    
#Tagging parts of speech
    
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
#punkt is pretrained tokenizer which can be trained again

#gettig raw data from corpus
train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

#Retraining punkt to create custom tokenizer
#This is a sent tokenizer hence it will return sentences
custom_sent_tokenizer= PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print(tagged)
    except Exception as e:
        print(str(e))

process_content()




#Chunking 

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}   """
            
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            
            #print(chunked)
            chunked.draw()
            
            
        
    except Exception as e:
        print(str(e))

process_content()

#Chinking









#Named Entity

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
    
            namedEnt = nltk.ne_chunk(tagged, binary=True)
            namedEnt.draw()
            
            
        
    except Exception as e:
        print(str(e))

process_content()





#Lemmatizing

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("better")) #better
#Default parameter is pos="n" i.e noun

#a for adjective
print(lemmatizer.lemmatize("better", pos="a")) #good




#To find nltk site pakage location
print(nltk.__file__)


#Accessing Corpora

# For me-Currently saved on desktop

from nltk.corpus import gutenberg

#bible-kjv.txt is a file in gutenberg corpus
sample = gutenberg.raw("bible-kjv.txt")
tok = sent_tokenize(sample)
print(tok[5:15])





            
