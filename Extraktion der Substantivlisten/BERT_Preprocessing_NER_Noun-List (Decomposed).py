#!/usr/bin/env python
# coding: utf-8

# In[3]:


# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 18:38:54 2021

@author: Goegg
"""

from spacy.lemmatizer import Lemmatizer, ADJ, NOUN, VERB
import pickle
import spacy
from charsplit import Splitter
import re


# In[4]:


liste = pickle.load( open( r"C:\Users\Goegg\OneDrive\Desktop\Durchgänge\3. TFIDF - BERT (FEHLT)\BERT_NE_Spacy_Nouns_COMPOSED_NOTFIDF.pickle", "rb" ) )
nlp = spacy.load("de_core_news_lg")

lemmatizer = nlp.vocab.morphology.lemmatizer
splitter = Splitter()


# In[17]:


def get_noun_and_ne(doc):
    all_docs= []
    number = 0
    for dic in liste:
        for key in dic:
            document = {}
            nouns_and_ents = []
            ents_dic = {}
            docdic_value = []
            #get all NE for this document
            for keys, values in dic[key][1].items():
                #add lemmatized NE and his tag in the dictionary
                ents_dic[keys] = values
                #split entities on whitespace
                splitted_entities = keys.split()
                #add those entities with more than one token separated to the NE_Nouns-List and if its possible to decompose them. If so, add them too.
                if len(splitted_entities) > 1:
                    for ent in splitted_entities:
                            lemmatized_ent = lemmatizer(ent.capitalize(), NOUN)[0]
                            #add lemmatized part to NE_Nouns-List
                            nouns_and_ents.append(lemmatized_ent)
                            #if possible, add decomposed components to NE_Nouns-List
                            if splitter.split_compound(lemmatized_ent)[0][0] >= 0.9:
                                decomposed_word_tuple = splitter.split_compound(str(lemmatized_ent))[0][1:]
                                for splitted_word in decomposed_word_tuple:
                                    nouns_and_ents.append(lemmatizer(splitted_word.capitalize(), NOUN)[0])
                #if there was no split on whitespace, just check if decomposing is possible without adding itself to NE_Noun-List
                else:
                    for ent in splitted_entities:
                            lemmatized_ent = lemmatizer(ent.capitalize(), NOUN)[0]
                            #if possible, add decomposed components to NE_Nouns-List
                            if splitter.split_compound(lemmatized_ent)[0][0] >= 0.9:
                                decomposed_word_tuple = splitter.split_compound(str(lemmatized_ent))[0][1:]
                                for splitted_word in decomposed_word_tuple:
                                    nouns_and_ents.append(lemmatizer(splitted_word.capitalize(), NOUN)[0])
            #get all nouns for this document 
            for noun in dic[key][0]:
                
                #add the lemma to the NE_Nouns-List
                nouns_and_ents.append(noun.capitalize())
                # if the lemma can be decomposed, add its decomposed parts
                if splitter.split_compound(noun)[0][0] >= 0.9:
                    decomposed_word_tuple = splitter.split_compound(str(noun))[0][1:]
                    for splitted_word in decomposed_word_tuple:
                        nouns_and_ents.append(lemmatizer(splitted_word.capitalize(), NOUN)[0])
        docdic_value.append(nouns_and_ents)
        docdic_value.append(ents_dic)
        document[key]=docdic_value
        all_docs.append(document)
        number += 1
        print("done with " + str(number))
    return all_docs

pickle_out = open(r"C:\Users\Goegg\OneDrive\Desktop\Durchgänge\3. TFIDF - BERT (FEHLT)\BERT_NE_Spacy_Nouns_DECOMPOSED_NOTFIDF.pickle", "wb")
pickle.dump(get_noun_and_ne(liste), pickle_out)
pickle_out.close()



