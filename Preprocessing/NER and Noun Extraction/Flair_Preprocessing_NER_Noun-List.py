#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 18:38:54 2021

@author: Goegg
"""

from spacy.lemmatizer import Lemmatizer, ADJ, NOUN, VERB
import pickle
import spacy
import re
from flair.data import Sentence
from flair.models import SequenceTagger
from segtok.segmenter import split_single


# In[2]:





# In[3]:


with open(r"C:\Users\Goegg\OneDrive\Desktop\Durchgänge\PI.pickle", "rb") as f:
    dic = pickle.load(f)


# In[4]:


def get_noun_and_ne(doc):
    all_docs= []
    number = 0
    for pi in doc:
        document = {}
        nouns_and_ents = []
        ents_dic = {}
        docdic_value = []
        #get all NE for this document
        sentences = [Sentence(sent, use_tokenizer=True) for sent in split_single(doc[pi].text)]
        tagger.predict(sentences)
        #jeder Satz
        for sent in sentences:
            sent_dic = sent.to_dict("ner")
            #jeder Eintrag in diesem Satz
            for entry in sent_dic["entities"]:
                ne = entry["text"].capitalize()
                label = str(entry["labels"][0]).split()[0]
                confidence = float(str(str((entry["labels"][0])).split()[1].translate(remove_paranthesis)))
                if confidence > 0.7:
                    ents_dic[lemmatizer(ne, NOUN)[0]] = label
                    nouns_and_ents.append(ne)
        #get all nouns for this document 
        for possible_nouns in doc[pi]:
            # without URLs
            if re.match(regex, possible_nouns.lemma_):
                continue
            else:
                if possible_nouns.pos_ == "NOUN":
                    nouns_and_ents.append(possible_nouns.lemma_)
        docdic_value.append(nouns_and_ents)
        docdic_value.append(ents_dic)
        document[pi]=docdic_value
        all_docs.append(document)
        number += 1
        print("done with " + str(number))
    print(len(all_docs))
    return all_docs

