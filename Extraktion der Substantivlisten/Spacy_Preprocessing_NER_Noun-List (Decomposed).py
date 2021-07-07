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
regex = "(ftp:\/\/|www\.|https?:\/\/){1}[a-zA-Z0-9u00a1-\uffff0-]{2,}\.[a-zA-Z0-9u00a1-\uffff0-]{2,}(\S*)"
dic = pickle.load( open( r"C:\Users\Goegg\OneDrive\Desktop\Durchgänge\PI.pickle", "rb" ) )
nlp = spacy.load("de_core_news_lg")

lemmatizer = nlp.vocab.morphology.lemmatizer
splitter = Splitter()

def get_noun_and_ne(doc):
    all_docs= []
    number = 0
    for pi in doc:
        document = {}
        nouns_and_ents = []
        ents_dic = {}
        docdic_value = []
        #get all NE for this document
        for entities in doc[pi].ents:
            #without URLs
            if re.match(regex, entities.text):
                continue
            else:
                lemmatized = lemmatizer(entities.text.capitalize(), NOUN)[0]
                #add lemmatized NE and his tag in the dictionary
                ents_dic[lemmatized] = entities.label_
                #add lemmatized NE to NE_Nouns-List
                nouns_and_ents.append(lemmatized)
                #split entities on whitespace
                splitted_entities = lemmatized.split()
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
        for possible_nouns in doc[pi]:
            #check the token for its pos-tag "noun"
            if possible_nouns.pos_ == "NOUN":
                #exclude URLs
                if re.match(regex, possible_nouns.lemma_):
                    continue
                else:
                    #add the lemma to the NE_Nouns-List
                    nouns_and_ents.append(possible_nouns.lemma_)
                    # if the lemma can be decomposed, add its decomposed parts
                    if splitter.split_compound(possible_nouns.lemma_)[0][0] >= 0.9:
                        decomposed_word_tuple = splitter.split_compound(str(possible_nouns))[0][1:]
                        for splitted_word in decomposed_word_tuple:
                            nouns_and_ents.append(lemmatizer(splitted_word.capitalize(), NOUN)[0])
        docdic_value.append(nouns_and_ents)
        docdic_value.append(ents_dic)
        document[pi]=docdic_value
        all_docs.append(document)
        number += 1
        print("done with " + str(number))
    return all_docs



pickle_out = open(r"C:\Users\Goegg\OneDrive\Desktop\Durchgänge\1. TFIDF - SPACY NE\NE_Nouns_DECOMPOSED_NOTFIDF.pickle", "wb")
pickle.dump(get_noun_and_ne(dic), pickle_out)
pickle_out.close()