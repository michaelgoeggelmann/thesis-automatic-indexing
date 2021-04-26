# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 18:38:54 2021

@author: Goegg
"""

from spacy.lemmatizer import Lemmatizer, ADJ, NOUN, VERB
import pickle
import spacy
import re
dic = pickle.load( open( r"C:\Users\Goegg\OneDrive\Dokumente\Uni\Master\Masterarbeit\Code\Pickle\Spacy Objekte\GC-OK-OBJDIC_Ohne Duplicates.pickle", "rb" ) )
nlp = spacy.load("de_core_news_lg")
regex = "(ftp:\/\/|www\.|https?:\/\/){1}[a-zA-Z0-9u00a1-\uffff0-]{2,}\.[a-zA-Z0-9u00a1-\uffff0-]{2,}(\S*)"


lemmatizer = nlp.vocab.morphology.lemmatizer


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
            # without URLs
            if re.match(regex, entities.text):
                continue
            else:
                # add them to the NE_Nouns-List and to a dictionary as key, with its label as value 
                ents_dic[lemmatizer(entities.text, NOUN)[0]] = entities.label_
                nouns_and_ents.append(lemmatizer(entities.text, NOUN)[0])
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
    return all_docs




pickle_out = open(r"C:\Users\Goegg\OneDrive\Dokumente\Uni\Master\Masterarbeit\Daten\Presseinfos\NE_NOUNS_ALL.pickle", "wb")
pickle.dump(get_noun_and_ne(dic), pickle_out)
pickle_out.close()