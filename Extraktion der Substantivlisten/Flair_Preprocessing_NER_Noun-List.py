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
dic = pickle.load( open( r"C:\Users\Goegg\OneDrive\Desktop\Durchgänge\PI.pickle", "rb" ) )
nlp = spacy.load("de_core_news_lg")
regex = "(ftp:\/\/|www\.|https?:\/\/){1}[a-zA-Z0-9u00a1-\uffff0-]{2,}\.[a-zA-Z0-9u00a1-\uffff0-]{2,}(\S*)"
tagger = SequenceTagger.load("flair/ner-german-large")
remove_paranthesis = str.maketrans({"(":None, ")":None})
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
        sentences = [Sentence(sent, use_tokenizer=True) for sent in split_single(dic[pi].text)]
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
    return all_docs


pickle_out = open(r"C:\Users\Goegg\OneDrive\Desktop\Durchgänge\2. TFIDF - FLAIR\NE_Nouns_COMPOSED_NOTFIDF.pickle", "wb")
pickle.dump(get_noun_and_ne(dic), pickle_out)
pickle_out.close()