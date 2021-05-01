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

for pi in dic:
    print(pi)
    break