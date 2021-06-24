# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 11:58:22 2021

@author: Goegg
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import spacy
import pandas as pd
import os
import pickle

diclist = pickle.load( open( r"C:\Users\Goegg\OneDrive\Desktop\Durchgänge\1. TFIDF - SPACY NE\Spacy_NE_Spacy_Nouns_DECOMPOSED_NOTFIDF.pickle", "rb" ) )


def spacy_tokenizer(tokenlist):
    return tokenlist

def getPInames(dicl):
    pi_names = []
    for doc in dicl:
        for dockey in doc:
            pi_names.append(dockey)
    return pi_names

def getNeNounLists(dic1):
    listwithNENOUN_Lists = []
    for dic in dic1:
        for dockey in dic:
            listwithNENOUN_Lists.append(dic[dockey][0])
    return listwithNENOUN_Lists

tokenizedList = getNeNounLists(diclist)
pi_names = getPInames(diclist)

vectorizer = TfidfVectorizer(tokenizer = spacy_tokenizer, max_df=.50, min_df=1, use_idf=True, norm=None, lowercase=False)
transformed_documents = vectorizer.fit_transform(tokenizedList)

transformed_documents_as_array = transformed_documents.toarray()


# make the output folder if it doesn't already exist
Path(r"C:\Users\Goegg\OneDrive\Desktop\Durchgänge\1. TFIDF - SPACY NE\Decomposed TFIDF Ranking").mkdir(parents=True, exist_ok=True)

# construct a list of output file paths using the previous list of text files the relative path for tf_idf_output
output_filenames = [os.path.join(r"C:\Users\Goegg\OneDrive\Desktop\Durchgänge\1. TFIDF - SPACY NE\Decomposed TFIDF Ranking", str(txt_file) + ".csv") for txt_file in pi_names]


# loop each item in transformed_documents_as_array, using enumerate to keep track of the current position
for counter, doc in enumerate(transformed_documents_as_array):
    # construct a dataframe
    tf_idf_tuples = list(zip(vectorizer.get_feature_names(), doc))
    sorted_by_second = sorted(tf_idf_tuples, key=lambda tup: tup[1], reverse=True)
    out_tup = [i for i in sorted_by_second if i[1] != 0.0]
    #reduced_list = sorted_by_second[:10]
    one_doc_as_df = pd.DataFrame.from_records(out_tup, columns=['term', 'score']).sort_values(by='score', ascending=False).reset_index(drop=True)
    # output to a csv using the enumerated value for the filename
    one_doc_as_df.to_csv(output_filenames[counter])
