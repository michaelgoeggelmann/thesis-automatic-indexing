# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 19:07:09 2020

@author: Goegg
"""
import os
import re
import pickle
import spacy


def get_filename(path):
    return [i.path for i in os.scandir(path) if i.is_file()]

def replaceHyphenated(s):
    matchList = re.findall(r"\w+-\W",s) # find combination of word-
    sOut = s
    for m in matchList:
        new = m.replace("-","")
        sOut = sOut.replace(m,new)
    return sOut

def get_PI(filepath):
    getFilename = os.path.basename(filepath)
    filename, ext = getFilename.split(".")
    if "_" in filename:
        filename, rest = filename.split("_", 1)
    return filename

def main():
    nlp = spacy.load("de_core_news_lg")
    merge_ents = nlp.create_pipe("merge_entities")
    nlp.add_pipe(merge_ents)
    docs = r"C:\Users\Goegg\OneDrive\Dokumente\Uni\Master\Masterarbeit\Daten\Presseinfos\final(txt)"
    
    files = get_filename(docs)
    print(len(files))
    final_dic = {}
    number = 0
    for filepath in files:
        with open(filepath, 'r', encoding="utf-8") as file_to_read:
            text = file_to_read.read()
            nText = replaceHyphenated(text)
            nText = nText.replace("\n"," ")
            PI = get_PI(filepath)
            if PI not in final_dic.keys():
                final_dic[PI] = nlp(nText)
            #remove duplicates
            else:
                continue
            number += 1
            print("done with " + str(number))
    
    pickle_out = open(r"C:\Users\Goegg\OneDrive\Dokumente\Uni\Master\Masterarbeit\Daten\Presseinfos\dict.pickle", "wb")
    pickle.dump(final_dic, pickle_out)
    pickle_out.close()
    
main()



