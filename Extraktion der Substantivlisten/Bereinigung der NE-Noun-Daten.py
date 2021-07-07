# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 14:15:50 2021

@author: Goegg
"""
import pickle
import re

regex1 = r"([!@#$%^&*(),?\":{}|<>€~])|(^ *\d[\d ]*$)" 

dic = pickle.load( open( r"C:\Users\Goegg\OneDrive\Desktop\Durchgänge\9. YAKE - BERT\BERT_NE_YAKE_DECOMPOSED_NOTFIDF.pickle", "rb" ) )
all_docs = []
for pi in dic:
    document = {}
    docdic_value = []
    nouns_and_ents = []
    ents_dic = {}
    for pi_name in pi:
        liste = pi[pi_name][0]
        nouns_and_ents = [word.lower() for word in liste if re.search(regex1, word) is None]
        dic1 = pi[pi_name][1]
        ents_dic = dict((k.lower(), v) for k, v in dic1.items() if re.search(regex1, k) is None) 
        docdic_value.append(nouns_and_ents)
        docdic_value.append(ents_dic)
        document[pi_name]=docdic_value
        all_docs.append(document)

print(dic[13])
print(all_docs[13])

pickle_out = open(r"C:\Users\Goegg\OneDrive\Desktop\Durchgänge\9. YAKE - BERT\BERT_NE_YAKE_DECOMPOSED_NOTFIDF.pickle", "wb")
pickle.dump(all_docs, pickle_out)
pickle_out.close()
