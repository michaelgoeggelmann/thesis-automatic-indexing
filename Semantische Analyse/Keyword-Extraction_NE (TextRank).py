# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 20:24:58 2021

@author: Goegg
"""
import pickle
import pandas as pd
import os

diclist = pickle.load( open( r"C:\Users\Goegg\OneDrive\Desktop\Durchgänge\9. YAKE - BERT\BERT_NE_YAKE_DECOMPOSED_NOTFIDF.pickle", "rb" ) )
to_be_vectorized = []
tfidf_tags = ["LOC", "PER", "ORG"]
output_path = r"C:\Users\Goegg\OneDrive\Desktop\Durchgänge\9. YAKE - BERT\MI_Schlagworte_Decomposed"
suffix = ".xlsx"
for doc in diclist:
    for pi_name in doc:
        all_rows = []
        ne_dic = doc[pi_name][1]
        rest = {pi_name : []}
        #get noun-list
        rest[pi_name] = [noun for noun in doc[pi_name][0] if noun not in ne_dic.keys()]

        for entity, label in ne_dic.items():
            if len(entity) <= 2:
                continue
            if label in tfidf_tags:
                if label == "LOC":
                    all_rows.append(["", entity, "", "", "", ""])
                elif label == "ORG":
                    all_rows.append(["", "", entity, "", "", ""])
                elif label == "PER":
                    all_rows.append(["", "", "", entity, "", ""])
            else:
                rest[pi_name].append(entity)
            
    to_be_vectorized.append(rest)
    output_df = pd.DataFrame(all_rows, columns=("TFIDF-Rank", "Geographischer Index:" , "Körperschaftsindex:", "Personen-Index", "Themen-Index", "Produkt-Index"))
    output_df.to_excel (os.path.join(output_path, pi_name + suffix), index = False, header=True)
    
pickle_out = open(r"C:\Users\Goegg\OneDrive\Desktop\Durchgänge\9. YAKE - BERT\vectorize_decomposed.pickle", "wb")
pickle.dump(to_be_vectorized, pickle_out)
pickle_out.close()
    
