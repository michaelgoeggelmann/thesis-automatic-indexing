# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 20:24:58 2021

@author: Goegg
"""
import pickle
import pandas as pd
import os

os.chdir(r"C:\Users\Goegg\OneDrive\Dokumente\Uni\Master\Masterarbeit\Code\Textstatistik und Machine Learning\TFIDF\first10_output")
diclist = pickle.load( open( r"C:\Users\Goegg\OneDrive\Dokumente\Uni\Master\Masterarbeit\Code\Pickle\Spacy Nouns_NE\8162_NO-URL_NE_NOUNS_ALL.pickle", "rb" ) )
to_be_vectorized = []
tfidf_tags = ["LOC", "PER", "ORG"]
output_path = r"C:\Users\Goegg\OneDrive\Dokumente\Uni\Master\Masterarbeit\Code\NER - Keyword Extraction\TEST_NER_OUTPUT"
suffix = ".xlsx"
for filename in os.listdir():
    input_df = pd.read_csv(filename)
    all_rows = []
    pi = os.path.splitext(filename)[0]
    ne_dic = [i[pi] for i in diclist if pi in i][0][1]
    rest = {pi : []}
    for index, row in input_df.iterrows():
        TFIDF_Rank = int(index) + 1
        print(TFIDF_Rank)
        if row["term"] in ne_dic.keys() and ne_dic[row["term"]] in tfidf_tags:
            if ne_dic[row["term"]] == "LOC":
                all_rows.append([TFIDF_Rank, row["term"], "", "", "", ""])
            elif ne_dic[row["term"]] == "ORG":
                all_rows.append([TFIDF_Rank, "", row["term"], "", "", ""])
            elif ne_dic[row["term"]] == "PER":
                all_rows.append([TFIDF_Rank, "", "", row["term"], "", ""])
        else:
            rest[pi].append(row["term"])
    to_be_vectorized.append(rest)
    
    output_df = pd.DataFrame(all_rows, columns=("TFIDF-Rank", "Geographischer Index:" , "Körperschaftsindex:", "Personen-Index", "Themen-Index", "Produkt-Index"))
    output_df.to_excel (os.path.join(output_path, pi + suffix), index = False, header=True)

pickle_out = open(r"C:\Users\Goegg\OneDrive\Dokumente\Uni\Master\Masterarbeit\Code\NER - Keyword Extraction\TEST_tbvectorized_pickle\first10_normal.pickle", "wb")
pickle.dump(to_be_vectorized, pickle_out)
pickle_out.close()
    
