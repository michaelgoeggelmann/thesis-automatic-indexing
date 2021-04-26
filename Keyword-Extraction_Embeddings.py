#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchtext
import pandas as pd
import pickle
import re
import os


# In[2]:


glove = torchtext.vocab.GloVe(name="6B", 
                              dim=300)   # embedding size = 300


# In[3]:


# Load and Preprocessing Products

df1 = pd.read_excel(r'C:\Users\Goegg\OneDrive\Dokumente\Uni\Master\Masterarbeit\Daten\Presseinfos\BOSCH_PRODUCTINDEX_DATA_TABLE.XLSX', header=0)

column_values = df1["DESCRIPTOR                  "].tolist()
products = []
remove_paranthesis = str.maketrans({"(":None, ")":None})
products_vectorized = []
for product in column_values:
    product = product.lower()
    product = product.rstrip()
    
    products.append(product.translate(remove_paranthesis))
    products = list(filter(("").__ne__, products))
for product in products:
    prod_dic = {}
    tokenized_list = re.split(" |/", product)
    vectors = []
    for word in tokenized_list:
        vectors.append(glove[word])
    prod_dic[product] = sum(vectors)
    products_vectorized.append(prod_dic)


# In[4]:


# Load and Preprocessing Topics

df2 = pd.read_excel(r'C:\Users\Goegg\OneDrive\Dokumente\Uni\Master\Masterarbeit\Daten\Presseinfos\BOSCH_TOPICINDEX_DATA_TABLE.XLSX', header=0)

column_values = df2["DESCRIPTOREN"].tolist()
topics = []
topics_vectorized = []
remove_paranthesis = str.maketrans({"(":None, ")":None})


for topic in column_values:
    topic = str(topic).lower()
    topic = topic.rstrip()
    topics.append(topic.translate(remove_paranthesis))
    topics = list(filter(("").__ne__, topics))
    
for topic in topics:
    topic_dic = {}
    tokenized_list = re.split(" |/", topic)
    vectors = []
    for word in tokenized_list:
        vectors.append(glove[word])
    topic_dic[topic] = sum(vectors)
    topics_vectorized.append(topic_dic)
    


# In[5]:


# Load ToBeVectorized

tbv = pickle.load(open( r"C:\Users\Goegg\OneDrive\Dokumente\Uni\Master\Masterarbeit\Code\NER - Keyword Extraction\TEST_tbvectorized_pickle\first10_normal.pickle", "rb" ))

for pi in tbv:
    for doc, words in pi.items():
        pi[doc] = [word.lower() for word in words]


# In[13]:


os.chdir(r"C:\Users\Goegg\OneDrive\Dokumente\Uni\Master\Masterarbeit\Code\NER - Keyword Extraction\TEST_NER_OUTPUT")

for pi in tbv:
    for doc, words in pi.items():
        df_output = pd.read_excel(os.path.join(str(doc)+".xlsx"))
        output_list = []
        for tbv_word in words:
            print("Zu untersuchendes Wort: " + tbv_word)
            product_highest_word = ""
            product_highest_number = 0
            topic_highest_word = ""
            topic_highest_number = 0
            output_dic = {}
            
            
            tokenized_list = re.split(" |/", tbv_word)
            vectors = []
            for word in tokenized_list:
                vectors.append(glove[word])
            x = sum(vectors)
            for product in products_vectorized:
                for product_word, vectors in product.items():
                    cs = torch.cosine_similarity(x.unsqueeze(0), vectors.unsqueeze(0)).item()
                    if cs > 0.4 and cs > product_highest_number:
                        product_highest_number = cs
                        product_highest_word = product_word
            for topic in topics_vectorized:
                for topic_word, vectors in topic.items():
                    cs = torch.cosine_similarity(x.unsqueeze(0), vectors.unsqueeze(0)).item()
                    if cs > 0.4 and cs > topic_highest_number:
                        topic_highest_number = cs
                        topic_highest_word = topic_word

            print ("Produkte: " + product_highest_word + " Score "+str(product_highest_number))
            print ("Themen: " + topic_highest_word + " Score "+str(topic_highest_number))
            if topic_highest_number == 0 and product_highest_number == 0:
                continue
            elif topic_highest_number > product_highest_number:
                output_dic["Themen-Index"] = tbv_word
            else:
                output_dic["Produkt-Index"] = tbv_word
            output_list.append(output_dic)
        df_output=df_output.append(output_list, ignore_index=True)
        df_output.to_excel(str(doc)+'.xlsx')




