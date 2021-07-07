#!/usr/bin/env python
# coding: utf-8

# In[1]:


#does it all: statistics (textrank), BERT-ner

from spacy.lemmatizer import Lemmatizer, ADJ, NOUN, VERB
import pickle
import spacy
import re
import yake


# In[2]:


#import spacy-objects
dic = pickle.load( open( r"C:\Users\Goegg\OneDrive\Desktop\Durchgänge\PI.pickle", "rb" ) )
liste = pickle.load( open( r"C:\Users\Goegg\OneDrive\Desktop\Durchgänge\3. TFIDF - BERT\BERT_NE_Spacy_Nouns_COMPOSED_NOTFIDF.pickle", "rb" ) )


# In[3]:


regex = "(ftp:\/\/|www\.|https?:\/\/){1}[a-zA-Z0-9u00a1-\uffff0-]{2,}\.[a-zA-Z0-9u00a1-\uffff0-]{2,}(\S*)"
remove_paranthesis = str.maketrans({"(":None, ")":None})


# In[4]:


nlp = spacy.load("de_core_news_lg")
lemmatizer = nlp.vocab.morphology.lemmatizer


# In[5]:


language = "de"
max_ngram_size = 1
deduplication_threshold = 0.9
numOfKeywords = 20
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)


# In[ ]:


output_top10 = []
n = 0
for index, pi in enumerate(dic):
    ents_dic = {}
    docdic_value = []
    output_dic = {}
    nouns_and_ents = []
    for keys, values in liste[index][pi][1].items():
        ents_dic[keys] = values
        nouns_and_ents.append(keys)
    for possible_nouns in dic[pi]:
            # without URLs
            if re.match(regex, possible_nouns.lemma_):
                continue
            else:
                if possible_nouns.pos_ == "NOUN":
                    nouns_and_ents.append(possible_nouns.lemma_.capitalize())
    keyword_tuples = custom_kw_extractor.extract_keywords(dic[pi].text)
    keyword_list = []
    for tuples in keyword_tuples:
        
        if lemmatizer(tuples[0], NOUN)[0].capitalize() in nouns_and_ents and lemmatizer(tuples[0], NOUN)[0].capitalize() not in ents_dic.keys() and len(keyword_list) < 10:
            keyword_list.append(lemmatizer(tuples[0], NOUN)[0].capitalize())
    docdic_value.append(keyword_list)
    docdic_value.append(ents_dic)
    output_dic[pi] = docdic_value
    output_top10.append(output_dic)
    n+=1
    print("done with " + str(n))
    print(output_top10)
    
pickle_out = open(r"C:\Users\Goegg\OneDrive\Desktop\Durchgänge\9. YAKE - BERT\BERT_NE_YAKE_COMPOSED_NOTFIDF.pickle", "wb")
pickle.dump(output_top10, pickle_out)
pickle_out.close()


# In[ ]:




