#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#does it all: statistics (textrank), BERT-ner

from spacy.lemmatizer import Lemmatizer, ADJ, NOUN, VERB
import pickle
import spacy
import re
from gensim.summarization import keywords


# In[ ]:


#import spacy-objects
dic = pickle.load( open( r"C:\Users\Goegg\OneDrive\Desktop\Durchgänge\PI.pickle", "rb" ) )
liste = pickle.load( open( r"C:\Users\Goegg\OneDrive\Desktop\Durchgänge\3. TFIDF - BERT\BERT_NE_Spacy_Nouns_COMPOSED_NOTFIDF.pickle", "rb" ) )


# In[ ]:


regex = "(ftp:\/\/|www\.|https?:\/\/){1}[a-zA-Z0-9u00a1-\uffff0-]{2,}\.[a-zA-Z0-9u00a1-\uffff0-]{2,}(\S*)"
remove_paranthesis = str.maketrans({"(":None, ")":None})


# In[ ]:


nlp = spacy.load("de_core_news_lg")
lemmatizer = nlp.vocab.morphology.lemmatizer


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
    potential_keywords = keywords(dic[pi].text).split("\n")
    lemmatized = [lemmatizer(word, NOUN)[0].capitalize() for word in potential_keywords]
    lemmatized_nouns = []
    for word in lemmatized:
        if word in nouns_and_ents and word not in ents_dic.keys():
            lemmatized_nouns.append(word)
    top10 = lemmatized_nouns[:10]
    docdic_value.append(top10)
    docdic_value.append(ents_dic)
    output_dic[pi] = docdic_value
    output_top10.append(output_dic)
    n+=1
    print("done with " + str(n))

pickle_out = open(r"C:\Users\Goegg\OneDrive\Desktop\Durchgänge\6. TextRank - BERT\BERT_NE_TextRank_COMPOSED_NOTFIDF.pickle", "wb")
pickle.dump(output_top10, pickle_out)
pickle_out.close()


# In[ ]:




