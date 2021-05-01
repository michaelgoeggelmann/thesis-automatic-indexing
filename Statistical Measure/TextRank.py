#!/usr/bin/env python
# coding: utf-8

# In[1]:


#does it all: statistics (textrank), flair-ner

from spacy.lemmatizer import Lemmatizer, ADJ, NOUN, VERB
import pickle
import spacy
import re
from gensim.summarization import keywords
from flair.data import Sentence
from flair.models import SequenceTagger
from segtok.segmenter import split_single


# In[2]:


#import spacy-objects
dic = pickle.load( open( r"C:\Users\Goegg\OneDrive\Dokumente\Uni\Master\Masterarbeit\Code\Pickle\Spacy Objekte\GC-OK-OBJDIC_Ohne Duplicates.pickle", "rb" ) )


# In[3]:


#load ner-tagger
tagger = SequenceTagger.load("flair/ner-german-large")


# In[4]:


regex = "(ftp:\/\/|www\.|https?:\/\/){1}[a-zA-Z0-9u00a1-\uffff0-]{2,}\.[a-zA-Z0-9u00a1-\uffff0-]{2,}(\S*)"
remove_paranthesis = str.maketrans({"(":None, ")":None})


# In[5]:


nlp = spacy.load("de_core_news_lg")
lemmatizer = nlp.vocab.morphology.lemmatizer


# In[6]:


output_top10 = []
for pi in dic:
    ents_dic = {}
    docdic_value = []
    output_dic = {}
    nouns_and_ents = []
    sentences = [Sentence(sent, use_tokenizer=True) for sent in split_single(dic[pi].text)]
    tagger.predict(sentences)
    #jeder Satz
    for sent in sentences:
        sent_dic = sent.to_dict("ner")
        #jeder Eintrag in diesem Satz
        for entry in sent_dic["entities"]:
            ne = entry["text"]
            label = str(entry["labels"][0]).split()[0]
            confidence = float(str(str((entry["labels"][0])).split()[1].translate(remove_paranthesis)))
            if confidence > 0.7:
                ents_dic[lemmatizer(ne, NOUN)[0]] = label
                nouns_and_ents.append(lemmatizer(ne, NOUN)[0])
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
        if word in nouns_and_ents:
            lemmatized_nouns.append(word)
    top10 = lemmatized_nouns[:10]
    docdic_value.append(top10)
    docdic_value.append(ents_dic)
    output_dic[pi] = docdic_value
    output_top10.append(output_dic)
    break
print(output_top10)

#pickle_out = open(r"", "wb")
#pickle.dump(output_top10, pickle_out)
#pickle_out.close()


# In[ ]:




