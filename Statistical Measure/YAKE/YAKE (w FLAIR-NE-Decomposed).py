#!/usr/bin/env python
# coding: utf-8

# In[1]:


#does it all: statistics (YAKE), flair-ner

from spacy.lemmatizer import Lemmatizer, ADJ, NOUN, VERB
import pickle
import spacy
import re
import yake
from charsplit import Splitter


# In[2]:


#import spacy-objects
dic = pickle.load( open( r"C:\Users\Goegg\OneDrive\Desktop\Durchgänge\PI.pickle", "rb" ) )
liste = pickle.load( open( r"C:\Users\Goegg\OneDrive\Desktop\Durchgänge\2. TFIDF - FLAIR\Flair_NE_Spacy_Nouns_COMPOSED_NOTFIDF.pickle", "rb" ) )


# In[3]:


regex = "(ftp:\/\/|www\.|https?:\/\/){1}[a-zA-Z0-9u00a1-\uffff0-]{2,}\.[a-zA-Z0-9u00a1-\uffff0-]{2,}(\S*)"
remove_paranthesis = str.maketrans({"(":None, ")":None})


# In[4]:


nlp = spacy.load("de_core_news_lg")
lemmatizer = nlp.vocab.morphology.lemmatizer
splitter = Splitter()


# In[5]:


language = "de"
max_ngram_size = 1
deduplication_threshold = 0.9
numOfKeywords = 20
custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=None)


# In[ ]:


output_top10 = []
counter = 0
for index, pi in enumerate(dic):
    ents_dic = {}
    docdic_value = []
    output_dic = {}
    nouns_and_ents = []
    for keys, values in liste[index][pi][1].items():
        ents_dic[keys] = values
        nouns_and_ents.append(keys)
        splitted_entities = keys.split()
        #add those entities with more than one token separated to the NE_Nouns-List and if its possible to decompose them. If so, add them too.
        if len(splitted_entities) > 1:
            for ent in splitted_entities:
                lemmatized_ent = lemmatizer(ent.capitalize(), NOUN)[0]
                #add lemmatized part to NE_Nouns-List
                nouns_and_ents.append(lemmatized_ent)
                #if possible, add decomposed components to NE_Nouns-List
                if splitter.split_compound(lemmatized_ent)[0][0] >= 0.9:
                    decomposed_word_tuple = splitter.split_compound(str(lemmatized_ent))[0][1:]
                    for splitted_word in decomposed_word_tuple:
                        nouns_and_ents.append(lemmatizer(splitted_word.capitalize(), NOUN)[0])
                #if there was no split on whitespace, just check if decomposing is possible without adding itself to NE_Noun-List
        else:
            for ent in splitted_entities:
                lemmatized_ent = lemmatizer(ent.capitalize(), NOUN)[0]
                #if possible, add decomposed components to NE_Nouns-List
                if splitter.split_compound(lemmatized_ent)[0][0] >= 0.9:
                    decomposed_word_tuple = splitter.split_compound(str(lemmatized_ent))[0][1:]
                    for splitted_word in decomposed_word_tuple:
                        nouns_and_ents.append(lemmatizer(splitted_word.capitalize(), NOUN)[0])
    #now nouns
    # now all nouns
    for possible_nouns in dic[pi]:
            # without URLs
            if re.match(regex, possible_nouns.lemma_):
                continue
            else:
                if possible_nouns.pos_ == "NOUN":
                    nouns_and_ents.append(possible_nouns.lemma_.capitalize())
                    if splitter.split_compound(possible_nouns.lemma_)[0][0] >= 0.9:
                        decomposed_word_tuple = splitter.split_compound(str(possible_nouns))[0][1:]
                        for splitted_word in decomposed_word_tuple:
                            nouns_and_ents.append(lemmatizer(splitted_word.capitalize(), NOUN)[0])
    n = 0
    keyword_tuples = custom_kw_extractor.extract_keywords(dic[pi].text)
    lemmatized = [lemmatizer(tuples[0], NOUN)[0].capitalize() for tuples in keyword_tuples]
    lemmatized_nouns = []
    for word in lemmatized:
        if word in nouns_and_ents and word not in ents_dic.keys():
            lemmatized_nouns.append(word)
            if splitter.split_compound(word)[0][0] >= 0.9:
                decomposed_word_tuple = splitter.split_compound(str(word))[0][1:]
                for splitted_word in decomposed_word_tuple:
                    n+=1
                    lemmatized_nouns.append(lemmatizer(splitted_word.capitalize(), NOUN)[0])
    top10 = lemmatized_nouns[:10+n]
    docdic_value.append(top10)
    docdic_value.append(ents_dic)
    output_dic[pi] = docdic_value
    output_top10.append(output_dic)
    counter+=1
    print("done with " + str(counter))
pickle_out = open(r"C:\Users\Goegg\OneDrive\Desktop\Durchgänge\8. YAKE - FLAIR\FLAIR_NE_YAKE_DECOMPOSED_NOTFIDF.pickle", "wb")
pickle.dump(output_top10, pickle_out)
pickle_out.close()


# In[ ]:




