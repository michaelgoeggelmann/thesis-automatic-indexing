#!/usr/bin/env python
# coding: utf-8

# In[2]:


#does it all: statistics (textrank), flair-ner

from spacy.lemmatizer import Lemmatizer, ADJ, NOUN, VERB
import pickle
import spacy
import re
from gensim.summarization import keywords
from charsplit import Splitter


# In[3]:


#import spacy-objects
dic = pickle.load( open( r"C:\Users\Goegg\OneDrive\Desktop\Durchgänge\PI.pickle", "rb" ) )


# In[5]:


regex = "(ftp:\/\/|www\.|https?:\/\/){1}[a-zA-Z0-9u00a1-\uffff0-]{2,}\.[a-zA-Z0-9u00a1-\uffff0-]{2,}(\S*)"
remove_paranthesis = str.maketrans({"(":None, ")":None})


# In[8]:


nlp = spacy.load("de_core_news_lg")
lemmatizer = nlp.vocab.morphology.lemmatizer
splitter = Splitter()


# In[24]:


output_top10 = []
counter = 0
for pi in dic:
    ents_dic = {}
    docdic_value = []
    output_dic = {}
    nouns_and_ents = []
    for entities in dic[pi].ents:
            # without URLs
            if re.match(regex, entities.text):
                continue
            else:
                # add them to the NE_Nouns-List and to a dictionary as key, with its label as value 
                lemmatized = lemmatizer(entities.text.capitalize(), NOUN)[0]
                ents_dic[lemmatized] = entities.label_
                nouns_and_ents.append(lemmatizer(entities.text.capitalize(), NOUN)[0])
                splitted_entities = lemmatized.split()
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
    potential_keywords = keywords(dic[pi].text).split("\n")
    lemmatized = [lemmatizer(word, NOUN)[0].capitalize() for word in potential_keywords]
    lemmatized_nouns = []
    for word in lemmatized:
        if word in nouns_and_ents and word not in ents_dic.keys():
            lemmatized_nouns.append(word)
            if splitter.split_compound(word)[0][0] >= 0.9:
                n+=1
                decomposed_word_tuple = splitter.split_compound(str(word))[0][1:]
                for splitted_word in decomposed_word_tuple:
                    lemmatized_nouns.append(lemmatizer(splitted_word.capitalize(), NOUN)[0])
    top10 = lemmatized_nouns[:10+n]
    docdic_value.append(top10)
    docdic_value.append(ents_dic)
    output_dic[pi] = docdic_value
    output_top10.append(output_dic)
    counter+=1
    print("done with " + str(counter))
pickle_out = open(r"C:\Users\Goegg\OneDrive\Desktop\Durchgänge\4. TextRank - SPACY NE\Spacy_NE_TextRank_DECOMPOSED_NOTFIDF.pickle", "wb")
pickle.dump(output_top10, pickle_out)
pickle_out.close()


# In[ ]:




