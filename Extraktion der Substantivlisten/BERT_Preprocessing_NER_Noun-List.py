#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from spacy.lemmatizer import Lemmatizer, ADJ, NOUN, VERB
import pickle
import spacy
import re
from segtok.segmenter import split_single
from simpletransformers.ner import NERModel, NERArgs
import nltk.data
import itertools


# In[ ]:


#load all we need
dic = pickle.load( open( r"C:\Users\Goegg\OneDrive\Desktop\Durchgänge\PI.pickle", "rb" ) )
nlp = spacy.load("de_core_news_lg")
regex = "(ftp:\/\/|www\.|https?:\/\/){1}[a-zA-Z0-9u00a1-\uffff0-]{2,}\.[a-zA-Z0-9u00a1-\uffff0-]{2,}(\S*)"
remove_paranthesis = str.maketrans({"(":None, ")":None})
lemmatizer = nlp.vocab.morphology.lemmatizer
sentences = nltk.data.load("tokenizers/punkt/german.pickle")
#initialize BERT-Model
model_args = NERArgs()
model_args.labels_list = ["O",
              "B-LOC",
              "B-LOCderiv",
              "B-LOCpart",
              "B-ORG",
              "B-ORGderiv",
              "B-ORGpart",
              "B-OTH",
              "B-OTHderiv",
              "B-OTHpart",
              "B-PER",
              "B-PERderiv",
              "B-PERpart",
              "I-LOC",
              "I-LOCderiv",
              "I-LOCpart",
              "I-ORG",
              "I-ORGderiv",
              "I-ORGpart",
              "I-OTH",
              "I-OTHderiv",
              "I-OTHpart",
              "I-PER",
              "I-PERderiv",
              "I-PERpart"
             ]
model = NERModel("bert", "fhswf/bert_de_ner", args=model_args, use_cuda=False)


MISC_Liste = ["B-OTH", "I-OTH"]

Beginning_Liste = ["B-PER", "B-ORG", "B-LOC", "B-OTH"]
Inner_Liste = ["I-PER", "I-ORG", "I-LOC", "I-OTH"]


# In[ ]:


# functions to join the extracted entities and labels

def join_tokens(tokens):
    res = ''
    if tokens:
        res = tokens[0]
        for token in tokens[1:]:
            if not (token.isalpha() and res[-1].isalpha()):
                res += " " + token  # punctuation
            else:
                res += ' ' + token  # regular word
    return res

def collapse(ner_result):
    # List with the result
    collapsed_result = []


    current_entity_tokens = []
    current_entity = None
    for entry in ner_result:
        # Iterate over the tagged tokens
        for token, tag in entry.items():

            if tag.startswith("B-"):
                # ... if we have a previous entity in the buffer, store it in the result list
                if current_entity is not None:
                    collapsed_result.append([join_tokens(current_entity_tokens), current_entity])

                current_entity = tag[2:]
                # The new entity has so far only one token
                current_entity_tokens = [token]

            # If the entity continues ...
            elif current_entity_tokens!= None and tag == "I-" + str(current_entity):
                # Just add the token buffer
                current_entity_tokens.append(token)
            else:
                collapsed_result.append([join_tokens(current_entity_tokens), current_entity])
                
                collapsed_result.append([token,tag[2:]])

                current_entity_tokens = []
                current_entity = None

                pass

        # The last entity is still in the buffer, so add it to the result
        # ... but only if there were some entity at all
    if current_entity is not None:
        collapsed_result.append([join_tokens(current_entity_tokens), current_entity])
        collapsed_result = sorted(collapsed_result)
        collapsed_result = list(k for k, _ in itertools.groupby(collapsed_result))
    without_none = [enti for enti in collapsed_result if not None in enti]
    collapsed_results = [[list[0], list[1]] for list in without_none if len(list[1]) == 3 and re.match(regex, list[0]) is None]
    return collapsed_results


# In[ ]:


# extracting NE (BERT) and NOUNS (SPACY)

def get_noun_and_ne(doc):
    all_docs= []
    number = 0
    for pi in doc:
        print(pi)
        document = {}
        docdic_value = []
        entity_list = []
        ents_dic = {}
    #get all NE for this document
        # tokenize whole text in sentences and remoive punctuation
        with_punc = sentences.tokenize(doc[pi].text)
        sents = [sentence[0: -1] for sentence in with_punc if sentence is not None]
        #let BERT predict NEs
        if len(sents):
            predictions, raw_outputs = model.predict(sents)
            #Remove "O"-Labels
            for sentence in predictions:
                for word in sentence:
                    label = list(word.values())[0]
                    if label == "O":
                        continue
                    else:
                        entity_list.append(word)
            #Join Entities and BIO-Scheme to whole entity, label and save in ents_dic:
            entity_list = collapse(entity_list)
            for words in entity_list:
                ents_dic[words[0]] = words[1]
            nouns_and_ents = [key for key in ents_dic.keys()]
            
    #get all nouns for this document 
        for possible_nouns in doc[pi]:
            # without URLs
            if re.match(regex, possible_nouns.lemma_):
                continue
            else:
                if possible_nouns.pos_ == "NOUN":
                    nouns_and_ents.append(possible_nouns.lemma_)
        docdic_value.append(nouns_and_ents)
        docdic_value.append(ents_dic)
        document[pi]=docdic_value
        all_docs.append(document)
        number += 1
        print("done with " + str(number))
    return all_docs


pickle_out = open(r"C:\Users\Goegg\OneDrive\Desktop\Durchgänge\3. TFIDF - BERT (FEHLT)\BERT_NE_Spacy_Nouns_COMPOSED_NOTFIDF.pickle", "wb")
pickle.dump(get_noun_and_ne(dic), pickle_out)
pickle_out.close()


# In[ ]:




