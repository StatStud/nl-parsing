from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import medspacy
from allennlp.predictors.predictor import Predictor
import spacy
import re
from collections import namedtuple 
import pandas as pd
from medspacy.ner import TargetRule
from collections import defaultdict
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer
import time
import torch
from scipy.spatial.distance import cosine
import pickle
import faiss
import numpy as np
import nltk

#embeddings_data = "embeddings_ddb.pickle"
embeddings_data = "ddb_embeddings_synonym.pickle"
model_name = "tekraj/avodamed-synonym-generator1"

datafile = "ddb_nodes_header.csv"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
srl_model = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
medspacy_model = medspacy.load()
simcse_model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
simcse_tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")





def get_embeddings(text_list, tokenizer, model):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v for k, v in encoded_input.items()}
    #move the encoded input to the device
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]


# setting device on GPU if available, else CPU
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

def run_pre_embedd(tokenizer, model):
    '''
    info: 
    tokenizer: huggingface model instance 
    model: huggingface model instance
    notes: assuming that column of entities is called 'node_name'
    '''
    print("starting pre-embedding")
    hugging_dataset = Dataset.from_pandas(pd.read_csv('ddb_nodes_header.csv'))    
    
    column_to_embedd = "node_name"

    embeddings_dataset = hugging_dataset.map(
        lambda x: {"embeddings": get_embeddings(x[column_to_embedd], tokenizer, model).detach().cpu().numpy()[0]}
    )
    print("embeddings done")

    embeddings_dataset.add_faiss_index(column='embeddings')
    print("faiss added")
    
    with open("embeddings.pickle", "wb") as f:
        pickle.dump(embeddings_dataset,f)
        
    print("embeddings saved")




def extract_sentences(text):
    return nltk.sent_tokenize(text)

def run_query_faiss(query,model,tokenizer):
    
    with open(embeddings_data, "rb") as f:
        hugging_dataset = pickle.load(f)    

    if (not isinstance(query, str)) or (not isinstance(query, list)):
        query = str(query)
        
    query_embedding = get_embeddings([query], tokenizer, model).cpu().detach().numpy()

    scores, samples = hugging_dataset.get_nearest_examples(
        "embeddings", query_embedding, k=10
    )

    samples_df = pd.DataFrame.from_dict(samples)
    samples_df["scores"] = scores
    samples_df.sort_values("scores", ascending=True, inplace=True)
    return samples_df
    
def extract_arg1_words(data):
    arg1_words = []
    for verb in data['verbs']:
        description = verb['description']
        arg1 = re.findall(r"ARG1:\s([\w\s]+)", description)
        if arg1:
            arg1_words.append(arg1[0].strip())
    return arg1_words    
    
def srl_extractor(sent, srl):
    """Returns a dictionary with all semantic role labels for an input sentence
       outputs a role-label : word dictionary structure.
    """
    pred = srl.predict(sentence=sent)
    return extract_arg1_words(pred)


        
def named_entity_extractor(sentence,nlp):
    target_matcher = nlp.get_pipe("medspacy_target_matcher")
    target_rules = [
        TargetRule("symptoms", "EVIDENCE_OF_symptoms", pattern=[{'LOWER': {'REGEX': 'lethargic'}}]),
        TargetRule("symptoms", "EVIDENCE_OF_symptoms", pattern=[{'LOWER': {'REGEX': 'headache'}}]),
        TargetRule("symptoms", "EVIDENCE_OF_symptoms", pattern=[{'LOWER': {'REGEX': 'headache'}}]),
        TargetRule("symptoms", "EVIDENCE_OF_symptoms", pattern=[{'LOWER': {'REGEX': 'hypertension'}}]),
        TargetRule("symptoms", "EVIDENCE_OF_symptoms", pattern=[{'LOWER': {'REGEX': 'chest palpitations'}}]),
        TargetRule("symptoms", "EVIDENCE_OF_symptoms", pattern=[{'LOWER': {'REGEX': 'chest pain'}}]),
        TargetRule("symptoms", "EVIDENCE_OF_symptoms", pattern=[{'LOWER': {'REGEX': 'shortness of breath'}}]),
        TargetRule("symptoms", "EVIDENCE_OF_symptoms", pattern=[{'LOWER': {'REGEX': 'throat'}}]),
        TargetRule("symptoms", "EVIDENCE_OF_symptoms", pattern=[{'LOWER': {'REGEX': 'inflamed'}}]),
        TargetRule("consolidation", "EVIDENCE_OF_symptoms"),
        TargetRule("EVIDENCE_OF_symptoms", "EVIDENCE_OF_symptoms", pattern=[{'LOWER': {'REGEX': 'infiltrat(e|es|ion)'}}]),
              ]
    target_matcher.add(target_rules)
    docs = nlp(sentence)
    result = list(docs.ents)
    return [str(x) for x in result]
        
def merge_lists(lst1, lst2):
    result = []
    for sublst1, sublst2 in zip(lst1, lst2):
        result.append(sublst1 + sublst2)
    return result

def string_match(query_lst):

    kgraph_nodes = pd.read_csv(datafile)[['node_name']].values.tolist()
    kgraph_nodes = [item for sublist in kgraph_nodes for item in sublist]
    
    result_lst = []
    for lst in query_lst:
        output_dict = {}
        for query in lst:
            tmp_lst = []
            for node in kgraph_nodes:
                fuzzy_score = fuzz.token_set_ratio(query, node) 
                if fuzzy_score > 70:
                    #take the inverse of the value to minimize rather than
                    #maximizing the value 
                    fuzzy_score = 100-fuzzy_score
                    tmp_lst.append((node,fuzzy_score))
            output_dict[query] = tmp_lst
        result_lst.append(output_dict)
    return result_lst
    
    
def get_embedding_matches(query_lst):
    result_lst = []
    for lst in query_lst:
        output_dict = {}
        for query in lst:
            query_results = run_query_faiss(query, model, tokenizer)[['node_name','scores']].values.tolist()
            output_dict[query] = [(result[0], result[1]) for result in query_results]
        result_lst.append(output_dict)
    return result_lst


def combine_scores(string_match, faiss_match, string_weight=0.75, faiss_weight = 0.25):
    # combine all dictionaries into a single dictionary
    combined_dict = {}
    for dict_item in string_match + faiss_match:
        for key, value in dict_item.items():
            if key in combined_dict:
                combined_dict[key].extend(value)
            else:
                combined_dict[key] = value
    
    # combine tuples with the same string name using a weighted score
    for key, value in combined_dict.items():
        string_scores = {}
        for string, score in value:
            if string in string_scores:
                string_scores[string].append(score)
            else:
                string_scores[string] = [score]
        
        new_value = []
        for string, scores in string_scores.items():
            if len(scores) > 1:
                if string_weight + faiss_weight != 1:
                    print("WARNING: weights do not add up to 1. Please check weight values")
                weighted_score = round((string_weight * scores[0]) + (faiss_weight * scores[1]))
                new_tuple = (string, weighted_score)
                new_value.append(new_tuple)
            else:
                new_value.append((string, scores[0]))
        
        combined_dict[key] = new_value
    
    return [combined_dict]


def flatten_list(lst):
    flatten_lst = []
    for item in lst:
        if type(item) == list:
            flatten_lst.extend(flatten_list(item))
        else:
            flatten_lst.append(item)
    return flatten_lst

def filter_best_matches(filtered_data):
    result = []
    for d in filtered_data:
        new_dict = {}
        for k, v in d.items():
            filtered_values = [t for t in v if t[1] < 3]
            if filtered_values:
                new_dict[k] = filtered_values
        result.append(new_dict)
    return result

def replace_dicts(lst1, lst2):
    final = []
    for i in range(len(lst1)):
        if lst2[i] != {}:
            final.append(lst2[i])
        else:
            final.append(lst1[i])
    return final


def combine_dicts_lists(dict_lst1, dict_lst2):
    combined_lst = []
    for i in range(len(dict_lst1)):
        combined = {key: dict_lst1[i].get(key, []) + dict_lst2[i].get(key, []) for key in dict_lst1[i].keys()}
        combined_lst.append(combined)
    return combined_lst

def print_stirng_faiss_output(test):
    for i in range(len(test['sent'])):
        print("***Input Sentence***: ",test['sent'][i])
        print("*****"*20)
        print("***ARG1 queries***: ",test['arg1'][i])
        print("*****"*20)
        print("***Entity queries***: ",test['entities'][i])
        print("*****"*20)
        print("***String Matches***: ",test['string_match'][i])
        print("*****"*20)
        print("***FAISS Matches***: ",test['faiss_match'][i])
        print("*****"*20)
        print("***RESULTS***: ")
        for key,value in test['best_results'][i].items():
            print("query: ", key)
            print("matches: ", value)
            print("")
        print("\n")
        
def combine_matches(string_matches, faiss_matches):
    combined_matches = []
    
    for string_match, faiss_match in zip(string_matches, faiss_matches):
        #each item (string_match and faiss_match) is 
        # a dictionary where the key is the query and the value is list of tuples
        combined_match = {}
        
        # First check for low score in string_matches
        for key, value in string_match.items():
            #for each dictionary from string matches,
            #the key is the query, value is the **list** of tuples

            #because value is *list* of tuples, loop through the list
            for tup_ele in value:
                if tup_ele[1] < 3:
                    combined_match[key] = value

                
        # Combine values from both matches for keys not in combined_match
        for key, value in faiss_match.items():
            if key in combined_match:
                continue
            
            if key in string_match:
                combined_value = []
                for string_tuple, faiss_tuple in zip(string_match[key], faiss_match[key]):
                    combined_score = 0.25 * string_tuple[1] + 0.75 * faiss_tuple[1]
                    combined_value.append((string_tuple[0], combined_score))
                combined_match[key] = combined_value
            else:
                combined_match[key] = value
        
        # Check for duplicate string names and merge them
        for key, value in combined_match.items():
            name_dict = {}
            for string_tuple in value:
                if string_tuple[0] in name_dict:
                    old_score = name_dict[string_tuple[0]][1]
                    new_score = 0.25 * string_tuple[1] + 0.75 * old_score
                    name_dict[string_tuple[0]] = (string_tuple[0], new_score)
                else:
                    name_dict[string_tuple[0]] = string_tuple
            combined_match[key] = list(name_dict.values())
        
        combined_matches.append(combined_match)
    
    return combined_matches


def run_all(text):
    """
    Link key phrases in the sentences to a set of nodes in the
    knowledge graph.
    """
    sent_set = extract_sentences(text)
    print("here is sent_set:", sent_set)
    
    
    arg1_lst = []
    named_entities = []
    
    for sent in sent_set:
        srl_output = srl_extractor(sent, srl_model)
        arg1_lst.append(srl_output)


    #named_entities = list(map(lambda sent: named_entity_extractor(sent,medspacy_model), sent_set))
    for sent in sent_set:
        named_entities.append(named_entity_extractor(sent,medspacy_model))

    query_lst = merge_lists(arg1_lst, named_entities)
    string_matches = string_match(query_lst)
    #tmp = best_string_matches(string_matches)
    b_string_matches = filter_best_matches(string_matches)
    
    final_string_matches = replace_dicts(string_matches,b_string_matches)
            
    faiss_dict = get_embedding_matches(query_lst)
    
    test = combine_matches(final_string_matches,faiss_dict)
    
    return test,final_string_matches,faiss_dict,query_lst
