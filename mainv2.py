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


def regex_entity_extract(sent):
    options = ["lethargic", "shortness of breath",
              "headache","hypertension","chest palpitations",
              "chest pain", "shortness of breath", "throat", 
              "inflamed"]
    pattern = r"\b(" + "|".join(map(re.escape, options)) + r")\b"
    matches = re.findall(pattern, sent)

    if matches:
        return matches
    else:
        return []

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
        TargetRule("symptoms", "EVIDENCE_OF_symptoms", pattern=[{'LOWER': {'REGEX': '\blethargic\b'}}]),
        TargetRule("symptoms", "EVIDENCE_OF_symptoms", pattern=[{'LOWER': {'REGEX': 'headache'}}]),
        TargetRule("symptoms", "EVIDENCE_OF_symptoms", pattern=[{'LOWER': {'REGEX': 'headache'}}]),
        TargetRule("symptoms", "EVIDENCE_OF_symptoms", pattern=[{'LOWER': {'REGEX': 'hypertension'}}]),
        TargetRule("symptoms", "EVIDENCE_OF_symptoms", pattern=[{'LOWER': {'REGEX': 'chest palpitations'}}]),
        TargetRule("symptoms", "EVIDENCE_OF_symptoms", pattern=[{'LOWER': {'REGEX': 'chest pain'}}]),
        TargetRule("symptoms", "EVIDENCE_OF_symptoms", pattern=[{'LOWER': {'REGEX': '\bshortness of breath\b'}}]),
        TargetRule("symptoms", "EVIDENCE_OF_symptoms", pattern=[{'LOWER': {'REGEX': 'throat'}}]),
        TargetRule("symptoms", "EVIDENCE_OF_symptoms", pattern=[{'LOWER': {'REGEX': 'inflamed'}}]),
        TargetRule("consolidation", "EVIDENCE_OF_symptoms"),
        TargetRule("EVIDENCE_OF_symptoms", "EVIDENCE_OF_symptoms", pattern=[{'LOWER': {'REGEX': 'infiltrat(e|es|ion)'}}]),
              ]
    target_matcher.add(target_rules)
    docs = nlp(sentence)
    result = list(docs.ents)
    result = [str(x) for x in result]
    
    if result:
        return result
    else:
        return regex_entity_extract(sentence)
    
def get_embedding_matches(query_lst):
    result_lst = []
    for lst in query_lst:
        output_dict = {}
        for query in lst:
            query_results = run_query_faiss(query, model, tokenizer)[['node_name','scores']].values.tolist()
            output_dict[query] = [(result[0], result[1]) for result in query_results]
        result_lst.append(output_dict)
    return result_lst


def flatten_list(lst):
    flatten_lst = []
    for item in lst:
        if type(item) == list:
            flatten_lst.extend(flatten_list(item))
        else:
            flatten_lst.append(item)
    return flatten_lst

def merge_lists(lst1, lst2):
    result = []
    for sublst1, sublst2 in zip(lst1, lst2):
        result.append(sublst1 + sublst2)
    return result

def print_log(sent_set,arg1_lst,named_entities,faiss_dict):
    for i in range(len(sent_set)):
        print("Sentence: ", sent_set[i])
        print("SRL Match: ",arg1_lst[i])
        print("Entity Extract Match: ",named_entities[i])
        print("FAISS Results: ")
        for k,v in faiss_dict[i].items():
            print("QUERY: ", k)
            print("Ranked Results")
            for ele in v:
                print(ele)
        print("\n")

def extract_top_strings(list_of_dicts):
    top_k = 1
    top_strings = {}
    for dictionary in list_of_dicts:
        for key, value in dictionary.items():
            if key in top_strings:
                top_strings[key].extend([t[0] for t in value[:top_k]])
            else:
                top_strings[key] = [t[0] for t in value[:top_k]]
    merged_list = list(set([s for sublist in top_strings.values() for s in sublist]))
    return merged_list

def run_all(text, show_log = True):
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
    faiss_dict = get_embedding_matches(query_lst)
    
    if show_log:
        print_log(sent_set,arg1_lst,named_entities,faiss_dict) 
        
    #return final list of input entities
    #for each sentence, get the top 3 FAISS RESULTS
    
    return extract_top_strings(faiss_dict)
