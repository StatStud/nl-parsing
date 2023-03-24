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
import jsonlines
import json

class entity_linker:
    def __init__(self, embeddings_data="primekg_embeddings_synonym.pickle",
                 datafile = "nodes_full.csv", 
                 model_name = "tekraj/avodamed-synonym-generator1"):
        self.embeddings_data = embeddings_data
        self.datafile = datafile
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.srl_model_link = "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
        self.srl_model = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
        self.medspacy_model = spacy.load("en_ner_bc5cdr_md")
        

    def run_pre_embedd(self):
        """
        Generate Emneddings and FAISS indices for given nodes.
        NOTE: Assuming column with node names is called "node_name"

        Parameters:
        -----------
        self.datafile : data csv file
            a csv file of the nodes. Column "node_name" must be present

        Returns:
        --------
        NO EXPLICT RETURN VALUE
        pickle file
            Pickle file of Huggingface dataset insatance is saved
        """
        print("starting pre-embedding")
        hugging_dataset = Dataset.from_pandas(pd.read_csv(self.datafile))    

        column_to_embedd = "node_name"

        embeddings_dataset = hugging_dataset.map(
            lambda x: {"embeddings": self.get_embeddings(x[column_to_embedd]).detach().cpu().numpy()[0]}
        )
        print("embeddings done")

        embeddings_dataset.add_faiss_index(column='embeddings')
        print("faiss added")
        
        output_file_name = f"{self.datafile[:-4]}_embeddings_{self.model_name.split('/')[-1] if '/' in self.model_name else self.model_name}.pickle"

        with open(output_file_name, "wb") as f:
            pickle.dump(embeddings_dataset,f)

        print("embeddings saved")
        
    def get_embeddings(self,text_list):
        """
        Sub function for self.run_pre_embedd().
        Applies the actual embedding to dataset.

        Parameters:
        -----------
        text_list : HuggingFace dataset column
            The column from the huggingface dataset instance containing the node name
            
        Returns:
        --------
        Embeddings
            Final embeddings.
        """
        encoded_input = self.tokenizer(
            text_list, padding=True, truncation=True, return_tensors="pt"
        )
        encoded_input = {k: v for k, v in encoded_input.items()}
        #move the encoded input to the device
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        model_output = self.model(**encoded_input)
        return self.cls_pooling(model_output)

    def cls_pooling(self,model_output):
        """
        Sub function of self.get_embeddings.
        This function takes in model_output as input, which is the output obtained after 
        passing the input text to the pre-trained BERT model, and returns the first (CLS) 
        token from the last hidden state of the model output.

        Parameters:
        -----------
        model_output : pytorch tensor
            A tensor representing the output of the pre-trained BERT model after passing the input text through it.

        Returns:
        --------
        pytorch tensor
            A tensor representing the first (CLS) token from the last hidden state of the pre-trained BERT model output.
        """
        return model_output.last_hidden_state[:, 0]
    
    def extract_sentences(self,text):
        """
        This function takes in a string of text as input and returns a list of sentences extracted from that text. 
        It uses the sent_tokenize function from the nltk library to tokenize the input text into sentences.

        Parameters:
        -----------
        text : String
            The context/paragraph 

        Returns:
        --------
        list[Str]
            A list of sentences from the context/paragraph
        """
        return nltk.sent_tokenize(text)
    
    def extract_arg1_words(self,data):
        """
        Sub function of self.srl_extractor. This function takes in a 
        dictionary data as input, which contains the output of the 
        Semantic Role Labeling (SRL) model for a given sentence, and 
        returns a list of words that correspond to the ARG1 (agent) role in the sentence.

        Parameters:
        -----------
        data : dictionary
            A dictionary containing the output of the SRL model for a given sentence.

        Returns:
        --------
        list[Str]
            A list of strings representing the words that correspond to the ARG1 role in the sentence.
        """
        arg1_words = []
        for verb in data['verbs']:
            description = verb['description']
            arg1 = re.findall(r"ARG1:\s([\w\s]+)", description)
            if arg1:
                arg1_words.append(arg1[0].strip())
        return arg1_words    

    def srl_extractor(self,sent):
        """
        This function takes in a string of text as input, 
        uses the Semantic Role Labeling (SRL) model to extract 
        the ARG1 words from each sentence in the input text, and 
        returns a list of all the ARG1 words extracted from the text.

        Parameters:
        -----------
        sent : String
            A string representing one sentence from the context

        Returns:
        --------
        list[Str]
            A list of strings representing the ARG1 words extracted from the input text.
        """
        pred = self.srl_model.predict(sentence=sent)
        return self.extract_arg1_words(pred)
    
    def regex_entity_extract(self,sent):
        """
        Sub function of self.named_entity_extractor.
        Regex entity extractor, in case medspacy returns
        empyt list.

        Parameters:
        -----------
        sent : String
            A string representing one sentence from the context

        Returns:
        --------
        list[Str]
            List of relevant string terms
        """
        options = ["lethargic", "shortness of breath",
                  "headache","hypertension","chest palpitations",
                  "chest pain", "shortness of breath", "throat", 
                  "inflamed", "interstitial fibrosis", 'aspirin',
                  "reticulonodular infiltrate","eggshell calcification",
                  "adenopathies", "lisinopril", "metoprolol","warfarin",
                  "left atrium", "enlarged left ventricle", "nausea",
                  'diaphoretic', 'epigastric area','asthma',
                  'diabetes mellitus', "chronic bronchitis"]
        pattern = r"\b(" + "|".join(map(re.escape, options)) + r")\b"
        matches = re.findall(pattern, sent)

        if matches:
            return matches
        else:
            return []
    
    def named_entity_extractor(self,sentence):
        """
        MedSpacy entity extractor.

        Parameters:
        -----------
        sentence : String
            A string representing one sentence from the context

        Returns:
        --------
        list[Str]
            Description of the return value.
        """
        docs = self.medspacy_model(sentence)
        result = list(docs.ents)
        result = [str(x) for x in result]

        if result:
            return result
        else:
            return self.regex_entity_extract(sentence)
        
    def merge_lists(self,lst1, lst2):
        """
        Sub function for self.run_all.
        Combines extracted queries from SRL and 
        NER and combines as a single list of queries 
        for each sentence in the context.

        Parameters:
        -----------
        lst : list[Str]
            Either SRL list of NER list. Order does not matter
        lst2 : list[Str]
            Either SRL list of NER list. Order does not matter

        Returns:
        --------
        list[Str]
            combined list of string queries, for each sentnece
        """
        result = []
        for sublst1, sublst2 in zip(lst1, lst2):
            result.append(sublst1 + sublst2)
        return result
    
    def run_query_faiss(self,query):
        """
        Sub function for self.get_embedding_matches.
        Obtain the top_k closest vector in vector space of nodes.
        This assumes that the pre-embedding file is saved and previously
        defined.

        Parameters:
        -----------
        query : String
            Either the SRL or NER string element representing the search query
        self.embeddings_data : pickle
            The pre-saved embeddings data. See self.run_pre_embedd function
        top_k: int
            The count of candidates we wish to return.

        Returns:
        --------
        pandas df
            Pandas dataframe containing the top_k node node_names (with 
            corresponding scores) that are closest in vector space to query
        """
        
        top_k = 1

        with open(self.embeddings_data, "rb") as f:
            hugging_dataset = pickle.load(f)    

        if (not isinstance(query, str)) or (not isinstance(query, list)):
            query = str(query)

        query_embedding = self.get_embeddings([query]).cpu().detach().numpy()

        scores, samples = hugging_dataset.get_nearest_examples(
            "embeddings", query_embedding, k=top_k
        )

        samples_df = pd.DataFrame.from_dict(samples)
        samples_df["scores"] = scores
        samples_df.sort_values("scores", ascending=True, inplace=True)
        return samples_df[['node_name']]
    
    def get_embedding_matches(self,query_lst):
        """
        Returns consolidated FAISS matching results for
        each context sentence.

        Parameters:
        -----------
        query_lst : list[Str]
            List of extracted string queries from SRL and NER

        Returns:
        --------
        List[Dict(Tuple)]
            A list of dictionary values, where each dictionary contains the 
            top_k nearest node matches for each query of each context sentence.
            The keys of the dictionary are the query for a given sentence, and the value
            is a tuple, where the first element is the node_name of the closest match, and
            the second element is the FAISS score (lower is better) of the match.
        """
        result_lst = []
        for lst in query_lst:
            output_dict = {}
            for query in lst:
                query_results = self.run_query_faiss(query).values.tolist()
                output_dict[query] = [(result[0]) for result in query_results][0]
            result_lst.append(output_dict)
        return result_lst
    
    def extract_top_strings(self,list_of_dicts):
        """
        Applied after running self.get_embedding_matches().
        Extracts the top 1 values (instead of top_k) final 
        node candidate for each sentence.

        Parameters:
        -----------
        list_of_dicts : List[Dict(Tuple)]
            Consolidated list of FAISS results. See output for self.get_embedding_matches()

        Returns:
        --------
        list[Str]
            A final list of entities that represents the best
            entity linking for the entire context
        """
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
    
    def print_log(self, sent_set, arg1_lst, named_entities, faiss_dict):
        """
        Tracking method for determining possible algorithmic improvments.
        This shows where each extracted query comes from (SRL or NER), the 
        quality of matches for each method, and --, all grouped on a per-sentnece
        basis for each context

        Parameters:
        -----------
        sent_set : list[Str]
            List of string sentences from context
        arg1_lst : list[Str]
            List of extracted string queries from SRL model, for each sentence
        named_entities : list[Str]
            List of extracted string queries from NER model, for each sentence 
        faiss_dict : List[Dict(Tuple)]
            List of consolidated top_k candiates for each query of the context sentence

        Returns:
        --------
        log.txt
            Text file containing all metrics.
        """
        with open('log.txt', 'w') as f:
            for i in range(len(sent_set)):
                f.write("Sentence: " + sent_set[i] + "\n")
                f.write("SRL Match: " + str(arg1_lst[i]) + "\n")
                f.write("Entity Extract Match: " + str(named_entities[i]) + "\n")
                f.write("FAISS Results: " + "\n")
                for k, v in faiss_dict[i].items():
                    f.write("QUERY: " + k + "\n")
                    f.write("Ranked Results" + "\n")
                    for ele in v:
                        f.write(str(ele) + "\n")
                f.write("\n")
    
    
    def print_model_log(self, file_path = "test.jsonl"):
        """
        Model Tracking of which algorithms and/or modifications were used, 
        and their resulting accuracy metrics.

        Parameters:
        -----------

        Returns:
        --------
        model_alter.txt
            Text file containing all model metrics with performance.
        """
        
        with open('model_alter.txt', 'w') as f:
            mean_accuracy, median_accuracy, \
            lowest_accuracy, highest_accuracy, context_count = self.get_accuracy_metrics(file_path)
            f.write("Model Summary\n"
                    f"Input Nodes Dataset: {self.datafile}\n"
                    f"Input Pre-Embedding File: {self.embeddings_data}\n"
                    f"FAISS Tokenizer: {self.tokenizer}\n"
                    f"{'*'*20} Query Extractors {'*'*20}\n"
                    f"SRL Model: {self.srl_model_link}\n"
                    f"NER Model: MedSpacy + Regex\n"
                    f"{'*'*20} Node Candidate Extractors {'*'*20}\n"
                    f"String Matching: NONE\n"
                    f"Vector Search: FAISS\n"
                    f"Top K Nodes per Sentence: 1"
                    f"{'*'*20} Final Accuracy Results {'*'*20}\n"
                    f"Number of Contexts: {context_count}\n"
                    f"Average Accuracy: {mean_accuracy}\n"
                    f"Median Accuracy: {median_accuracy}\n"
                    f"Lowest Accuracy: {lowest_accuracy}\n"
                    f"Highest Accuracy: {highest_accuracy}\n")

            

    def get_accuracy_metrics(self, file_path = "test.jsonl"):
        """
        Parses through output file and obtains accuracies metrics for
        all contextes in the file.

        Parameters:
        -----------
        file_path : .jsonl file
            The file containing all the contexts after have been parsed with entity linking

        Returns:
        --------
        [mean_accuracy, median_accuracy, lowest_accuracy, highest_accuracy] floats
            Specific accuracy metrics, as specified by variable name.
        """
        with open(file_path) as f:
            accuracies = []
            for line in f:
                data = json.loads(line)
                accuracies.append(data["accuracy"])

            mean_accuracy = np.mean(accuracies)
            median_accuracy = np.median(accuracies)
            lowest_accuracy = min(accuracies)
            highest_accuracy = max(accuracies)

            return mean_accuracy, median_accuracy, lowest_accuracy, highest_accuracy, len(accuracies)
#######################################################################

    def predict(self,text, show_log = True):
        """
        The mother of all functions. This is the main
        function used to parse a context sentence.

        Parameters:
        -----------
        text : String
            The entire string context
        show_log : boolean, optional (default=True)
            Whehter or not to create the log.txt file for tracking and debugging

        Returns:
        --------
        list[Str]
            The final list of entity linking used to represent the entire context
        """
        sent_set = self.extract_sentences(text)

        arg1_lst = []
        named_entities = []

        for sent in sent_set:
            srl_output = self.srl_extractor(sent)
            arg1_lst.append(srl_output)


        #named_entities = list(map(lambda sent: named_entity_extractor(sent,medspacy_model), sent_set))
        for sent in sent_set:
            named_entities.append(self.named_entity_extractor(sent))

        query_lst = self.merge_lists(arg1_lst, named_entities)            
        faiss_dict = self.get_embedding_matches(query_lst)
        
        # remove duplicate faiss keys
        faiss_dict = remove_duplicate_keys(faiss_dict)
        
        #remove default nodes
        default_nodes = ['all', 'sad', 'old age', 'elderly',
                        'abt-493', 'surgery complication',
                        'fits','symptoms and signs']
        
        faiss_dict = [{k:v for k,v in d.items() if v not in default_nodes} for d in faiss_dict]
        
        
        #get linked entities
        linked_entities = get_unique_values(faiss_dict)
        
        

#         if show_log:
#             self.print_log() 

        #return final list of input entities
        #for each sentence, get the top 1 FAISS RESULTS

        return faiss_dict,linked_entities


    def save_output(self, input_data="data/medqa-ddb-handmap.jsonl", output_data="test.txt"):
        """
        Similar to self.run_all, only instead of manually inputting the 
        context as the argument, the user can instead pass a file containing
        many contexts to parse.

        Parameters:
        -----------
        input_data : .jsonl file
            The dataset containing all the contexts over which we wish to parse
        output_data : .txt file
            A copy of the input_data, but with the added fields after processing.
            Namely: "linked_auto" (the final entity linking list) and "accuracy" 
            (how well "linked_auto" matches to "linked_entities" field)

        Returns:
        --------
        .txt file
            A modified version of the input_data containing the parsing fields
        """
        # Open the JSONL input file and create a new output file
        with jsonlines.open(input_data) as reader, open(output_data, mode='w') as writer:

            # Loop through all records in the input file
            for record in reader:

                # Apply the run_all function to the question
                faiss_dict,linked_auto = self.predict((record['question']))

                # Add the linked_auto list to the record as a new field
                record['linked_auto'] = linked_auto
                record['query_node_mapping'] = faiss_dict

                # Compare the linked_entities and linked_auto lists and calculate accuracy
                linked_entities = record['linked_entities']
                num_correct = len(set([i.lower() for i in linked_entities]).intersection(set([i.lower() for i in linked_auto])))
                accuracy = (num_correct / len(linked_entities))

                # Add the accuracy to the record as a new field
                record['accuracy'] = accuracy

                # Write the updated record to the output file
                writer.write("question\n")
                writer.write(record['question'] + "\n")
                writer.write("linked_entities\n")
                writer.write(str(record['linked_entities']) + "\n")
                writer.write("linked_auto\n")
                writer.write(str(record['linked_auto']) + "\n")
                writer.write("query_node_mapping\n")
                writer.write(str(record['query_node_mapping']) + "\n")
                writer.write("accuracy\n")
                writer.write(str(record['accuracy']) + "\n\n")

                
def remove_duplicate_keys(dicts_list):
    """
    Removes duplicate keys in a list of dictionaries.

    Args:
    - dicts_list: a list of dictionaries

    Returns:
    - a list of dictionaries with duplicate keys removed
    """

    output_list = []
    keys_set = set()

    for d in dicts_list:
        new_dict = {}
        for k, v in d.items():
            if k not in keys_set:
                new_dict[k] = v
                keys_set.add(k)
        output_list.append(new_dict)

    return output_list

def get_unique_values(data):
    unique_values = set()
    for d in data:
        for v in d.values():
            unique_values.add(v)
    return list(unique_values)

def parse_sentence(sentence):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(sentence)
    output = []
    
    if len(doc) == 3 and doc[1].text == 'and':
        output.append(doc[0].text)
        output.append(doc[2].text)
        return output
    else:
        return sentence
    
    for i, token in enumerate(doc):
        if token.pos_ == 'ADJ' and doc[i+1].pos_ == 'NOUN':
            if doc[i+2].text == 'and':
                output.append(token.text + ' ' + doc[i+1].text)
            elif doc[i+2].pos_ == 'ADP' and doc[i+3].pos_ == 'ADJ' and doc[i+4].pos_ == 'NOUN':
                output.append(token.text + ' ' + doc[i+1].text + ' ' + doc[i+3].text + ' ' + doc[i+4].text)
    
    return output
