
datafile = "ddb_nodes_header.csv"
embeddings_data = "embeddings_ddb.pickle"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = "michiyasunaga/BioLinkBERT-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
srl_model = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
medspacy_model = medspacy.load()
simcse_model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
simcse_tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")


sent_set = ["my throat hurts like hell lethargic",
            "Have had a terrible headache",
           "Her hiccups are not stopping, is this serious?",
           "I have been losing a lot of hair lately"]




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
    '''

    hugging_dataset = Dataset.from_pandas(pd.read_csv('ddb_nodes_header.csv'))
    print("dataset in")
    
    
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


# res = faiss.StandardGpuResources()  # use a single GPU
# print("Faiss on GPU status:", res)

# ngpus = faiss.get_num_gpus()
# print("number of GPUs:", ngpus)

# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.current_device())
# print(torch.cuda.device(0))
# print(torch.cuda.get_device_name(0))




# #Additional Info when using cuda
# if device.type == 'cuda':
#     print(torch.cuda.get_device_name(0))
#     print('Memory Usage:')
#     print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
#     print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')


print("starting pre-embedding")
#run_pre_embedd()
print("done")

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

#nlp = medspacy.load()
    
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
                if fuzzy_score > 80:
                    tmp_lst.append(node)
            output_dict[query] = tmp_lst
        result_lst.append(output_dict)
    return result_lst
    
    
def get_embedding_matches(query_lst):
    result_lst = []
    for lst in query_lst:
        output_dict = {}
        for query in lst:
            result = flatten_list(run_query_faiss(query,model,tokenizer)[['node_name']].values.tolist())
            output_dict[query] = result
        result_lst.append(output_dict)
    return result_lst

def compare(sentence1, sentence2, model, tokenizer):
    # Tokenize input texts
    texts = [sentence1, sentence2]
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    # Get the embeddings
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

    # Calculate cosine similarities
    # Cosine similarities are in [-1, 1]. Higher means more similar
    cosine_sim_0_1 = 1 - cosine(embeddings[0], embeddings[1])

    return cosine_sim_0_1

def get_sent_embedding(sent_set, final, model, tokenizer):
    output = {}
    result = []
    score_lst = []
    for sentence in range(len(final)):
        for key,lst in final[sentence].items():
            for ele in lst:
                score = compare(sent_set[i], ele, model, tokenizer)
                score_lst.append((ele, score))
            sorted_scores = sorted(score_lst, key=lambda x: x[1], reverse=True)
            output[key] = sorted_scores
            score_lst = []
        result.append(output)
        output = {}
        print("sentence ", sentence+1, "out of ", len(sent_set), "completed")
    return result


def flatten_list(lst):
    flatten_lst = []
    for item in lst:
        if type(item) == list:
            flatten_lst.extend(flatten_list(item))
        else:
            flatten_lst.append(item)
    return flatten_lst

def combine_dicts_lists(dict_lst1, dict_lst2):
    combined_lst = []
    for i in range(len(dict_lst1)):
        combined = {key: dict_lst1[i].get(key, []) + dict_lst2[i].get(key, []) for key in dict_lst1[i].keys()}
        combined_lst.append(combined)
    return combined_lst


def run_all(sent_set):
    """
    Link key phrases in the sentences to a set of nodes in the
    knowledge graph.
    """
    print("here is sent_set:", sent_set)
    
    #debugger list will match length of sent_set
    
    
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
    faiss_dict = get_embedding_matches(query_lst)
    
    combined_result = combine_dicts_lists(string_matches,faiss_dict)
 
    print("***** RUNNING COSINE SIM ****")    
    context_embedding = get_sent_embedding(sent_set, combined_result, model, tokenizer) 
    
    final = {
    "sent":sent_set,
    "arg1": arg1_lst,
    'entities': named_entities,
    "string_match": string_matches,
    "faiss_match": faiss_dict,
    "best_results": context_embedding
    }
    
    return final
    
def print(final):
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
