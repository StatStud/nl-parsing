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
        self.srl_model = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
        self.medspacy_model = medspacy.load()
        

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
                  "inflamed"]
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
        target_matcher = self.medspacy_model.get_pipe("medspacy_target_matcher")
        target_rules = [
            TargetRule("symptoms", "EVIDENCE_OF_symptoms", pattern=[{'LOWER': {'REGEX': '\blethargic\b'}}]),
            TargetRule("symptoms", "EVIDENCE_OF_symptoms", pattern=[{'LOWER': {'REGEX': 'headache'}}]),
            TargetRule("consolidation", "EVIDENCE_OF_symptoms"),
            TargetRule("EVIDENCE_OF_symptoms", "EVIDENCE_OF_symptoms", pattern=[{'LOWER': {'REGEX': 'infiltrat(e|es|ion)'}}]),
                  ]
        target_matcher.add(target_rules)
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
        
        top_k = 10

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
        return samples_df
    
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
                query_results = self.run_query_faiss(query)[['node_name','scores']].values.tolist()
                output_dict[query] = [(result[0], result[1]) for result in query_results]
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
    
#######################################################################

    def run_all(self,text, show_log = True):
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

        if show_log:
            self.print_log(sent_set,arg1_lst,named_entities,faiss_dict) 

        #return final list of input entities
        #for each sentence, get the top 1 FAISS RESULTS

        return self.extract_top_strings(faiss_dict)


    def save_output(self, input_data = "data/medqa.jsonl", output_data = "test.jsonl"):
        """
        Similar to self.run_all, only instead of manually inputting the 
        context as the argument, the user can instead pass a file containing
        many contexts to parse.

        Parameters:
        -----------
        input_data : .jsonl file
            The dataset containing all the contexts over which we wish to parse
        output_data : .jsonl file
            A copy of the input_data, but with the added fields after processing.
            Namely: "linked_auto" (the final entity linking list) and "accuracy" 
            (how well "linked_auto" matches to "linked_entities" field)

        Returns:
        --------
        .jsonl file
            A modified version of the input_data containing the parsing fields
        """
        # Open the JSONL input file and create a new output file
        with jsonlines.open(input_data) as reader, jsonlines.open(output_data, mode='w') as writer:

            # Loop through all records in the input file
            for record in reader:

                # Apply the run_all function to the question
                linked_auto = self.run_all((record['question']))

                # Add the linked_auto list to the record as a new field
                record['linked_auto'] = linked_auto

                # Compare the linked_entities and linked_auto lists and calculate accuracy
                linked_entities = record['linked_entities']
                num_correct = len(set([i.lower() for i in linked_entities]).intersection(set([i.lower() for i in linked_auto])))
                accuracy = (num_correct / len(linked_entities))

                # Add the accuracy to the record as a new field
                record['accuracy'] = accuracy

                # Write the updated record to the output file
                writer.write(record)
