# import argparse
# import pandas as pd
# from datasets import Dataset
# from transformers import AutoModel, AutoTokenizer
# import time
# import torch
# from scipy.spatial.distance import cosine
# import pickle
# import faiss
# import numpy as np


# lst = [
#     'Tom stated that he has been feeling tired',
#     'He “is not sleeping well, around 3-4 hours a night”',
#     'Tom is having difficulty completing everyday tasks and is not socializing frequently',
#     'Tom says he feels like running inside the house through the day',
#     'He states he feels “isolated and alone”',
#     'Tom presented with a slow speech and flat affect during the session',
#     'Relationships with family and friends are reduced',
#     'Toms sleeping patterns are irregular',
#     'Normal food intake',
#     'Tom reported seeing Jesus Christ in his sleep',
#     'Weight remains unchanged',
#     'Tom presented with mild depressive symptomatology',
#     'Tom was calm and adequately responsive',
#     'There are signs of mild anxiety',
#     'Tom has another session next week at 1pm on 12/05/2022',
#     'He has a goal to reach out to a close friend and open up about how he has been feeling'
# ]

# df = pd.DataFrame({'text': lst})


# class vectors:
#     def __init__(self,args,model_name = "tekraj/avodamed-synonym-generator1"):
#         self.datafile = args.datafile
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model_name = model_name
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModel.from_pretrained(model_name).to(self.device)
#         self.embedd_col = args.embedd_col

#     def run_pre_embedd(self):
#             """
#             Generate Emneddings and FAISS indices for given nodes.
#             NOTE: Assuming column with node names is called "node_name"

#             Parameters:
#             -----------
#             self.datafile : data csv file
#                 a csv file of the nodes. Column "node_name" must be present

#             Returns:
#             --------
#             NO EXPLICT RETURN VALUE
#             pickle file
#                 Pickle file of Huggingface dataset insatance is saved
#             """
#             print("starting pre-embedding")
#             hugging_dataset = Dataset.from_pandas(df)    

#             column_to_embedd = self.embedd_col

#             embeddings_dataset = hugging_dataset.map(
#                 lambda x: {"embeddings": self.get_embeddings(x[column_to_embedd]).detach().cpu().numpy()[0]}
#             )
#             print("embeddings done")

#             embeddings_dataset.add_faiss_index(column='embeddings')
#             print("faiss added")
            
#             output_file_name = f"{self.datafile[:-4]}_embeddings_{self.model_name.split('/')[-1] if '/' in self.model_name else self.model_name}.pickle"

#             with open(output_file_name, "wb") as f:
#                 pickle.dump(embeddings_dataset,f)

#             print("embeddings saved")
        
#     def get_embeddings(self,text_list):
#         """
#         Sub function for self.run_pre_embedd().
#         Applies the actual embedding to dataset.

#         Parameters:
#         -----------
#         text_list : HuggingFace dataset column
#             The column from the huggingface dataset instance containing the node name
            
#         Returns:
#         --------
#         Embeddings
#             Final embeddings.
#         """
#         encoded_input = self.tokenizer(
#             text_list, padding=True, truncation=True, return_tensors="pt"
#         )
#         encoded_input = {k: v for k, v in encoded_input.items()}
#         #move the encoded input to the device
#         encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
#         model_output = self.model(**encoded_input)
#         return self.cls_pooling(model_output)

#     def cls_pooling(self,model_output):
#         """
#         Sub function of self.get_embeddings.
#         This function takes in model_output as input, which is the output obtained after 
#         passing the input text to the pre-trained BERT model, and returns the first (CLS) 
#         token from the last hidden state of the model output.

#         Parameters:
#         -----------
#         model_output : pytorch tensor
#             A tensor representing the output of the pre-trained BERT model after passing the input text through it.

#         Returns:
#         --------
#         pytorch tensor
#             A tensor representing the first (CLS) token from the last hidden state of the pre-trained BERT model output.
#         """
#         return model_output.last_hidden_state[:, 0]

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--datafile", type=str, default="deleteme.csv")
#     parser.add_argument("--embedd_col", type=str, default="text")

#     args = parser.parse_args()

#     final = vectors(args)
#     final.run_pre_embedd()

##################################################################################################
##################################################################################################
##################################################################################################

import pandas as pd
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer
import time
import torch
from scipy.spatial.distance import cosine
import pickle
import faiss
import numpy as np

class CosineSim():
    model_name = "tekraj/avodamed-synonym-generator1"
    datafile = "agent-prompts.json"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embeddings_data="deleteme_embeddings_avodamed-synonym-generator1.pickle"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    def __init__(self):
        pass
        

    #@staticmethod
    def get_embeddings(text_list):
        encoded_input = CosineSim.tokenizer(
            text_list, padding=True, truncation=True, return_tensors="pt"
        )
        encoded_input = {k: v for k, v in encoded_input.items()}
        encoded_input = {k: v.to(CosineSim.device) for k, v in encoded_input.items()}
        model_output = CosineSim.model(**encoded_input)
        return model_output.last_hidden_state[:, 0]


    @staticmethod
    def run_query_faiss(query, top_k):
        with open(CosineSim.embeddings_data, "rb") as f:
            hugging_dataset = pickle.load(f)

        if (not isinstance(query, str)) or (not isinstance(query, list)):
            query = str(query)

        query_embedding = CosineSim.get_embeddings([query]).cpu().detach().numpy()

        scores, samples = hugging_dataset.get_nearest_examples(
            "embeddings", query_embedding, k=top_k
        )

        samples_df = pd.DataFrame.from_dict(samples)
        samples_df["scores"] = scores
        samples_df.sort_values("scores", ascending=True, inplace=True)
        samples_df = samples_df[['text','scores']]
        samples_df = samples_df.to_dict('records')
        return samples_df

        

if __name__ == "__main__":
    print("hello")
    test = CosineSim().run_query_faiss(query="Patient reports seeing God in his sleep", top_k=3)
    #test.run_pre_embedd()
    #final = test.run_query_faiss(query="What should I eat?", top_k=1)
    print(test)
