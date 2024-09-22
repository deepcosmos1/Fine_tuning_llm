import gradio as gr
from transformers import  AutoTokenizer
import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset, Dataset
import torch
from huggingface_hub import notebook_login
import pandas as pd
import tqdm.asyncio
import faiss
import os
import asyncio

class Preprocess:
    def __init__(self) -> None:
        self.num_proc=torch.cuda.device_count()
        print(f"Number of GPUS {self.num_proc}")
        self.dataset=None
        self.tokenizer = None
        self.DATASET_TYPE = None
        self.DATASET_FORMAT = None
        self.FILTER = None
        self.FILTER_FIELD = None
        self.FILTER_NUM_TOKENS = None
        self.DEDUP = None
        self.DEDUP_FIELD = None
        self.EMB_MODEL =  None
        self.DEDUP_METRIC = None
        self.DEDUP_THRESHOLD = None
        self.NA_HANDLING = None
        self.preprocessed_Path=""
    def filter(self):
        print('Started Filtering')
        self.dataset = self.dataset.filter(lambda element:len(self.tokenizer.tokenize(element[self.FILTER_FIELD]))>self.FILTER_NUM_TOKENS)
        
    async def deduplicate(self):
        print('Started Deduplication')
        print(self.EMB_MODEL)
        emb_model = SentenceTransformer(self.EMB_MODEL)
 
        
        pool=emb_model.start_multi_process_pool(target_devices=None)            
        emb_arr = np.zeros((0,emb_model.get_sentence_embedding_dimension()))
        for i in tqdm.asyncio.tqdm(iterable=range(0,len(self.dataset['temporary_text']),1000),desc="Generating Embeddings"):
            emb =  emb_model.encode_multi_process(self.dataset[i:i+1000]['temporary_text'],pool=pool)
            emb_arr =  np.vstack((emb_arr,emb))
        emb_model.stop_multi_process_pool(pool=pool)
        emb_arr=emb_arr.astype(np.float32)
        ngpus = faiss.get_num_gpus()
        print("number of GPUs:", ngpus)
        dim = emb_arr.shape[1]
        if self.DEDUP_METRIC=="cosine" :
            cpu_index = faiss.IndexFlatIP(dim)
        else:
            cpu_index=faiss.IndexBinaryFlat(dim)
        gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
        gpu_index.add(emb_arr) 
        D, I = gpu_index.search(emb_arr, 2)          
        duplicates = np.where(D[:, 1] < self.DEDUP_THRESHOLD)[0] 
        unique_indices = set(range(len(emb_arr)))
        for dup in  tqdm.asyncio.tqdm(iterable=duplicates, desc="Removing duplicates"):
            unique_indices.remove(dup)
        self.dataset = self.dataset.select(sorted(unique_indices))
        
    async def embedding(self,input):
        if(self.EMB_MODEL is None):
            raise gr.Error("The embedding model is not defined in preprocessing Tab")
        emb_model = SentenceTransformer(self.EMB_MODEL)
        pool=emb_model.start_multi_process_pool(target_devices=None)            
        emb_arr = np.zeros((0,emb_model.get_sentence_embedding_dimension()))
        emb = emb_model.encode_multi_process([input],pool=pool)
        emb_arr =  np.vstack((emb_arr,emb))
        emb_model.stop_multi_process_pool(pool=pool)
        emb_arr=emb_arr.astype(np.float32)
        return emb_arr

    def checkNans(self):
        self.dataset=self.dataset.to_pandas()
        print('Checking NAN')
        for i in self.dataset.columns:
            if self.dataset[i].dtype == 'object':
                self.dataset[i]=self.dataset[i].str.lower().replace('nan', np.nan).replace('?', np.nan).replace('', np.nan)
            if self.dataset[i].isnull().all():
                self.dataset[i] = 0
        if(self.NA_HANDLING=="Replace NA"):
            print('Replacing NA')
            self.dataset = self.dataset.fillna(self.dataset.iloc[0])
        else:
            print('Dropping NA')
            self.dataset.dropna(inplace=True)
            
        self.dataset.reset_index(drop=True, inplace=True)
        self.dataset=Dataset.from_pandas(self.dataset)

            
    async def preprocess(self,dataset_type, dataset_path, split, uploaded_file_csv, uploaded_file_json, dataset_format, filtering, tokenizer_name, filter_field, filter_num_tokens, deduplication, model_name, dedup_threshold, dedup_metric, na_handling):
        if dataset_type == 'HuggingFace':
            try:
                self.dataset=load_dataset(dataset_path,split=split)
            except:
                raise gr.Error('Unable To Load Dataset')
        elif dataset_type == 'CSV':
            try:
                self.dataset = load_dataset('csv', data_files =  uploaded_file_csv,split='train')
            except:
                raise gr.Error('Unable To Load Dataset')
        elif dataset_type == 'JSON':
            try:
                self.dataset = load_dataset('json', data_files = uploaded_file_json,split='train')
            except:
                raise gr.Error('Unable To Load Dataset')
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
        except:
            raise gr.Error('Could not fetch tokenizer')

        self.DATASET_TYPE = dataset_type
        self.DATASET_FORMAT = dataset_format
        self.FILTER = filtering
        self.FILTER_FIELD = filter_field
        self.FILTER_NUM_TOKENS = filter_num_tokens
        self.DEDUP = deduplication
        self.DEDUP_FIELD = None
        self.EMB_MODEL =  model_name
        self.DEDUP_METRIC = dedup_metric
        self.DEDUP_THRESHOLD = dedup_threshold
        self.NA_HANDLING = na_handling
        self.preprocessed_Path=os.path.join("/src","Preprocess_Data.csv")
        if (os.path.exists(self.preprocessed_Path)):
            os.remove(self.preprocessed_Path)
        print("Started")
        self.checkNans()
        # print(self.FILTER_FIELD)
        # print(list(self.dataset.column_names))
        if(self.DATASET_FORMAT!="Table" and self.FILTER_FIELD not in list(self.dataset.column_names)):
            raise gr.Error("Filter Field in incorrect")
        if(self.DATASET_FORMAT=="Table"):
            self.FILTER_FIELD="temporary_text"
        def filter(element,columns_names):
            tem = []
            for i in range(len(element[columns_names[0]])):
                tem.append(",".join([j+" : "+str(element[j][i]) for j in columns_names]))
            element["temporary_text"]=tem
            return element
        self.dataset=self.dataset.map(lambda element:filter(element,self.dataset.column_names),batched=True,num_proc=self.num_proc)
        
        if self.FILTER:
            self.filter()
        if self.DEDUP:
            await self.deduplicate()
        self.dataset=self.dataset.remove_columns(["temporary_text"])
        self.dataset.to_csv(self.preprocessed_Path,index=False)
        return self.dataset
    def get_Preprocess_Dataset(self):
        return self.preprocessed_Path
        


# def preprocess_and_show(dataset, split, tokenizer, emb_model, filter, dedup, filter_field, filter_num_tokens, dedup_metric, dedup_threshold):
#     # Convert string inputs to their correct data types
#     filter = filter == "True"
#     dedup = dedup == "True"
#     filter_num_tokens = int(filter_num_tokens)
#     dedup_threshold = float(dedup_threshold)

#     obj = Preprocess(dataset=dataset,
#                      split=split,
#                      tokenizer=tokenizer,
#                      emb_model=emb_model,
#                      filter=filter,
#                      dedup=dedup,
#                      filter_field=filter_field,
#                      filter_num_tokens=filter_num_tokens,
#                      dedup_metric=dedup_metric,
#                      dedup_threshold=dedup_threshold)
#     obj.preprocess()
    
#     summary = f"Processed Dataset Summary: \n- Total Entries: {len(obj.dataset)}"
#     sample_entries = "\n".join([entry['text'] for entry in obj.dataset.select(range(2))])
    
#     return summary, sample_entries

# # Gradio interface
# iface = gr.Interface(fn=preprocess_and_show,
#                      inputs=[
#                          gr.Textbox(value="mhenrichsen/alpaca_2k_test", label="Dataset Name", placeholder="Enter dataset name here"),
#                          gr.Textbox(value="train", label="Split", placeholder="Enter split here"),
#                          gr.Textbox(value="NousResearch/Llama-2-7b-hf", label="Tokenizer", placeholder="Enter tokenizer here"),
#                          gr.Textbox(value="thenlper/gte-small", label="Embedding Model", placeholder="Enter embedding model here"),
#                          gr.Radio(choices=["True", "False"], value="True", label="Filter"),
#                          gr.Radio(choices=["True", "False"], value="True", label="Deduplicate"),
#                          gr.Textbox(value="output", label="Filter Field", placeholder="Enter filter field here"),
#                          gr.Textbox(value="100", label="Filter Num Tokens", placeholder="Enter number of tokens for filtering"),
#                          gr.Radio(choices=["cosine", "jaccard"], value="cosine", label="Dedup Metric"),
#                          gr.Slider(minimum=0, maximum=1, value=0.95, step=0.01, label="Dedup Threshold"),
#                      ],
#                      outputs=[
#                          gr.Textbox(label="Summary"),
#                          gr.Textbox(label="Sample Entries")
#                      ],
#                      title="Text Preprocessing with Gradio",
#                      description="Fill in the details to preprocess your dataset.")

# # Run the interface with sharing enabled
# if __name__ == "__main__":
#     iface.launch(share=True)
