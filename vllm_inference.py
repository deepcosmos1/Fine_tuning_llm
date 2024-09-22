import pandas as pd
import random
from tqdm import tqdm
import json
import time
from datasets import load_dataset,Dataset
import torch
import gradio as gr
import os
from sdmetrics.reports.single_table import QualityReport
import pickle
import gc
import asyncio
import fire

report = QualityReport()


num_gpu = torch.cuda.device_count()
class Prompt_test(object):
    def  __init__(self):
        self.llm=None
        self.synthetic_dataset=None
        self.dataset=None
        self.sampling_params=None
    async def load_model(self,model_name,temperature=0.7, top_p=0.9,max_tokens=2048):
            
        if(model_name=="./finetune-out/merged"):
            model_name=model_name.replace(".","").strip()
            model_name=os.path.join("/src",f"axolotl{model_name}")
        
        from vllm import LLM, SamplingParams
        from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
        try:
            self.llm = LLM(model=model_name,max_model_len=4096,tensor_parallel_size=num_gpu,enforce_eager=True)
            self.sampling_params = SamplingParams(temperature=temperature, top_p=top_p,max_tokens=max_tokens)  
        except:
            raise gr.Error("VLLM failed to initiate")
        
        
    async def single_prompt(self,model_name,temperature, top_p,max_tokens,input_prompt_path):
        f = open(input_prompt_path, "r")
        input_prompt_list=f.readlines()
        prompt="\n".join(input_prompt_list)
        f.close()
        await self.load_model(model_name,temperature,top_p,max_tokens)
        prompts = []
        prompts.append(prompt)
        outputs = self.llm.generate(prompts, self.sampling_params)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            out_text=f"{prompt}{generated_text}".split("<<ASSISTANT>>")[-1]
            
            if("}" not in out_text):
                percentage=100
                sign="-"
            else:
                shortened_text=out_text.split("}")[0]+'}'
                percentage=(len(shortened_text)/len(out_text))*100
                sign=""
 
        return f"[OUT_TEXT]{out_text}[/OUT_TEXT] [HALLUCINATION]{sign}{percentage}[/HALLUCINATION]"
        
    async def generate_dataset(self,model_name,temperature, top_p,max_tokens, dataset_type, dataset_path, split, uploaded_file_csv, uploaded_file_json, num_samples_input, batch_size_input, input_prompt_path, output_format,progress=gr.Progress(track_tqdm=True)):
        f = open(input_prompt_path, "r")
        input_prompt_list=f.readlines()
        input_prompt="\n".join(input_prompt_list)
        f.close()
        await self.load_model(model_name,temperature,top_p,max_tokens)
        self.synthetic_dataset=None
        self.dataset=None
        jsons_out = []
        NUM_SAMPLES = num_samples_input
        BATCH_SIZE = batch_size_input
        start=time.time()
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
               
        self.dataset=self.dataset.to_pandas()
        while len(jsons_out) < NUM_SAMPLES:
            prompts=[]
            columns=self.dataset.columns
            for _ in tqdm(range(BATCH_SIZE), desc="Generating Prompt for Synthetic Data"):
                random_slot=self.dataset.loc[random.randrange(len(self.dataset)),[columns[0],columns[1]]]
                
                input_text = f'''{input_prompt}'''
                if output_format=="JSON":
                    assistant_prompt=f'''<<ASSISTANT>>{{'''
                else:
                    assistant_prompt=f'''<<ASSISTANT>>{{'''
                for j in range(2):
                    random_choice=random_slot[columns[j]]
                    if(True if(type(random_choice)==int or (type (random_choice)==float)) else False):
                        if(output_format=="JSON"):
                            assistant_prompt+=f"\"{columns[j]}\" :  {random_choice}, "
                        elif(output_format=="RealTabFormer"):
                            assistant_prompt+=f"\"{columns[j]}\" is  {random_choice}, "
                        elif(output_format=="Tabula"):
                            assistant_prompt+=f"\"{columns[j]}\" {random_choice}, "
                    else:
                        if(output_format=="JSON"):
                            assistant_prompt+=f"\"{columns[j]}\" : \"{random_choice}\", "  
                        elif(output_format=="RealTabFormer"):
                            assistant_prompt+=f"\"{columns[j]}\" is \"{random_choice}\", "  
                        elif(output_format=="Tabula"):
                            assistant_prompt+=f"\"{columns[j]}\" \"{random_choice}\", "  
                input_text+=assistant_prompt
                prompts.append(input_text)
            print("Generating")
            outputs = self.llm.generate(prompts, self.sampling_params)
            print("Parsing")
            for output in tqdm(outputs,desc="Processing prompt for Synthetic Data"):
                if len(jsons_out)==NUM_SAMPLES:
                    break
                prompt = output.prompt
                # print(prompt)
                generated_text = output.outputs[0].text
                if(output_format=="JSON"):
                    new_row = f"{prompt}{generated_text}".split('<<ASSISTANT>>')[1].split('}')[0]+"}"
                    try:
                        # print(new_row+"\n\n")
                        var=json.loads(new_row)
                        # var = new_row
                        jsons_out.append(var)
                    except:
                        pass
                else:
                    new_row = f"{prompt}{generated_text}".split('<<ASSISTANT>>')[1]
                    new_row=new_row.replace("{","").split('}')[0].replace("\n","")
    
                    temp=[]
                    try:
                        if(output_format=="Tabula"):
                            for i in range(1,len(columns)):
                                to_split=f"\"{columns[i]}\""
                                prev_split=f"\"{columns[i-1]}\""
                                temp_row=new_row.split(to_split)[0].strip()[:-1].split(prev_split)[-1].strip()
                                temp.append(":".join([prev_split,temp_row]))
                                new_row=to_split + new_row.split(to_split)[1]
                            temp.append(":".join([f"\"{columns[-1]}\"",new_row.split(f"\"{columns[-1]}\"")[-1]]))
                        else:
                            for i in range(1,len(columns)):
                                to_split=f"\"{columns[i]}\""
                                prev_split=f"\"{columns[i-1]}\""
                                temp_row=new_row.split(to_split)[0].strip()[:-1].split(prev_split)[-1].strip()[3:]
                                temp.append(":".join([prev_split,temp_row]) )
                                new_row=to_split+new_row.split(to_split)[1]
                            temp.append(":".join([f"\"{columns[-1]}\"",new_row.split(f"\"{columns[-1]}\" is ")[-1]]))    
                        out=f"""{{ {",".join(temp)} }}"""  
                        # print(out)
                        jsons_out.append(json.loads(out))
                        del temp
                        gc.collect()
                    except:
                        pass
                    
            print("Time took to generate",time.time()-start)
        if output_format=="JSON":
            self.synthetic_dataset=pd.DataFrame(jsons_out)
        else:
            self.synthetic_dataset=pd.DataFrame(jsons_out)
        try:
            self.synthetic_dataset=self.synthetic_dataset[self.dataset.columns]
            numerical_columns = self.dataset.select_dtypes(include=['int64', 'float64']).columns
            for i in numerical_columns:
                self.synthetic_dataset[i]= pd.to_numeric(self.synthetic_dataset[i], errors="coerce")
            self.synthetic_dataset.to_csv(os.path.join("/src","SyntheticData.csv"),index=False)
        except:
            raise gr.Error("Generated dataset does not conform the structural properties")
        return os.path.join("/src","SyntheticData.csv")


    
    
        
if __name__ == '__main__':
  fire.Fire(Prompt_test)
        
