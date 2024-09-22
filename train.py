import gradio as gr
import os
import asyncio
import gc
import json
from datasets import load_dataset, Dataset
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import time
class Trainer:
    def __init__(self):
        self.config_path = os.path.expanduser('~/.cache/huggingface/accelerate/default_config.yaml')
        acc_path = os.path.expanduser('~/.cache/huggingface/accelerate')
        os.makedirs(acc_path,exist_ok=True)
        if(not os.path.exists):
            with open(self.config_path, 'w') as fp:
                pass
            fp.close()  
    if 'axolotl' not in os.getcwd() and os.path.exists(os.path.join(os.getcwd(),"axolotl")):
        os.chdir("axolotl")

    async def check_and_prepare_environment(self):
        progress=gr.Progress()
        progress((0,8),desc="Installing Dependencies",total=8)
        start=time.time()
        print("Installing Dependencies")
        
        if os.path.split(os.getcwd())[-1]!="axolotl":
            proc= await asyncio.create_subprocess_shell(f"git clone -b main --depth 1 https://github.com/OpenAccess-AI-Collective/axolotl",stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                error_message = stderr.decode() if stderr else 'Unknown error'
                raise gr.Error(f"Could not clone axolotl: {error_message}")
            else:
                os.chdir("axolotl")
        
        print(1)
        progress((1,8),desc="Installing Dependencies",total=8)
        print(3)
        progress((3,8),desc="Installing Dependencies",total=8)
        print(4)
        progress((4,8),desc="Installing Dependencies",total=8)
        print(5) 
        progress((5,8),desc="Installing Dependencies",total=8)
        print(6)
        progress((6,8),desc="Installing Dependencies",total=8)

        print(7)
        progress((7,8),desc="Installing Dependencies",total=8)
        if not os.path.exists(os.path.join(os.getcwd(),"llama.cpp")):
            proc= await asyncio.create_subprocess_shell(f"git clone https://github.com/ggerganov/llama.cpp.git" ,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await proc.communicate()
            # print(stdout)
            if proc.returncode != 0:
                error_message = stderr.decode() if stderr else 'Unknown error'
                raise gr.Error(f"Could not clone llama cpp\n {error_message}")
        print(8)
        progress((8,8),desc="Installing Dependencies",total=8)
        print("Environment Setup Complete")
        print(f"Time to install dependencies {(time.time()-start)/60} minutes")
        progress(None)
        self.dataset=None
        self.save_format=None
        self.tab_is_last=False

    
    async def applyAlgorithm(self,algorithm):
        if(self.dataset is None):
            raise gr.Error("Dataset Not Defined")
        instruction_template="You are a helpful assistant for creating synthetic data"
        if(len(list(self.dataset.column_names))<3):
            raise gr.Error("Dataset must have more than three columns")
        if(algorithm=="realTabFormer"):
            print("RealTabFormer")
            def mapping(element,columns):
                text=[]
                instruction=[]
                input=[]
                for i in element[columns[0]]:
                    row=[]
                    for j in columns:
                        is_numeric=True if(type(element[j][i])==int or (type (element[j][i])==float)) else False
                        if is_numeric:
                            row.append(f"\"{j}\" is {str(element[j][i])}")
                        else:
                             row.append(f"\"{j}\" is \"{str(element[j][i])}\"")
                    text.append(f'{{ {", ".join(row)} }}')
                    instruction.append(instruction_template)
                    current_input=""
                    for j in range(3):
                        if(True if(type(element[columns[j]][i])==int or (type (element[columns[j]][i])==float)) else False):
                            current_input+=f"\"{columns[j]}\" is {element[columns[j]][i]}, "
                        else:
                            current_input+=f"\"{columns[j]}\" is \"{element[columns[j]][i]}\", "                    
                    input.append(current_input)
                element["output"]=text
                element["input"]=input
                element["instruction"]=instruction
                return element
            #temp1 = mapping(element,self.dataset.column_names)
            #print(temp1)    
            self.dataset=self.dataset.map(lambda element:mapping(element,self.dataset.column_names),batched=True,num_proc=4)
            
        elif(algorithm=="Tabula"):
            print("Tabula")
            def mapping(element,columns):
                text=[]
                instruction=[]
                input=[]
                for i in element[columns[0]]:
                    row=[]
                    for j in columns:
                        is_numeric=True if(type(element[j][i])==int or (type (element[j][i])==float)) else False
                        if is_numeric:
                            row.append(f"\"{j}\" {str(element[j][i])}")
                        else:
                             row.append(f"\"{j}\" \"{str(element[j][i])}\"")
                    text.append(f'{{ {", ".join(row)} }}')
                    instruction.append(instruction_template)
                    current_input=""
                    for j in range(3):
                        if(True if(type(element[columns[j]][i])==int or (type (element[columns[j]][i])==float)) else False):
                            current_input+=f"\"{columns[j]}\" {element[columns[j]][i]}, "
                        else:
                            current_input+=f"\"{columns[j]}\" \"{element[columns[j]][i]}\", "                    
                    input.append(current_input)
                element["output"]=text
                element["input"]=input
                element["instruction"]=instruction
                return element
            self.dataset=self.dataset.map(lambda element:mapping(element,self.dataset.column_names),batched=True,num_proc=4)

        
        elif(algorithm=="JSON"):
            print("Applied JSON")
            def mapping(element,columns):
                text=[]
                instruction=[]
                input=[]
            
                for i in range(len(element[columns[0]])):
                    row={}
                    for j in columns:
                        row[j]=element[j][i]
                    text_t = json.dumps([row])
                    text_t = text_t.replace("[","")
                    text_t = text_t.replace("]","")
                    text.append(text_t)
                    instruction.append(instruction_template)
                    current_input="{ "
                    for j in range(3):
                        if(True if(type(element[columns[j]][i])==int or (type (element[columns[j]][i])==float)) else False):
                            current_input+=f"\"{columns[j]}\": {element[columns[j]][i]}, "
                        else:
                            current_input+=f"\"{columns[j]}\": \"{element[columns[j]][i]}\", "                    
                    input.append(current_input)
                element["output"]=text
                element["input"]=input
                element["instruction"]=instruction
                return element
            self.dataset=self.dataset.map(lambda element:mapping(element,self.dataset.column_names),batched=True,num_proc=4)
                
        else:
            raise gr.Error("Invalid algorithm for table dataset")
        self.dataset=self.dataset.select_columns(["instruction","input","output"])
        self.dataset.to_csv(f"Temporary_Algorithm.csv")
        
        
    async def generate_configs(self,algorithm, max_steps, base_model, model_type, tokenizer_type, is_llama_derived_model,
            strict, datasets_path, dataset_format, output_format, shards,
            val_set_size, output_dir, adapter, lora_model_dir, sequence_len, sample_packing,
            pad_to_sequence_len, lora_r, lora_alpha, lora_dropout,
            lora_target_modules, lora_target_linear, lora_fan_in_fan_out, gradient_accumulation_steps,
            micro_batch_size, num_epochs, optimizer, lr_scheduler, learning_rate, train_on_inputs,
            group_by_length, bf16, fp16, tf32, gradient_checkpointing,
            resume_from_checkpoint, local_rank, logging_steps, xformers_attention, flash_attention,
            load_best_model_at_end, warmup_steps, evals_per_epoch, eval_table_size, saves_per_epoch,
            debug, weight_decay, wandb_project, wandb_entity, wandb_watch,
            wandb_name, wandb_log_model,Use_deepspeed,deepspeed_config,tab_is_last,save_format=".safetensors",progress=gr.Progress(track_tqdm=True)):
        self.save_format=save_format
        self.tab_is_last=tab_is_last
        import torch
        if (torch.cuda.device_count()==0):
            raise gr.Error("NO GPU FOUND! gpu is required for training")
        # Generate qlora.yml configuration
        lora_modules_string=""
        for i in lora_target_modules.split(","):
            lora_modules_string+=(" - "+i.strip()+"\n")
    
        load_4_bit_bool="true" if adapter=="qlora" else "false"
        load_8_bit_bool="false" if adapter=="qlora" else "true"
        

        algorithm_applicable=dataset_format=="Table"

        if(algorithm_applicable):
            print("Table Selected")
            if (not os.path.exists(datasets_path)):
                try:
                    self.dataset=load_dataset(datasets_path,split='train')
                except:
                    raise gr.Error('Unable To Load Dataset')
            elif os.path.splitext(datasets_path)[-1]==".csv":
                try:
                    self.dataset = load_dataset('csv', data_files =  datasets_path,split='train')
                except:
                    raise gr.Error('Unable To Load Dataset')
            elif os.path.splitext(datasets_path)[-1]==".json":
                try:
                    self.dataset = load_dataset('json', data_files = datasets_path,split='train')
                except:
                    raise gr.Error('Unable To Load Dataset')
            if(output_format=="CSV"):
                await self.applyAlgorithm(algorithm)
            else:
                await self.applyAlgorithm("JSON")
            datasets_path=f"Temporary_Algorithm.csv"
            dataset_format=f"Alpaca"

        if os.path.splitext(datasets_path)[-1]==".json":
            gr.Error("Please use .jsonl instead of json")
         # {(max_steps)}
        qlora_config = f'''
        max_steps: 
        base_model: {base_model}  #model name
        model_type: {model_type}
        tokenizer_type: {tokenizer_type}
        is_llama_derived_model: {str(is_llama_derived_model).lower()}
        
        load_in_8bit: {load_8_bit_bool}
        load_in_4bit:  {load_4_bit_bool}
        strict: {str(strict).lower()}
        
        datasets:
          - path: {datasets_path} #dataset name
            type: {str(dataset_format).lower()}
            shards: {shards if shards=="" else int(shards)}
        dataset_prepared_path: 
        val_set_size: {val_set_size}
        output_dir: {output_dir}
        
        adapter: {adapter}
        lora_model_dir: {lora_model_dir}

        sequence_len: {int(sequence_len)}
        sample_packing: {str(sample_packing).lower()}
        pad_to_sequence_len: {str(pad_to_sequence_len).lower()}
        eval_sample_packing: false
        lora_r: {int(lora_r)}
        lora_alpha: {int(lora_alpha)}
        lora_dropout: {lora_dropout}
        lora_target_modules: 
        lora_target_linear: {str(lora_target_linear).lower()}
        lora_fan_in_fan_out: {str(lora_fan_in_fan_out).lower()}
        save_safetensors: {"true" if save_format==".safetensors" else ""}
        wandb_mode: offline
        wandb_project: 
        wandb_entity: 
        wandb_watch: 
        wandb_name: 
        wandb_log_model:
        
     
    
        gradient_accumulation_steps: {int(gradient_accumulation_steps)}
        micro_batch_size: {int(micro_batch_size)}
        num_epochs: {int(num_epochs)}
        optimizer: {optimizer}
        lr_scheduler: {lr_scheduler}
        learning_rate: {learning_rate}
        
        train_on_inputs: {str(train_on_inputs).lower()}
        group_by_length: {str(group_by_length).lower()}
        bf16: {str(bf16).lower()}
        fp16: {str(fp16).lower()}
        tf32: {str(tf32).lower()}
        
        gradient_checkpointing: {str(gradient_checkpointing).lower()}
        

        early_stopping_patience:
        resume_from_checkpoint: {resume_from_checkpoint}
        local_rank: {local_rank if local_rank=="" else int(local_rank)}
        logging_steps: {logging_steps if logging_steps=="" else int(logging_steps)}
        xformers_attention: {str(xformers_attention).lower()}
        flash_attention: {str(flash_attention).lower()}
        load_best_model_at_end: {str(load_best_model_at_end).lower()}
        
        # warmup_steps: {warmup_steps if warmup_steps=="" else int()}
        # evals_per_epoch: {evals_per_epoch if evals_per_epoch=="" else int(evals_per_epoch)}
        # eval_table_size: {eval_table_size if eval_table_size=="" else int(eval_table_size)}
        # saves_per_epoch: {saves_per_epoch if saves_per_epoch=="" else int(saves_per_epoch)}
        # debug: {str(debug).lower()}
        # deepspeed: 
        # weight_decay: {weight_decay}
        fsdp:
        fsdp_config:
        special_tokens:
          bos_token: "<s>"
          eos_token: "</s>"
          unk_token: "<unk>"

        do_causal_lm_eval: true
        '''
        
        with open("qlora.yml", "w") as f:
            f.write(qlora_config)


      
        if(deepspeed_config=="Zero 1"):
            zero3_init_flag="false"
            deepspeed_config_file="deepspeed_configs/zero1.json"
        elif(deepspeed_config=="Zero 1"):
            zero3_init_flag="false"
            deepspeed_config_file="deepspeed_configs/zero2.json"
        else:
            zero3_init_flag="true"
            deepspeed_config_file="deepspeed_configs/zero3.json"
        if(Use_deepspeed):
            # Generate Accelerate default_config.yaml
            accelerate_config = f"""
            compute_environment: LOCAL_MACHINE
            debug: true
            deepspeed_config:
              deepspeed_config_file: {deepspeed_config_file}
              zero3_init_flag: {zero3_init_flag}
            distributed_type: DEEPSPEED
            downcast_bf16: 'no'
            machine_rank: 0
            main_training_function: main
            num_machines: 1
            num_processes: {torch.cuda.device_count()}
            rdzv_backend: static
            same_network: true
            tpu_env: []
            tpu_use_cluster: false
            tpu_use_sudo: false
            use_cpu: false
            """
        else:
            accelerate_config =f"""
            compute_environment: LOCAL_MACHINE
            debug: false
            distributed_type: MULTI_GPU
            downcast_bf16: 'no'
            gpu_ids: all
            machine_rank: 0
            main_training_function: main
            mixed_precision: fp16
            num_machines: 1
            num_processes: {torch.cuda.device_count()}
            rdzv_backend: static
            same_network: true
            tpu_env: []
            tpu_use_cluster: false
            tpu_use_sudo: false
            use_cpu: false
            """
        with open(self.config_path, "w") as f:
            f.write(accelerate_config)
        print("Configuration Complete")
        return await self.start_train(output_dir = output_dir)

    async def start_train(self, output_dir, merge=True,progress=gr.Progress(track_tqdm=True)):
        print("Training Started !!!\n\n")
        f = open(os.path.join(os.getcwd(),"train_logs.json"), mode='w', encoding='utf-8')
        f.close()
        f = open(os.path.join(os.getcwd(),"eval_logs.json"), mode='w', encoding='utf-8')
        f.close()
        # os.system(f"accelerate launch -m axolotl.cli.train qlora.yml")
        
        proc= await asyncio.create_subprocess_shell(f"accelerate launch -m axolotl.cli.train qlora.yml",stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
    
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error_message = stderr.decode() if stderr else 'Unknown error'
            raise gr.Error(f"Training error: {error_message}")
        print("Training Completed !!!\n\n")
        self.TrainLogsDataFrame,self.TrainLogsFigure,self.EvalLogsFigure=await self.checkStats()
        if merge:
            return await self.start_merge(output_dir)
        else:
            response_dict={"code":200,"message":f"Training Completed model saved to {output_dir}","logs":dataframe,"fig":self.TrainLogsFigure,"eval_fig":self.EvalLogsFigure}
            return response_dict

    
    async def checkStats(self):
        # logsDataFrame.to_csv(os.path.expanduser(os.path.join('~',"src","TrainingLogs.csv")),index=False)
        train_rows=[]
        with open(os.path.join(os.getcwd(),"train_logs.json")) as user_file:
              parsed_json = json.load(user_file)
        for i in range(len(parsed_json)):
            dictionary_epoch={"epoch":parsed_json[i]["epoch"]} | parsed_json[i]["train_logs"]
            train_rows.append(dictionary_epoch)
        logsDataFrame=pd.DataFrame.from_dict(train_rows)
        os.remove(os.path.join(os.getcwd(),"train_logs.json"))
        pd.options.plotting.backend = "plotly"
        train_fig=logsDataFrame.plot(x="epoch", y=logsDataFrame.columns[1:],kind="line",labels={
                             "epoch": "Epoch",
                             "variable": "Metric",
                             "value": "Value"
                         },title="Training Logs",markers=True)
        train_fig.update_layout(xaxis = dict(tickmode = 'array',tickvals = logsDataFrame['epoch']))
        del logsDataFrame,train_rows

        
        test_rows=[]
        with open(os.path.join(os.getcwd(),"eval_logs.json")) as user_file:
              parsed_json = json.load(user_file)
        for i in range(len(parsed_json)-1):
            dictionary_epoch={"epoch":parsed_json[i]["epoch"]} | parsed_json[i]["eval_logs"] 
            test_rows.append(dictionary_epoch)
        logsDataFrame=pd.DataFrame.from_dict(test_rows)
        pd.options.plotting.backend = "plotly"
        eval_fig=logsDataFrame.plot(x="epoch", y=logsDataFrame.columns[1:],kind="line",labels={
                             "epoch": "Epoch",
                             "variable": "Metric",
                             "value": "Value"
                         },title="Training Logs",markers=True)
        eval_fig.update_layout(xaxis = dict(tickmode = 'array',tickvals = logsDataFrame['epoch']))
        del logsDataFrame,test_rows

        
        with open(os.path.join(os.getcwd(),"eval_logs.json")) as user_file:
              parsed_json = json.load(user_file)[-1]["Final Summary"]
        os.remove(os.path.join(os.getcwd(),"eval_logs.json"))
        logsDataFrame=pd.DataFrame.from_dict([parsed_json])   
        logsDataFrame.drop(["epoch"],axis=1)    
        gc.collect()
        return logsDataFrame,train_fig,eval_fig
                        
    async def start_merge(self, output_dir):
        print("Merging Started !!!\n\n")
        proc= await asyncio.create_subprocess_shell(f"python3 -m axolotl.cli.merge_lora qlora.yml --lora_model_dir={output_dir}",stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
        
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            error_message = stderr.decode() if stderr else 'Unknown error'
            raise gr.Error(f"Merging error: {error_message}")
        else:
            print("Merging Completed !!!\n\n")
            response_dict={"code":200,"message":f"Merging Completed model saved to {os.path.join(output_dir,'merged')}","logs" : self.TrainLogsDataFrame, "fig":self.TrainLogsFigure,"eval_fig":self.EvalLogsFigure}
            if(os.path.exists("Temporary_Algorithm.csv")):
                os.remove("Temporary_Algorithm.csv")
            
            return response_dict
            # if(self.tab_is_last):
            #     return await self.model_to_disk()
            # else:
            #     return response_dict

    async def model_to_disk(self):
       
     
        print("Saving")
        if self.save_format==".GGUF":
        
            gguf_path=os.path.join("/src","finetune_merged.gguf")
            response_dict={"code":200,"message":f"Merging Completed model saved to {gguf_path}" ,"logs":self.TrainLogsDataFrame, "fig":self.TrainLogsFigure,"eval_fig":self.EvalLogsFigure}
            return response_dict
        else:        
            savetensors_path=os.path.join("/src","finetune_merged.zip")
            parent_path=os.path.join("/src","axolotl","finetune-out","merged")
                      
           
            await self.save_file(savetensors_path,parent_path)
            
            if os.path.exists(savetensors_path):
                response_dict={"code":200,"message":f"Merging Completed model saved to {savetensors_path}","logs":self.TrainLogsDataFrame, "fig":self.TrainLogsFigure,"eval_fig":self.EvalLogsFigure}
                return response_dict
            else:
               raise gr.Error("Could not convert to safetensors format")


                

    async def save_file(self,savetensors_path,parent_path):
        import zipfile
        files=[]
        if(os.path.exists(savetensors_path)):
            os.remove(savetensors_path)
        for file in tqdm.tqdm(os.listdir(parent_path),desc="Saving Files"):
                    files.append(os.path.join(parent_path,file))
        with zipfile.ZipFile(savetensors_path, 'w') as zip_file:
            for file in tqdm.tqdm(files):
                zip_file.write(file)

        

# # Define the Gradio function outside the class to avoid issues with class methods
# def generate_configs_interface(model, dataset, dataset_type, output_dir, epochs, max_steps, train, merge):
#     configurator = Trainer()
#     return configurator.generate_configs(model, dataset, dataset_type, output_dir, epochs, max_steps, train, merge)

# # Setup Gradio Interface
# iface = gr.Interface(
#     generate_configs_interface,
#     [
#         gr.Textbox(label="Model", value='NousResearch/Llama-2-7b-hf'),
#         gr.Textbox(label="Dataset", value='mhenrichsen/alpaca_2k_test'),
#         gr.Textbox(label="Dataset Type", value='alpaca'),
#         gr.Textbox(label="Output Directory", value='./qlora-out'),
#         gr.Slider(minimum=1, maximum=50, value=4, step=1, label="Epochs"),
#         gr.Textbox(label="Max Steps (Keep it empty to run eposhs)", value='10'),
#         gr.Radio(choices=["True", "False"], value="False", label="Train"),
#         gr.Radio(choices=["True", "False"], value="False", label="Merge With Base"),
#     ],
#     outputs=[gr.Textbox(label="Output")],
#     title="Trainer",
#     description="Fill in the details to Start Training")

# if __name__ == "__main__":
#     iface.launch(share=True)
