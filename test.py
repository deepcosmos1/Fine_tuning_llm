import gradio as gr
from preprocess import Preprocess
from datasets import Dataset
import asyncio
from train import Trainer
from prompt_test import Prompt_test
import os
from huggingface_hub import HfApi


class Main:
    
    def __init__(self):
        self.utility=Prompt_test()
        self.trainer=Trainer()
        self.algorithm=""
        self.outputFormat=""
        self.model_output_format=""
        self.PreprocessObject=Preprocess()

    def set_Algorithm(self,output_format,algorithm,model_output_format):
        self.outputFormat=output_format
        self.algorithm=algorithm
        self.model_output_format=model_output_format
        if self.outputFormat == "JSON":
            return f"Output Format {self.outputFormat}" 
        else:
            return f"Output Format {self.outputFormat} and Algorithm {self.algorithm}"

    async def preprocess_data(self,dataset_type, dataset_path, split, uploaded_file_csv, uploaded_file_json, dataset_format, filtering, tokenizer_name, filter_field, filter_num_tokens, deduplication, model_name, dedup_threshold, dedup_metric, na_handling,progress=gr.Progress(track_tqdm=True)):      
        if dataset_type != 'HuggingFace':
            if uploaded_file_csv is None and uploaded_file_json is None:
                raise gr.Error('Please Upload A file')
     
        result =await self.PreprocessObject.preprocess(dataset_type, dataset_path, split, uploaded_file_csv, uploaded_file_json, dataset_format, filtering, tokenizer_name, filter_field, filter_num_tokens, deduplication, model_name, dedup_threshold, dedup_metric, na_handling)
        
        result = result.to_pandas()
        datasave_path = self.PreprocessObject.get_Preprocess_Dataset()
        return datasave_path, result[:5]
            
    
    async def train_model(self,max_steps, base_model, model_type, tokenizer_type, is_llama_derived_model,
                    strict, datasets_path, dataset_format, shards,
                    val_set_size, output_dir, adapter, lora_model_dir, sequence_len, sample_packing,
                    pad_to_sequence_len, lora_r, lora_alpha, lora_dropout,
                    lora_target_modules, lora_target_linear, lora_fan_in_fan_out, gradient_accumulation_steps,
                    micro_batch_size, num_epochs, optimizer, lr_scheduler, learning_rate, train_on_inputs,
                    group_by_length, bf16, fp16, tf32, gradient_checkpointing,
                    resume_from_checkpoint, local_rank, logging_steps, xformers_attention, flash_attention,
                    load_best_model_at_end, warmup_steps, evals_per_epoch, eval_table_size, saves_per_epoch,
                    debug, weight_decay, wandb_project, wandb_entity, wandb_watch,
                    wandb_name, wandb_log_model,Use_debug,deepspeed_config,last_tab,progress=gr.Progress(track_tqdm=True)):
        
        
        await self.trainer.check_and_prepare_environment()
        response=await self.trainer.generate_configs(self.algorithm, max_steps, base_model, model_type, tokenizer_type, is_llama_derived_model,
                strict, datasets_path, dataset_format, self.outputFormat, shards, 
                val_set_size, output_dir, adapter, lora_model_dir, sequence_len, sample_packing,
                pad_to_sequence_len, lora_r, lora_alpha, lora_dropout,
                lora_target_modules, lora_target_linear, lora_fan_in_fan_out, gradient_accumulation_steps,
                micro_batch_size, num_epochs, optimizer, lr_scheduler, learning_rate, train_on_inputs,
                group_by_length, bf16, fp16, tf32, gradient_checkpointing,
                resume_from_checkpoint, local_rank, logging_steps, xformers_attention, flash_attention,
                load_best_model_at_end, warmup_steps, evals_per_epoch, eval_table_size, saves_per_epoch,
                debug, weight_decay, wandb_project, wandb_entity, wandb_watch,
                wandb_name, wandb_log_model,Use_debug,deepspeed_config,last_tab,self.model_output_format)
        # return str(response)
        if(response["code"]==200):
            return [response["message"],response["logs"],response["fig"],response["eval_fig"]]
    async def get_embedding(self,input,progress=gr.Progress(track_tqdm=True)):
        return await self.PreprocessObject.embedding(input)
    def upload_to_hf(self,path,Repo_Name,HF_KEY,Type,progress=gr.Progress(track_tqdm=True)):
        self.login_to_hf(HF_KEY)
        if(type=="Dataset"):
            try:
                temp_data = load_dataset('csv', data_files = path,split='train')
            except:
                raise gr.Error('Unable To Load Dataset')
            try:
                temp.data.push_to_hub(Repo_Name)
            except:
                raise gr.Error("Could not push to hugging face")
            return (f"Dataset uploaded to {Repo_Name}")
        else:
            hf_api = HfApi()
            try:
                hf_api.create_repo(repo_id=Repo_Name,token=HF_KEY,repo_type=Type)
            except:
                raise gr.Error(f"Repo {Repo_Name} already exists")
            api.upload_folder(
                folder_path=path,
                repo_id=Repo_Name,
                repo_type=Type,
            )
            return (f"Model uploaded to {Repo_Name}")
    
    def login_to_hf(self,HF_KEY):
        from huggingface_hub import login
        try:
            os.environ["HF_KEY"] = HF_KEY
            login(os.environ["HF_KEY"])
            return "Login successful!"
        except Exception as e:
            raise gr.Error(f"Login failed: {e}")

       
    def initiate_userInterface(self):
        with gr.Blocks() as self.app:
            gr.Markdown("### Using Generative AI LLM To Generate Synthetic Structured Data")
    
            # Config Tab
            with gr.Tab("Configuration"):
                with gr.Column():
                    with gr.Group():
                        output_format = gr.Radio(choices=['CSV'], label = 'Output format', value='CSV',interactive=True)
                        algorithm = gr.Radio(choices=["realTabFormer", "Tabula"], label="Algorithm", value='Tabula')
                        model_output_format = gr.Radio(choices=['.safetensors','.GGUF'], label = 'Model Output format', value='.safetensors',interactive=False)
                        output = gr.Textbox(label='Status',visible=False)
                        def check_output(value):
                            return gr.Radio(interactive = value == 'CSV')
                        output_format.change(check_output,inputs=output_format,outputs=algorithm)
                        Upload = gr.Button('Set Algorithm')
                        Upload.click(self.set_Algorithm,inputs=[output_format,algorithm,model_output_format],outputs=[output])
                    with gr.Group():
                        HF_KEY = gr.Textbox(label="HuggingFace API Key",type='password')
                        output = gr.Textbox(label="Status")
                        Upload = gr.Button('Login To HuggingFace')
                        Upload.click(self.login_to_hf,inputs=[HF_KEY],outputs=[output])
                    # with gr.Group():
                    #     WANDB_KEY = gr.Textbox(label="Wandb API Key",type='password')
                    #     output = gr.Textbox(label="Result")
                    #     Upload = gr.Button('Login To Wandb')
                    #     Upload.click(self.login_to_hf,inputs=[WANDB_KEY],outputs=[output])
            
            # Preprocessing Tab
            with gr.Tab("Preprocessing"):
                with gr.Accordion("Input & Models"):
                    dataset_type = gr.Radio(choices=['CSV', 'JSON'], label="File Type", value='CSV')
                    with gr.Tab("Input From huggingface",visible=False) as Tab_HuggingFace:
                        dataset_path = gr.Textbox(label="Dataset Path", value="mhenrichsen/alpaca_2k_test")
                        split = gr.Textbox(label="Split", value="train")
                    with gr.Tab("Upload CSV",visible = True) as Tab_CSV:
                        uploaded_file_csv = gr.File(label="Upload File", file_types=['.csv','.csv.gz'])
                    with gr.Tab("Upload JSON",visible = False) as Tab_JSON:
                        uploaded_file_json = gr.File(label="Upload File", file_types=['.json'])
    
                    def check(value):
                        dict_v = {Tab_CSV: gr.Tab(visible = value == 'CSV'),
                                Tab_JSON: gr.Tab(visible= value == 'JSON'),}
                        return dict_v
                    dataset_type.select(check,inputs=dataset_type,outputs=[Tab_CSV,Tab_JSON])
                    
                    dataset_format = gr.Radio(choices=['Alpaca', 'Table'], label="Dataset Format", value='Alpaca')
                    
                with gr.Tab("Filtering"):
                    filtering = gr.Checkbox(label="Enable Filtering", value=True)
                    tokenizer_name = gr.Textbox(label="Tokenizer Model Name", value="/src/LLama-2-7b-hf")
                    filter_field = gr.Textbox(label="Name of Filter Field", value="output")
                    filter_num_tokens = gr.Slider(minimum=0, maximum=4098, label="Exclude Data Below Token Count", value=100)
                
                with gr.Tab("Deduplication"):
                    deduplication = gr.Checkbox(label="Enable Deduplication", value=True)
                    model_name = gr.Textbox(label="Model Name for Embedding", value="/src/gte_embedding")
                    dedup_threshold = gr.Slider(minimum=0, maximum=1, step=0.1, label="Deduplication Threshold", value=0.9)
                    dedup_metric = gr.Radio(choices=["cosine", "jaccard"], value="cosine", label="Dedup Metric")
                
                with gr.Tab("NA Handling"):
                    na_handling = gr.Radio(choices=['Drop NA', 'Replace NA'], label="NA Handling", value='Drop NA')
                
                process_btn = gr.Button("Process Data")
                process_btn.click(
                    self.preprocess_data,
                    inputs=[dataset_type, dataset_path, split, uploaded_file_csv, uploaded_file_json, dataset_format, filtering, tokenizer_name, filter_field, filter_num_tokens, deduplication, model_name, dedup_threshold, dedup_metric, na_handling],
                    outputs=[gr.File(label="Preprocessing Result",interactive=False),gr.Dataframe()]
                )
    
            with gr.Tab("Emedding" ,visible=False):
                input = gr.Textbox()
                output = gr.Textbox()
                submit = gr.Button()
                submit.click(
                    self.get_embedding,
                    inputs=[input],
                    outputs=[output]
                )
    
            # Finetuning Tab
            with gr.Tab("FineTuning For Chain of Thoughts"):
                base_model = gr.Textbox(label="base_model", value="/src/LLama-2-7b-hf")
                datasets_path = gr.Textbox(label="datasets_path", value="/src/ChainOfThoughts.csv")
                dataset_format = gr.Radio(choices=['Alpaca'], label="Dataset Format", value='Alpaca')
                shards = gr.Slider(minimum=0, maximum=20, step=1, label="shards", value=10)
                last_tab = gr.Checkbox(label='last_tab',value=False,visible=False)
               
                with gr.Accordion("Advanced Settings",open=False):
                    with gr.Tab("YAML Configuration"):
                        model_type = gr.Textbox(label="model_type", value="LlamaForCausalLM",info="",visible=False)
                        tokenizer_type = gr.Textbox(label="tokenizer_type", value="LlamaTokenizer",visible=False)
                        is_llama_derived_model = gr.Checkbox(label="is_llama_derived_model", value=True,info="Determines the padding strategy based on the parent type of the model")
                        strict = gr.Checkbox(label="strict", value=False,visible=False)
                        val_set_size = gr.Slider(minimum=0, maximum=1, step=0.1, label="val_set_size", value=0.05,info="Percentage of training data to be used for validation")
                        output_dir = gr.Textbox(label="output_dir", value="./finetune-out",info="Output directory of the finetuned model")
                        adapter = gr.Radio(choices=["qlora", "lora"], label="adapter",value='qlora',info="Parameter efficient training strategy")
                        lora_model_dir = gr.Textbox(label="lora_model_dir",info="Directory of a custom adapter can be provided",visible=False)
                        sequence_len = gr.Slider(minimum=512, maximum=4096, step=10,label="sequence_len", value=1024,info="The maximum length input allowed to train")
                        sample_packing = gr.Checkbox(label="sample_packing", value=True,info="Speeds up data preparation but recommended false for small datasets")
                        pad_to_sequence_len = gr.Checkbox(label="pad_to_sequence_len", value=True, info="Pads the input to match sequence length to avoid memory fragmentation and out of memory issues. Recommended true")
                        # eval_sample_packing = gr.Checkbox(label="eval_sample_packing", value=False)
                        lora_r = gr.Slider(minimum=8, maximum=64, step=2,label="lora_r", value=32,info="The number of parameters in adaptation layers.")
                        lora_alpha = gr.Slider(minimum=8, maximum=64, step=0.1,label="lora_alpha", value=16,info="How much adapted weights affect base model's")
                        lora_dropout = gr.Slider(minimum=0, maximum=1, label="lora_dropout", value=0.05, step=0.01,info="The ratio of weights ignored randomly within adapted weights")
                        lora_target_modules = gr.Textbox(label="lora_target_modules", value="q_proj, v_proj, k_proj",info="All dense layers can be targeted using parameter efficient tuning")
                        lora_target_linear = gr.Checkbox(label="lora_target_linear", value=True,info="Lora Target Modules will be ignored and all linear layers will be used")
                        lora_fan_in_fan_out = gr.Textbox(label="lora_fan_in_fan_out",visible=False)
                        
                        gradient_accumulation_steps = gr.Slider(minimum=4, maximum=64, step=1,label="gradient_accumulation_steps", value=4,info="Number of steps required to update the weights with cumulative gradients")
                        micro_batch_size = gr.Slider(minimum=1, maximum=64, step=2,label="micro_batch_size", value=2,info="Number of samples sent to each gpu")
                        num_epochs = gr.Slider(minimum=1, maximum=4, step=1,label="num_epochs", value=1)
                        max_steps = gr.Textbox(label="max_steps",value='1',info="Maximum number of steps to be trained. Overwrites the number of epochs",visible=False)
                        optimizer = gr.Radio(choices=["adamw_hf",'adamw_torch','adamw_torch_fused','adamw_torch_xla','adamw_apex_fused','adafactor','adamw_anyprecision','sgd','adagrad','adamw_bnb_8bit','lion_8bit','lion_32bit','paged_adamw_32bit','paged_adamw_8bit','paged_lion_32bit','paged_lion_8bit'], value="paged_adamw_32bit",label='optimizer',info="Use an optimizer which aligns with the quantization of model")
                        lr_scheduler = gr.Radio(label="lr_scheduler", choices=['one_cycle', 'log_sweep', 'cosine'],value="cosine",info="Determines dynamic learning rate based on current step")
                        learning_rate = gr.Textbox(label="max_learning_rate", value="2e-5",info="")
                        train_on_inputs = gr.Checkbox(label="train_on_inputs", value=False,visible=False)
                        group_by_length = gr.Checkbox(label="group_by_length", value=False,visible=False)
                        bf16 = gr.Checkbox(label="bfloat16", value=False,info="Enable bfloat16 precision for tensors; supported only on Ampere or newer GPUs.")
                        fp16 = gr.Checkbox(label="Half Precision", value=True,info="Enable half precision (FP16) for tensor processing.")
                        tf32 = gr.Checkbox(label="TensorFloat32", value=False,info="Enable TensorFloat32 precision for tensors; supported only on Ampere or newer GPUs.")
                        gradient_checkpointing = gr.Checkbox(label="gradient_checkpointing", value=True,info='',visible=False)
                        resume_from_checkpoint = gr.Textbox(label="resume_from_checkpoint",visible=False)
                        local_rank = gr.Textbox(label="local_rank",visible=False)
                        logging_steps = gr.Slider(minimum=1, maximum=100, step=1,label="logging_steps", value=1,info='',visible=False)
                        xformers_attention = gr.Checkbox(label="xformers_attention", value=False,visible=False)
                        flash_attention = gr.Checkbox(label="flash_attention", value=False,info='',visible=False)
                        load_best_model_at_end = gr.Checkbox(label="load_best_model_at_end", value=False,visible=False)
                        warmup_steps = gr.Slider(minimum=1, maximum=100, step=1,label="warmup_steps", value=10,visible=False)
                        evals_per_epoch = gr.Slider(minimum=1, maximum=100, step=1,label="evals_per_epoch", value=4,info='No. of Evaluation Per Epoch',visible=False)
                        eval_table_size = gr.Textbox(label="eval_table_size",visible=False)
                        saves_per_epoch = gr.Slider(minimum=1, maximum=100, step=1,label="saves_per_epoch", value=1,info='No. of checkpoints to be saved')
                        
                        debug = gr.Checkbox(label="debug", value=False,visible=False)
                        
                        weight_decay = gr.Number(label="weight_decay", value=0.0,visible=False)
                        wandb_watch = gr.Checkbox(label="wandb_watch", value=False,visible=False)
                        wandb_log_model = gr.Checkbox(label="wandb_log_model", value=False,visible=False)
                        wandb_project = gr.Textbox(label="wandb_project",visible=False)
                        wandb_entity = gr.Textbox(label="wandb_entity",visible=False)
                        wandb_name = gr.Textbox(label="wandb_name",visible=False)
        
                        
                    with gr.Tab("Accelerate Configuration"):
                        Use_debug = gr.Checkbox(label="Use DeepSpeed", value=True)
                        deepspeed_config = gr.Radio(label="deepspeed_config", choices=['Zero 1', 'Zero 2',' Zero 3'],value="Zero 1")
    
                    # Continue adding the rest of the Accelerate config inputs similarly)
                    
                train_btn = gr.Button("Start Training")
                output_dataframe= gr.Dataframe(label="Training Results")
                output_graph = gr.Plot(label='Training Plot')
                output_graph2 = gr.Plot(label='Training Plot')
                train_btn.click(
                    self.train_model,
                    inputs=[max_steps, base_model, model_type, tokenizer_type, is_llama_derived_model,
                    strict, datasets_path, dataset_format, shards,
                    val_set_size, output_dir, adapter, lora_model_dir, sequence_len, sample_packing,
                    pad_to_sequence_len, lora_r, lora_alpha, lora_dropout,
                    lora_target_modules, lora_target_linear, lora_fan_in_fan_out, gradient_accumulation_steps,
                    micro_batch_size, num_epochs, optimizer, lr_scheduler, learning_rate, train_on_inputs,
                    group_by_length, bf16, fp16, tf32, gradient_checkpointing,
                    resume_from_checkpoint, local_rank, logging_steps, xformers_attention, flash_attention,
                    load_best_model_at_end, warmup_steps, evals_per_epoch, eval_table_size, saves_per_epoch,
                    debug, weight_decay, wandb_project, wandb_entity, wandb_watch,
                    wandb_name, wandb_log_model,Use_debug,deepspeed_config,last_tab],
                    outputs=[gr.Textbox(label="Training Output",interactive=False),output_dataframe,output_graph,output_graph2]
                )
            
            # Finetuning Tab
            with gr.Tab("FineTuning For Structured Data"):
                base_model = gr.Textbox(label="base_model", value="./finetune-out/merged",visible=False)
                datasets_path = gr.Textbox(label="datasets_path", value="/src/FunctionCalling.csv")
                dataset_format = gr.Radio(choices=['Alpaca'], label="Dataset Format", value='Alpaca')
                shards = gr.Slider(minimum=0, maximum=20, step=1, label="shards", value=10)
                last_tab = gr.Checkbox(label='last_tab',value=False,visible=False)
               
                with gr.Accordion("Advanced Settings",open=False):
                    with gr.Tab("YAML Configuration"):
                        model_type = gr.Textbox(label="model_type", value="LlamaForCausalLM",info="",visible=False)
                        tokenizer_type = gr.Textbox(label="tokenizer_type", value="LlamaTokenizer",visible=False)
                        is_llama_derived_model = gr.Checkbox(label="is_llama_derived_model", value=True,info="Determines the padding strategy based on the parent type of the model")
                        strict = gr.Checkbox(label="strict", value=False,visible=False)
                        val_set_size = gr.Slider(minimum=0, maximum=1, step=0.1, label="val_set_size", value=0.05,info="Percentage of training data to be used for validation")
                        output_dir = gr.Textbox(label="output_dir", value="./finetune-out",info="Output directory of the finetuned model")
                        adapter = gr.Radio(choices=["qlora", "lora"], label="adapter",value='qlora',info="Parameter efficient training strategy")
                        lora_model_dir = gr.Textbox(label="lora_model_dir",info="Directory of a custom adapter can be provided",visible=False)
                        sequence_len = gr.Slider(minimum=512, maximum=4096, step=10,label="sequence_len", value=1024,info="The maximum length input allowed to train")
                        sample_packing = gr.Checkbox(label="sample_packing", value=True,info="Speeds up data preparation but recommended false for small datasets")
                        pad_to_sequence_len = gr.Checkbox(label="pad_to_sequence_len", value=True, info="Pads the input to match sequence length to avoid memory fragmentation and out of memory issues. Recommended true")
                        # eval_sample_packing = gr.Checkbox(label="eval_sample_packing", value=False)
                        lora_r = gr.Slider(minimum=8, maximum=64, step=2,label="lora_r", value=32,info="The number of parameters in adaptation layers.")
                        lora_alpha = gr.Slider(minimum=8, maximum=64, step=0.1,label="lora_alpha", value=16,info="How much adapted weights affect base model's")
                        lora_dropout = gr.Slider(minimum=0, maximum=1, label="lora_dropout", value=0.05, step=0.01,info="The ratio of weights ignored randomly within adapted weights")
                        lora_target_modules = gr.Textbox(label="lora_target_modules", value="q_proj, v_proj, k_proj",info="All dense layers can be targeted using parameter efficient tuning")
                        lora_target_linear = gr.Checkbox(label="lora_target_linear", value=True,info="Lora Target Modules will be ignored and all linear layers will be used")
                        lora_fan_in_fan_out = gr.Textbox(label="lora_fan_in_fan_out",visible=False)
                        
                        gradient_accumulation_steps = gr.Slider(minimum=4, maximum=64, step=1,label="gradient_accumulation_steps", value=4,info="Number of steps required to update the weights with cumulative gradients")
                        micro_batch_size = gr.Slider(minimum=1, maximum=64, step=2,label="micro_batch_size", value=2,info="Number of samples sent to each gpu")
                        num_epochs = gr.Slider(minimum=1, maximum=4, step=1,label="num_epochs", value=4)
                        max_steps = gr.Textbox(label="max_steps",value='1',info="Maximum number of steps to be trained. Overwrites the number of epochs",visible=False)
                        optimizer = gr.Radio(choices=["adamw_hf",'adamw_torch','adamw_torch_fused','adamw_torch_xla','adamw_apex_fused','adafactor','adamw_anyprecision','sgd','adagrad','adamw_bnb_8bit','lion_8bit','lion_32bit','paged_adamw_32bit','paged_adamw_8bit','paged_lion_32bit','paged_lion_8bit'], value="paged_adamw_32bit",label='optimizer',info="Use an optimizer which aligns with the quantization of model")
                        lr_scheduler = gr.Radio(label="lr_scheduler", choices=['one_cycle', 'log_sweep', 'cosine'],value="cosine",info="Determines dynamic learning rate based on current step")
                        learning_rate = gr.Textbox(label="max_learning_rate", value="2e-5",info="")
                        train_on_inputs = gr.Checkbox(label="train_on_inputs", value=False,visible=False)
                        group_by_length = gr.Checkbox(label="group_by_length", value=False,visible=False)
                        bf16 = gr.Checkbox(label="bfloat16", value=False,info="Enable bfloat16 precision for tensors; supported only on Ampere or newer GPUs.")
                        fp16 = gr.Checkbox(label="Half Precision", value=True,info="Enable half precision (FP16) for tensor processing.")
                        tf32 = gr.Checkbox(label="TensorFloat32", value=False,info="Enable TensorFloat32 precision for tensors; supported only on Ampere or newer GPUs.")
                        gradient_checkpointing = gr.Checkbox(label="gradient_checkpointing", value=True,info='',visible=False)
                        resume_from_checkpoint = gr.Textbox(label="resume_from_checkpoint",visible=False)
                        local_rank = gr.Textbox(label="local_rank",visible=False)
                        logging_steps = gr.Slider(minimum=1, maximum=100, step=1,label="logging_steps", value=1,info='',visible=False)
                        xformers_attention = gr.Checkbox(label="xformers_attention", value=False,visible=False)
                        flash_attention = gr.Checkbox(label="flash_attention", value=False,info='',visible=False)
                        load_best_model_at_end = gr.Checkbox(label="load_best_model_at_end", value=False,visible=False)
                        warmup_steps = gr.Slider(minimum=1, maximum=100, step=1,label="warmup_steps", value=10,visible=False)
                        evals_per_epoch = gr.Slider(minimum=1, maximum=100, step=1,label="evals_per_epoch", value=4,info='No. of Evaluation Per Epoch',visible=False)
                        eval_table_size = gr.Textbox(label="eval_table_size",visible=False)
                        saves_per_epoch = gr.Slider(minimum=1, maximum=100, step=1,label="saves_per_epoch", value=1,info='No. of checkpoints to be saved')
                        
                        debug = gr.Checkbox(label="debug", value=False,visible=False)
                        
                        weight_decay = gr.Number(label="weight_decay", value=0.0,visible=False)
                        wandb_watch = gr.Checkbox(label="wandb_watch", value=False,visible=False)
                        wandb_log_model = gr.Checkbox(label="wandb_log_model", value=False,visible=False)
                        wandb_project = gr.Textbox(label="wandb_project",visible=False)
                        wandb_entity = gr.Textbox(label="wandb_entity",visible=False)
                        wandb_name = gr.Textbox(label="wandb_name",visible=False)
        
                        
                    with gr.Tab("Accelerate Configuration"):
                        Use_debug = gr.Checkbox(label="Use DeepSpeed", value=True)
                        deepspeed_config = gr.Radio(label="deepspeed_config", choices=['Zero 1', 'Zero 2',' Zero 3'],value="Zero 1")
    
                    # Continue adding the rest of the Accelerate config inputs similarly)
                    
                train_btn = gr.Button("Start Training")
                output_dataframe= gr.Dataframe(label="Training Results")
                output_graph = gr.Plot(label='Training Plot')
                output_graph2 = gr.Plot(label='Training Plot')
                train_btn.click(
                    self.train_model,
                    inputs=[max_steps, base_model, model_type, tokenizer_type, is_llama_derived_model,
                    strict, datasets_path, dataset_format, shards,
                    val_set_size, output_dir, adapter, lora_model_dir, sequence_len, sample_packing,
                    pad_to_sequence_len, lora_r, lora_alpha, lora_dropout,
                    lora_target_modules, lora_target_linear, lora_fan_in_fan_out, gradient_accumulation_steps,
                    micro_batch_size, num_epochs, optimizer, lr_scheduler, learning_rate, train_on_inputs,
                    group_by_length, bf16, fp16, tf32, gradient_checkpointing,
                    resume_from_checkpoint, local_rank, logging_steps, xformers_attention, flash_attention,
                    load_best_model_at_end, warmup_steps, evals_per_epoch, eval_table_size, saves_per_epoch,
                    debug, weight_decay, wandb_project, wandb_entity, wandb_watch,
                    wandb_name, wandb_log_model,Use_debug,deepspeed_config,last_tab],
                    outputs=[gr.Textbox(label="Training Output",interactive=False),output_dataframe,output_graph,output_graph2]
                )
    
                # Finetuning Tab
            with gr.Tab("FineTuning on Your Dataset"):
                base_model = gr.Textbox(label="base_model", value="./finetune-out/merged",visible=False)
                datasets_path = gr.Textbox(label="datasets_path", value="/src/Preprocess_Data.csv",visible=False)
                dataset_format = gr.Radio(choices=['Alpaca', 'Table'], label="Dataset Format", value='Alpaca')
                shards = gr.Slider(minimum=0, maximum=20, step=1, label="shards", value=10)
                last_tab = gr.Checkbox(label='last_tab',value=True,visible=False)
               
                with gr.Accordion("Advanced Settings",open=False):
                    with gr.Tab("YAML Configuration"):
                        model_type = gr.Textbox(label="model_type", value="LlamaForCausalLM",info="",visible=False)
                        tokenizer_type = gr.Textbox(label="tokenizer_type", value="LlamaTokenizer",visible=False)
                        is_llama_derived_model = gr.Checkbox(label="is_llama_derived_model", value=True,info="Determines the padding strategy based on the parent type of the model")
                        strict = gr.Checkbox(label="strict", value=False,visible=False)
                        val_set_size = gr.Slider(minimum=0, maximum=1, step=0.1, label="val_set_size", value=0.05,info="Percentage of training data to be used for validation")
                        output_dir = gr.Textbox(label="output_dir", value="./finetune-out",info="Output directory of the finetuned model")
                        adapter = gr.Radio(choices=["qlora", "lora"], label="adapter",value='qlora',info="Parameter efficient training strategy")
                        lora_model_dir = gr.Textbox(label="lora_model_dir",info="Directory of a custom adapter can be provided",visible=False)
                        sequence_len = gr.Slider(minimum=512, maximum=4096, step=10,label="sequence_len", value=1024,info="The maximum length input allowed to train")
                        sample_packing = gr.Checkbox(label="sample_packing", value=True,info="Speeds up data preparation but recommended false for small datasets")
                        pad_to_sequence_len = gr.Checkbox(label="pad_to_sequence_len", value=True, info="Pads the input to match sequence length to avoid memory fragmentation and out of memory issues. Recommended true")
                        # eval_sample_packing = gr.Checkbox(label="eval_sample_packing", value=False)
                        lora_r = gr.Slider(minimum=8, maximum=64, step=2,label="lora_r", value=32,info="The number of parameters in adaptation layers.")
                        lora_alpha = gr.Slider(minimum=8, maximum=64, step=0.1,label="lora_alpha", value=16,info="How much adapted weights affect base models")
                        lora_dropout = gr.Slider(minimum=0, maximum=1, label="lora_dropout", value=0.05, step=0.01,info="The ratio of weights ignored randomly within adapted weights")
                        lora_target_modules = gr.Textbox(label="lora_target_modules", value="q_proj, v_proj, k_proj",info="All dense layers can be targeted using parameter efficient tuning")
                        lora_target_linear = gr.Checkbox(label="lora_target_linear", value=True,info="Lora Target Modules will be ignored and all linear layers will be used")
                        lora_fan_in_fan_out = gr.Textbox(label="lora_fan_in_fan_out",visible=False)
                        
                        gradient_accumulation_steps = gr.Slider(minimum=4, maximum=64, step=1,label="gradient_accumulation_steps", value=4,info="Number of steps required to update the weights with cumulative gradients")
                        micro_batch_size = gr.Slider(minimum=1, maximum=64, step=2,label="micro_batch_size", value=2,info="Number of samples sent to each gpu")
                        num_epochs = gr.Slider(minimum=1, maximum=4, step=1,label="num_epochs", value=4)
                        max_steps = gr.Textbox(label="max_steps",value='1',info="Maximum number of steps to be trained. Overwrites the number of epochs",visible=False)
                        optimizer = gr.Radio(choices=["adamw_hf",'adamw_torch','adamw_torch_fused','adamw_torch_xla','adamw_apex_fused','adafactor','adamw_anyprecision','sgd','adagrad','adamw_bnb_8bit','lion_8bit','lion_32bit','paged_adamw_32bit','paged_adamw_8bit','paged_lion_32bit','paged_lion_8bit'], value="paged_adamw_32bit",label='optimizer',info="Use an optimizer which aligns with the quantization of model")
                        lr_scheduler = gr.Radio(label="lr_scheduler", choices=['one_cycle', 'log_sweep', 'cosine'],value="cosine",info="Determines dynamic learning rate based on current step")
                        learning_rate = gr.Textbox(label="max_learning_rate", value="2e-5",info="")
                        train_on_inputs = gr.Checkbox(label="train_on_inputs", value=False,visible=False)
                        group_by_length = gr.Checkbox(label="group_by_length", value=False,visible=False)
                        bf16 = gr.Checkbox(label="bfloat16", value=False,info="Enable bfloat16 precision for tensors; supported only on Ampere or newer GPUs.")
                        fp16 = gr.Checkbox(label="Half Precision", value=True,info="Enable half precision (FP16) for tensor processing.")
                        tf32 = gr.Checkbox(label="TensorFloat32", value=False,info="Enable TensorFloat32 precision for tensors; supported only on Ampere or newer GPUs.")
                        gradient_checkpointing = gr.Checkbox(label="gradient_checkpointing", value=True,info='',visible=False)
                        resume_from_checkpoint = gr.Textbox(label="resume_from_checkpoint",visible=False)
                        local_rank = gr.Textbox(label="local_rank",visible=False)
                        logging_steps = gr.Slider(minimum=1, maximum=100, step=1,label="logging_steps", value=1,info='',visible=False)
                        xformers_attention = gr.Checkbox(label="xformers_attention", value=False,visible=False)
                        flash_attention = gr.Checkbox(label="flash_attention", value=False,info='',visible=False)
                        load_best_model_at_end = gr.Checkbox(label="load_best_model_at_end", value=False,visible=False)
                        warmup_steps = gr.Slider(minimum=1, maximum=100, step=1,label="warmup_steps", value=10,visible=False)
                        evals_per_epoch = gr.Slider(minimum=1, maximum=100, step=1,label="evals_per_epoch", value=4,info='No. of Evaluation Per Epoch',visible=False)
                        eval_table_size = gr.Textbox(label="eval_table_size",visible=False)
                        saves_per_epoch = gr.Slider(minimum=1, maximum=100, step=1,label="saves_per_epoch", value=1,info='No. of checkpoints to be saved')
                        
                        debug = gr.Checkbox(label="debug", value=False,visible=False)
                        
                        weight_decay = gr.Number(label="weight_decay", value=0.0,visible=False)
                        wandb_watch = gr.Checkbox(label="wandb_watch", value=False,visible=False)
                        wandb_log_model = gr.Checkbox(label="wandb_log_model", value=False,visible=False)
                        wandb_project = gr.Textbox(label="wandb_project",visible=False)
                        wandb_entity = gr.Textbox(label="wandb_entity",visible=False)
                        wandb_name = gr.Textbox(label="wandb_name",visible=False)
        
                        
                    with gr.Tab("Accelerate Configuration"):
                        Use_debug = gr.Checkbox(label="Use DeepSpeed", value=True)
                        deepspeed_config = gr.Radio(label="deepspeed_config", choices=['Zero 1', 'Zero 2',' Zero 3'],value="Zero 1")
    
                    # Continue adding the rest of the Accelerate config inputs similarly)
                    
                train_btn = gr.Button("Start Training")
                output_dataframe= gr.Dataframe(label="Training Results")
                output_graph = gr.Plot(label='Training Plot')
                output_graph2 = gr.Plot(label='Training Plot')
                train_btn.click(
                    self.train_model,
                    inputs=[max_steps, base_model, model_type, tokenizer_type, is_llama_derived_model,
                    strict, datasets_path, dataset_format, shards,
                    val_set_size, output_dir, adapter, lora_model_dir, sequence_len, sample_packing,
                    pad_to_sequence_len, lora_r, lora_alpha, lora_dropout,
                    lora_target_modules, lora_target_linear, lora_fan_in_fan_out, gradient_accumulation_steps,
                    micro_batch_size, num_epochs, optimizer, lr_scheduler, learning_rate, train_on_inputs,
                    group_by_length, bf16, fp16, tf32, gradient_checkpointing,
                    resume_from_checkpoint, local_rank, logging_steps, xformers_attention, flash_attention,
                    load_best_model_at_end, warmup_steps, evals_per_epoch, eval_table_size, saves_per_epoch,
                    debug, weight_decay, wandb_project, wandb_entity, wandb_watch,
                    wandb_name, wandb_log_model,Use_debug,deepspeed_config,last_tab],
                    outputs=[gr.Textbox(label="Training Output",interactive=False),output_dataframe,output_graph,output_graph2]
                )
    
            
            # Prompt Testing Tab
            with gr.Tab("Testing"):
                model_name = gr.Textbox(label='Model Name',value='./finetune-out/merged',info='For Base Model Give Full Path')
                output = gr.Textbox(label="Model Loding Respose",interactive=False,value="Model Loaded")
                temperature = gr.Slider(minimum=0.1,maximum=1,label='temperature',value=0.75,step=0.1,info='Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.')
                top_p = gr.Slider(minimum=0.1,maximum=1,label='top_p',value=0.9,step=0.1,info='When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens.')
                max_tokens = gr.Slider(minimum=512,maximum=4096,label='max_tokens',value=1024,step=10,info='Maximum number of tokens to generate. A word is generally 2-3 tokens.')
                submit = gr.Button('Load Model')
                submit.click(self.utility.load_model,inputs=[model_name,temperature,top_p,max_tokens],outputs=output)
                with gr.Accordion(label="Text Completion Testing"):
                    with gr.Column():
                        with gr.Group():
                            gr.Markdown("### Prompt Test 1")
                            with gr.Row(equal_height=True):
                                with gr.Group():
                                    prompt_input_sys = gr.TextArea(label="System Prompt",max_lines=15,lines=15, value="""
                                                COT:
            
            To generate data for an individual based on specific age and workclass attributes, we'll need to carefully consider the guidelines provided, \
            which cover various socio-economic factors. These factors include age, workclass, education, marital status, occupation, and others, all of \
            which play a crucial role in determining an individual's income level and socio-economic status. Here's how we'll approach this:
        
            1. Age and Workclass: These are the primary attributes provided by the user. Age is directly related to experience and potentially, to \
            income level. Workclass categorizes the individual's employment sector, which can significantly influence earning potential.
            
            2. Other Attributes: While the user has specified age and workclass, other attributes like education, marital status, occupation, etc., \
            are not specified. We'll infer these based on typical correlations with the given age and workclass, adhering to statistical norms and logical \
            deductions. For instance, higher education levels often correlate with professional or white-collar jobs, and marital status can reflect stability, \
            influencing income.
            
            3. Socio-Economic Indicators: Capital gain, capital loss, and hours per week are direct indicators of financial status and work engagement. \
            We'll align these with the workclass and potential occupations to reflect realistic economic activities.
            
            4. Demographic Factors: Race, sex, and native country also affect socio-economic status due to systemic factors and demographic trends. \
            While adhering to the principle of equality, we acknowledge statistical disparities that exist and will consider these in generating a realistic \
            profile.
            
            5. Income Level: This is the outcome we're particularly interested in, determined by the interplay of all the above factors. The income level, \
            either ">50k" or "<=50k", will be deduced based on the cumulative effect of the individual's attributes, reflecting the guidelines' insights into \
            how each factor typically influences income.
            
            - For Age: Given the specified age, we consider typical life milestones and career stages associated with this age group, influencing education \
            level, marital status, and likely income bracket.
            - For Workclass: The workclass will guide our assumptions about the individual's occupation, hours worked per week, and potential for capital \
            gains, which are closely tied to the type of employment and industry standards.
            - Inference of Other Attributes: Based on age and workclass, we'll infer the most probable education level, marital status, and occupation. \
            For example, a younger individual in a professional workclass might be single and have a higher education level, while an older individual might \
            have more work experience and financial responsibilities.
            - Determining Income: Integrating all these factors, we'll estimate the income level, considering how each attribute statistically influences \
            earning potential It should be Either '>50k' or '<=50k'.
            
            This structured approach allows us to generate a comprehensive profile that not only meets the user's request but also provides insights into the \
            socio-economic dynamics at play. Let's proceed with creating the prompt based on these considerations. 
        
            Given the structured approach and chain of thought described, here is the prompt to execute with the "generate_data" function:
            {{
              "age": 45,
              "workclass": "Private",
              "fnlwgt": 120000,
              "education": "Bachelors",
              "education-num": 13,
              "marital-status": "Married-civ-spouse",
              "occupation": "Exec-managerial",
              "relationship": "Husband",
              "race": "White",
              "sex": "Male",
              "capital-gain": 5000,
              "capital-loss": 0,
              "hours-per-week": 40,
              "native-country": "United States",
              "income": ">50k"
            }}
        
            This JSON object represents a data generation request for an individual who is 45 years old and works in the private sector. \
            The selections for education, marital status, occupation, and other factors are inferred based on typical correlations with the \
            specified age and workclass, aimed at generating a realistic socio-economic profile leading to an income level of ">50k".
            
            You have access to the following function. Use them if required:
            {{
              "type": "function",
              "function": {{
                "name": "generate_data",
                "description": "Generates data in a specified format.",
                "parameters": {{
                  "type": "object",
                  "properties": {{
                    "age": {{"type": "integer", "description": "Age"}},
                    "workclass": {{"type": "string", "description": "Workclass type"}},
                    "fnlwgt": {{"type": "integer", "description": "fnlwgt as per the COT"}},
                    "education": {{"type": "string", "description": "Education level as per the COT"}},
                    "education-num": {{"type": "integer", "description": "Education numeric value as per the COT"}},
                    "marital-status": {{"type": "string", "description": "Marital status as per the COT"}},
                    "occupation": {{"type": "string", "description": "Occupation as per the COT"}},
                    "relationship": {{"type": "string", "description": "Relationship as per the COT"}},
                    "race": {{"type": "string", "description": "Race as per the COT"}},
                    "sex": {{"type": "string", "description": "Sex as per the COT"}},
                    "capital-gain": {{"type": "integer", "description": "Capital gain as per the COT"}},
                    "capital-loss": {{"type": "integer", "description": "Capital loss as per the COT"}},
                    "hours-per-week": {{"type": "integer", "description": "Weekly working hours as per the COT"}},
                    "native-country": {{"type": "string", "description": "Name Of Native country as per the COT"}},
                    "income": {{"type": " ">50k" or "<=50k" ", "description": "Either ">50k" or "<=50k" "}}
                  }},
                  "required": [
                    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", 
                    "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
                  ]
                }}
              }}
            }}
                                    """)
                                    prompt_input_user = gr.TextArea(label="User Prompt",lines=5,value="Use the generate_data function to generate Data for a person of age 39 and workclass Private and fnlwgt value 1020")
                                    prompt_input_assi =  gr.TextArea(label="Assistant Prompt",lines=1,value="""{"age": 39,"workplace": "Private", "fnlwgt": """)
                                    submit = gr.Button("Submit")
                                with gr.Group():
                                    output = gr.TextArea(label="Output",lines=25)
                                    score= gr.Textbox(label="Hallucination Score",info="Increase tokens or assistant response is not closing if negative. Decrease tokens if positive",value='Hallucination value')
                                    submit.click(self.utility.single_prompt, inputs=[prompt_input_sys,prompt_input_user,prompt_input_assi], outputs=[output,score])
                        with gr.Group():
                            gr.Markdown("### Prompt Test 2")
                            with gr.Row(equal_height=True):
                                with gr.Group():
                                    prompt_input_sys_2 = gr.TextArea(label="System Prompt",max_lines=15,lines=15)
                                    prompt_input_user_2 = gr.TextArea(label="User Prompt",lines=5)
                                    prompt_input_assi_2 =  gr.TextArea(label="Assistant Prompt",lines=1)
                                    submit_2 = gr.Button("Submit")
                                with gr.Group():
                                    output_2 = gr.TextArea(label="Output",lines=25)
                                    score_2= gr.Textbox(label="Hallucination Score",info="Increase tokens or assistant response is not closing if negative. Decrease tokens if positive",value='Hallucination value')
                                    submit_2.click(self.utility.single_prompt, inputs=[prompt_input_sys_2,prompt_input_user_2,prompt_input_assi_2], outputs=[output_2,score_2])
                        with gr.Group():
                            gr.Markdown("### Prompt Test 3")
                            with gr.Row(equal_height=True):
                                with gr.Group():
                                    prompt_input_sys_3 = gr.TextArea(label="System Prompt",max_lines=15,lines=15)
                                    prompt_input_user_3 = gr.TextArea(label="User Prompt",lines=5)
                                    prompt_input_assi_3 =  gr.TextArea(label="Assistant Prompt",lines=1)
                                    submit_3 = gr.Button("Submit")
                                with gr.Group():
                                    output_3 = gr.TextArea(label="Output",lines=25)
                                    score_3= gr.Textbox(label="Hallucination Score",info="Increase tokens or assistant response is not closing if negative. Decrease tokens if positive",value='Hallucination value')
                                    submit_3.click(self.utility.single_prompt, inputs=[prompt_input_sys_3,prompt_input_user_3,prompt_input_assi_3], outputs=[output_3,score_3])
                        with gr.Group():
                            gr.Markdown("### Prompt Test 4")
                            with gr.Row(equal_height=True):
                                with gr.Group():
                                    prompt_input_sys_4 = gr.TextArea(label="System Prompt",max_lines=15,lines=15)
                                    prompt_input_user_4 = gr.TextArea(label="User Prompt",lines=5)
                                    prompt_input_assi_4 =  gr.TextArea(label="Assistant Prompt",lines=1)
                                    submit_4 = gr.Button("Submit")
                                with gr.Group():
                                    output_4 = gr.TextArea(label="Output",lines=25)
                                    score_4= gr.Textbox(label="Hallucination Score",info="Increase tokens or assistant response is not closing if negative. Decrease tokens if positive",value='Hallucination value')
                                    submit_4.click(self.utility.single_prompt, inputs=[prompt_input_sys_4,prompt_input_user_4,prompt_input_assi_4], outputs=[output_4,score_4])
    
            
            # Synthetic Dataset Generation Tab
            with gr.Tab("Structured Dataset Generation"):
                dataset_type = gr.Radio(choices=['CSV', 'JSON'], label="File Type", value='CSV')
                with gr.Tab("Input From huggingface",visible=False) as Tab_HuggingFace_syn:
                    dataset_path = gr.Textbox(label="Dataset Path", value="mhenrichsen/alpaca_2k_test")
                    split = gr.Textbox(label="Split", value="train")
                with gr.Tab("Upload CSV",visible = True) as Tab_CSV_syn:
                    uploaded_file_csv = gr.File(label="File Type", file_types=['.csv','.csv.gz'])
                with gr.Tab("Upload JSON",visible = False) as Tab_JSON_syn:
                    uploaded_file_json = gr.File(label="File Type", file_types=['.json'])
                input_prompt_sys = gr.TextArea(label="System prompt",lines=15,value="""
                                                COT:
            
            To generate data for an individual based on specific age and workclass attributes, we'll need to carefully consider the guidelines provided, \
            which cover various socio-economic factors. These factors include age, workclass, education, marital status, occupation, and others, all of \
            which play a crucial role in determining an individual's income level and socio-economic status. Here's how we'll approach this:
        
            1. Age and Workclass: These are the primary attributes provided by the user. Age is directly related to experience and potentially, to \
            income level. Workclass categorizes the individual's employment sector, which can significantly influence earning potential.
            
            2. Other Attributes: While the user has specified age and workclass, other attributes like education, marital status, occupation, etc., \
            are not specified. We'll infer these based on typical correlations with the given age and workclass, adhering to statistical norms and logical \
            deductions. For instance, higher education levels often correlate with professional or white-collar jobs, and marital status can reflect stability, \
            influencing income.
            
            3. Socio-Economic Indicators: Capital gain, capital loss, and hours per week are direct indicators of financial status and work engagement. \
            We'll align these with the workclass and potential occupations to reflect realistic economic activities.
            
            4. Demographic Factors: Race, sex, and native country also affect socio-economic status due to systemic factors and demographic trends. \
            While adhering to the principle of equality, we acknowledge statistical disparities that exist and will consider these in generating a realistic \
            profile.
            
            5. Income Level: This is the outcome we're particularly interested in, determined by the interplay of all the above factors. The income level, \
            either ">50k" or "<=50k", will be deduced based on the cumulative effect of the individual's attributes, reflecting the guidelines' insights into \
            how each factor typically influences income.
            
            - For Age: Given the specified age, we consider typical life milestones and career stages associated with this age group, influencing education \
            level, marital status, and likely income bracket.
            - For Workclass: The workclass will guide our assumptions about the individual's occupation, hours worked per week, and potential for capital \
            gains, which are closely tied to the type of employment and industry standards.
            - Inference of Other Attributes: Based on age and workclass, we'll infer the most probable education level, marital status, and occupation. \
            For example, a younger individual in a professional workclass might be single and have a higher education level, while an older individual might \
            have more work experience and financial responsibilities.
            - Determining Income: Integrating all these factors, we'll estimate the income level, considering how each attribute statistically influences \
            earning potential It should be Either '>50k' or '<=50k'.
            
            This structured approach allows us to generate a comprehensive profile that not only meets the user's request but also provides insights into the \
            socio-economic dynamics at play. Let's proceed with creating the prompt based on these considerations. 
        
            Given the structured approach and chain of thought described, here is the prompt to execute with the "generate_data" function:
            {{
              "age": 45,
              "workclass": "Private",
              "fnlwgt": 120000,
              "education": "Bachelors",
              "education-num": 13,
              "marital-status": "Married-civ-spouse",
              "occupation": "Exec-managerial",
              "relationship": "Husband",
              "race": "White",
              "sex": "Male",
              "capital-gain": 5000,
              "capital-loss": 0,
              "hours-per-week": 40,
              "native-country": "United States",
              "income": ">50k"
            }}
        
            This JSON object represents a data generation request for an individual who is 45 years old and works in the private sector. \
            The selections for education, marital status, occupation, and other factors are inferred based on typical correlations with the \
            specified age and workclass, aimed at generating a realistic socio-economic profile leading to an income level of ">50k".
            
            You have access to the following function. Use them if required:
            {{
              "type": "function",
              "function": {{
                "name": "generate_data",
                "description": "Generates data in a specified format.",
                "parameters": {{
                  "type": "object",
                  "properties": {{
                    "age": {{"type": "integer", "description": "Age"}},
                    "workclass": {{"type": "string", "description": "Workclass type"}},
                    "fnlwgt": {{"type": "integer", "description": "fnlwgt as per the COT"}},
                    "education": {{"type": "string", "description": "Education level as per the COT"}},
                    "education-num": {{"type": "integer", "description": "Education numeric value as per the COT"}},
                    "marital-status": {{"type": "string", "description": "Marital status as per the COT"}},
                    "occupation": {{"type": "string", "description": "Occupation as per the COT"}},
                    "relationship": {{"type": "string", "description": "Relationship as per the COT"}},
                    "race": {{"type": "string", "description": "Race as per the COT"}},
                    "sex": {{"type": "string", "description": "Sex as per the COT"}},
                    "capital-gain": {{"type": "integer", "description": "Capital gain as per the COT"}},
                    "capital-loss": {{"type": "integer", "description": "Capital loss as per the COT"}},
                    "hours-per-week": {{"type": "integer", "description": "Weekly working hours as per the COT"}},
                    "native-country": {{"type": "string", "description": "Name Of Native country as per the COT"}},
                    "income": {{"type": " ">50k" or "<=50k" ", "description": "Either ">50k" or "<=50k" "}}
                  }},
                  "required": [
                    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", 
                    "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
                  ]
                }}
              }}
            }}
                                    """)
                input_prompt_user = gr.TextArea(label="User prompt",lines=2,value="Use the generate_data function to generate Data for a person of age 39 and workclass Private and fnlwgt value 1020")
                output_format = gr.Radio(choices=['RealTabFormer','Tabula'],label='Output Format',value='RealTabFormer')
                
    
                def check_syn(value):
                    dict_v = {Tab_CSV_syn: gr.Tab(visible = value == 'CSV'),
                            Tab_JSON_syn: gr.Tab(visible= value == 'JSON'),}
                    return dict_v
                dataset_type.select(check_syn,inputs=dataset_type,outputs=[Tab_CSV_syn,Tab_JSON_syn])
                num_samples_input = gr.Number(label="Num Samples", value=40)
                batch_size_input = gr.Number(label="Batch Size", value=20)
                generate_button = gr.Button("Generate Dataset")
                output_dataframe = gr.Dataframe(label="Output dataframe")
                output_data = gr.File(label="Output File")
                generate_button.click(
                    self.utility.generate_dataset,
                    inputs=[dataset_type,dataset_path,split,uploaded_file_csv,uploaded_file_json,num_samples_input,batch_size_input,input_prompt_sys,input_prompt_user, output_format],
                    outputs=[output_dataframe,output_data]
                )
                eval_button = gr.Button("Evaluate")
                output_graph1 = gr.Plot(label='Graph')
                output_graph2 = gr.Plot(label='Graph')
                output_graph3 = gr.Plot(label='Graph')
                output_graph4 = gr.Plot(label='Graph')
                output_graph5 = gr.Plot(label='Graph')
                output_graph6 = gr.Plot(label='Graph')
                eval_button.click(
                    self.utility.evaluate,
                    inputs=[],
                    outputs=[output_graph1,output_graph2,output_graph3,output_graph4,output_graph5,output_graph6]
                )
    
                
        return self.app

   
if __name__ == "__main__":
    main=Main()
    app=main.initiate_userInterface()                   
    app.queue().launch(share=True,server_name='0.0.0.0')




# info="Format Should be: ' [INST] <<SYS>> System Prompt <<SYS>> <<USER>> user prompt <<USER>> [/INST] '",value='''[INST] <<SYS>>
            # COT:
            
            # To generate data for an individual based on specific age and workclass attributes, we'll need to carefully consider the guidelines provided, \
            # which cover various socio-economic factors. These factors include age, workclass, education, marital status, occupation, and others, all of \
            # which play a crucial role in determining an individual's income level and socio-economic status. Here's how we'll approach this:
        
            # 1. Age and Workclass: These are the primary attributes provided by the user. Age is directly related to experience and potentially, to \
            # income level. Workclass categorizes the individual's employment sector, which can significantly influence earning potential.
            
            # 2. Other Attributes: While the user has specified age and workclass, other attributes like education, marital status, occupation, etc., \
            # are not specified. We'll infer these based on typical correlations with the given age and workclass, adhering to statistical norms and logical \
            # deductions. For instance, higher education levels often correlate with professional or white-collar jobs, and marital status can reflect stability, \
            # influencing income.
            
            # 3. Socio-Economic Indicators: Capital gain, capital loss, and hours per week are direct indicators of financial status and work engagement. \
            # We'll align these with the workclass and potential occupations to reflect realistic economic activities.
            
            # 4. Demographic Factors: Race, sex, and native country also affect socio-economic status due to systemic factors and demographic trends. \
            # While adhering to the principle of equality, we acknowledge statistical disparities that exist and will consider these in generating a realistic \
            # profile.
            
            # 5. Income Level: This is the outcome we're particularly interested in, determined by the interplay of all the above factors. The income level, \
            # either ">50k" or "<=50k", will be deduced based on the cumulative effect of the individual's attributes, reflecting the guidelines' insights into \
            # how each factor typically influences income.
            
            # - For Age: Given the specified age, we consider typical life milestones and career stages associated with this age group, influencing education \
            # level, marital status, and likely income bracket.
            # - For Workclass: The workclass will guide our assumptions about the individual's occupation, hours worked per week, and potential for capital \
            # gains, which are closely tied to the type of employment and industry standards.
            # - Inference of Other Attributes: Based on age and workclass, we'll infer the most probable education level, marital status, and occupation. \
            # For example, a younger individual in a professional workclass might be single and have a higher education level, while an older individual might \
            # have more work experience and financial responsibilities.
            # - Determining Income: Integrating all these factors, we'll estimate the income level, considering how each attribute statistically influences \
            # earning potential It should be Either '>50k' or '<=50k'.
            
            # This structured approach allows us to generate a comprehensive profile that not only meets the user's request but also provides insights into the \
            # socio-economic dynamics at play. Let's proceed with creating the prompt based on these considerations. 
        
            # Given the structured approach and chain of thought described, here is the prompt to execute with the "generate_data" function:
            # {{
            #   "age": 45,
            #   "workclass": "Private",
            #   "fnlwgt": 120000,
            #   "education": "Bachelors",
            #   "education-num": 13,
            #   "marital-status": "Married-civ-spouse",
            #   "occupation": "Exec-managerial",
            #   "relationship": "Husband",
            #   "race": "White",
            #   "sex": "Male",
            #   "capital-gain": 5000,
            #   "capital-loss": 0,
            #   "hours-per-week": 40,
            #   "native-country": "United States",
            #   "income": ">50k"
            # }}
        
            # This JSON object represents a data generation request for an individual who is 45 years old and works in the private sector. \
            # The selections for education, marital status, occupation, and other factors are inferred based on typical correlations with the \
            # specified age and workclass, aimed at generating a realistic socio-economic profile leading to an income level of ">50k".
            
            # You have access to the following function. Use them if required:
            # {{
            #   "type": "function",
            #   "function": {{
            #     "name": "generate_data",
            #     "description": "Generates data in a specified format.",
            #     "parameters": {{
            #       "type": "object",
            #       "properties": {{
            #         "age": {{"type": "integer", "description": "Age"}},
            #         "workclass": {{"type": "string", "description": "Workclass type"}},
            #         "fnlwgt": {{"type": "integer", "description": "fnlwgt as per the COT"}},
            #         "education": {{"type": "string", "description": "Education level as per the COT"}},
            #         "education-num": {{"type": "integer", "description": "Education numeric value as per the COT"}},
            #         "marital-status": {{"type": "string", "description": "Marital status as per the COT"}},
            #         "occupation": {{"type": "string", "description": "Occupation as per the COT"}},
            #         "relationship": {{"type": "string", "description": "Relationship as per the COT"}},
            #         "race": {{"type": "string", "description": "Race as per the COT"}},
            #         "sex": {{"type": "string", "description": "Sex as per the COT"}},
            #         "capital-gain": {{"type": "integer", "description": "Capital gain as per the COT"}},
            #         "capital-loss": {{"type": "integer", "description": "Capital loss as per the COT"}},
            #         "hours-per-week": {{"type": "integer", "description": "Weekly working hours as per the COT"}},
            #         "native-country": {{"type": "string", "description": "Name Of Native country as per the COT"}},
            #         "income": {{"type": " ">50k" or "<=50k" ", "description": "Either ">50k" or "<=50k" "}}
            #       }},
            #       "required": [
            #         "age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", 
            #         "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
            #       ]
            #     }}
            #   }}
            # }}
#             <</SYS>> 
#             <<USER>> Use the generate_data function to generate Data for a person of age 39 and workclass Private and \
#             fnlwgt value 1020<</USER>>
#             [/INST] 
# '''
